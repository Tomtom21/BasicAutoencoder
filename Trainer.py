import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class Trainer:
    def __init__(self, 
                 model: nn.Module, 
                 train_dataset: Dataset, 
                 test_dataset: Dataset,
                 loss_func: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_dir: Path):

        # Defining our argument parser
        self.parser = argparse.ArgumentParser(description="Train a neural network")
        self.parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
        self.parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
        self.args = self.parser.parse_args()

        # Getting the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Saving our parameters
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.save_dir = save_dir

        # Move datasets to the device
        train_dataset.images = train_dataset.images.to(self.device)
        test_dataset.images = test_dataset.images.to(self.device)

        # Create data loaders
        self.train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

        # Loss tracking
        self.loss_list = []
        self.test_loss_list = []

    def _batch_preprocessing(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Override class for any batch preprocessing
        (e.g. unsqueezing the batch)
        """
        return batch

    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch and return the average loss."""
        self.model.train()
        epoch_loss = 0.0

        with tqdm(total=len(self.train_dataset), desc=f"Epoch {epoch + 1}/{self.args.epochs}") as pbar:
            for batch in self.train_loader:
                # (b, 1, 32, 32)
                batch = self._batch_preprocessing(batch)
                self.model.zero_grad()

                output = self.model(batch)
                loss = self.loss_func(output, batch)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * batch.shape[0]
                pbar.update(batch.shape[0])

        return epoch_loss / len(self.train_loader.dataset)

    def _eval_epoch(self, epoch: int) -> float:
        """Evaluate on the test set and return the average loss."""
        self.model.eval()
        test_loss = 0.0

        with torch.no_grad():
            with tqdm(total=len(self.test_dataset), desc=f"Epoch {epoch + 1}/{self.args.epochs} (test)") as pbar:
                for batch in self.test_loader:
                    # (b, 1, 32, 32)
                    batch = self._batch_preprocessing(batch)

                    output = self.model(batch)
                    loss = self.loss_func(output, batch)
                    test_loss += loss.item() * batch.shape[0]
                    pbar.update(batch.shape[0])

        return test_loss / len(self.test_loader.dataset)

    def train(self):
        """The main training loop for the autoencoder"""
        print("Beginning training...")
        for epoch in range(self.args.epochs):
            train_loss = self._train_epoch(epoch)
            test_loss = self._eval_epoch(epoch)

            self.loss_list.append(train_loss)
            self.test_loss_list.append(test_loss)
            print(f"Epoch {epoch + 1} - train loss: {round(train_loss, 5)}  test loss: {round(test_loss, 5)}")

        # Plot and save results
        self._plot_losses()
        self._save_model()

    def _plot_losses(self):
        """Plot and save the training and test loss curves."""
        epochs = range(1, self.args.epochs + 1)
        plt.figure()
        plt.plot(epochs, self.loss_list, label="Train loss")
        plt.plot(epochs, self.test_loss_list, label="Test loss")

        plt.xlabel("Epoch")
        plt.ylabel("Mean loss")
        plt.title("Training and Test Loss")
        plt.legend()
        plt.tight_layout()

        plot_path = self.save_dir / "loss.png"
        plt.savefig(plot_path)
        print(f"Loss plot saved to {plot_path}")

    def _save_model(self):
        """Save the trained model."""
        model_path = self.save_dir / "model.pth"
        torch.save(self.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
