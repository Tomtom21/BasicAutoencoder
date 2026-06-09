import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from Monochrome32Dataset import Monochrome32Dataset
from model import MonochromeAutoEncoder

# Parsing arguments
parser = argparse.ArgumentParser(description="Train the Monochrome 32x32 Autoencoder")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
args = parser.parse_args()

# Getting the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Creating the dataset objects
dataset_dir = Path(__file__).parent.parent / "dataset"
train_dataset = Monochrome32Dataset(dataset_dir / "train")
train_dataset.images = train_dataset.images.to(device)
test_dataset = Monochrome32Dataset(dataset_dir / "test")
test_dataset.images = test_dataset.images.to(device)

# Setting up the data loaders for training
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# Preparing the model
print("Dataset loaded, loading model...")
model = MonochromeAutoEncoder()
model.to(device)

# Training
print("Beginning training...")
loss_func = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_list = []
test_loss_list = []

for epoch in range(args.epochs):
    epoch_loss = 0.0

    # Setting up a pbar for progress tracking
    with tqdm(total=len(train_dataset), desc=f"Epoch {epoch + 1}/{args.epochs}") as pbar:
        for batch in train_loader:
            # For an autoencoder the target is the input itself
            batch = batch.unsqueeze(1)  # (B, 1, 32, 32)
            model.zero_grad()

            output = model(batch)
            loss = loss_func(output, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch.shape[0]
            pbar.update(batch.shape[0])

    # Evaluating on the test set
    test_loss = 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_dataset), desc=f"Epoch {epoch + 1}/{args.epochs} (test)") as pbar:
            for batch in test_loader:
                batch = batch.unsqueeze(1)  # (B, 1, 32, 32)

                output = model(batch)
                loss = loss_func(output, batch)
                test_loss += loss.item() * batch.shape[0]
                pbar.update(batch.shape[0])
    model.train()

    # Keeping track of the losses
    loss_list.append(epoch_loss / len(train_dataset))
    test_loss_list.append(test_loss / len(test_dataset))
    print(f"Epoch {epoch + 1} - train loss: {round(epoch_loss / len(train_dataset), 5)}  test loss: {round(test_loss / len(test_dataset), 5)}")

# Plotting the loss curves
epochs = range(1, args.epochs + 1)
plt.figure()
plt.plot(epochs, loss_list, label="Train loss")
plt.plot(epochs, test_loss_list, label="Test loss")

plt.xlabel("Epoch")
plt.ylabel("Mean loss")
plt.title("Training and Test Loss")
plt.legend()
plt.tight_layout()

plot_path = Path(__file__).parent / "loss.png"
plt.savefig(plot_path)
print(f"Loss plot saved to {plot_path}")

# Saving the model
model_path = Path(__file__).parent / "model.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")
