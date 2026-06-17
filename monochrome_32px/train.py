from pathlib import Path

import torch

from Trainer import Trainer
from monochrome_32px.Monochrome32Dataset import Monochrome32Dataset
from monochrome_32px.model import MonochromeAutoEncoder


if __name__ == "__main__":
    # Load datasets
    dataset_dir = Path(__file__).parent.parent / "dataset"
    train_dataset = Monochrome32Dataset(dataset_dir / "train")
    test_dataset = Monochrome32Dataset(dataset_dir / "test")

    # Create model
    model = MonochromeAutoEncoder()

    # Create loss function and optimizer
    loss_func = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Save directory
    save_dir = Path(__file__).parent

    # Create and run trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        loss_func=loss_func,
        optimizer=optimizer,
        save_dir=save_dir
    )
    trainer.train()
