from pathlib import Path

import torch

from Trainer import Trainer
from color_32px.Color32Dataset import Color32Dataset
from color_32px.model import ColorAutoEncoder

if __name__ == "__main__":
    # Load datasets
    dataset_dir = Path(__file__).parent.parent / "dataset"
    train_dataset = Color32Dataset(dataset_dir / "train")
    test_dataset = Color32Dataset(dataset_dir / "test")

    # Create model
    model = ColorAutoEncoder()

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
