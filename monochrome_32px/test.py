import argparse
import random
from pathlib import Path

import torch

from Tester import Tester
from monochrome_32px.Monochrome32Dataset import Monochrome32Dataset
from monochrome_32px.model import MonochromeAutoEncoder


class MonochromeTester(Tester):
    def _batch_preprocessing(self, batch: torch.Tensor):
        """Overriding the method from the Tester class"""
        return batch.unsqueeze(1)


if __name__ == "__main__":
    # Load validation dataset
    dataset_dir = Path(__file__).parent.parent / "dataset"
    val_dataset = Monochrome32Dataset(dataset_dir / "validation")

    # Create model
    model = MonochromeAutoEncoder()

    # Model path
    model_path = Path(__file__).parent / "model.pth"

    # Create and run tester
    tester = MonochromeTester(
        model=model,
        val_dataset=val_dataset,
        model_path=model_path,
    )
    tester.test()
