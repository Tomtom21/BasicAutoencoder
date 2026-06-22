import argparse
import random
from pathlib import Path

import torch

from Tester import Tester
from color_32px.Color32Dataset import Color32Dataset
from color_32px.model import ColorAutoEncoder


if __name__ == "__main__":
    # Load validation dataset
    dataset_dir = Path(__file__).parent.parent / "dataset"
    val_dataset = Color32Dataset(dataset_dir / "validation")

    # Create model
    model = ColorAutoEncoder()

    # Model path
    model_path = Path(__file__).parent / "model.pth"

    # Create and run tester
    tester = Tester(
        model=model,
        val_dataset=val_dataset,
        model_path=model_path,
    )
    tester.test()
