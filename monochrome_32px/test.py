import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from Monochrome32Dataset import Monochrome32Dataset
from model import MonochromeAutoEncoder

parser = argparse.ArgumentParser(description="Visualise autoencoder reconstructions on validation images")
parser.add_argument("--n", type=int, default=8, help="Number of images to display")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load validation dataset
dataset_dir = Path(__file__).parent.parent / "dataset"
val_dataset = Monochrome32Dataset(dataset_dir / "validation")

# Load model
model = MonochromeAutoEncoder()
model.load_state_dict(torch.load(Path(__file__).parent / "model.pth", map_location=device))
model.to(device)
model.eval()

# Pick a random handful of images
n = min(args.n, len(val_dataset))
indices = random.sample(range(len(val_dataset)), n)
images = val_dataset.images[indices].to(device)  # (n, 32, 32)

with torch.no_grad():
    reconstructions = model(images.unsqueeze(1))  # (n, 1, 32, 32)

# Plot originals vs reconstructions
fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
for i in range(n):
    axes[0, i].imshow(images[i].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[0, i].axis("off")
    axes[1, i].imshow(reconstructions[i, 0].cpu(), cmap="gray", vmin=0, vmax=1)
    axes[1, i].axis("off")

axes[0, 0].set_title("Original", loc="left", fontsize=9)
axes[1, 0].set_title("Reconstructed", loc="left", fontsize=9)
plt.suptitle("Autoencoder Validation Reconstructions")
plt.tight_layout()
plt.show()
