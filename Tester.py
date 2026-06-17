import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset


class Tester:
    def __init__(self,
                 model: nn.Module,
                 val_dataset: Dataset,
                 model_path: Path):

        # Defining our argument parser
        self.parser = argparse.ArgumentParser(description="Visualise autoencoder on validation images")
        self.parser.add_argument("--n", type=int, default=8, help="Number of images to display")
        self.args = self.parser.parse_args()

        # Getting the device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Saving our parameters
        self.model = model.to(self.device)
        self.val_dataset = val_dataset

        # Move dataset to device
        self.val_dataset.images = self.val_dataset.images.to(self.device)

        # Load model weights
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def _batch_preprocessing(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Override class for any batch preprocessing
        (e.g. unsqueezing the batch)
        """
        return batch

    def _select_images(self) -> torch.Tensor:
        """Select random images from the validation dataset."""
        n = min(self.args.n, len(self.val_dataset))
        indices = random.sample(range(len(self.val_dataset)), n)
        return self.val_dataset.images[indices]

    def _reconstruct_images(self, images: torch.Tensor) -> torch.Tensor:
        """Run the model on the selected images"""
        with torch.no_grad():
            images = self._batch_preprocessing(images)
            output = self.model(images)
        return output

    def _plot_comparison(self, originals: torch.Tensor, reconstructed: torch.Tensor):
        """Plot original vs reconstructed images"""
        n = originals.shape[0]
        fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))

        # Determine if images are grayscale or color based on channel dimension
        # Images are in (batch, channels, height, width) format
        num_channels = originals.shape[1]
        is_grayscale = num_channels == 1
        cmap = "gray" if is_grayscale else None

        for i in range(n):
            orig_img = originals[i].cpu()
            recon_img = reconstructed[i].cpu()

            # converting from (c, h, w) to correct format
            if num_channels == 1:
                # if grayscale, squeeze channel dimension
                orig_img = orig_img.squeeze(0)
                recon_img = recon_img.squeeze(0)
            else:
                # if rgb, convert from (c, h, w) to (h, w, c)
                orig_img = orig_img.permute(1, 2, 0)
                recon_img = recon_img.permute(1, 2, 0)

            axes[0, i].imshow(orig_img, cmap=cmap, vmin=0, vmax=1)
            axes[0, i].axis("off")
            axes[1, i].imshow(recon_img, cmap=cmap, vmin=0, vmax=1)
            axes[1, i].axis("off")

        axes[0, 0].set_title("Original", loc="left", fontsize=9)
        axes[1, 0].set_title("Reconstructed", loc="left", fontsize=9)
        plt.suptitle("Autoencoder Output Comparison")
        plt.tight_layout()
        plt.show()

    def test(self):
        """Main testing function that selects images, reconstructs them, and plots."""
        print("Selecting random images...")
        images = self._select_images()

        print("Reconstructing images...")
        reconstructed = self._reconstruct_images(images)

        print("Plotting comparison...")
        self._plot_comparison(images, reconstructed)
