import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class ImageDataset(Dataset):
    def __init__(self,
                 data_dir_path,
                 color_mode="L",
                 target_size=None,
                 normalize=True,
                 image_file_extensions=('.png', '.jpg', '.jpeg')):
        self.data_dir_path = data_dir_path
        self.color_mode = color_mode
        self.target_size = target_size
        self.normalize = normalize

        # Getting a list of all image files
        self.image_files = sorted(
            f for f in os.listdir(self.data_dir_path)
            if f.lower().endswith(image_file_extensions) and not f.startswith(".")
        )

        # Loading the images into memory
        images = []
        for image_file in tqdm(self.image_files, desc="Loading images"):
            image_path = os.path.join(self.data_dir_path, image_file)

            img = Image.open(image_path)

            # Changing to the desired target size
            if self.target_size is not None:
                img = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)

            # Converting to the proper color space
            img = img.convert(self.color_mode)

            # Converting over to a numpy array and normalizing
            img_array = np.array(img, dtype=np.float32)
            if self.normalize:
                img_array = img_array / 255.0

            # Appending and closing the image
            images.append(img_array)
            img.close()

        # Stacking the images and convert to a torch tensor
        image_stack = np.stack(images, axis=0)
        self.images = torch.from_numpy(image_stack)

        if self.color_mode == "RGB":
            # (n, h, w, c) -> (n, c, h, w)
            self.images = self.images.permute(0, 3, 1, 2)  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
            
