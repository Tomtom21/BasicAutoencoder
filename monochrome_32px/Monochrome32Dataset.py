import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

class Monochrome32Dataset(Dataset):
    def __init__(self, data_dir_path):
        self.data_dir_path = data_dir_path
        self.image_files = sorted(
            f for f in os.listdir(self.data_dir_path)
            if f.endswith(".png") and not f.startswith(".")
        )

        images = []
        for image_file in tqdm(self.image_files, desc="Loading images"):
            image_path = os.path.join(self.data_dir_path, image_file)
            
            # Loading the image as a grayscale image and normalizing to [0, 1]
            img = Image.open(image_path).convert("L")
            img_array = np.array(img, dtype=np.float32) / 255.0
            
            images.append(img_array)
            img.close()

        self.images = torch.from_numpy(np.stack(images, axis=0))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]

