from pathlib import Path
import random
from tqdm import tqdm
from PIL import Image

from ImageGenerator import ImageGenerator

# Creating the dataset directory structure
script_dir = Path(__file__).parent
dataset_dir = script_dir / "dataset"

# Defining our image generator
image_generator = ImageGenerator(image_size=(32, 32))

# Asking the user how many images to generate total (Assuming train, test, split of 80-10-10)
total_images = int(input("Enter the total number of images to generate: "))
image_counts = {
    "train": int(total_images * 0.8),
    "test": int(total_images * 0.1),
    "validation": total_images - int(total_images * 0.8) - int(total_images * 0.1)
}

print(f"Generating {image_counts['train']} training images, {image_counts['test']} test images, and {image_counts['validation']} validation images.")

# Looping through our splits and generating images for each one
for split in ("train", "test", "validation"):
    # Making sure that the directory for this split exists
    (dataset_dir / split).mkdir(parents=True, exist_ok=True)
    for _ in tqdm(range(image_counts[split]), desc=f"Generating {split} images"):
        # Generating a random image background
        generate_background = random.choice([
            image_generator.generate_random_solid_background,
            image_generator.generate_random_gradient_background,
        ])
        background = generate_background()

        # Adding randomized text to the image (1 char in this case)
        image = image_generator.add_random_text(background, min_chars=1, max_chars=1)

        # Saving the image to the appropriate directory
        image_path = dataset_dir / split / f"{split}_{_}.png"
        Image.fromarray(image).save(image_path)
