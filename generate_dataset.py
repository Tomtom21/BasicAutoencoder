from pathlib import Path
import numpy as np

# Creating the dataset directory structure
script_dir = Path(__file__).parent
dataset_dir = script_dir / "dataset"

for split in ("training", "test", "validation"):
    (dataset_dir / split).mkdir(parents=True, exist_ok=True)

def generate_background_image(size=(64, 64)):
    """Generates a random background image."""
    return np.random.randint(0, 256, size + (3,), dtype=np.uint8)
