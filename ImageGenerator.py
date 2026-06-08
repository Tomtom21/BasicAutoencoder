import numpy as np
import random
import string
from PIL import Image, ImageDraw

class ImageGenerator:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size

    def generate_gradient_background(self):
        """Generates a gradient background image."""
        x = np.linspace(0, 255, self.image_size[0], dtype=np.uint8)
        y = np.linspace(0, 255, self.image_size[1], dtype=np.uint8)
        xv, yv = np.meshgrid(x, y)
        return np.stack((xv, yv, np.zeros_like(xv)), axis=-1)

    def generate_solid_background(self, color=(255, 0, 0)):
        """Generates a solid color background image."""
        return np.full(self.image_size + (3,), color, dtype=np.uint8)

    def add_random_text(self, image):
        """Adds random garbled text at a random position and rotation to the image."""
        pil_image = Image.fromarray(image).convert('RGBA')

        # Random garbled text
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(3, 12)))

        # Transparent overlay for text
        txt_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Random position and rotation
        x = random.randint(0, self.image_size[1] - 1)
        y = random.randint(0, self.image_size[0] - 1)
        angle = random.uniform(0, 360)

        draw.text((x, y), text, fill=(255, 255, 255, 255))
        txt_layer = txt_layer.rotate(angle, expand=False)

        result = Image.alpha_composite(pil_image, txt_layer).convert('RGB')
        return np.array(result)

    def generate_random_image(self):
        """Generates a random RGB image."""
        return np.random.randint(0, 256, self.image_size + (3,), dtype=np.uint8)