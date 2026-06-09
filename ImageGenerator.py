import numpy as np
import random
import string
from PIL import Image, ImageDraw, ImageFont

class ImageGenerator:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size

    def generate_random_gradient_background(self):
        """Generates a random gradient background image."""
        # Random start and end colors for each RGB channel
        start_color = np.array([random.randint(0, 255) for _ in range(3)])
        end_color = np.array([random.randint(0, 255) for _ in range(3)])

        # Random gradient angle (0 to 2π)
        angle = random.uniform(0, 2 * np.pi)

        # Create coordinate grids
        x = np.arange(self.image_size[1])
        y = np.arange(self.image_size[0])
        xv, yv = np.meshgrid(x, y)

        # Project coordinates onto the gradient direction
        gradient = xv * np.cos(angle) + yv * np.sin(angle)

        # Normalize to 0-1 range
        gradient = (gradient - gradient.min()) / (gradient.max() - gradient.min())
        gradient = gradient[:, :, np.newaxis]

        result = start_color + gradient * (end_color - start_color)
        return result.astype(np.uint8)

    def generate_random_solid_background(self):
        """Generates a solid background image with a random color."""
        color = np.array([random.randint(0, 255) for _ in range(3)], dtype=np.uint8)
        return np.full(self.image_size + (3,), color, dtype=np.uint8)

    def add_random_text(self, image, min_chars = 3, max_chars = 12):
        """Adds random garbled text at a random position and rotation to the image."""
        pil_image = Image.fromarray(image).convert('RGBA')

        # Random garbled text
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(min_chars, max_chars)))

        # Transparent overlay for text
        txt_layer = Image.new('RGBA', pil_image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(txt_layer)

        # Use a larger font size
        try:
            font = ImageFont.truetype("arial.ttf", int(self.image_size[0] // 1.8))
        except OSError:
            font = ImageFont.load_default()

        # Center position with random offsets
        center_x = self.image_size[1] // 2
        center_y = self.image_size[0] // 2

        # Random offsets (up to 30% of image size in each direction)
        offset_x = random.randint(-self.image_size[1] // 3, self.image_size[1] // 3)
        offset_y = random.randint(-self.image_size[0] // 3, self.image_size[0] // 3)

        x = center_x + offset_x
        y = center_y + offset_y
        angle = random.uniform(0, 15)

        draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
        txt_layer = txt_layer.rotate(angle, expand=False)

        result = Image.alpha_composite(pil_image, txt_layer).convert('RGB')
        return np.array(result)
