import numpy as np
from PIL import Image

class ImageCropper():
    """Crop image to a wanted size using randomly selected crops."""
    def __init__(self, img):
        self.img = img
    
    def crop(self, target_size):
        """Crop image to target_size."""
        actual_size = self.img.size
        if (actual_size[0] < target_size[0]) or (actual_size[1] < target_size[1]):
            raise ValueError("img is smaller than target_size: {} < {}".format(actual_size, target_size))
        diffx = actual_size[0] - target_size[0] + 1
        diffy = actual_size[1] - target_size[1] + 1
        
        start_at_x = np.random.randint(0, diffx)
        start_at_y = np.random.randint(0, diffy)
        self.img = self.img.crop((start_at_x, start_at_y,
                        start_at_x + target_size[0], start_at_y + target_size[1]))
        
    def get_pil_image(self):
        """Return PIL image."""
        return self.img