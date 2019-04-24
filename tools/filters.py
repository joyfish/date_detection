from PIL import Image
import numpy as np


class VintageFilter:
    """
    Make an image look vintage (more yellowish).
    Note: Image is changed in place, hence you do not need a get method.

    Rescources:
        Nicholas White (https://github.com/nickwah/vintage-pil/blob/master/vintage.py)
    """
    VINTAGE_COLOR_LEVELS = {
        'r': [0, 0, 0, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 7, 7, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10, 11,
              11, 12, 12, 12, 12, 13, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 19, 19, 20, 21, 22, 22, 23, 24,
              25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 47, 48, 49, 52, 54, 55, 57,
              59, 60, 62, 65, 67, 69, 70, 72, 74, 77, 79, 81, 83, 86, 88, 90, 92, 94, 97, 99, 101, 103, 107, 109, 111,
              112, 116, 118, 120, 124, 126, 127, 129, 133, 135, 136, 140, 142, 143, 145, 149, 150, 152, 155, 157, 159,
              162, 163, 165, 167, 170, 171, 173, 176, 177, 178, 180, 183, 184, 185, 188, 189, 190, 192, 194, 195, 196,
              198, 200, 201, 202, 203, 204, 206, 207, 208, 209, 211, 212, 213, 214, 215, 216, 218, 219, 219, 220, 221,
              222, 223, 224, 225, 226, 227, 227, 228, 229, 229, 230, 231, 232, 232, 233, 234, 234, 235, 236, 236, 237,
              238, 238, 239, 239, 240, 241, 241, 242, 242, 243, 244, 244, 245, 245, 245, 246, 247, 247, 248, 248, 249,
              249, 249, 250, 251, 251, 252, 252, 252, 253, 254, 254, 254, 255, 255, 255, 255, 255, 255, 255, 255, 255,
              255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
        'g': [0, 0, 1, 2, 2, 3, 5, 5, 6, 7, 8, 8, 10, 11, 11, 12, 13, 15, 15, 16, 17, 18, 18, 19, 21, 22, 22, 23, 24,
              26, 26, 27, 28, 29, 31, 31, 32, 33, 34, 35, 35, 37, 38, 39, 40, 41, 43, 44, 44, 45, 46, 47, 48, 50, 51,
              52, 53, 54, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 68, 69, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81,
              83, 84, 85, 86, 88, 89, 90, 92, 93, 94, 95, 96, 97, 100, 101, 102, 103, 105, 106, 107, 108, 109, 111, 113,
              114, 115, 117, 118, 119, 120, 122, 123, 124, 126, 127, 128, 129, 131, 132, 133, 135, 136, 137, 138, 140,
              141, 142, 144, 145, 146, 148, 149, 150, 151, 153, 154, 155, 157, 158, 159, 160, 162, 163, 164, 166, 167,
              168, 169, 171, 172, 173, 174, 175, 176, 177, 178, 179, 181, 182, 183, 184, 186, 186, 187, 188, 189, 190,
              192, 193, 194, 195, 195, 196, 197, 199, 200, 201, 202, 202, 203, 204, 205, 206, 207, 208, 208, 209, 210,
              211, 212, 213, 214, 214, 215, 216, 217, 218, 219, 219, 220, 221, 222, 223, 223, 224, 225, 226, 226, 227,
              228, 228, 229, 230, 231, 232, 232, 232, 233, 234, 235, 235, 236, 236, 237, 238, 238, 239, 239, 240, 240,
              241, 242, 242, 242, 243, 244, 245, 245, 246, 246, 247, 247, 248, 249, 249, 249, 250, 251, 251, 252, 252,
              252, 253, 254, 255],
        'b': [53, 53, 53, 54, 54, 54, 55, 55, 55, 56, 57, 57, 57, 58, 58, 58, 59, 59, 59, 60, 61, 61, 61, 62, 62, 63,
              63, 63, 64, 65, 65, 65, 66, 66, 67, 67, 67, 68, 69, 69, 69, 70, 70, 71, 71, 72, 73, 73, 73, 74, 74, 75,
              75, 76, 77, 77, 78, 78, 79, 79, 80, 81, 81, 82, 82, 83, 83, 84, 85, 85, 86, 86, 87, 87, 88, 89, 89, 90,
              90, 91, 91, 93, 93, 94, 94, 95, 95, 96, 97, 98, 98, 99, 99, 100, 101, 102, 102, 103, 104, 105, 105, 106,
              106, 107, 108, 109, 109, 110, 111, 111, 112, 113, 114, 114, 115, 116, 117, 117, 118, 119, 119, 121, 121,
              122, 122, 123, 124, 125, 126, 126, 127, 128, 129, 129, 130, 131, 132, 132, 133, 134, 134, 135, 136, 137,
              137, 138, 139, 140, 140, 141, 142, 142, 143, 144, 145, 145, 146, 146, 148, 148, 149, 149, 150, 151, 152,
              152, 153, 153, 154, 155, 156, 156, 157, 157, 158, 159, 160, 160, 161, 161, 162, 162, 163, 164, 164, 165,
              165, 166, 166, 167, 168, 168, 169, 169, 170, 170, 171, 172, 172, 173, 173, 174, 174, 175, 176, 176, 177,
              177, 177, 178, 178, 179, 180, 180, 181, 181, 181, 182, 182, 183, 184, 184, 184, 185, 185, 186, 186, 186,
              187, 188, 188, 188, 189, 189, 189, 190, 190, 191, 191, 192, 192, 193, 193, 193, 194, 194, 194, 195, 196,
              196, 196, 197, 197, 197, 198, 199]
    }

    def __init__(self, img):
        """
        Args:
            img: A PIL Image
        """
        self.img = img

    def _modify_all_pixels(self):
        width, height = self.img.size
        pixels = self.img.load()
        for x in range(width):
            for y in range(height):
                pixels[x, y] = self._adjust_levels(*pixels[x, y])

    def _adjust_levels(self, r, g, b):  # expect rgb; rgba will blow up
        r_map = self.VINTAGE_COLOR_LEVELS['r']
        g_map = self.VINTAGE_COLOR_LEVELS['g']
        b_map = self.VINTAGE_COLOR_LEVELS['b']
        return r_map[r], g_map[g], b_map[b]

    def process(self):
        """
        Process the image that it looks vintage.
        """
        self._modify_all_pixels()
        return self.img


class NoiseFilter:
    """
    Add different types of noise (gauss, salt&pepper, poission, speckle) to PIL image.
    Note: Image is not changed in place, hence you do need a get method.

    References:
        Shubham Pachori https://stackoverflow.com/questions/22937589/how-to-add-noise-gaussian-salt-and-pepper-etc-to-img_np-in-python-with-opencv
    """
    def __init__(self, img):
        """
        Args:
            img (PIL.Image): Input image.
        """
        self.img = img
        self.img_np = np.array(img)

    def add_gauss_noise(self, mean=0, var=1000):
        """
        Add gaussian noise to image.

        Args:
            mean: Mean of Gauss distribution.
            var: Variance of Gauss distribuation.
        """
        row, col, ch = self.img_np.shape

        sigma = var**0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)

        noisy = self.img_np + gauss
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = np.uint8(noisy)
        
        self.img = Image.fromarray(noisy)
        self.img_np = noisy

    def add_saltpepper_noise(self, s_vs_p=.5, amount=.04):
        """
        Add Salt&pepper noise to image.

        Args:
            s_vs_p (float): Ratio of salt to pepper noise (between 0. and 1.).
            amount (float): Amout of pixel which will get noise value (between 0. and 1.).
        """
        row, col, ch = self.img_np.shape
        noisy = np.copy(self.img_np)
        # Salt mode
        num_salt = np.ceil(amount * self.img_np.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.img_np.shape]
        noisy[coords] = 1
        # Pepper mode
        num_pepper = np.ceil(amount* self.img_np.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.img_np.shape]
        noisy[coords] = 0
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = np.uint8(noisy)
        
        self.img = Image.fromarray(noisy)
        self.img_np = noisy

    def add_poisson_noise(self):
        """Add poisson noise to image."""
        vals = len(np.unique(self.img_np))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(self.img_np * vals) / float(vals)
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = np.uint8(noisy)
        
        self.img = Image.fromarray(noisy)
        self.img_np = noisy

    def add_speckle_noise(self):
        """Add speckle noise to image."""
        row, col, ch = self.img_np.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = self.img_np + self.img_np * gauss
        noisy[noisy > 255] = 255
        noisy[noisy < 0] = 0
        noisy = np.uint8(noisy)
        
        self.img = Image.fromarray(noisy)
        self.img_np = noisy
        
    def get_pil_image(self):
        """Get image as PIL image."""
        return self.img
    
    def get_numpy_image(self):
        """Get image as numpy array."""
        return self.img_np
