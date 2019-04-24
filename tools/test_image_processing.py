import pytest
from image_processing import ImageCropper
from PIL import Image

def test_imagecropper_tosmaller():
    """Test if target_size equals cropped image size."""
    img = Image.new("RGB", (10, 10), "red")
    target_size = (5, 5)
    cr = ImageCropper(img)
    cr.crop(target_size = target_size)
    img_after = cr.get_pil_image()
    assert img_after.size == target_size


def test_imagecropper_tolarger():
    """Test if it throws error when target_size is larger than image size."""
    img = Image.new("RGB", (10, 10), "red")
    target_size = (50, 50)
    cr = ImageCropper(img)
    with pytest.raises(ValueError):
        cr.crop(target_size = target_size)
