import pytest
from filters import VintageFilter, NoiseFilter
from PIL import Image
import numpy as np

@pytest.mark.parametrize("original_color,vintage_color", [
    ('red', [255, 0, 53]),
    ('white', [255, 255, 199]),
    ('blue', [0, 0, 199])
])
def test_vintagefilter(original_color, vintage_color):
    image = Image.new('RGB', (1, 1), original_color)
    vm = VintageFilter(image)
    vm.process()
    image_numpy = np.array(image)
    assert image_numpy[0][0].tolist() == vintage_color


@pytest.mark.parametrize("original_color,var,max_mean_values", [
    ('red', 1000, [255, 20, 20]),
    ('green', 1000, [20, 255, 20]),
    ('blue', 1000, [20, 20, 255]),
    
    ('red', 10, [255, 2, 2]),
    ('green', 10, [2, 255, 2]),
    ('blue', 10, [2, 2, 255])
])
def test_gaussnoise(original_color, var, max_mean_values):
    image = Image.new('RGB', (10, 10), original_color)
    nf = NoiseFilter(image)
    nf.add_gauss_noise(var=var)
    assert nf.get_numpy_image()[:, :, 0].mean() < max_mean_values[0]
    assert nf.get_numpy_image()[:, :, 1].mean() < max_mean_values[1]
    assert nf.get_numpy_image()[:, :, 2].mean() < max_mean_values[2]