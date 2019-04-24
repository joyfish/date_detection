# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Process original images

# ## Goals
#
# * Test filters to proprocess images: 
#     * Make generated images look like original ones.
#     * Make original images look like generated ones.
# * Test resize function as an alternative to cropping.

# ## TL;DR: Findings
#
# * fastNlMeansDenoisingColored works fine,...
#     * with h=10 (default) does not work on black surface.
#     * with h=20, works on black surface but blurs a lot.
# * MaxFilter:
#     May look bright date-text look fatter on dark surface.
# * MinFilter:
#     May look dark date-text look fatter on bright surface.
# * EMBOSS filter make date-text look nice, when surface is plank/has no structure. (but still not useful)

# #### Imports

# +
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import os

import IPython.display #import Image, display
import PIL
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps
from cv2 import fastNlMeansDenoisingColored
# -

# ### Load data

# Set path to sample images
s0 = "./samples/1984-02-26 Dia 01aL009.jpg"
s1 = "./samples/1984-02-29 Dia 01aL024.jpg"

# Original images
img0 = Image.open(s0).convert("RGB")
img1 = Image.open(s1).convert("RGB")
type(img1)

# Whole date
date0 = img0.crop(box=(4979-1400, 3248-500, 4979-400, 3248-150))
date1 = img1.crop(box=(4779-1350, 3186-400, 4779-350, 3186-50))

# Part of date
img0c = img0.crop((4779-700, 3186-400, 4779-300, 3186-150))
img1c = img1.crop((4779-900, 3186-400, 4779-500, 3186-50))

date0

date1

# ## Processing
#
# ### 1) Image Filters
#
# * [done] All filters from PIL https://pillow.readthedocs.io/en/5.1.x/reference/ImageFilter.html#imagefilter-module
# * [done] cv2.fastNlMeansDenoisingColored
#
# ### 2) Change size of image
#
# * [ongoing] Resize
#     * Resize-Filters
#     * Size = (224, 500)
# * [ongoing] Crop
#     * Crop lower-right only

# # Image Filters

# > * ImageFilter.GaussianBlur(radius
# > * ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
# > * ImageFilter.MedianFilter
# > * ImageFilter.MinFilter
# > * ImageFilter.MaxFilter

def plot_different_filter_values(img, pil_filter, filter_parameters_list):
    """Plot filtered images below each other using specified pil_filter and filter_parameters_list"""
    # since values are parsed to pil_filter via *v make sure
    # values consists of list of lists.
    fig, ax = plt.subplots(len(filter_parameters_list)+1, 1, figsize=(32, 32))
    if not isinstance(filter_parameters_list[0], list):
        filter_parameters_list=[[x] for x in filter_parameters_list]
    ax[0].set_title('Original')
    ax[0].imshow(img)
    ax[0].axis('off')
    for i, param in enumerate(filter_parameters_list):
        im = img.filter(pil_filter(*param))
        ax[i+1].set_title('{} value: {}'.format(str(pil_filter), param))
        ax[i+1].imshow(im)
        ax[i+1].axis('off')

plot_different_filter_values(date1, ImageFilter.GaussianBlur, filter_parameters_list=[2,5])

# ### UnsharpMask

val_list = [[2, 150, 3],
 [5, 150, 3],
 [9, 150, 3],
 [2, 100, 3],
 [2, 200, 3],
 [9, 100, 3]]
plot_different_filter_values(date1, ImageFilter.UnsharpMask, val_list)

# ### MedianFilter

plot_different_filter_values(date1, ImageFilter.MedianFilter, [5, 7, 9])

# ### MinFilter

plot_different_filter_values(date1, ImageFilter.MinFilter, [5, 7])

# ### MaxFilter

plot_different_filter_values(date1, ImageFilter.MaxFilter, [5, 7])

plot_different_filter_values(date0, ImageFilter.MaxFilter, [5, 7])

# ## Some default filters (without parameters to set)
#
# > "The current version of the library provides the following set of predefined image enhancement filters"
# >
# > * BLUR
# > * CONTOUR
# > * DETAIL
# > * EDGE_ENHANCE
# > * EDGE_ENHANCE_MORE
# > * EMBOSS
# > * FIND_EDGES
# > * SHARPEN
# > * SMOOTH
# > * SMOOTH_MORE

# Different filters, not different filter-parameters
def plot_different_filters(img, filter_list):
    fig, ax = plt.subplots(len(filter_list)+1, 1, 
                           figsize=(32*2, 32*2))
    ax[0].set_title('Original')
    ax[0].imshow(img)
    ax[0].axis('off')
    for i, the_filter in enumerate(filter_list):
        im = img.filter(the_filter())
        ax[i+1].set_title('{}'.format(str(the_filter)))
        ax[i+1].imshow(im)
        ax[i+1].axis('off')

# all **predefined** filters from https://pillow.readthedocs.io/en/5.1.x/reference/ImageFilter.html#filters
filters = [ImageFilter.BLUR, ImageFilter.CONTOUR, ImageFilter.DETAIL, ImageFilter.EDGE_ENHANCE, ImageFilter.EDGE_ENHANCE_MORE, ImageFilter.EMBOSS, ImageFilter.FIND_EDGES, ImageFilter.SHARPEN, ImageFilter.SMOOTH, ImageFilter.SMOOTH_MORE]

plot_different_filters(date1, filters)

plot_different_filters(date0, filters)

# # fastNlMeansDenoisingColored

def plot_cvDN(h=10,tws=7,sws=21):
    """https://www.bogotobogo.com/python/OpenCV_Python/python_opencv3_Image_Non-local_Means_Denoising_Algorithm_Noise_Reduction.php"""
    # date0
    open_cv_image = np.array(date0) 
    img = open_cv_image[:, :, ::-1].copy() 
    b,g,r = cv2.split(img)           # get b,g,r
    rgb_img0 = cv2.merge([r,g,b])     # switch it to rgb
    dst = cv2.fastNlMeansDenoisingColored(img,None,h,tws,sws)
    b,g,r = cv2.split(dst)           # get b,g,r
    rgb_dst0 = cv2.merge([r,g,b])     # switch it to rgb

    # date1
    open_cv_image = np.array(date1) 
    img = open_cv_image[:, :, ::-1].copy() 
    b,g,r = cv2.split(img)           # get b,g,r
    rgb_img1 = cv2.merge([r,g,b])     # switch it to rgb
    dst = cv2.fastNlMeansDenoisingColored(img,None,h,tws,sws)
    b,g,r = cv2.split(dst)           # get b,g,r
    rgb_dst1 = cv2.merge([r,g,b])     # switch it to rgb

    f = plt.figure(figsize=(15,5))
    ax0o = f.add_subplot(221)
    ax0p = f.add_subplot(222)
    ax1o = f.add_subplot(223)
    ax1p = f.add_subplot(224)
    ax0o.imshow(rgb_img0)
    ax0p.imshow(rgb_dst0)
    ax1o.imshow(rgb_img1)
    ax1p.imshow(rgb_dst1)
plot_cvDN()

plot_cvDN(15)

plot_cvDN(20)

# # Resize

# > * PIL.Image.NEAREST (use nearest neighbour), 
# > * PIL.Image.LANCZOS (a high-quality downsampling filter).
# > * PIL.Image.BILINEAR (linear interpolation), 
# > * PIL.Image.BICUBIC (cubic spline interpolation)

# Just point to ints
modes = [PIL.Image.NEAREST, PIL.Image.LANCZOS, PIL.Image.BILINEAR, PIL.Image.BICUBIC]
print(modes)
# Use eval() to get corresponsing ints
modes = ["PIL.Image.NEAREST", "PIL.Image.LANCZOS", "PIL.Image.BILINEAR", "PIL.Image.BICUBIC"]

def plot_different_resamplingfilters(img, mode_list, size=(224, 224), crop_box=(150,190, 224, 224)):
    fig, ax = plt.subplots(1, len(mode_list), figsize=(18, 18))
    for i, m in enumerate(mode_list):
        im = img.resize(size, eval(m))
        im = im.crop(box=crop_box)
        ax[i].set_title('{}: {}'.format(eval(m), m))
        ax[i].imshow(im)
        ax[i].axis('off')

plot_different_resamplingfilters(img1, modes)

plot_different_resamplingfilters(img0, modes)

plot_different_resamplingfilters(img1, modes,size=(500, 500), crop_box=(340, 430, 480, 480))

plot_different_resamplingfilters(img0, modes,size=(500, 500), crop_box=(340, 430, 480, 480))
