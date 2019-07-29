# ---
# jupyter:
#   jupytext:
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

import os
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import glob
from glob import glob as gglob
from tqdm import tqdm
import urllib.request

from keras.models import load_model
from keras.preprocessing import image

#
# Colab
#

def is_colab_notebook():
    return 'google.colab' in sys.modules

#
# IO
#

import shutil
def cp(src_path, dest_path):
    """Copy-paste folder"""
    try:
        shutil.copytree(src_path, dest_path)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)

#
# Get data
#

def get_coco_dataset():
    """Downloads COCO-dataset and unpacks it to val2017 folder"""
    if os.path.exists("val2017"):
        print("val2017 directory already exists.")
    else:
        print("Downloading val2017. This will take a while.")
        res = requests.get("http://images.cocodataset.org/zips/val2017.zip")
        if res.ok:
            z = zipfile.ZipFile(io.BytesIO(res.content))
            z.extractall()
            print("Done. The images are in folder val2017.")
        else:
            print("Something went wrong. The zip-file was not downloaded.")

def get_blackwhite_images(folder_name = 'blackwhiteimages'): 
    """Download same black-white image in a seperate folder."""
    black_white_images_url = [
        'https://png.pngtree.com/thumb_back/fw800/back_pic/04/27/63/50583c00f940c29.jpg',
        'https://image.freepik.com/foto-gratuito/stucco-parete-nera_1194-7151.jpg',
        'https://images.unsplash.com/photo-1529753253655-470be9a42781?ixlib=rb-1.2.1&ixid=eyJhcHBfaWQiOjEyMDd9&w=1000&q=80',
        'http://images.unsplash.com/photo-1513569771920-c9e1d31714af?ixlib=rb-1.2.1&q=80&fm=jpg&crop=entropy&cs=tinysrgb&w=1080&fit=max&ixid=eyJhcHBfaWQiOjEyMDd9',
        'https://i.pinimg.com/736x/c6/da/6c/c6da6cc0413ac055c6b821fcbf17524c--black-sea-black-water.jpg',
        'https://www.ziro.de/sites/default/files/styles/colorbox/public/content/dekore/ziro-vinylan-plus-fantasy-white.jpg?itok=PzPeOPgc',
        'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRMvXpgbPw3CA7zob0I5tWe2NKBCwIS09HWdS215kF0cS_ihhAJNA',
        'https://qph.fs.quoracdn.net/main-qimg-75429bf3f0b5bc31752226b865dd4e38'
        ]
    
    if not os.path.exists(folder_name):
        print('Create folder <{}>.'.format(folder_name))
        os.makedirs(folder_name)  
        for i, url in enumerate(black_white_images_url):
            name = os.path.join(folder_name, "bw_{:02}.jpg".format(i))
            urllib.request.urlretrieve(url, name)
        n_jpgs = len(gglob(os.path.join(folder_name, '*')))
        print("Done! Downloaded {} jpg images.".format(n_jpgs))
    else:
        n_jpgs = len(gglob(os.path.join(folder_name, '*')))
        print("Folder '{}' already exists and has {} jpg images stored.".format(folder_name, n_jpgs))

def get_black_images(folder_name = 'blackimages'): 
    """Download same black-white image in a seperate folder."""
    black_images_url = ['http://image.fg-a.com/backgrounds/black-planet-surface-1920.jpg']
    
    if not os.path.exists(folder_name):
        print('Create folder <{}>.'.format(folder_name))
        os.makedirs(folder_name)  
        for i, url in enumerate(black_images_url):
            name = os.path.join(folder_name, "bw_{:02}.jpg".format(i))
            urllib.request.urlretrieve(url, name)
        n_jpgs = len(gglob(os.path.join(folder_name, '*')))
        print("Done! Downloaded {} jpg images.".format(n_jpgs))
    else:
        n_jpgs = len(gglob(os.path.join(folder_name, '*')))
        print("Folder '{}' already exists and has {} jpg images stored.".format(folder_name, n_jpgs))

fontpaths = [
     #'7SEGG-CHAN/DSEG7SEGGCHAN-Regular.ttf',
     #'7SEGG-CHAN/DSEG7SEGGCHANMINI-Regular.ttf',
     'Classic/DSEG7Classic-Bold.ttf',
     'Classic/DSEG7Classic-BoldItalic.ttf',
     'Classic/DSEG7Classic-Italic.ttf',
     'Classic/DSEG7Classic-Light.ttf',
     'Classic/DSEG7Classic-LightItalic.ttf',
     'Classic/DSEG7Classic-Regular.ttf',
     'Classic-MINI/DSEG7ClassicMini-Bold.ttf',
     'Classic-MINI/DSEG7ClassicMini-BoldItalic.ttf',
     'Classic-MINI/DSEG7ClassicMini-Italic.ttf',
     'Classic-MINI/DSEG7ClassicMini-Light.ttf',
     'Classic-MINI/DSEG7ClassicMini-LightItalic.ttf',
     'Classic-MINI/DSEG7ClassicMini-Regular.ttf',
     'Modern/DSEG7Modern-Bold.ttf',
     'Modern/DSEG7Modern-BoldItalic.ttf',
     'Modern/DSEG7Modern-Italic.ttf',
     'Modern/DSEG7Modern-Light.ttf',
     'Modern/DSEG7Modern-LightItalic.ttf',
     'Modern/DSEG7Modern-Regular.ttf',
     'Modern-MINI/DSEG7ModernMini-Bold.ttf',
     'Modern-MINI/DSEG7ModernMini-BoldItalic.ttf',
     'Modern-MINI/DSEG7ModernMini-Italic.ttf',
     'Modern-MINI/DSEG7ModernMini-Light.ttf',
     'Modern-MINI/DSEG7ModernMini-LightItalic.ttf',
     'Modern-MINI/DSEG7ModernMini-Regular.ttf']


def get_dseg_fonts():
    """Downloads the fonts for the data generator into the fonts folder"""
    url = "https://github.com/rbtdev/spiroplot/raw/master/fonts/DSEG7/{}"
    target_folder = "fonts/"
    # returned list
    fontnames_ok = []

    if not os.path.exists(target_folder):
        os.makedirs(target_folder, exist_ok=True)
        for fontname in tqdm(fontpaths):
            r = requests.get(url.format(fontname), stream=True)
            fontname = target_folder + fontname.replace("/", "-")
            if r.ok:
                with open(fontname, 'wb') as f:
                    for chunk in r:
                        f.write(chunk)
                fontnames_ok.append(fontname)
            else:
                print("Something went wrong. Font {} was not downloaded".format(fontname))
        print("Done. Fonts are downloaded.")
    else:
        fontnames_ok = glob.glob(target_folder + '*.ttf')
        #fontnames_ok = [target_folder.replace("/", "-") + x for x in fontnames_ok]
        num_ttfs = len(fontnames_ok)
        print("Folder '{}' already exists and has {} fonts stored.".format(target_folder, num_ttfs))
    return fontnames_ok

def get_dseg_font():
    print("Please use <get_dseg_fonts()>")
    get_dseg_fonts()

# Process images
#
# Naming convestion: img_*, where the asterix defines the image type.
#
# img_pil: PIL (RGB).
# img_cv: cv2 (it is a numpy array with BGR).
# img_keras: A numpy array with shape (1, width, height, channels), which can be passed to Keras' model.predict().
# img_list: List of img_pil.

def get_all_jpg_files(path):
    jpg_paths = [os.path.join(path, file) for file in os.listdir(path) if file.lower().endswith(".jpg")]
    return jpg_paths


# Plot 

def plot_cv_img(img_cv, figsize=None):
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=figsize)
    plt.imshow(img_cv)


def plt_img_pil(img_pil):
    im_array = np.asarray(img_pil)
    plt.imshow(im_array)
    plt.show()

# Read images

def read_cv_image(image_path):
    #TODO read as RGB image
    return cv2.imread(image_path)


def read_pil_image(image_path):
    return Image.open(image_path).convert('RGB')


# Convert images

def cv2pil(img_cv):
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv)
    return img_pil


def pil2cv(img_pil):
    """https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format"""
    img_cv = np.array(img_pil) 
    # Convert RGB to BGR 
    img_cv = img_cv[:, :, ::-1].copy() 
    return cv2.cvtColor(img_cv, cv2.CV_BGR2RGB)


def pil2keras(img_pil, normalize=True):
    """Convert PIL (values between 0 and 255) to numpy array which can be used
    from Keras' model.predict(). Usually Keras needs normalized values (between 0 and 1).

    Args:
       img_pil: PIL Image
       normalize: If values should be between 0 and 1.

    Returns:
        np.array in format which can be used from Keras' model.predict.
    """
    img_keras = image.img_to_array(img_pil)
    img_keras = np.expand_dims(img_keras, axis=0)
    if normalize:
        img_keras = img_keras/255
    return img_keras


def np2pil(img_keras, normalized=True):
    """img_keras with shape like (100,200,3) to img_pil"""
    if normalized:
        img_keras = img_keras*255
    return Image.fromarray(np.uint8(img_keras))


# Denoise images using cv2.fastNlMeansDenoisingColored

def denoise_cv_img(img_cv, h=2):
    """Apply denoising. Depending on the the size of the image, different values of h are reasonable.
    For small images h=6 might be already too high, for large images h=10 might be too low. Other 
    parameters are hardcoded (they are quite stable).

    h=2 is a good value for images from samples/composed."""
    img_cv = cv2.fastNlMeansDenoisingColored(img_cv, None, h, 7, 21)
    return img_cv


# Several steps in one

def read_denoise_img_keras_pil(image_path, h=2):
    """Read image from path, apply denoising and return as PIL image."""
    img_cv = read_cv_image(image_path)
    img_cv = denoise_cv_img(img_cv, h=h)
    img_pil = cv2pil(img_cv) # TODO cv2keras
    img_keras = pil2keras(img_pil)
    return [img_keras, img_pil]


def read_denoise_images_keras_pil(jpgs):
    """Read images from list of paths (jpgs) and apply denoising. Return as list of PIL images."""
    imgs =  [read_denoise_img_keras_pil(x) for x in jpgs]
    return imgs


# Convert between differnt values for bouding boxes, points, etc.

def convert_xywh_to_ullr_with_expansion(xywh, expansion_factor=1.):
    """Convert xywh set to (ul, lr) set but with expansion to x and y direction.
    
    Args:
        xywh (tuple): (x-center, y-center, width, height)
        expansion_factor: Defines expansion (make boudning box wider and higher). Default=1.0 (no expansion).
    """
    ul = (xywh[0] - expansion_factor * .5 * xywh[2], xywh[1] - expansion_factor * .5 * xywh[3])
    lr = (xywh[0] + expansion_factor * .5 * xywh[2], xywh[1] + expansion_factor * .5 * xywh[3])
    return (ul, lr)

def xywh2ullr(xywh):
    """Convert xywh set to (ul, lr) set."""
    return convert_xywh_to_ullr_with_expansion(xywh, expansion_factor=1.0)


#
# Make mask
#

def make_mask(img_pil_size, bounding_box):
    """Make mask with 0 at background and 1 at date."""
    # reshape img size
    img_size = (img_pil_size[1], img_pil_size[0])

    # create black array
    mask_np = np.zeros(img_size)
    
    # add ones at date position
    ul, lr = xywh2ullr(bounding_box)
    ullr = [int(x) for x in ul + lr] # float to ints
    mask_np[ullr[1]:ullr[3], ullr[0]:ullr[2]] = 1
    return mask_np
