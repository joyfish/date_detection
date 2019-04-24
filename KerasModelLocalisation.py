# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Date Location (Keras)

# %% [markdown]
#
# Main steps:
#
#  1. Get raw data (COCO): 5000 images 
#  2. ImageDataLocationGenerator class and fit_generator
#  3. CNN
#  4. fit_generator, which generates data on-the-fly
#  5. Save in Google Drive


# %%
# Colab setup
import sys
def is_colab_notebook():
    return 'google.colab' in sys.modules

# Mount your Drive drive. This is needed to save results to your Google Drive
if is_colab_notebook():
    from google.colab import drive
    drive.mount('/content/gdrive')

# %%
!cp -f /content/gdrive/My\ Drive/date_datetection/TrainingDataGeneration.py TrainingDataGeneration.py 
!cp -rf /content/gdrive/My\ Drive/date_datetection/tools tools
!ls tools

# %%
from PIL import Image, ImageDraw
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
from IPython.display import display

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard
from keras import backend as K

K.tensorflow_backend._get_available_gpus()

from TrainingDataGeneration import DateImageGenerator
from tools import basic, draw

# %%
# Create folders to store model in
os.makedirs("model", exist_ok=True)
os.makedirs("model/weights", exist_ok=True)

# %%
# Download data and fonts
basic.get_coco_dataset()
fontnames = basic.get_dseg_fonts()

# %% [markdown]
# ### Init DateImageGenerator
#

# %%
#
# Set parameters for image generation:
#
bg_image_path = os.path.expanduser("val2017/")
target_size = (200, 100)
font_lettersize_range = (10, 16)
h = (40/360, 100/360) 
s = (80/100, 100/100)
v = (80/100, 100/100)

# image generator
gen = DateImageGenerator(bg_image_path=bg_image_path, target_size=target_size,
    font_paths=fontnames, font_lettersize_range=font_lettersize_range,
    h=h, s=s, v=v)

mains, font_dict, noise_dict, misc_dict = gen.sample_localisation(realistic_date=False)

b_xywh = mains['bounding_box']
pos = misc_dict['position_upperleft_corner']

# Try it out (generates one sample).
#img, date, center, font, font_lettersize, font_width_height, position, color, rotation, is_vintage, blur_value, noise_saltpepper_value, noise_gauss_value, img_original, img_original_cropped = gen.sample(add_point_at_target=False)
lower_right = (pos[0] + b_xywh[2], pos[1] + b_xywh[3])
#print("date: {}, center: {}, font_width_height: {}, position: {}, color: {}, rotation: {}, is_vintage: {}, blur_value: {}, noise_sp_value: {}, noise_gauss_value: {}".format(date, center, font_width_height, position, color, rotation, is_vintage, blur_value, noise_saltpepper_value, noise_gauss_value))

print("Image with date: <{}> | <{}>".format(mains['date_iso'], misc_dict['date_formated']))
display(mains['img'])
print("Image with date and box around it:")
sol = draw.draw_box(mains['img'], pos, lower_right, fill=0, width=3)
display(draw.draw_box(sol, pos, lower_right, fill=255, width=1))

# %%
def generator(batch_size=100, normalize=True):
    """generator function which will passed to keras' fit_generation function.
    This function makes use of the ImageDateLocationGenerator class."""
    batch_x = np.zeros((batch_size, 100, 200, 3))
    batch_y = np.zeros((batch_size, 4))
    while True:
        for i in range(batch_size):
            mains, _, _, _ = gen.sample_localisation(realistic_date=True)
            y = mains['bounding_box_relative'] # i.e. (x, y, w, h) with values between 0. and 1.
            x = np.array(mains['img'])
            if normalize:
              x = x/255
            batch_x[i] = x
            batch_y[i] = y
        yield batch_x, batch_y

# %%
# %%time
# ### Generate Validation Data

# generate validation data using the generator function. Returns a tuple (x_val, y_val).
if is_colab_notebook():
    valid_generator = generator(batch_size=2000, normalize=True)
    valid = next(valid_generator)
else:
    valid_generator = generator(batch_size=200, normalize=True)
    valid = next(valid_generator)

def keras2pil(x):
    """img_keras with shape like (100,200,3) to img_pil"""
    return Image.fromarray(np.uint8(x*255))

image_index = 100
x_example = valid[0][image_index]
print(x_example.shape)
y_example = valid[1][image_index]
y_example_ = (y_example[0]* 200, y_example[0] *100, y_example[0] *200 , y_example[0] * 100)
print(y_example_)
keras2pil(x_example)

# %%
# Check if GPU
K.tensorflow_backend._get_available_gpus()

# %% [markdown]
# #### Build model

# %%
# # Model
input = Input(name='data_input', shape=(100, 200, 3))

with K.name_scope('conv1'):
    x = Conv2D(64, (3, 3), activation="elu", padding="same")(input)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

with K.name_scope('conv2'):
    x = Conv2D(32, (3, 3), activation="elu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

with K.name_scope('conv3'):
    x = Conv2D(16, (3, 3), activation="elu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

with K.name_scope('conv4'):
    x = Conv2D(8, (3, 3), activation="elu", padding="same")(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding="same")(x)

x = Flatten(name='flatten')(x)

with K.name_scope('dense1'):
    x = Dense(units=128, activation="elu")(x)
    #x = Dropout(0.5)(x)

with K.name_scope('dense2'):
    x = Dense(units= 64, activation="elu")(x)
    #x = Dropout(0.5)(x)
    
with K.name_scope('dense3'):
    x = Dense(units= 32, activation="elu")(x)
    #x = Dropout(0.5)(x)

with K.name_scope('out'):
    out = Dense(name="y_xywh", units=4)(x)

model = Model(inputs=input, outputs=out)


model.summary()

# %%
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# %%
# Fitting
DEBUG = True

if DEBUG:
    batch_size=3
    steps_per_epoch=3
    epochs=1
elif is_colab_notebook():
    batch_size=60
    steps_per_epoch=70
    epochs=120
else:
    batch_size=60
    steps_per_epoch=35
    epochs=10
print("That are {:,} observations in total.".format(batch_size * steps_per_epoch * epochs))

# %%
# checkpoint
#filepath="model/weights/weights-improvement-{epoch:02d}-{val_loss:.2f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
logger = CSVLogger('model/measures.csv', append=True, separator=';')
tbcallback = TensorBoard(log_dir='model/tbc2', histogram_freq=1,
                         write_graph=True,
                         write_grads=False,
                         batch_size=batch_size,
                         write_images=False)
#callbacks_list = [checkpoint, logger]
callbacks_list = [logger, tbcallback]

# %% [markdown]
# #### Fit model
#
# * using pretrained model or 
# * fresh version

# %%
USE_PRETRAINED_MODEL = False
PRETRAINED_PATH_NAME = 'model_localisation2_20190114/model.h5'

if USE_PRETRAINED_MODEL:  
    print("Using pretrained model!")
    from keras.models import load_model
    gdrive_path = '/content/gdrive/My Drive/date_detection_models/'
    model = load_model(gdrive_path + PRETRAINED_PATH_NAME)

# %%
if USE_PRETRAINED_MODEL:  
    history = model.fit_generator(generator(batch_size=batch_size), steps_per_epoch=steps_per_epoch, epochs=epochs,
                             validation_data=valid, callbacks=callbacks_list, initial_epoch=79)
else:
  print("Do not train on pretrained model.")

# %%
if not USE_PRETRAINED_MODEL:  
    print("Train fresh model.")
    history = model.fit_generator(generator(batch_size=batch_size), 
                                  steps_per_epoch=steps_per_epoch, 
                                  epochs=epochs,
                                  validation_data=valid, 
                                  callbacks=callbacks_list)

# %% [markdown]
# #### Save model

# %%
model.save('model/model.h5')

# %%
# cp model from colab to gdrive
if is_colab_notebook():
    drive_path="/content/gdrive/My Drive/date_detection_models/model_localisation2_"
    date = datetime.now().strftime('%Y%m%d')
    save_path = drive_path + date 
    print(save_path)
    basic.cp('model', save_path)
