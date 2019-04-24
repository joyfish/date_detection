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

# %%
import colorsys
import os

import numpy as np
import random
import time
import datetime
from PIL import Image, ImageFilter, ImageDraw, ImageOps, ImageFont
from tools.filters import VintageFilter, NoiseFilter
from tools.image_processing import ImageCropper

try:
    from deprecated import deprecated
except:
    print("Install deprecated module with 'pip install Deprecated'.")

# %%
# TODO move hsv from init to function call

class DateImageGenerator():
    def __init__(self, bg_image_path=os.path.expanduser("val2017/"),
                target_size=(320, 320), font_paths=["fonts/DSEG7Classic-Bold.ttf", "fonts/DSEG7Classic-Regular.ttf"],
                font_lettersize_range=(8, 13),
                h=(40/360, 100/360), s=(80/100, 100/100), v=(80/100, 100/100)):
        self.bg_image_path = bg_image_path
        self.bg_jpegs = [os.path.join(bg_image_path, file) for file in os.listdir(bg_image_path) if file.lower().endswith(".jpg")]
        self.target_size = target_size
        self.font_paths = font_paths
        self.font_lettersize_range = font_lettersize_range
        self.h = h
        self.s = s
        self.v = v

    def _add_random_vintage(self, img, fraction=.5):
        """
        Apply a vinage filter to image. 

        Args:
            img: The image
            fraction: Define the faction how often a vinage filter is applied to a picture.
        """
        random_number = np.random.random()
        if random_number < fraction:
            vm = VintageFilter(img)
            vm.process() # inplace change
            is_vintage = True
        else:
            is_vintage = False
        return img, is_vintage

    @staticmethod
    def _add_random_blur(img, fraction=.5, low=0.0, high=1.75):
        """Add Gaussian blur to the image with a random radius between low and high.
        Define the faction how often the filter is applied."""
        random_number_fraction = np.random.random()
        random_number = 0
        if random_number_fraction < fraction:
            random_number = np.random.uniform(low=low, high=high)
            img = img.filter(ImageFilter.GaussianBlur(radius=random_number))
        return img, random_number

    @staticmethod
    def _add_gaussian_noise(img, fraction=.5, low=50, high=500):
        """Add Gaussian noise to an image. Define fraction of the cases with a random noise amount between low and high."""
        random_number_fraction = np.random.random()
        random_number = 0
        if random_number_fraction < fraction:
            random_number = np.random.uniform(low=low, high=high)
            nf = NoiseFilter(img)
            nf.add_gauss_noise(var=random_number)
            img = nf.get_pil_image()
        return img, random_number

    @staticmethod
    def _add_digit(img, position, digit, font, color, rotation):
        """
        Add text to an image at upper left position with corresponding text (=digit), font, color, and rotation values.

        This function can be used to 
        * add whole date-text (like 31-12-1990) or 
        * one single digit (the function _add_digits make use of _add_digit).

        This function cannot
        * handle whitespaces properly since the font does not support that. (Solution: print every digit one by one 
        using _add_digits function)

        Args:
            img: Image
            position: Define the upper left postion where the digit should be printed.
            digit (str): The digit to print.
            font: The font.
            color (tuple): The color in RGB.
            rotation: The rotation (in degree) of the digit

        Returns:
            img: The processed Image.
            center_box (tuple): The bounding box (x, y, w, h) with x and y being the center of the box.
            position (tuple): Upper left postion of the digit which was printed to the image.
            position_ur: Upper right postion of the digit. This is useful when _add_digits is used.
        """
        # font of input-digit
        font_width_height = font.getsize(digit)
        font_width = font_width_height[0]
        font_height = font_width_height[1]

        # font of digit=8
        font_width_height_default = font.getsize("8")
        font_width_default = font_width_height_default[0]
        font_height_default = font_width_height_default[1]
        txt = Image.new('L', font_width_height)

        # add digit
        draw = ImageDraw.Draw(txt)
        draw.text((0, 0), digit, font=font, fill=225)
        w = txt.rotate(rotation, expand=1)
        img.paste(ImageOps.colorize(w, color, color), position, w)
        
        # outputs
        center_box = (position[0] + font_width/2, position[1] + font_height/2, 
                      font_width, font_height)
        position_ur = (position[0] + font_width_default, position[1])
        return img, center_box, position, position_ur

    def _add_digits(self, img, position, digits, font, color, rotation):
        """
        Adds many digits to the image while adding proper whitespaces (_add_digit cannot do that).
        Return the postions of each and every digit (useful for object detection/segmentation task).

        Args:
            img: Image
            postion: The upper left postion of the first digit to print. Passed to _add_digit.
            digits (str): The string which should be printed with whitespaces where whitespaces should be 
                printed. The width of a whitespace is the width of the digit '8' (this is not the case when 
                you use _add_digit).
            font: The font. Passed to _add_digit.
            color: The color. Passed to _add_digit.
            rotation: The rotation. Passed to _add_digit.
        """
        p_next = position
        targets = []
        for digit in digits:
            img, cb, p, p_next = self._add_digit(img, p_next, digit, font, color, rotation)
            # only store valid digits
            if not digit == ' ':
                targets.append((digit, cb, p))
        return img, targets

    @staticmethod
    def _generate_random_rotation(low=-1.5, high=1.5):
        """Generate a random number for rotation of digit(s)."""
        return np.random.uniform(low=low, high=high)

    def _generate_crop(self, img):
        """Get a random crop of an image with the wanted size (self.target_size).

        Args:
            img: PIL.Image.Image

        Returns:
            img (PIL.Image.Image): Image with new size.
        """
        cr = ImageCropper(img)
        cr.crop(target_size = self.target_size)
        img = cr.get_pil_image()
        return img

    @staticmethod
    def _generate_realistic_random_date(start="01011950", end="31122049",
                                        format_in='%d%m%Y', format_out='%d%m%y'):
        """Generate a random date which really exists.

        Use _generate_digits(realistic_date=False) when you want to generate all digits uniformly.

        Args:
            start (str): Start date in form of `format_in`.
            end (str): End date in form of `format_in`.
            format_in (str): Format of date in `start` and `end`.
            format_out (str): Format of generated date which will be returned.
        """
        start = time.mktime(time.strptime(start, format_in))
        end = time.mktime(time.strptime(end, format_in))
        date = start + random.random() * (end - start)
        return time.strftime(format_out, time.localtime(date))

    def _generate_digits(self, number_of_digits=6, realistic_date=False):
        """Generate digits which can be printed on the image. 
        
        Args:
            realistic_date (bool): If true a real date will be generated.
            
        Returns:
            A six digits long string with eigher a realisitc date or random numbers with 
            values between 0 and 9 or withespace.
            """
        if realistic_date:
            return self._generate_realistic_random_date()
        else:
            digits = ""
            for i in range(number_of_digits):
                v = str(np.random.randint(0, 11))
                if v == "10":
                    v = " "
                digits += v
            return digits

    @staticmethod
    def _format_digits(digits, first_space_rand=(1,2), second_space_rand=(1,2)):
        """Format digits: Add spaces after second and 4th value and create a ISO(-like) format.
        This only works for digits with 6 numbers.

        Args:
            digits (str): Six digits.
            first_space_rand (tuple): Random number for first space. Use (1,2) to get exact one space.
            second_space_rand (tuple): Random number for second space. Use (1,2) to get exact one space.

        Return:
            digits_formated (str): Properly formated version for printing.
            digits_isolike (str): Date in ISO (realistic date) or ISO-like (uniformly generated) format. 

        Examples:
            >>> gen._format_digits('311290')
            ('31 12 90', '1990-12-31')
            >>> gen._format_digits(' 1 190')
            (' 1  1 90', '1990-01-01')
        """
        # split digits
        assert len(digits) == 6
        d0, d1, d2 = digits[0:2], digits[2:4], digits[4:6]

        # add spaces 
        space0 = np.random.randint(*first_space_rand)
        space0 = " " * space0
        space1 = np.random.randint(*second_space_rand)
        space1 = " " * space1
        digits_formated = "{:2}{}{:2}{}{:2}".format(d0, space0, d1, space1, d2)

        # generate iso
        d0 = d0.replace("  ", "00").replace(" ", "0")
        d1 = d1.replace("  ", "00").replace(" ", "0")
        d2 = d2.replace("  ", "00").replace(" ", "0")
        if int(d2) < 50:
            d2 = "20"+d2
        else:
            d2 = "19"+d2
        digits_isolike = "{:04}-{:02}-{:02}".format(int(d2),int(d1),int(d0))

        #return
        return digits_formated, digits_isolike

    def _generate_font(self):
        """Randomly select font size within given range and generate font from font_path."""
        font_name = random.choice(self.font_paths)
        font_lettersize = np.random.randint(self.font_lettersize_range[0], self.font_lettersize_range[1])
        font = ImageFont.truetype(font_name, font_lettersize)
        return font, font_name, font_lettersize
    
    def _generate_rgb_variation(self):
        """Creates a random variation of given hsv color.
        
        Pass a tuple with (min number, max number) when value should be randomly sampled or 
        one value when a fix number (i.e. color) should be used.
        
        Args:
            h: hue with number(s) between 0 and 1.
            s: saturation with number(s) between 0 and 1.
            v: value with number(s) between 0 and 1.
        Returns:
            (r, g, b) with ints between 0 and 255.
        """
        h = self.h
        s = self.s
        v = self.v
        if type(h) == tuple:
            h = np.random.uniform(*h)
            s = np.random.uniform(*s)
            v = np.random.uniform(*v)
        msg = "Values for h, s, and v must be between 0 and 1, but is {}."
        assert 0 <= h <= 1, msg.format(h)
        assert 0 <= s <= 1, msg.format(s)
        assert 0 <= v <= 1, msg.format(v)
        #r,g,b = colorsys.hsv_to_rgb(h,s,v)
        color = tuple(int(x * 255) for x in colorsys.hsv_to_rgb(h, s, v))
        return color

    @staticmethod
    def _generate_text_position(img_size, font_width_height):
        """Return upper-left position to put the text
        Args:
            img_size: (width, height)
            font_size: (width, height)
        Return:
            (x, y) coordiante where to put the text
        """
        diffx = img_size[0] - font_width_height[0]
        diffy = img_size[1] - font_width_height[1]
        start_at_x = np.random.randint(0, diffx)
        start_at_y = np.random.randint(0, diffy)
        position = (start_at_x, start_at_y)
        return position

    def _unified_processing_step(self, vintage_fraction=.5, gauss_noise_fraction=.25, gauss_noise_high = 1000,
                                first_blur_fraction=.5):
        """This processing step can be used for several own defined sampling functions. 
        Currently only used for sample_localisation_and_multiclassif()"""
        for i in range(1000):
            random_idx = np.random.choice(len(self.bg_jpegs), 1)[0]
            img = Image.open(self.bg_jpegs[random_idx]).convert('RGB')
            if (img.size[0] > self.target_size[0]) and (img.size[1] > self.target_size[1]):
                break
            else:
                pass
        img_original = img

        # create cropped image
        img = self._generate_crop(img)
        img_original_cropped = img

        # add vintage, blur and noise
        img, is_vintage = self._add_random_vintage(img, vintage_fraction)
        img, noise_gauss_value = self._add_gaussian_noise(img, 
                                                        fraction=gauss_noise_fraction,  
                                                        low=50, high=gauss_noise_high)
        img, first_blur_value = self._add_random_blur(img, fraction=first_blur_fraction,low=0.0, high=1.25)

        # generate color
        color = self._generate_rgb_variation()

        # generate rotation
        rotation = self._generate_random_rotation(low=-1.5, high=1.5)
            
        # return
        # other is only passed to sample's return statement and is therfore pooled here
        pool = {'img_original': img_original, 'img_original_cropped': img_original_cropped, 
                 'is_vintage': is_vintage, 'noise_gauss_value': noise_gauss_value, 
                 'first_blur_value': first_blur_value}
        return img, color, rotation, pool

    def sample_localisation_and_multiclassif(self, realistic_date=True, add_point_at_target=False, hardcode_date=None,
            vintage_fraction=.5, gauss_noise_fraction=.25, gauss_noise_high = 1000, first_blur_fraction=.5, 
                                             second_blur_fraction=.5):
        """Generate one sample for OBJECT LOCALISATION task, i.e. bounding box of date and date itself.
        This function uses _unified_processing_step to ensure DRY.

        Args:
            realistic_date (bool): Weather or not a real date should be used. 
            add_point_at_target (bool): If point should be added at center of date-text.
            hardcode_date (str): Use this harcoded six digits. Cannot be used with realistic_code. Default is None.

        Returns:
            mains: Contains main values for modelling: Image, date (ISO-like), digits, bounding box (in pixels), 
                bounding box relative values (between 0 and 1).
            font_dict: Contains infos about font: Name, size, color.
            noise_dict: Contains infos about noise which was added to image: vinage, Gaussian noise, 
                first Gaussian blur, second Gaussian blur, and rotation.
            misc_dict: Misc infos.
        """
        img, color, rotation, pool = self._unified_processing_step(vintage_fraction, 
            gauss_noise_fraction, gauss_noise_high, first_blur_fraction)
        # generate date
        if hardcode_date:
            msg = "hardcoded_date needs to be {}, but is {}."
            assert isinstance(hardcode_date, str), msg.format('string type', type(hardcode_date))
            assert len(hardcode_date) == 6, msg.format('of length 6', len(hardcoded_date))
            digits = hardcode_date
        else:
            digits = self._generate_digits(realistic_date=realistic_date)
        date_formated, date_iso = self._format_digits(digits, first_space_rand=(4, 15), second_space_rand=(4, 9))

        # generate font
        font, font_name, font_lettersize = self._generate_font()
        b_wh = font.getsize(date_formated)

        # generate position of text (upper left position)
        position = self._generate_text_position(img_size=img.size, font_width_height=b_wh)
        # calc center position
        b_xy = (int(position[0] + .5 * b_wh[0]),
                  int(position[1] + .5 * b_wh[1]))
        # bounding_box
        bounding_box = b_xy + b_wh
        
        # add date-text to image (use _add_digit to add one date-string)
        img, _, _ ,_ = self._add_digit(img, position=position, digit=date_formated,
                                       font=font, color=color, rotation=rotation)

        # add blur after date-text was added to image (to blur date-text)
        img, second_blur_value = self._add_random_blur(img, fraction=second_blur_fraction, low=.0, high=.75)

        # add target point at center
        if add_point_at_target:
            draw = ImageDraw.Draw(img)
            draw.ellipse((b_xy[0]-5, b_xy[1]-5, b_xy[0]+5, b_xy[1]+5), fill='yellow', outline='black')
            #del draw
        
        # relative bounding box (between 0 and 1)
        target_size_double = self.target_size + self.target_size
        bounding_box_relative = [bb/tsd for bb, tsd in zip(bounding_box, target_size_double)]
        
        # one-hot-encode target
        #
        
        # prepare return
        mains = {'img': img, 'date_iso': date_iso, 'digits': digits,
                 'bounding_box': bounding_box, 'bounding_box_relative': bounding_box_relative}
        font_dict = {'font': font_name, 'font_lettersize': font_lettersize, 'color': color}
        noise_dict = {'is_vintage': pool['is_vintage'], 'noise_gauss_value': pool['noise_gauss_value'], 
                      'first_blur_value': pool['first_blur_value'], 'second_blur_value': second_blur_value,
                      'rotation': rotation}
        misc_dict = {'position_upperleft_corner': position, 'date_formated': date_formated,
                     #'img_original': pool['img_original'], 'img_original_cropped': pool['img_original_cropped']
                    }
        return mains, font_dict, noise_dict, misc_dict
        
        
    def sample_detection(self, realistic_date=False, add_point_at_target=False,
                         vintage_fraction=0., gauss_noise_fraction=.1, gauss_noise_high = 50, 
                         first_blur_fraction=.5, second_blur_fraction=.5):
        """Generate one sample for OBJECT DETECTION task, i.e. value and bounding box of EACH single digit.
        This function uses _unified_processing_step to ensure DRY.

        Args:
            add_point_at_target (bool): If point should be added at center of date-text.

        Returns:
            mains: contains main values for modelling: The digits and their postions.
                img: The image
                date_isolike (str): The printed date (since this might be unrealistic (1231-23-98) it is called iso-'like'.
                targets (list): List of all digits with a tuple (digit, bounding_box with xywh, upper_left_postion with xy).
            font_dict: Contains infos about font.
            noise_dict: Contains infos about noise which was added to image.
            misc_dict: Misc infos, like intermediate images, postion upper-left and date-formated.
        """
        img, color, rotation, pool = self._unified_processing_step(vintage_fraction=vintage_fraction, 
                                                                   gauss_noise_fraction=gauss_noise_fraction, 
                                                                   gauss_noise_high=gauss_noise_high,
                                                                   first_blur_fraction=first_blur_fraction)

        # generate date
        digits = self._generate_digits(realistic_date=realistic_date)
        date_formated, date_isolike = self._format_digits(digits, 
                                                          first_space_rand=(1, 2), 
                                                          second_space_rand=(1, 2)) #*_space_rand=(1, 2) means that exactyl one whitespace will be used.

        # generate font
        font, font_name, font_lettersize = self._generate_font()
        date_len = len(date_formated)
        date_using8s = "8" * date_len
        font_width_height = font.getsize(date_using8s)

        # position of text (upper left position)
        position = self._generate_text_position(img_size=img.size, font_width_height=font_width_height)

        # calc center position
        center = (int(position[0] + .5 * font_width_height[0]),
                  int(position[1] + .5 * font_width_height[1]))

        # add date to image (use _add_digits to add several date-string)
        img, targets  = self._add_digits(img, position=position, digits=date_formated,
                             font=font, color=color, rotation=rotation)

        # add blur after date-text was added to image (to blur date-text)
        img, second_blur_value = self._add_random_blur(img, fraction=second_blur_fraction, low=.0, high=.75)

        # add target point at center
        if add_point_at_target:
            draw = ImageDraw.Draw(img)
            draw.ellipse((center[0]-5, center[1]-5, center[0]+5, center[1]+5), fill='yellow', outline='black')
            #del draw
            
        mains = {'img': img, 'date_isolike': date_isolike, 'targets': targets}
        font_dict = {'font': font_name, 'font_lettersize': font_lettersize, 'color': color}
        noise_dict = {'is_vintage': pool['is_vintage'], 'noise_gauss_value': pool['noise_gauss_value'], 
                      'first_blur_value': pool['first_blur_value'], 'second_blur_value': second_blur_value,
                      'rotation': rotation}
        misc_dict = {'position_upperleft_corner': position, 'digits': digits,
                     'date_formated': date_formated, 'date_isolike': date_isolike,
                     'img_original': pool['img_original'], 'img_original_cropped': pool['img_original_cropped']}
        return mains, font_dict, noise_dict, misc_dict
    
    @staticmethod
    def _digits_to_one_hot(digits):
        """Creates a numpy one_hot_encoding of the date
        
        Args:
            digits: ISO date as string only the 2 least significant digits of the year are used
            
        Returns:
            one-hot encoding np.array, each row is one digit
        """
        encoding = np.zeros((11, 11), int)
        np.fill_diagonal(encoding, 1)
        one_hot = np.zeros((6, 11))
        for idx, val in enumerate(digits):
            try:
                val = int(val)
            except ValueError:
                val = 10
            one_hot[idx] = encoding[val]
        return one_hot

    @staticmethod
    def one_hot_to_digits(one_hot: np.array):
        digits = ""
        for digit in np.argmax(one_hot, axis=1):
            if digit == 10:
                digits += " "
            else:
                digits += str(digit)
        return digits

    def pytorch_generator(self, batch_size, device, realistic_date=False, transform=None):
        """returns a data generator function for use with PyTorch

        Args:
            batch_size (int): size of the batches the generator returns
            device (torch.device): selects if cpu or cuda tensor is returned
            realistic_date (bool): if false the function can return any 6 digits, not necessarily a valid date
            task_type (str): selects task type of the target

        Example:
            >> all(type(item) == torch.Tensor for item in next(get.pytorch_generator(16, "cpu")))
            True
        """

        import torch
        from torchvision.transforms.functional import to_tensor
        from pytorch_helper import combine_labels

        if type(device) == str and device in ["cpu", "cuda"]:
            device = torch.device(device)

        np_images = np.zeros([batch_size] + list(self.target_size)[::-1] + [3])
        np_pos_labels = np.zeros((batch_size, 4))
        np_digit_labels = np.zeros((batch_size, 6, 11))
        while True:
            images = list()
            for i in range(batch_size):
                mains, font_dict, noise_dict, misc_dict = self.sample_localisation_and_multiclassif(realistic_date)
                np_pos_labels[i,:] = mains["bounding_box_relative"]
                np_digit_labels[i, :, :] = self._digits_to_one_hot(mains["digits"])
                images.append(to_tensor(mains["img"]))
            pos_labels = torch.tensor(np_pos_labels, dtype=torch.float)
            digit_labels = torch.tensor(np_digit_labels, dtype=torch.float)
            x = torch.utils.data.dataloader.default_collate(images)
            y = combine_labels(pos_labels, digit_labels, batch_size)
            if device.type == "cuda":
                yield x.cuda(), y.cuda()
            else:
                yield x, y


# %%
if __name__ == "__main__":
    #
    # Set parameters for image generation:
    #
    
    #To make it reproducable when an error occurs
    if 'n' not in locals():
        n=1
        random.seed(n)
        np.random.seed(n)
    print("Current running random number for reproducibility: {}\n".format(n))
    n += 1

    # import and prep
    #from helper import get_dseg_fonts, draw_box
    from tools import basic, draw
    font_paths = basic.get_dseg_fonts()
    font_paths
    
    # parameters for ImageGenerator
    bg_image_path = os.path.expanduser("val2017/")
    target_size = (200, 100)
    #font_paths = ["fonts/DSEG7Classic-Bold.ttf", "fonts/DSEG7Classic-Regular.ttf"]
    font_lettersize_range = (14, 20)
    h = (30/360, 100/360)
    s = (70/100, 100/100)
    v = (70/100, 100/100)

    # image generator
    gen = DateImageGenerator(bg_image_path=bg_image_path, target_size=target_size,
                             font_paths=font_paths, font_lettersize_range=font_lettersize_range)

    # Try it out (generates one sample)
    # use sample_localisation() for object localisation and sample_detect() for object detection
    print("\n\nObject localisation add digits with a random number of whitespaces.")
    (mains, font_dict, noise_dict, misc_dict) = gen.sample_localisation_and_multiclassif(realistic_date=False, 
        vintage_fraction=0, gauss_noise_fraction=0,first_blur_fraction=0, second_blur_fraction=0)
    print("\nReturns:\n", font_dict)
    print(noise_dict)

    try:
        print("\nCreated digits (w/ spaces) [digits]:\t\t<{}>".format(mains['digits']))
        print("Printed digits (w/ spaces) [date_formated]:\t<{}>".format(misc_dict['date_formated']))
        print("Digit formated as ISO date [date_iso]:\t\t<{}>\nImage:".format(mains['date_iso']))
    except:
        print("\nLegend:('digit', (center-x, center-y, width, height), (upper-left-x, upper-left-y))\n", mains['targets'],"\n")
        print("Image with date: {} -> {}".format(misc_dict['date_formated'], mains['date_isolike']))

    display(mains['img'])
    print("Image with date and box around it:")
    ul, lr = basic.xywh2ullr(mains['bounding_box'])
    #sol = draw.draw_box(img, position, lower_right, fill=0, width=3)
    display(draw.draw_box(mains['img'], ul, lr, fill=255, width=1))
    # print mask
    #print("Mask location [0=b(l)ackground; 1=date]:")
    #mask_np = basic.make_mask(img_pil_size=mains['img'].size, bounding_box=mains['bounding_box'])
    #mask_pil = basic.np2pil(mask_np)
    #display(mask_pil)

# %%
mains, font_dict, noise_dict, misc_dict = gen.sample_detection()
mains['img']


# %%
mains, misc_dict

# %%

