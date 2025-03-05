"""
DSC 20 Project
Name(s): Mac Dang, Brian Liu
PID(s):  A18631118, A18235912
Sources: Lecture slides
"""

import numpy as np
import os
from PIL import Image
import wave
import struct
# import matplotlib.pyplot as plt

NUM_CHANNELS = 3

# 0 --> no intensity (completely dark)
# 255 --> maximum intensity (full brightness)

# 0 0 0 --> Black
# 255 255 255 --> White

# 255 0 0 --> Red
# 0 255 0 --> Green
# 0 0 255 --> Blue

# 255 255 0 --> Yellow
# 0 255 255 --> Cyan
# 225 0 255 --> Magenta

# --------------------------------------------------------------------------- #

# YOU SHOULD NOT MODIFY THESE TWO METHODS

def img_read_helper(path):
    """
    Creates an RGBImage object from the given image file
    """
    # Open the image in RGB
    img = Image.open(path).convert("RGB")
    # Convert to numpy array and then to a list
    matrix = np.array(img).tolist()
    # Use student's code to create an RGBImage object
    return RGBImage(matrix)


def img_save_helper(path, image):
    """
    Saves the given RGBImage instance to the given path
    """
    # Convert list to numpy array
    img_array = np.array(image.get_pixels())
    # Convert numpy array to PIL Image object
    img = Image.fromarray(img_array.astype(np.uint8))
    # Save the image object to path
    img.save(path)


# --------------------------------------------------------------------------- #

# Part 1: RGB Image #
class RGBImage:
    """
    Represents an image in RGB format
    """


    def __init__(self, pixels):
        """
        Initializes a new RGBImage object

        # Test with non-rectangular list
        >>> pixels = [
        ...              [[255, 255, 255], [255, 255, 255]],
        ...              [[255, 255, 255]]
        ...          ]
        >>> RGBImage(pixels)
        Traceback (most recent call last):
        ...
        TypeError

        # Test instance variables
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.pixels
        [[[255, 255, 255], [0, 0, 0]]]
        >>> img.num_rows
        1
        >>> img.num_cols
        2
        """
        
        # If there is an empty image
        if not pixels or all(row == [] for row in pixels):
            self.pixels = []
            self.num_rows = 0
            self.num_cols = 0
            return
        
        # Get the number of rows of the image
        num_rows = len(pixels)
        
        # Get the number of columns by getting the length of the first row
        num_cols = len(pixels[0]) if pixels and pixels [0] else 0
        
        # Set the instance vars
        self.pixels = pixels
        self.num_rows = num_rows
        self.num_cols = num_cols
        
        # Raise exceptions here

        # Ensure 'pixels' is a non empty list
        if not isinstance(pixels, list) or len(pixels) < 1:
            raise TypeError()
        
        # Ensure every row in 'pixels' is a non empty list
        if not all([isinstance(row, list) for row in pixels]) or not all([len(row) >= 1 for row in pixels]):
            raise TypeError()
        
        # Ensure all rows in 'pixels' have the same length for a rectangular shape
        if not all([len(row) == len(pixels[0]) for row in pixels]):
            raise TypeError()

        # Ensure all elements in each row is a list
        if not all([isinstance(col, list) for row in pixels for col in row]):
            raise TypeError()
        
        # Ensure every pixel list has three elements (for R, G, B)
        if not all(isinstance(col, list) and len(col) == 3 for row in pixels for col in row):
            raise TypeError()

        # Ensure each value in the channel is within [0, 255]
        if not all([channel in range(0, 256) for row in pixels for col in row for channel in col]):
            raise ValueError()

    def size(self):
        """
        Returns the size of the image in (rows, cols) format

        # Make sure to complete __init__ first
        
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (1, 2)
        
        >>> pixels = [
        ...              [[1, 1, 1], [2, 2, 2], [3, 3, 3]], # Row 0
        ...              [[4, 4, 4], [5, 5, 5], [6, 6, 6]]  # Row 1
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.size()
        (2, 3)
        
        """
    
        # Returns the number of rows, columns
        return self.num_rows, self.num_cols
        
    def get_pixels(self):
        """
        Returns a copy of the image pixel array
        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_pixels = img.get_pixels()

        # Check if this is a deep copy
        >>> img_pixels                               # Check the values
        [[[255, 255, 255], [0, 0, 0]]]
        >>> id(pixels) != id(img_pixels)             # Check outer list
        True
        >>> id(pixels[0]) != id(img_pixels[0])       # Check row
        True
        >>> id(pixels[0][0]) != id(img_pixels[0][0]) # Check pixel
        True
        """
        
        new_pixels = [[list(col) for col in row] for row in self.pixels]
        
        return new_pixels

    def copy(self):
        """
        Returns a copy of this RGBImage object

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_copy = img.copy()

        # Check that this is a new instance
        >>> id(img_copy) != id(img)
        True
        """
        
        return RGBImage(self.get_pixels())

    def get_pixel(self, row, col):
        """
        Returns the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid index
        >>> img.get_pixel(1, 0)
        Traceback (most recent call last):
        ...
        ValueError

        # Run and check the returned value
        >>> img.get_pixel(0, 0)
        (255, 255, 255)
        """
        
        if not ((row >= 0 and row < self.num_rows) and (col >= 0 and col < self.num_cols)):
            raise ValueError()
        
        return tuple(self.pixels[row][col])
    
    def set_pixel(self, row, col, new_color):
        """
        Sets the (R, G, B) value at the given position

        # Make sure to complete __init__ first
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)

        # Test with an invalid new_color tuple
        >>> img.set_pixel(0, 0, (256, 0, 0))
        Traceback (most recent call last):
        ...
        ValueError

        # Check that the R/G/B value with negative is unchanged
        >>> img.set_pixel(0, 0, (-1, 0, 0))
        >>> img.pixels
        [[[255, 0, 0], [0, 0, 0]]]
        
        # pixels = [[[255, 255, 255], [0, 0, 0]]]
        # (num_to_change, num_to_change, (red, green, blue))
        # red = -1 --> don't change it, only green and blue
        # [[[255, 0, 0], [0, 0, 0]]]
        
        >>> pixels = [
        ...             [[255, 255, 255], [128, 128, 128], [0, 0, 0]],      # Row 0
        ...             [[100, 100, 100], [50, 50, 50], [200, 200, 200]]    # Row 1
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img.set_pixel(1, 2, (3, 4, 5))
        >>> img.pixels
        [[[255, 255, 255], [128, 128, 128], [0, 0, 0]], [[100, 100, 100], [50, 50, 50], [3, 4, 5]]]
        
        """
        
        # Ensure correct row and column types
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        
        # Ensure correct row and column bounds
        if row not in range(self.num_rows) or col not in range(self.num_cols):
            raise ValueError()
        
        # Ensure new_color is a tuple of exactly three ints
        if not isinstance(new_color, tuple) or len(new_color) != 3 or not all(isinstance(channel, int) for channel in new_color):
            raise TypeError() 

        if any(channel > 255 for channel in new_color):
            raise ValueError()
        
        for channel in range(len(self.pixels[row][col])):
            if new_color[channel] >= 0:
                self.pixels[row][col][channel] = new_color[channel]
    

# Part 2: Image Processing Template Methods #
class ImageProcessingTemplate:
    """
    Contains assorted image processing methods
    Intended to be used as a parent class
    """

    def __init__(self):
        """
        Creates a new ImageProcessingTemplate object

        # Check that the cost was assigned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost
        0
        """
        
        self.cost = 0

    def get_cost(self):
        """
        Returns the current total incurred cost

        # Check that the cost value is returned
        >>> img_proc = ImageProcessingTemplate()
        >>> img_proc.cost = 50 # Manually modify cost
        >>> img_proc.get_cost()
        50
        """
        
        return self.cost

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check if this is returning a new RGBImage instance
        >>> img_proc = ImageProcessingTemplate()
        >>> pixels = [
        ...              [[255, 255, 255], [0, 0, 0]]
        ...          ]
        >>> img = RGBImage(pixels)
        >>> img_negate = img_proc.negate(img)
        >>> id(img) != id(img_negate) # Check for new RGBImage instance
        True

        # The following is a description of how this test works
        # 1 Create a processor
        # 2/3 Read in the input and expected output
        # 4 Modify the input
        # 5 Compare the modified and expected
        # 6 Write the output to file
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()                            # 1
        >>> img = img_read_helper('img/test_image_32x32.png')                 # 2
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')  # 3
        >>> img_negate = img_proc.negate(img)                               # 4
        >>> img_negate.pixels == img_exp.pixels # Check negate output       # 5
        True
        >>> img_save_helper('img/out/test_image_32x32_negate.png', img_negate)# 6
        """
        
        # Invert each pixel by (255 - intensity_value)
        inverted_pixels = [[[255 - val for val in pixel] for pixel in row] for row in image.get_pixels()]
        
        return RGBImage(inverted_pixels)

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_gray.png')
        >>> img_gray = img_proc.grayscale(img)
        >>> img_gray.pixels == img_exp.pixels # Check grayscale output
        True
        >>> img_save_helper('img/out/test_image_32x32_gray.png', img_gray)
        """
        
        
        # (R, G, B) -> ((R + G + B) / 3, (R + G + B) / 3, (R + G + B) / 3)
        # [sum([R, G, B]) // 3] * 3
        grayscale_pixels = [[[sum(pixel) // 3] * 3 for pixel in row] for row in image.get_pixels()]
        
        return RGBImage(grayscale_pixels)
        
    def rotate_180(self, image):
        """
        Returns a rotated version of the given image

        # See negate for info on this test
        # You can view the output in the img/out/ directory
        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_rotate.png')
        >>> img_rotate = img_proc.rotate_180(img)
        >>> img_rotate.pixels == img_exp.pixels # Check rotate_180 output
        True
        >>> img_save_helper('img/out/test_image_32x32_rotate.png', img_rotate)
        """
        
        # Start by reversing the rows
        # Then reverse columns
        # [::-1]
        
        rotated180_pixels = [row[::-1] for row in image.get_pixels()[::-1]]
        
        return RGBImage(rotated180_pixels)
        
    def get_average_brightness(self, image):
        """
        Returns the average brightness for the given image

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.get_average_brightness(img)
        86
        """
        
        # pix_avg = (100 + 101 + 101) // 3 = 100
        # all_avg = sum of all pix_avg // number of pixels
        # the sum of the sum of the sum of each pixel (and averaging them) of each row of each column
        
        total_brightness_pixels = sum(sum(sum(pixel) // 3 for pixel in row) for row in image.get_pixels())
        
        all_pixels = image.num_rows * image.num_cols
        
        return total_brightness_pixels // all_pixels

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level

        >>> img_proc = ImageProcessingTemplate()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_adjusted.png')
        >>> img_adjust = img_proc.adjust_brightness(img, 1.2)
        >>> img_adjust.pixels == img_exp.pixels # Check adjust_brightness
        True
        >>> img_save_helper('img/out/test_image_32x32_adjusted.png', img_adjust)
        """
        
        new_img = [[list(map(lambda x: int(x * intensity) if x * intensity <= 255 and x * intensity >= 0 else 0 if x * intensity < 0 else 255, pixel)) for pixel in row] for row in image.get_pixels()]

        if not isinstance(intensity, float):
            raise TypeError()
        
        return RGBImage(new_img)
        
    
# Part 3: Standard Image Processing Methods #
class StandardImageProcessing(ImageProcessingTemplate):
    """
    Represents a standard tier of an image processor
    """

    def __init__(self):
        """
        Creates a new StandardImageProcessing object

        # Check that the cost was assigned
        >>> img_proc = StandardImageProcessing()
        >>> img_proc.cost
        0
        """
        
        super().__init__()
        
        self.cost = 0
        self.free_uses = 0

    def negate(self, image):
        """
        Returns a negated copy of the given image

        # Check the expected cost
        >>> img_proc = StandardImageProcessing()
        >>> img_in = img_read_helper('img/square_32x32.png')
        >>> negated = img_proc.negate(img_in)
        >>> img_proc.get_cost()
        5

        # Check that negate works the same as in the parent class
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_negate.png')
        >>> img_negate = img_proc.negate(img)
        >>> img_negate.pixels == img_exp.pixels # Check negate output
        True
        """
        
        negate_monetized = super().negate(image)
        
        if self.free_uses > 0:
            self.free_uses -= 1
        else:
            self.cost += 5
        
        return negate_monetized

    def grayscale(self, image):
        """
        Returns a grayscale copy of the given image

        """
        
        grayscale_monetized = super().grayscale(image)

        if self.free_uses > 0:
            self.free_uses -= 1
        else:
            self.cost += 6
        
        return grayscale_monetized

    def rotate_180(self, image):
        """
        Returns a rotated version of the given image
        """
        
        rotate_180_monetized = super().rotate_180(image)

        if self.free_uses > 0:
            self.free_uses -= 1
        else:
            self.cost += 10
        
        return rotate_180_monetized

    def adjust_brightness(self, image, intensity):
        """
        Returns a new image with adjusted brightness level
        """
        
        adjust_brightness_monetized = super().adjust_brightness(image, intensity)

        if self.free_uses > 0:
            self.free_uses -= 1
        else:
            self.cost += 1
        
        return adjust_brightness_monetized

    def redeem_coupon(self, amount):
        """
        Makes the given number of methods calls free

        # Check that the cost does not change for a call to negate
        # when a coupon is redeemed
        >>> img_proc = StandardImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_proc.redeem_coupon(1)
        >>> img = img_proc.rotate_180(img)
        >>> img_proc.get_cost()
        0
        """
        
        if not isinstance(amount, int):
            raise TypeError()
        
        if amount <= 0:
            raise ValueError()
        
        self.free_uses += amount
