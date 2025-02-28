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
        # YOUR CODE GOES HERE #
        
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

        '''
        return tuple(self.pixels[row][col][0], self.pixels[row][col][1], self.pixels[row][col][2])
        
        # TypeError: tuple expected at most 1 argument, got 3
        # These are three separate arguments
        
        '''
    
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
