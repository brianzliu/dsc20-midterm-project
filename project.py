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
        num_rows = len(pixels)
        num_cols = len(pixels[0])
        self.pixels = pixels
        self.num_rows = num_rows
        self.num_cols = num_cols
        # Raise exceptions here

        if not isinstance(pixels, list) or len(pixels) < 1:
            raise TypeError()
        
        # num_rows needs to be of type list and have at least one element
        if not all([isinstance(row, list) for row in pixels]) or not all([len(row) >= 1 for row in pixels]):
            raise TypeError()
        
        if not all([len(row) == len(pixels[0]) for row in pixels]):
            raise TypeError()

        if not all([isinstance(col, list) for row in pixels for col in row]):
            raise TypeError()
            
        if not all([len(col) == 3 for row in pixels for col in row]):
            raise TypeError()

        # ValueError one
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
        """

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

        return (self.pixels[row][col][0], self.pixels[row][col][1], self.pixels[row][col][2])

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
        """
        if not isinstance(row, int) or not isinstance(col, int):
            raise TypeError()
        if row not in range(0, self.size()[0]) or col not in range(0, self.size()[1]):
            raise ValueError()
        if not isinstance(new_color, tuple) or len(new_color) != 3 or not all([isinstance(channel, int) for channel in new_color]):
            raise TypeError() 

        if any([channel > 255 for channel in new_color]):
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
        grayscale_pixels = [[[sum(pixel) // 3 for _ in range(len(pixel))] for pixel in row] for row in image.get_pixels()]
        
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
        total_brightness_pixels = sum([sum([sum(pixel) // 3 for pixel in row]) for row in image.get_pixels()])
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



# Part 4: Premium Image Processing Methods #
class PremiumImageProcessing(ImageProcessingTemplate):
    """
    Represents a paid tier of an image processor
    """

    def __init__(self):
        """
        Creates a new PremiumImageProcessing object

        # Check the expected cost
        >>> img_proc = PremiumImageProcessing()
        >>> img_proc.get_cost()
        50
        """
        
        super().__init__()
        
        self.cost = 50

    def pixelate(self, image, block_dim):
        """
        Returns a pixelated version of the image, where block_dim is the size of 
        the square blocks.

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_pixelate = img_proc.pixelate(img, 4)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_pixelate.png')
        >>> img_exp.pixels == img_pixelate.pixels # Check pixelate output
        True
        >>> img_save_helper('img/out/test_image_32x32_pixelate.png', img_pixelate)
        """
        
        # Image dimensions
        image_pixels = image.pixels
        height = len(image_pixels)
        width = len(image_pixels[0])
        
        
        # Deep copy needed of original pixels
        pixelated_image = [[list(pixel) for pixel in row] for row in image_pixels]
        
        # Process image in specified block_dim regions
        for row in range(0, height, block_dim):
            for col in range(0, width, block_dim):
                row_range = min(row + block_dim, height)
                col_range = min(col + block_dim, width)
                
                red_count, green_count, blue_count = 0, 0, 0
                pixel_count = 0
                
                # Calculate average color in the block
                for i in range(row, row_range):
                    for j in range(col, col_range):
                        red, green, blue = image_pixels[i][j]
                        
                        red_count += red
                        green_count += green
                        blue_count += blue
                        pixel_count += 1
                
                red_average = red_count // pixel_count
                green_average = green_count // pixel_count
                blue_average = blue_count // pixel_count
                
                # Change all pixels in the block to the average color
                
                for i in range(row, row_range):
                    for j in range(col, col_range):
                        pixelated_image[i][j] = [red_average, green_average, blue_average]
        
        return RGBImage(pixelated_image)
            

    def edge_highlight(self, image):
        """
        Returns a new image with the edges highlighted

        >>> img_proc = PremiumImageProcessing()
        >>> img = img_read_helper('img/test_image_32x32.png')
        >>> img_edge = img_proc.edge_highlight(img)
        >>> img_exp = img_read_helper('img/exp/test_image_32x32_edge.png')
        >>> img_exp.pixels == img_edge.pixels # Check edge_highlight output
        True
        >>> img_save_helper('img/out/test_image_32x32_edge.png', img_edge)
        """

        # Convert to grayscale first
        grayscale_image = self.grayscale(image)
        grayscale_pixels = grayscale_image.get_pixels()

        # Image dimensions
        height = len(grayscale_pixels)
        width = len(grayscale_pixels[0])

        kernel = [[-1, -1, -1],
                  [-1, 8, -1],
                  [-1, -1, -1]]

        edge_highlight_pixels = [[[0, 0, 0] for j in range(width)] for i in range(height)]

        for row in range(height):
            for col in range(width):
            
                edge_value = 0

                # Apply bounds check
                for i in range(-1, 2):
                    for j in range(-1, 2):
                    
                        new_row, new_col = row + i, col + j
                        
                        if 0 <= new_row < height and 0 <= new_col < width:
                        
                            weight = kernel[i + 1][j + 1]
                            pixel_value = grayscale_pixels[new_row][new_col][0]
                            edge_value += pixel_value * weight

                # Fix values to be between 0 and 255
                edge_value = max(0, min(255, edge_value))
                # Convert back to RGB format
                edge_highlight_pixels[row][col] = [edge_value] * 3

        return RGBImage(edge_highlight_pixels)


# --------------------------------------------------------------------------- #

# YOU SHOULD NOT MODIFY THESE THREE METHODS

def audio_read_helper(path, visualize=False):
    """
    Creates an AudioWave object from the given WAV file
    """
    with wave.open(path, "rb") as wav_file:
        num_frames = wav_file.getnframes()  # Total number of frames
        num_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
        sample_width = wav_file.getsampwidth()  # Number of bytes per sample (e.g., 2 for 16-bit)
        
        # Read the frames as bytes
        raw_bytes = wav_file.readframes(num_frames)
        
        # Determine the format string for struct.unpack()
        fmt = f"{num_frames * num_channels}{'h' if sample_width == 2 else 'B'}"
        
        # Convert bytes to a list of integers
        audio_data = list(struct.unpack(fmt, raw_bytes))

    return AudioWave(audio_data)


def audio_save_helper(path, audio, sample_rate = 44100):
    """
    Saves the given AudioWave instance to the given path as a WAV file
    """
    sample_rate = 44100  # 44.1 kHz standard sample rate
    num_channels = 1  # Mono
    sample_width = 2  # 16-bit PCM

    # Convert list to bytes
    byte_data = struct.pack(f"{len(audio.wave)}h", *audio.wave)
    with wave.open(path, "wb") as wav_file:
        wav_file.setnchannels(num_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(byte_data)


def audio_visualizer(path, start=0, end=5):
    """
    Visualizes the given WAV file
    x-axis: time (sec)
    y-axis: wave amplitude
    ---
    Parameters: 
        path (str): path to the WAV file
        start (int): start timestamp in seconds, default 0
        end (int): end timestamp in seconds, default 5
    """
    with wave.open(path, "rb") as wav_file:
        sample_freq = wav_file.getframerate()  # Sample rate
        n_samples = wav_file.getnframes()  # Total number of samples
        duration = n_samples/sample_freq # Duration of audio, in seconds

        if any([type(param) != int for param in [start, end]]):
            raise TypeError("start and end should be integers.")
        if (start < 0) or (start > duration) or (end < 0) or (end > duration) or start >= end:
            raise ValueError(f"Invalid timestamp: start and end should be between 0 and {int(duration)}, and start < end.")
        
        num_frames = wav_file.getnframes()  # Total number of frames
        num_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
        sample_width = wav_file.getsampwidth()  # Number of bytes per sample (e.g., 2 for 16-bit)
        
        # Extract audio wave as list
        raw_bytes = wav_file.readframes(num_frames)
        fmt = f"{num_frames * num_channels}{'h' if sample_width == 2 else 'B'}"
        audio_data = list(struct.unpack(fmt, raw_bytes))

        # Plot the audio wave
        time = np.linspace(start, end, num=(end - start)*sample_freq)
        audio_data = audio_data[start*sample_freq:end*sample_freq]
        plt.figure(figsize=(15, 5))
        plt.ylim([-32768, 32767])
        plt.plot(time, audio_data)
        plt.title(f'Audio Plot of {path} from {start}s to {end}s')
        plt.ylabel('sound wave')
        plt.xlabel('time (s)')
        plt.xlim(start, end)
        plt.show()


# --------------------------------------------------------------------------- #

# Part 5: Multimedia Processing
class AudioWave():
    """
        Represents audio through a 1-dimensional array of amplitudes
    """
    def __init__(self, amplitudes):
        self.wave = amplitudes

class PremiumPlusMultimediaProcessing(PremiumImageProcessing):
    """
        Represents the paid tier of multimedia processing
    """
    def __init__(self):
        """
        Creates a new PremiumPlusMultimediaProcessing object

        # Check the expected cost
        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> multi_proc.get_cost()
        75
        """
        super().__init__()
        self.cost = 75
    
    def reverse_song(self, audio):
        """
        Reverses the audio of the song.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_reversed = multi_proc.reverse_song(audio)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_reversed.wav')
        >>> audio_exp.wave == audio_reversed.wave # Check reverse_song output
        True
        >>> audio_save_helper('audio/out/one_summers_day_reversed.wav', audio_reversed)
        """
        if not isinstance(audio, AudioWave):
            raise TypeError()
        
        return AudioWave([amp for amp in audio.wave[::-1]])
    
    def slow_down(self, audio, factor):
        """
        Slows down the song by a certain factor.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_slow = multi_proc.slow_down(audio, 2)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_slow.wav')
        >>> audio_exp.wave == audio_slow.wave # Check slow_down output
        True
        >>> audio_save_helper('audio/out/one_summers_day_slow.wav', audio_slow)

        # >>> audio = audio_read_helper('audio/what_once_was.wav')
        # >>> audio_slow = multi_proc.slow_down(audio, 2)
        # >>> audio_save_helper('audio/out/what_once_was_slow.wav', audio_slow)
        """
        if not isinstance(audio, AudioWave):
            raise TypeError()
        if not isinstance(factor, int):
            raise TypeError()
        if not factor > 0:
            raise ValueError()
        
        return AudioWave([amp for amp in audio.wave for _ in range(factor)])
    
    def speed_up(self, audio, factor):
        """
        Speeds up the song by a certain factor.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_sped_up = multi_proc.speed_up(audio, 2)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_sped_up.wav')
        >>> audio_exp.wave == audio_sped_up.wave # Check speed_up output
        True
        >>> audio_save_helper('audio/out/one_summers_day_sped_up.wav', audio_sped_up)
        """
        if not isinstance(audio, AudioWave):
            raise TypeError()
        if not isinstance(factor, int):
            raise TypeError()
        if not factor > 0:
            raise ValueError()
        
        return AudioWave([amp for amp in audio.wave[::factor]])

    def reverb(self, audio):
        """
        Adds a reverb/echo effect to the song.

        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_reverb = multi_proc.reverb(audio)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_reverb.wav')
        >>> audio_exp.wave == audio_reverb.wave # Check reverb output
        True
        >>> audio_save_helper('audio/out/one_summers_day_reverb.wav', audio_reverb)

        # >>> audio = audio_read_helper('audio/what_once_was.wav')
        # >>> audio_reverb = multi_proc.reverb(audio)
        # >>> audio_save_helper('audio/out/what_once_was_reverb.wav', audio_reverb)
        """
        if not isinstance(audio, AudioWave):
            raise TypeError()
        
        return AudioWave(audio.wave[:4] + [min(32767, max(-32768, round(sum([audio.wave[i-5+j] * (j/5) for j in range(1,6)])))) for i in range(4, len(audio.wave))])
        
    def clip_song(self, audio, start, end):
        """
        Clips a song based on a specified start and end.
        
        >>> multi_proc = PremiumPlusMultimediaProcessing()
        >>> audio = audio_read_helper('audio/one_summers_day.wav')
        >>> audio_clipped = multi_proc.clip_song(audio, 30, 70)
        >>> audio_exp = audio_read_helper('audio/exp/one_summers_day_clipped.wav')
        >>> audio_exp.wave == audio_clipped.wave # Check clip_song output
        True
        >>> audio_save_helper('audio/out/one_summers_day_clipped.wav', audio_clipped)
        """
        if not isinstance(audio, AudioWave):
            raise TypeError()
        if not isinstance(start, int) or not isinstance(end, int):
            raise TypeError()
        if not 0 <= start <= 100:
            raise ValueError()
        if not 0 <= end <= 100:
            raise ValueError()
        
        if end <= start:
            return AudioWave([])
        
        start_index = int(len(audio.wave) * (start / 100))
        end_index = int(len(audio.wave) * (end / 100))

        return AudioWave(audio.wave[start_index:end_index+1])


# Part 6: Image KNN Classifier #
class ImageKNNClassifier:
    """
    Represents a simple KNNClassifier
    """

    def __init__(self, k_neighbors):
        """
        Creates a new KNN classifier object
        """
        
        self.k_neighbors = k_neighbors
        
        # List of (image, label) tuples where 'image' is a RGBImage and 'label' is a string
        self.data = []

    def fit(self, data):
        """
        Stores the given set of data and labels for later
        """
        
        assert isinstance(data, list)
        assert all(isinstance(item, tuple) and len(item) == 2 for item in data)
        assert all(isinstance(item[1], str) for item in data)
        
        if len(data) < self.k_neighbors:
            raise ValueError
        
        self.data = data

    def distance(self, image1, image2):
        """
        Returns the distance between the given images

        >>> img1 = img_read_helper('img/steve.png')
        >>> img2 = img_read_helper('img/knn_test_img.png')
        >>> knn = ImageKNNClassifier(3)
        >>> knn.distance(img1, img2)
        15946.312896716909
        """
        
        # Parameters
        # - Image1: RGBImage instance
        # - Image2: RGBImage instance
        
        # Exceptions
        # - If either (or both) are not RGBImage instances, raise a TypeError().
        # - If they are not the same size, raise a ValueError().
        
        if not isinstance(image1, RGBImage) or not isinstance(image2, RGBImage):
            raise TypeError()
            
        if image1.size() != image2.size():
            raise ValueError()
            
        flat_image1 = [channel for row in image1.pixels for pixel in row for channel in pixel]
        flat_image2 = [channel for row in image2.pixels for pixel in row for channel in pixel]
        
        sum_squared_difference = sum(map(lambda p1, p2: (p1 - p2) ** 2, flat_image1, flat_image2))
        
        euclidean_distance = sum_squared_difference ** 0.5
        
        return euclidean_distance

    def vote(self, candidates):
        """
        Returns the most frequent label in the given list

        >>> knn = ImageKNNClassifier(3)
        >>> knn.vote(['label1', 'label2', 'label2', 'label2', 'label1'])
        'label2'
        """
        return max(set(candidates), key=candidates.count)

    def predict(self, image):
        """
        Predicts the label of the given image using the labels of
        the K closest neighbors to this image

        The test for this method is located in the knn_tests method below
        """
        if len(self.data) == 0:
            raise ValueError()
        
        dists = list(map(lambda fit_img: (self.distance(fit_img[0], image), fit_img[1]), self.data))
        sorted_dists = sorted(dists, key=lambda x: x[0])
        k_sorted_dists = sorted_dists[:self.k_neighbors]
        return self.vote(list(map(lambda img: img[1], k_sorted_dists)))


def knn_tests(test_img_path):
    """
    Function to run knn tests

    >>> knn_tests('img/knn_test_img.png')
    'nighttime'
    """
    # Read all of the sub-folder names in the knn_data folder
    # These will be treated as labels
    path = 'knn_data'
    data = []
    for label in os.listdir(path):
        label_path = os.path.join(path, label)
        # Ignore non-folder items
        if not os.path.isdir(label_path):
            continue
        # Read in each image in the sub-folder
        for img_file in os.listdir(label_path):
            train_img_path = os.path.join(label_path, img_file)
            img = img_read_helper(train_img_path)
            # Add the image object and the label to the dataset
            data.append((img, label))

    # Create a KNN-classifier using the dataset
    knn = ImageKNNClassifier(5)

    # Train the classifier by providing the dataset
    knn.fit(data)

    # Create an RGBImage object of the tested image
    test_img = img_read_helper(test_img_path)

    # Return the KNN's prediction
    predicted_label = knn.predict(test_img)
    return predicted_label