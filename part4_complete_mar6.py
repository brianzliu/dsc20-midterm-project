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

        edge_highlight_pixels = [[[0, 0, 0] for i in range(width)] for i in range(height)]

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


# def audio_visualizer(path, start=0, end=5):
#     """
#     Visualizes the given WAV file
#     x-axis: time (sec)
#     y-axis: wave amplitude
#     ---
#     Parameters:
#         path (str): path to the WAV file
#         start (int): start timestamp in seconds, default 0
#         end (int): end timestamp in seconds, default 5
#     """
#     with wave.open(path, "rb") as wav_file:
#         sample_freq = wav_file.getframerate()  # Sample rate
#         n_samples = wav_file.getnframes()  # Total number of samples
#         duration = n_samples/sample_freq # Duration of audio, in seconds

#         if any([type(param) != int for param in [start, end]]):
#             raise TypeError("start and end should be integers.")
#         if (start < 0) or (start > duration) or (end < 0) or (end > duration) or start >= end:
#             raise ValueError(f"Invalid timestamp: start and end should be between 0 and {int(duration)}, and start < end.")
        
#         num_frames = wav_file.getnframes()  # Total number of frames
#         num_channels = wav_file.getnchannels()  # Number of channels (1 for mono, 2 for stereo)
#         sample_width = wav_file.getsampwidth()  # Number of bytes per sample (e.g., 2 for 16-bit)
        
#         # Extract audio wave as list
#         raw_bytes = wav_file.readframes(num_frames)
#         fmt = f"{num_frames * num_channels}{'h' if sample_width == 2 else 'B'}"
#         audio_data = list(struct.unpack(fmt, raw_bytes))

#         # Plot the audio wave
#         time = np.linspace(start, end, num=(end - start)*sample_freq)
#         audio_data = audio_data[start*sample_freq:end*sample_freq]
#         plt.figure(figsize=(15, 5))
#         plt.ylim([-32768, 32767])
#         plt.plot(time, audio_data)
#         plt.title(f'Audio Plot of {path} from {start}s to {end}s')
#         plt.ylabel('sound wave')
#         plt.xlabel('time (s)')
#         plt.xlim(start, end)
#         plt.show()


# --------------------------------------------------------------------------- #
