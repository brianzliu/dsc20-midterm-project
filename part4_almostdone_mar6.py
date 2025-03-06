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
        
        # (i, j)    -1          0           1
        # -1        (-1, -1)    (-1,0)      (-1,1)
        # 0         (0,-1)      (0,0)       (0,1)
        # 1         (1,-1)      (1,0)       (1,1)
        
        # Need to turn it into grayscale first
        grayscale_image = self.grayscale(image)
        grayscale_pixels = grayscale_image.get_pixels()
            
        # Image dimensions
        height = len(grayscale_pixels)
        width = len(grayscale_pixels[0])
        
        kernel = [[-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]]
                

        edge_highlight_pixels = [[0 for i in range(width)] for i in range(height)]

        for row in range(1, height - 1):
            for col in range(1, width - 1):
                edge_value = 0
                
                # Generate values -1, 0, 1
                for i in range(3):
                    for j in range(3):
                        weight = kernel[i][j]
                        pixel_value = grayscale_pixels[row + i - 1][col + j - 1][0]
                        edge_value += pixel_value * weight
                
                # Need values within 0, 255
                edge_value = max(0, min(255, edge_value))
                edge_highlight_pixels[row][col] = edge_value
        
        final_image = [[[edge_highlight_pixels[row][col]] * 3 for col in range(width)] for row in range(height)]

        return RGBImage(final_image)
