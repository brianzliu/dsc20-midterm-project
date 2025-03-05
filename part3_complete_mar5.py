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
