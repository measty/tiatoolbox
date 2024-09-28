import numpy as np

# from demux_fixed import VirtualRestainer
from PIL import Image

from tiatoolbox.tools.stainnorm import (
    CustomNormalizer,
    MacenkoNormalizer,
    VahadaneNormalizer,
)


class StainNormWrapper:
    def __init__(self, stain_norm_method, stain_mat_source=None):
        """Wrapper for stain normalizer classes to use as a postproc function
        Args:
            stain_norm_method (str): stain normalization method
            stain_mat_source (str | ndarray): stain matrix source. Can be a path
            to an image or a stain matrix for 'custom' method.
        """
        self.stain_norm_method = stain_norm_method
        self.stain_norm_source = stain_mat_source
        self.normalizer = None

    def fit(self, img):
        if self.stain_norm_method == "vahadane":
            self.normalizer = VahadaneNormalizer()
        elif self.stain_norm_method == "macenko":
            self.normalizer = MacenkoNormalizer()
        elif self.stain_norm_method == "custom":
            self.normalizer = CustomNormalizer(stain_matrix=self.stain_norm_source)
            return
        else:
            raise ValueError("Unknown stain normalization method")
        # load image in stain_mat_source to learn stain mat
        img = Image.open(self.stain_norm_source)
        self.normalizer.fit(np.array(img))

    def __call__(self, img):
        return self.normalizer.transform(img)


class Fluorescent2RGB:
    """Converts a multi-channel fluorescent image to RGB image,
    by associating each channel with a color.
    """

    def __init__(self, channel_map):
        """Initializes the class
        Args:
            channel_map (dict): dictionary of channel to color mapping
        """
        self.channel_map = channel_map
        # build cmap matrix
        self.colour_matrix = np.array([color for color in self.channel_map.values()])

    def __call__(self, img):
        """Converts a multi-channel fluorescent image to RGB image
        Args:
            img (ndarray): input image
        Returns:
            ndarray: RGB image
        """
        # convert to RGB
        return np.einsum("ijk,kl->ijl", img, self.colour_matrix)
