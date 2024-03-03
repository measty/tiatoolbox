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


class StainFalseColor:
    """Does stain separation on a H&E image, and then makes a false color image
    by associating each channel with a color.
    """

    def __init__(self, channel_map):
        """Initializes the class
        Args:
            channel_map (dict): dictionary of channel to color mapping
        """
        self.channel_map = channel_map
        self.cdx2_muc5_stain_dict = np.array(
            [[0.299, 0.762, 0.575], [0.739, 0.279, 0.614], [0.497, 0.39, -0.775]],
        )
        self.muc2_stain_dict = np.array(
            [[0.309, 0.716, 0.626], [0.168, 0.372, 0.913], [0.922, -0.388, -0.012]],
        )
        # build cmap matrix
        self.colour_matrix = np.array([color for color in self.channel_map.values()])

    def __call__(self, img):
        """Converts a H&E image to a false color image depicting the stains
        Args:
            img (ndarray): input image
        Returns:
            ndarray: RGB image
        """
        # convert to RGB
        img = img.copy()
        img = img[:, :, 0:3]

        # extract cdx2 and muc5 stains
        norm = CustomNormalizer(self.cdx2_muc5_stain_dict)
        # import pdb; pdb.set_trace()
        stain_img = norm.get_concentrations(img, self.cdx2_muc5_stain_dict)
        stain_img = np.reshape(stain_img, [img.shape[0], img.shape[1], 3])
        cdx2 = stain_img[:, :, 0]
        muc5 = stain_img[:, :, 1]

        # extract muc2 stains
        norm = CustomNormalizer(self.muc2_stain_dict)
        stain_img = norm.get_concentrations(img, self.muc2_stain_dict)
        stain_img = np.reshape(stain_img, [img.shape[0], img.shape[1], 3])
        muc2 = stain_img[:, :, 1]

        cdx2[cdx2 < 0.30] = 0
        muc5[muc5 < 0.30] = 0
        muc2[muc2 < 0.10] = 0

        stain_img = np.concatenate(
            [cdx2[:, :, np.newaxis], muc2[:, :, np.newaxis], muc5[:, :, np.newaxis]],
            axis=2,
        )

        # return cdx2, muc2, muc5
        # import pdb; pdb.set_trace()
        return (np.einsum("ijk,kl->ijl", stain_img, self.colour_matrix) * 255).astype(
            np.uint8,
        )
