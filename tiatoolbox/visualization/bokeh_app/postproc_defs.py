import numpy as np
import torch
import torchvision.transforms.functional as TF

# from demux_fixed import VirtualRestainer
from PIL import Image
from torch import nn

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
        self.channel_map = {
            "CDX2": [0.6, 0.2, 0.4],
            "MUC2": [1, 1, 0],
            "MUC5": [0, 1, 0],
        }
        self.cdx2_muc5_stain_dict = np.array(
            [[0.299, 0.762, 0.575], [0.739, 0.279, 0.614], [0.497, 0.39, -0.775]],
        )
        self.muc2_stain_dict = np.array(
            [[0.309, 0.716, 0.626], [0.168, 0.372, 0.913], [0.922, -0.388, -0.012]],
        )
        # build cmap matrix
        self.colour_matrix = np.array([color for color in self.channel_map.values()])
        self.channels = [0, 1, 2]

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

        channels = [cdx2, muc2, muc5]

        stain_img = np.zeros((img.shape[0], img.shape[1], len(channels)))
        for i in self.channels:
            stain_img[:, :, i] = channels[i]

        # return selected chanels mapped to their colors
        return (np.einsum("ijk,kl->ijl", stain_img, self.colour_matrix) * 255).astype(
            np.uint8,
        )


class StainLayer(nn.Module):
    """Stain Layer to perform stain deconvolution"""

    def __init__(self, M="ruifrok", cin=3, cout=3, normalize=False, use_bias=False):
        """Parameters
        ----------
        M : TYPE: String ('ruifrok'), None (random initiallization) or numpy matrix, optional
            DESCRIPTION. Stain Matrix The default is 'ruifrok'.
        cin : TYPE integer, optional
            DESCRIPTION. number of channels of input image. Only needed if M is None. The default is 3.
        cout : TYPE, optional
            DESCRIPTION.  number of channels of output image. Only needed if M is None.  The default is 3.
        normalize : TYPE, optional
            DESCRIPTION. Whether to row normalize the stain matrix. The default is False.
        use_bias : TYPE, optional
            DESCRIPTION. Whether to add a bias vecotr to the stain normalized image. The default is False.

        Returns:
        -------
        None.

        """
        super().__init__()
        self.use_bias = use_bias
        self.normalize = normalize
        if M == "mIHC":
            # M = np.random.randn(cin, cout) / 3.0
            M = np.array(
                [
                    [0.62, 0.637, 0.458],  # H
                    [0.29, 0.832, 0.473],  # CDX2 (pink)
                    [0.3, 0.491, 0.818],  # CDX8 (brown)
                    [0.033, 0.343, 0.939],  # MUC2 (yellow)
                    [0.741, 0.294, 0.604],
                ],  # MUC5 (green),
                dtype=np.float64,
            )
            # initialize above using #H,CDX2,DAB,MUC2,MUC5
        elif type(M) == str and M.lower() == "ruifrok":
            M = np.array(
                [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
                dtype=np.float64,
            )  # initialize using Ruifrok
        M = np.linalg.pinv(M)
        self.cin, self.cout = M.shape
        M = M.astype(np.float32)
        weights = torch.tensor(M)
        self.weights = nn.Parameter(weights)
        if self.use_bias:
            bias = torch.Tensor(self.cout)
            self.bias = nn.Parameter(bias)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)  # bias init

    def forward(self, x):
        """Perform stain separation with stain matrix and bias

        Parameters
        ----------
        x : TYPE Torch Tensor NxCinxHxW (N images, Cin channels, H Heighth W Width)
            DESCRIPTION.

        Returns:
        -------
        Z : TYPE Torch Tensor NxCoutxHxW (N images, Cout channels, H Heighth W Width)
            DESCRIPTION.

        """
        # operates in NCHW
        X = torch.max(x, 1e-2 * torch.ones_like(x))
        X = X.permute(0, 2, 3, 1)  # Move to NHWC
        Z = (torch.log(X) / np.log(1e-2)) @ self.StainMatrix
        if self.use_bias:
            Z += self.bias
        Z = Z.permute(0, 3, 1, 2)  # ... and move back
        return Z

    def reverse(self, x):
        """Perform stain recombination with stain matrix and bias

        Parameters
        ----------
        x : TYPE Torch Tensor NxCoutxHxW (N images, Cout channels, H Heighth W Width)
            DESCRIPTION.

        Returns:
        -------
        Z : TYPE Torch Tensor NxCinxHxW (N images, Cin channels, H Heighth W Width)
            DESCRIPTION.

        """
        log_adjust = -np.log(1e-2)
        Z = x.permute(0, 2, 3, 1)
        if self.use_bias:
            Z -= self.bias
        log_rgb = -(Z * log_adjust) @ torch.pinverse(self.StainMatrix)
        rgb = torch.exp(log_rgb)
        rgb = torch.clamp(rgb, 0, 1)
        Z = rgb.permute(0, 3, 1, 2)
        return Z

    @property
    def StainMatrix(self):
        """Return the stain matrix being used (not biases)

        Returns:
        -------
        TYPE
            DESCRIPTION.

        """
        if self.normalize:
            return fcn.normalize(self.weights)
        return self.weights

    def RemapChannel(self, Z, toPIL=True, channels=None, separate=True):
        """For a SINGLE stain separated image Z, return the list of images mapped to original colors
        For example, if Z is an image containing the stain separated H, E and D channels,
        then this function will return 3 images -- one for H mapped back to the color for H,
        one for E mapped to the color of E and one for D mapped to the color of D

        Parameters
        ----------
        Z : TYPE Torch tensor of size CoutxHxW (a single stain separated Image)
            DESCRIPTION.
        toPIL : TYPE, optional
            DESCRIPTION. Whether to return images in PIL. The default is True. Should be set to False if you want to keep things in torch tensors
        channels : TYPE, list of channels to return optional
            DESCRIPTION. The default is None. (all channels)
        separate : TYPE, optional
            DESCRIPTION. Whether to return the images as separate torch tensors. The default is True.

        Returns:
        -------
        out : TYPE
            DESCRIPTION.

        """
        # Works for a single torch CHW image at a time
        out = []
        if channels is None:
            channels = range(Z.shape[0])
        if not separate:
            out = torch.zeros_like(Z)
            for i in channels:
                out[i] = Z[i]
            out.unsqueeze_(0)
            out = self.reverse(out)[0]
            if toPIL:
                out = TF.to_pil_image(out)  # output in HWC if toPIL = True
            # import pdb; pdb.set_trace()
            return out
        for i in channels:
            null = torch.zeros_like(Z)
            null[i] = Z[i]
            # h0t = torch.tensor(null.astype(np.float32)).unsqueeze_(0)
            null.unsqueeze_(0)
            ih0t = self.reverse(null)[0]  # recombine and map (in CHW)
            if toPIL:
                ih0t = TF.to_pil_image(ih0t)  # output in HWC if toPIL = True
            out.append(ih0t)
        return out


class TorchStainSep:
    def __init__(
        self,
        stain_matrix=None,
        normalize=False,
        use_bias=False,
        channels=None,
        to_rgb=True,
    ):
        """Initializes the class
        Args:
            stain_matrix (ndarray): stain matrix
            normalize (bool): whether to normalize the stain matrix
            use_bias (bool): whether to use bias
        """
        if stain_matrix is None:
            stain_matrix = "mIHC"
        self.stain_layer = StainLayer(
            stain_matrix,
            normalize=normalize,
            use_bias=use_bias,
        )
        if channels is None:
            self.channels = [0, 1, 2, 3, 4]
        else:
            self.channels = channels
        self.to_rgb = to_rgb

    def __call__(self, img):
        with torch.no_grad():
            img = TF.to_tensor(img).unsqueeze(0)
            img = self.stain_layer.forward(img)
            img = img.squeeze(0)
            if self.to_rgb:
                # map the channels we are keeping back to rgb
                img = self.stain_layer.RemapChannel(
                    img,
                    toPIL=False,
                    channels=self.channels,
                    separate=False,
                )
            img = img.permute(1, 2, 0)

        return (img.detach().cpu().numpy() * 255).astype(np.uint8)
