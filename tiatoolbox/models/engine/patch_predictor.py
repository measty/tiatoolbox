# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# The Original Code is Copyright (C) 2021, TIA Centre, University of Warwick
# All rights reserved.
# ***** END GPL LICENSE BLOCK *****

"""This module implements patch-level prediction."""

import copy
import os
import pathlib
import warnings
from collections import OrderedDict
from typing import Callable, Tuple, Union

import numpy as np
import torch
import tqdm

from tiatoolbox.models.architecture import get_pretrained_model
from tiatoolbox.models.dataset.classification import PatchDataset, WSIPatchDataset
from tiatoolbox.models.engine.semantic_segmentor import IOSegmentorConfig
from tiatoolbox.utils import misc
from tiatoolbox.utils.misc import save_as_json
from tiatoolbox.wsicore.wsireader import VirtualWSIReader, get_wsireader


class IOPatchPredictorConfig(IOSegmentorConfig):
    """Contain patch predictor input and output information."""

    def __init__(
        self,
        patch_input_shape=None,
        input_resolutions=None,
        stride_shape=None,
        **kwargs,
    ):
        stride_shape = patch_input_shape if stride_shape is None else stride_shape
        super().__init__(
            input_resolutions=input_resolutions,
            output_resolutions=[],
            stride_shape=stride_shape,
            patch_input_shape=patch_input_shape,
            patch_output_shape=patch_input_shape,
            save_resolution=None,
            **kwargs,
        )


class PatchPredictor:
    """Patch-level predictor.

    Args:
        model (nn.Module): Use externally defined PyTorch model for prediction with.
          weights already loaded. Default is `None`. If provided,
          `pretrained_model` argument is ignored.
        pretrained_model (str): Name of the existing models support by tiatoolbox
          for processing the data. Refer to
          `tiatoolbox.models.classification.get_pretrained_model` for details.
          By default, the corresponding pretrained weights will also be
          downloaded. However, you can override with your own set of weights
          via the `pretrained_weights` argument. Argument is case insensitive.
        pretrained_weights (str): Path to the weight of the corresponding
          `pretrained_model`.
          >>> predictor = PatchPredictor(
          ...    pretrained_model="resnet18-kather100k",
          ...    pretrained_weights="resnet18_local_weight")
        batch_size (int) : Number of images fed into the model each time.
        num_loader_workers (int) : Number of workers to load the data.
          Take note that they will also perform preprocessing.
        verbose (bool): Whether to output logging information.

    Attributes:
        img (:obj:`str` or :obj:`pathlib.Path` or :class:`numpy.ndarray`):
          A HWC image or a path to WSI.
        mode (str): Type of input to process. Choose from either `patch`, `tile`
          or `wsi`.
        model (nn.Module): Defined PyTorch model.
        pretrained_model (str): Name of the existing models support by tiatoolbox
          for processing the data. Refer to
          `tiatoolbox.models.classification.get_pretrained_model` for details.
          By default, the corresponding pretrained weights will also be
          downloaded. However, you can override with your own set of weights
          via the `pretrained_weights` argument. Argument is case insensitive.
        batch_size (int) : Number of images fed into the model each time.
        num_loader_workers (int): Number of workers used in torch.utils.data.DataLoader.
        verbose (bool): Whether to output logging information.

    Examples:
        >>> # list of 2 image patches as input
        >>> data = [img1, img2]
        >>> predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> # array of list of 2 image patches as input
        >>> data = np.array([img1, img2])
        >>> predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> # list of 2 image patch files as input
        >>> data = ['path/img.png', 'path/img.png']
        >>> predictor = PatchPredictor(pretrained_model="resnet18-kather100k")
        >>> output = predictor.predict(data, mode='patch')

        >>> # list of 2 image tile files as input
        >>> tile_file = ['path/tile1.png', 'path/tile2.png']
        >>> predictor = PatchPredictor(pretraind_model="resnet18-kather100k")
        >>> output = predictor.predict(tile_file, mode='tile')

        >>> # list of 2 wsi files as input
        >>> wsi_file = ['path/wsi1.svs', 'path/wsi2.svs']
        >>> predictor = PatchPredictor(pretraind_model="resnet18-kather100k")
        >>> output = predictor.predict(wsi_file, mode='wsi')

    """

    def __init__(
        self,
        batch_size=8,
        num_loader_workers=0,
        model=None,
        pretrained_model=None,
        pretrained_weights=None,
        verbose=True,
    ):
        super().__init__()

        self.imgs = None
        self.mode = None

        if model is None and pretrained_model is None:
            raise ValueError("Must provide either of `model` or `pretrained_model`")

        if model is not None:
            self.model = model
            ioconfig = None  # retrieve iostate from provided model ?
        else:
            model, ioconfig = get_pretrained_model(pretrained_model, pretrained_weights)

        self.ioconfig = ioconfig  # for storing original
        self._ioconfig = None  # for storing runtime
        self.model = model  # for runtime, such as after wrapping with nn.DataParallel
        self.pretrained_model = pretrained_model
        self.batch_size = batch_size
        self.num_loader_worker = num_loader_workers
        self.verbose = verbose

    @staticmethod
    def merge_predictions(
        img: Union[str, pathlib.Path, np.ndarray],
        output: dict,
        resolution: float = None,
        units: str = None,
        postproc_func: Callable = None,
        return_raw: bool = False,
    ):
        """Merge patch-level predictions to form a 2-dimensional prediction map.

        #! Improve how the below reads.
        The prediction map will contain values from 0 to N, where N is the number
        of classes. Here, 0 is the background which has not been processed by the
        model and N is the number of classes predicted by the model.

        Args:
            img (:obj:`str` or :obj:`pathlib.Path` or :class:`numpy.ndarray`):
              A HWC image or a path to WSI.
            output (dict): Ouput generated by the model.
            resolution (float): Resolution of merged predictions.
            units (str): Units of resolution used when merging predictions. This
              must be the same `units` used when processing the data.
            postproc_func (callable): A function to post-process raw prediction
              from model. By default, internal code uses the `np.argmax` function.
            return_raw (bool): Return raw result without applying the `postproc_func`
              on the assembled image.
        Returns:
            prediction_map (ndarray): Merged predictions as a 2D array.

        Examples:
            >>> # pseudo output dict from model with 2 patches
            >>> output = {
            ...     'resolution': 1.0,
            ...     'units': 'baseline',
            ...     'probabilities': [[0.45, 0.55], [0.90, 0.10]],
            ...     'predictions': [1, 0],
            ...     'coordinates': [[0, 0, 2, 2], [2, 2, 4, 4]],
            ... }
            >>> merged = PatchPredictor.merge_predictions(
            ...         np.zeros([4, 4]),
            ...         output,
            ...         resolution=1.0,
            ...         units='baseline'
            ... )
            >>> merged
            array([[2, 2, 0, 0],
                   [2, 2, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1]])

        """
        reader = get_wsireader(img)
        if isinstance(reader, VirtualWSIReader):
            warnings.warn(
                (
                    "Image is not pyramidal hence read is forced to be "
                    "at `units='baseline'` and `resolution=1.0`."
                )
            )
            resolution = 1.0
            units = "baseline"

        canvas_shape = reader.slide_dimensions(resolution=resolution, units=units)
        canvas_shape = canvas_shape[::-1]  # XY to YX

        # may crash here, do we need to deal with this ?
        output_shape = reader.slide_dimensions(
            resolution=output["resolution"], units=output["units"]
        )
        output_shape = output_shape[::-1]  # XY to YX
        fx = np.array(canvas_shape) / np.array(output_shape)

        if "probabilities" not in output.keys():
            coordinates = output["coordinates"]
            predictions = output["predictions"]
            denominator = None
            output = np.zeros(list(canvas_shape), dtype=np.float32)
        else:
            coordinates = output["coordinates"]
            predictions = output["probabilities"]
            num_class = np.array(predictions[0]).shape[0]
            denominator = np.zeros(canvas_shape)
            output = np.zeros(list(canvas_shape) + [num_class], dtype=np.float32)

        for idx, bound in enumerate(coordinates):
            prediction = predictions[idx]
            # assumed to be in XY
            # top-left for output placement
            tl = np.ceil(np.array(bound[:2]) * fx).astype(np.int32)
            # bot-right for output placement
            br = np.ceil(np.array(bound[2:]) * fx).astype(np.int32)
            output[tl[1] : br[1], tl[0] : br[0]] = prediction
            if denominator is not None:
                denominator[tl[1] : br[1], tl[0] : br[0]] += 1

        # deal with overlapping regions
        if denominator is not None:
            output = output / (np.expand_dims(denominator, -1) + 1.0e-8)
            if not return_raw:
                # convert raw probabilities to predictions
                if postproc_func is not None:
                    output = postproc_func(output)
                else:
                    output = np.argmax(output, axis=-1)
                # to make sure background is 0 while class will be 1..N
                output[denominator > 0] += 1
        return output

    def _predict_engine(
        self,
        dataset,
        return_probabilities=False,
        return_labels=False,
        return_coordinates=False,
        on_gpu=True,
    ):
        """Make a prediction on a dataset. The dataset may be mutated.

        Args:
            dataset (torch.utils.data.Dataset): PyTorch dataset object created using
              tiatoolbox.models.data.classification.Patch_Dataset.
            return_probabilities (bool): Whether to return per-class probabilities.
            return_labels (bool): Whether to return labels.
            return_coordinates (bool): Whether to return patch coordinates.
            on_gpu (bool): whether to run model on the GPU.

        Returns:
            output (ndarray): Model predictions of the input dataset

        """
        dataset.preproc_func = self.model.preproc_func

        # preprocessing must be defined with the dataset
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=self.num_loader_worker,
            batch_size=self.batch_size,
            drop_last=False,
            shuffle=False,
        )

        if self.verbose:
            pbar = tqdm.tqdm(
                total=int(len(dataloader)), leave=True, ncols=80, ascii=True, position=0
            )

        # use external for testing
        model = misc.model_to(on_gpu, self.model)

        cum_output = {
            "probabilities": [],
            "predictions": [],
            "coordinates": [],
            "labels": [],
        }
        for _, batch_data in enumerate(dataloader):

            batch_output_probabilities = self.model.infer_batch(
                model, batch_data["image"], on_gpu
            )
            # We get the index of the class with the maximum probability
            batch_output_predictions = self.model.postproc_func(
                batch_output_probabilities
            )

            # tolist might be very expensive
            cum_output["probabilities"].extend(batch_output_probabilities.tolist())
            cum_output["predictions"].extend(batch_output_predictions.tolist())
            if return_coordinates:
                cum_output["coordinates"].extend(batch_data["coords"].tolist())
            if return_labels:  # be careful of `s`
                # We do not use tolist here because label may be of mixed types
                # and hence collated as list by torch
                cum_output["labels"].extend(list(batch_data["label"]))

            if self.verbose:
                pbar.update()
        if self.verbose:
            pbar.close()

        if not return_probabilities:
            cum_output.pop("probabilities")
        if not return_labels:
            cum_output.pop("labels")
        if not return_coordinates:
            cum_output.pop("coordinates")

        return cum_output

    def predict(
        self,
        imgs,
        masks=None,
        labels=None,
        mode="patch",
        return_probabilities=False,
        return_labels=False,
        on_gpu=True,
        ioconfig: IOPatchPredictorConfig = None,
        patch_input_shape: Tuple[int, int] = None,
        stride_shape: Tuple[int, int] = None,
        resolution=None,
        units=None,
        merge_predictions=False,
        save_dir=None,
        save_output=False,
    ):
        """Make a prediction for a list of input data.

        Args:
            imgs (list, ndarray): List of inputs to process. When using `patch`
              mode, the input must be either a list of images, a list of image
              file paths or a numpy array of an image list. When using `tile` or
              `wsi` mode, the input must be a list of file paths.
            masks (list): List of masks. Only utilised when processing image tiles
              and whole-slide images. Patches are only processed if they are
              within a masked area. If not provided, then a tissue mask will be
              automatically generated for whole-slide images or the entire image
              is processed for image tiles.
            labels: List of labels. If using `tile` or `wsi` mode, then only a
              single label per image tile or whole-slide image is supported.
            mode (str): Type of input to process. Choose from either `patch`, `tile`
              or `wsi`.
            return_probabilities (bool): Whether to return per-class probabilities.
            return_labels (bool): Whether to return the labels with the predictions.
            on_gpu (bool): whether to run model on the GPU.
            patch_input_shape (tuple): Size of patches input to the model. Patches
              are at requested read resolution, not with respect to level 0, and must be
              positive.
            stride_shape (tuple): Stride using during tile and WSI processing.
              Stride is at requested read resolution, not with respect to to level
              0, and must be positive. If not provided,
              `stride_shape=patch_input_shape`.
            resolution (float): Resolution used for reading the image. Please see
                :obj:`WSIReader` for details.
            units (str): Units of resolution used for reading the image. Choose from
              either `level`, `power` or `mpp`. Please see
                :obj:`WSIReader` for details.
            merge_predictions (bool): Whether to merge the predictions to form a
              2-dimensional map. This is only applicable for `mode='wsi'`
              or `mode='tile'`.
            save_dir (str or pathlib.Path): Output directory when processing
              multiple tiles and whole-slide images. By default, it is folder `output`
              where the running script is invoked.
            save_output (bool): Whether to save output for a single file. default=False

        Returns:
            output (ndarray, dict): Model predictions of the input dataset. If
              multiple image tiles or whole-slide images are provided as input,
              or save_output is True, then results are saved to `save_dir` and a
              dictionary indicating save location for each input is return.
              The dict has following format:
                - img_path: path of the input image.
                    - raw: path to save location for raw prediction, saved in .json.
                    - merged: path to .npy contain merged predictions if
                      `merge_predictions` is `True`.

        Examples:
            >>> wsis = ['wsi1.svs', 'wsi2.svs']
            >>> predictor = PatchPredictor(
            ...                 pretrained_model="resnet18-kather100k")
            >>> output = predictor.predict(wsis, mode="wsi")
            >>> output.keys()
            ['wsi1.svs', 'wsi2.svs']
            >>> output['wsi1.svs']
            {'raw': '0.raw.json', 'merged': '0.merged.npy}
            >>> output['wsi2.svs']
            {'raw': '1.raw.json', 'merged': '1.merged.npy}

        """
        if mode not in ["patch", "wsi", "tile"]:
            raise ValueError(
                f"{mode} is not a valid mode. Use either `patch`, `tile` or `wsi`"
            )
        if mode == "patch" and labels is not None:
            # if a labels is provided, then return with the prediction
            return_labels = bool(labels)
            if len(labels) != len(imgs):
                raise ValueError(
                    f"len(labels) != len(imgs) : " f"{len(labels)} != {len(imgs)}"
                )
        if mode == "wsi" and masks is not None and len(masks) != len(imgs):
            raise ValueError(
                f"len(masks) != len(imgs) : " f"{len(masks)} != {len(imgs)}"
            )

        if mode == "patch":
            # don't return coordinates if patches are already extracted
            return_coordinates = False
            dataset = PatchDataset(imgs, labels)
            output = self._predict_engine(
                dataset, return_probabilities, return_labels, return_coordinates, on_gpu
            )

        else:
            if stride_shape is None:
                stride_shape = patch_input_shape

            # ! not sure if there is any way to make this nicer
            make_config_flag = (
                patch_input_shape is None,
                resolution is None,
                units is None,
            )

            if ioconfig is None and self.ioconfig is None and any(make_config_flag):
                raise ValueError(
                    "Must provide either `ioconfig` or "
                    "`patch_input_shape`, `resolution`, and `units`."
                )
            if ioconfig is None and self.ioconfig:
                ioconfig = copy.deepcopy(self.ioconfig)
                # ! not sure if there is a nicer way to set this
                if patch_input_shape is not None:
                    ioconfig.patch_input_shape = patch_input_shape
                if stride_shape is not None:
                    ioconfig.stride_shape = stride_shape
                if resolution is not None:
                    ioconfig.input_resolutions[0]["resolution"] = resolution
                if units is not None:
                    ioconfig.input_resolutions[0]["units"] = units
            elif ioconfig is None and all(not v for v in make_config_flag):
                ioconfig = IOPatchPredictorConfig(
                    input_resolutions=[{"resolution": resolution, "units": units}],
                    patch_input_shape=patch_input_shape,
                    stride_shape=stride_shape,
                )

            fx_list = ioconfig.scale_to_highest(
                ioconfig.input_resolutions, ioconfig.input_resolutions[0]["units"]
            )
            fx_list = zip(fx_list, ioconfig.input_resolutions)
            fx_list = sorted(fx_list, key=lambda x: x[0])
            highest_input_resolution = fx_list[0][1]

            if mode == "tile":
                warnings.warn(
                    "WSIPatchDataset only reads image tile at "
                    '`units="baseline"`. Resolutions will be converted '
                    "to baseline value."
                )
                ioconfig = ioconfig.to_baseline()

            if len(imgs) > 1:
                warnings.warn(
                    "When providing multiple whole-slide images / tiles, "
                    "we save the outputs and return the locations "
                    "to the corresponding files."
                )

            if len(imgs) > 1:
                warnings.warn(
                    "When providing multiple whole-slide images / tiles, "
                    "we save the outputs and return the locations "
                    "to the corresponding files."
                )
                if save_dir is None:
                    warnings.warn(
                        "> 1 WSIs detected but there is no save directory set."
                        "All subsequent output will be saved to current runtime"
                        "location under folder 'output'. Overwriting may happen!"
                    )
                    save_dir = pathlib.Path(os.getcwd()).joinpath("output")

                save_dir = pathlib.Path(save_dir)

            if save_dir is not None:
                save_dir = pathlib.Path(save_dir)
                save_dir.mkdir(parents=True, exist_ok=False)

            # return coordinates of patches processed within a tile / whole-slide image
            return_coordinates = True
            if not isinstance(imgs, list):
                raise ValueError(
                    "Input to `tile` and `wsi` mode must be a list of file paths."
                )

            # None if no output
            outputs = None

            self._ioconfig = ioconfig
            # generate a list of output file paths if number of input images > 1
            file_dict = OrderedDict()
            for idx, img_path in enumerate(imgs):
                img_path = pathlib.Path(img_path)
                img_label = None if labels is None else labels[idx]
                img_mask = None if masks is None else masks[idx]

                dataset = WSIPatchDataset(
                    img_path,
                    mode=mode,
                    mask_path=img_mask,
                    patch_input_shape=ioconfig.patch_input_shape,
                    stride_shape=ioconfig.stride_shape,
                    resolution=ioconfig.input_resolutions[0]["resolution"],
                    units=ioconfig.input_resolutions[0]["units"],
                )
                output_model = self._predict_engine(
                    dataset,
                    return_labels=False,
                    return_probabilities=return_probabilities,
                    return_coordinates=return_coordinates,
                    on_gpu=on_gpu,
                )
                output_model["label"] = img_label
                # add extra information useful for downstream analysis
                output_model["pretrained_model"] = self.pretrained_model
                output_model["resolution"] = highest_input_resolution["resolution"]
                output_model["units"] = highest_input_resolution["units"]

                outputs = [output_model]  # assign to a list
                merged_prediction = None
                if merge_predictions:
                    merged_prediction = self.merge_predictions(
                        img_path,
                        output_model,
                        resolution=output_model["resolution"],
                        units=output_model["units"],
                        postproc_func=self.model.postproc,
                    )
                    outputs.append(merged_prediction)

                if len(imgs) > 1 or save_output:
                    # dynamic 0 padding
                    img_code = f"{idx:0{len(str(len(imgs)))}d}"

                    save_info = {}
                    save_path = os.path.join(str(save_dir), img_code)
                    raw_save_path = f"{save_path}.raw.json"
                    save_info["raw"] = raw_save_path
                    save_as_json(output_model, raw_save_path)
                    if merge_predictions:
                        merged_file_path = f"{save_path}.merged.npy"
                        np.save(merged_file_path, merged_prediction)
                        save_info["merged"] = merged_file_path
                    file_dict[str(img_path)] = save_info

            output = file_dict if len(imgs) > 1 or save_output else outputs

        return output