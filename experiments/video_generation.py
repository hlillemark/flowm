from datasets.video import (
    MNISTWorldVideoDataset,
    BlockWorldVideoDataset,
)
from algorithms import DFoTVideo, FloWMVideo
from .base_exp import BaseLightningExperiment
from .data_modules import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        flowm_video=FloWMVideo,
    )

    compatible_datasets = dict(
        # video datasets
        mnist_world=MNISTWorldVideoDataset,
        mnist_world_fernn=MNISTWorldVideoDataset,
        blockworld=BlockWorldVideoDataset
    )

    data_module_cls = _data_module_cls
