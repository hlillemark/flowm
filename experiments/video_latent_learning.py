from datasets.video import (
    MNISTWorldVideoDataset,
    BlockWorldVideoDataset
)
from algorithms.vae.mae_vit_trainer import MAEVITImageVAETrainer
from .base_exp import BaseLightningExperiment
from .data_modules import _data_module_cls


class VideoLatentLearningExperiment(BaseLightningExperiment):
    """
    An experiment for training & validating the first stage model (e.g. VAE)
    that learns the latent representation of the data
    """

    compatible_algorithms = dict(
        train_mae_vae=MAEVITImageVAETrainer,
    )

    compatible_datasets = dict(
        # video datasets
        mnist_world=MNISTWorldVideoDataset,
        blockworld=BlockWorldVideoDataset
    )

    data_module_cls = _data_module_cls
 