from datasets.video import (
    BlockWorldVideoDataset,
)
from algorithms.vae.mae_vit_vae_preprocessor import MAE_ViT_VAE_Preprocessor
from .base_exp import BaseLightningExperiment
from .data_modules import ValDataModule


class VideoLatentPreprocessingExperiment(BaseLightningExperiment):
    """
    Experiment for preprocessing videos to latents using a pretrained ImageVAE model.
    """

    compatible_algorithms = dict(
        mae_vit_vae_preprocessor=MAE_ViT_VAE_Preprocessor
    )

    compatible_datasets = dict(
        blockworld=BlockWorldVideoDataset,
    )

    data_module_cls = ValDataModule

    def training(self) -> None:
        raise NotImplementedError(
            "Training not implemented for video preprocessing experiments"
        )

    def testing(self) -> None:
        raise NotImplementedError(
            "Testing not implemented for video preprocessing experiments"
        )
