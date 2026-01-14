import lightning.pytorch as pl
from torch.utils.data import DataLoader, Subset
from lightning.pytorch.strategies import DDPStrategy
from algorithms.vae import MAE_ViT
from datasets.video import (MNISTWorldVideoDataset, BlockWorldVideoDataset)
from utils.torch_utils import freeze_model
from omegaconf import OmegaConf
import torch
import yaml
from tqdm import tqdm
from einops import rearrange
import numpy as np
from algorithms.common.base_pytorch_algo import BasePytorchAlgo
import os
import hydra
from omegaconf import DictConfig, open_dict
import random
from utils.print_utils import cyan
import torchvision 
from utils.distributed_utils import rank_zero_print

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def _vae_normalize_z(z, latent_mean, latent_std):
    shape = [1] * (z.ndim - latent_mean.ndim) + list(latent_mean.shape)
    mean = latent_mean.reshape(shape)
    std = latent_std.reshape(shape)
    return (z - mean) / std

def _normalize_depth(depth_video: torch.Tensor, depth_min: float, depth_max: float, normalize_depth_with_log: bool = True) -> torch.Tensor:
    """
    Normalize the depth video to [0, 1]
    
    Args:
        depth_video: Input depth video tensor
        depth_min: Minimum depth value for normalization
        depth_max: Maximum depth value for normalization
    """
    if normalize_depth_with_log:
        depth_video = torch.log(depth_video)
        depth_min = torch.log(depth_min)
        depth_max = torch.log(depth_max)
    return (depth_video - depth_min) / (depth_max - depth_min)


def recursive_round(val, decimals=4):
    if isinstance(val, float):
        return round(val, decimals)
    elif isinstance(val, list):
        return [recursive_round(v, decimals) for v in val]
    else:
        return val  # e.g., int or other types

# Example usage: 
# python -m algorithms.vae.parallel_estimate_latent_stats
# Used for calculating stats of multiscale vae models originally trained on the image vae latent space
class VAEStatsModel(BasePytorchAlgo):
    def __init__(self, cfg, dataset_cls):
        super().__init__(cfg)
        self.means = []
        self.stds = []
        self.subset_size = cfg.subset_size
        self.cfg = cfg
        self.image_width, self.image_height = self.cfg.dataset.resolution
        self.which_channel = self.cfg.which_channel
        self.dataset_cls = dataset_cls
        if self.which_channel == 'depths':
            self.depth_min = torch.tensor(self.cfg.dataset.depth_min)
            self.depth_max = torch.tensor(self.cfg.dataset.depth_max)
            self.normalize_depth_with_log = self.cfg.dataset.normalize_depth_with_log
        
        self.dataset_split = cfg.training_splits[0]
        
        # Check if we should only test existing latent stats
        self.test_only_mode = cfg.get('test_only_existing_latent_stats', False)
        
        rank_zero_print(cyan(f"Initializing VAE with channel: {self.which_channel}"))
        if self.test_only_mode:
            rank_zero_print(cyan("Running in TEST-ONLY mode: will validate existing latent stats without computing new ones"))
        
        self.configure_model()
    
    def configure_model(self):
        self.vae_path = self.cfg.algorithm.vae.pretrained_path
        
        rank_zero_print(cyan(f"Loading from this vae path: {self.vae_path}"))
        self.vae = MAE_ViT.from_pretrained(path=self.vae_path, **self.cfg.algorithm.vae.pretrained_kwargs)
        freeze_model(self.vae)

    def forward(self, x):
        raise NotImplementedError

    def training_step(self, *args, **kwargs):
        raise NotImplementedError
    
    def validate_vae_loaded_correctly_on_first_validation_batch(self, x):
        # Create test video output, to validate that the correct trained vae was used.
        # Just do the first one in the batch
        with torch.no_grad():
            z = self.vae.vae_encode(x, image_height=self.image_height, image_width=self.image_width)
            xhat = self.vae.vae_decode(z[:1, ...])[0]
        output_file = 'test_vae_output.mp4'
        fps = 15
        xhat = rearrange(xhat, "t c h w -> t h w c")
        xhat = (np.clip(xhat.cpu().numpy(), a_min=0.0, a_max=1.0) * 255).astype(np.int8)
        torchvision.io.write_video(output_file, xhat, fps=fps)
        rank_zero_print(cyan(f"Check at {output_file} for a reconstruction video to make sure you loaded the vae correctly!"))

    def val_dataloader(self):
        dataset = self.dataset_cls(self.cfg.dataset, self.dataset_split)
        total_size = len(dataset)
        
        # Get a random subset of indices
        indices = torch.randperm(total_size)[:self.subset_size]
        subset = Subset(dataset, indices)

        # Create the DataLoader
        return DataLoader(subset, batch_size=32, shuffle=True, num_workers=4)

    def validation_step(self, batch, batch_idx):
        x = batch[self.which_channel]
        if self.which_channel == 'depths':
            x = _normalize_depth(x, self.depth_min, self.depth_max, self.normalize_depth_with_log)
            
        # Only run on first trial with first rank
        if batch_idx == 0 and self.global_rank == 0:
            self.validate_vae_loaded_correctly_on_first_validation_batch(x)
        
        # If in test-only mode, only run validation on first batch and exit
        if self.test_only_mode:
            if batch_idx == 0 and self.global_rank == 0:
                # Get existing latent stats from config
                existing_mean = self.cfg.algorithm.vae.pretrained_kwargs.get('latent_mean', None)
                existing_std = self.cfg.algorithm.vae.pretrained_kwargs.get('latent_std', None)
                
                if existing_mean is None or existing_std is None:
                    rank_zero_print("ERROR: test_only_existing_latent_stats=True but no existing latent_mean/latent_std found in config!")
                    return
                
                rank_zero_print(cyan(f"Testing existing latent stats:"))
                rank_zero_print(cyan(f"  Mean: {existing_mean}"))
                rank_zero_print(cyan(f"  Std: {existing_std}"))
                
                self._test_existing_latent_normalization(x)
            return
            
        z = self.vae.vae_encode(x, image_height=self.image_height, image_width=self.image_width)
        # output z shape is [b, t, c, h, w]
        
        # calc stats across num_channels
        mean = z.mean(dim=(0, 1, 3, 4)).cpu()
        std = z.std(dim=(0, 1, 3, 4)).cpu()

        self.means.append(mean)
        self.stds.append(std)

    def on_validation_epoch_end(self):
        # If in test-only mode, we don't compute new stats
        if self.test_only_mode:
            rank_zero_print(cyan("Test-only mode completed. No new latent stats computed."))
            return
            
        # Stack the per-batch stats on each process
        local_means = torch.stack(self.means).to(self.device)  # (N, C)
        local_stds = torch.stack(self.stds).to(self.device)

        # Use Lightning's all_gather to collect from all devices
        all_means = self.all_gather(local_means)  # (world_size, N, C)
        all_stds = self.all_gather(local_stds)

        # Only do the final print on global rank 0
        if self.global_rank == 0:
            all_means_flat = all_means.reshape(-1, all_means.shape[-1])
            all_stds_flat = all_stds.reshape(-1, all_stds.shape[-1])

            overall_mean_tensor = all_means_flat.mean(dim=0).reshape(-1, 1, 1)
            overall_std_tensor = all_stds_flat.mean(dim=0).reshape(-1, 1, 1)

            overall_mean = recursive_round(overall_mean_tensor.tolist(), decimals=4)
            overall_std = recursive_round(overall_std_tensor.tolist(), decimals=4)

            rank_zero_print(cyan("Latent stats computed across all devices.\n"))
            rank_zero_print(cyan("Overall mean: " + str(overall_mean)))
            rank_zero_print(cyan("Overall std: " + str(overall_std)))
            
            # Flattened versions
            flat_mean = recursive_round(overall_mean_tensor.flatten().tolist(), decimals=4)
            flat_std = recursive_round(overall_std_tensor.flatten().tolist(), decimals=4)

            rank_zero_print(cyan("Overall mean (flat):" + str(flat_mean)))
            rank_zero_print(cyan("Overall std (flat):" + str(flat_std)))
            
            self._validate_latent_normalization(overall_mean, overall_std)

        # Clear for next epoch
        self.means.clear()
        self.stds.clear()
    
    def _validate_latent_normalization(self, overall_mean, overall_std):
        # Only on global rank 0
        if self.global_rank != 0:
            return

        rank_zero_print(cyan("\nValidating normalization on a single batch..."))

        # Sample one batch from the validation dataloader
        loader = self.val_dataloader()
        x = next(iter(loader))[self.which_channel].to(self.device)
        if self.which_channel == 'depths':
            x = _normalize_depth(x, self.depth_min, self.depth_max, self.normalize_depth_with_log)
        
        # Process videos directly with MAE-ViT
        with torch.no_grad():
            z = self.vae.vae_encode(x, image_height=self.image_height, image_width=self.image_width)

        # Normalize the latents using the _vae_normalize_x function
        overall_mean_tensor = torch.tensor(overall_mean, device=self.device)
        overall_std_tensor = torch.tensor(overall_std, device=self.device)
        z_norm = _vae_normalize_z(z, latent_mean=overall_mean_tensor, latent_std=overall_std_tensor)
        
        # Calculate stats after normalization
        norm_mean = z_norm.mean(dim=(0, 1, 3, 4)).tolist()
        norm_std = z_norm.std(dim=(0, 1, 3, 4)).tolist()

        rank_zero_print(cyan(f"Post-normalization mean: {norm_mean}"))
        rank_zero_print(cyan(f"Post-normalization std:  {norm_std}"))
    
    def _test_existing_latent_normalization(self, x, existing_mean = 0.0, existing_std = 1.0):
        """Test normalization using existing latent stats from config"""
        rank_zero_print(cyan("\nTesting existing latent normalization on a single batch..."))

        # Process videos directly with MAE-ViT
        with torch.no_grad():
            z = self.vae.vae_encode(x, image_height=self.image_height, image_width=self.image_width)

        # Convert existing stats to tensor format if needed
        if not isinstance(existing_mean, torch.Tensor):
            if isinstance(existing_mean, (int, float)):
                # Scalar case
                existing_mean_tensor = torch.ones((z.shape[2], 1, 1), device=self.device) * existing_mean
            else:
                # List/array case
                existing_mean_tensor = torch.tensor(existing_mean, device=self.device)
                if existing_mean_tensor.ndim == 1:
                    existing_mean_tensor = existing_mean_tensor.reshape(-1, 1, 1)
        else:
            existing_mean_tensor = existing_mean.to(self.device)
            
        if not isinstance(existing_std, torch.Tensor):
            if isinstance(existing_std, (int, float)):
                # Scalar case  
                existing_std_tensor = torch.ones((z.shape[2], 1, 1), device=self.device) * existing_std
            else:
                # List/array case
                existing_std_tensor = torch.tensor(existing_std, device=self.device)
                if existing_std_tensor.ndim == 1:
                    existing_std_tensor = existing_std_tensor.reshape(-1, 1, 1)
        else:
            existing_std_tensor = existing_std.to(self.device)

        # Normalize the latents using existing stats
        z_norm = _vae_normalize_z(z, latent_mean=existing_mean_tensor, latent_std=existing_std_tensor)
        
        # Calculate stats before normalization
        orig_mean = z.mean(dim=(0, 1, 3, 4)).tolist()
        orig_std = z.std(dim=(0, 1, 3, 4)).tolist()
        
        # Calculate stats after normalization
        norm_mean = z_norm.mean(dim=(0, 1, 3, 4)).tolist()
        norm_std = z_norm.std(dim=(0, 1, 3, 4)).tolist()

        rank_zero_print(cyan(f"Original latent mean: {recursive_round(orig_mean)}"))
        rank_zero_print(cyan(f"Original latent std:  {recursive_round(orig_std)}"))
        rank_zero_print(cyan(f"Post-normalization mean: {recursive_round(norm_mean)}"))
        rank_zero_print(cyan(f"Post-normalization std:  {recursive_round(norm_std)}"))
        
        # Check if normalization is working correctly
        mean_close_to_zero = all(abs(m) < 0.1 for m in norm_mean)
        std_close_to_one = all(abs(s - 1.0) < 0.2 for s in norm_std)
        
        if mean_close_to_zero and std_close_to_one:
            rank_zero_print(cyan("✓ Normalization test PASSED: stats are close to N(0,1)"))
        else:
            rank_zero_print(cyan("⚠ Normalization test WARNING: stats may not be optimal"))
            if not mean_close_to_zero:
                rank_zero_print(cyan(f"  - Mean not close to 0: {norm_mean}"))
            if not std_close_to_one:
                rank_zero_print(cyan(f"  - Std not close to 1: {norm_std}"))

"""
Example run commands:

1. Compute new latent stats (default behavior):
python -m algorithms.vae.parallel_estimate_latent_stats dataset=blockworld dataset.latent.type=online algorithm=train_mae_vae experiment=video_latent_learning ckpt_map=default

2. Test existing latent stats (test-only mode):
python -m algorithms.vae.parallel_estimate_latent_stats dataset=blockworld dataset.latent.type=online algorithm=train_mae_vae experiment=video_latent_learning ckpt_map=default +test_only_existing_latent_stats=true

Make sure to format your 'configurations/algorithm/vae/name.yaml correctly
and that your algorithm has the correct vae pointer.
and that your cluster has the right pretrained path.

For mode 1: paste the computed latent_mean and latent_std into the corresponding configurations.algorithm.vae file
For mode 2: make sure your config already has latent_mean and latent_std values to test

Write the data split you want to use in the following `main` function.

It will also paste out one validation video to ./test_vae_output.mp4
python -m algorithms.vae.parallel_estimate_latent_stats dataset=blockworld algorithm=train_mae_vae experiment=video_latent_learning ckpt_map=default
TODO: find a way to make the vae override come from here without a shortcode. need to currently manually modify 
train_mae_vae to have the right vae override.
"""
@hydra.main(config_path="../../configurations", config_name="config", version_base=None,)
def main(cfg: DictConfig):
    # Make sure if running on cluster's compute node that it distributes properly
    os.environ.pop("SLURM_JOB_ID", None)
    os.environ.pop("SLURM_NTASKS", None)
    set_seed(42)
    
    # TODO: Change the dataset class respectively
    # dataset_cls = MNISTWorldVideoDataset
    dataset_cls = BlockWorldVideoDataset
    
    # TODO: Set training split here:
    override_training_split = ["tex_training"]

    other_params = DictConfig({
        "which_channel": "videos", # videos, depths,
        "debug": False,
        "external_cond_stack": False,
        "training_splits": cfg.experiment.training.splits_included if override_training_split is None else override_training_split,
        "subset_size": 50_000,
    })
    
    if len(other_params.training_splits) != 1:
        raise RuntimeError("Not allowed to pass multiple datasets as training split to est. latent stats, not implemented yet")
    
    with open_dict(cfg):
        for key, value in other_params.items():
            cfg[key] = value
    
    # Manual override for computing new stats (but not in test-only mode)
    test_only_mode = cfg.get('test_only_existing_latent_stats', False)
    if not test_only_mode:
        cfg.algorithm.vae.pretrained_kwargs.latent_mean = 0
        cfg.algorithm.vae.pretrained_kwargs.latent_std = 1
    else:
        rank_zero_print(cyan("Test-only mode: preserving existing latent_mean and latent_std from config"))

    model = VAEStatsModel(cfg, dataset_cls = dataset_cls)

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=4,
        strategy=DDPStrategy(find_unused_parameters=False),
        max_epochs=1,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.validate(model)

if __name__ == '__main__':
    main()
