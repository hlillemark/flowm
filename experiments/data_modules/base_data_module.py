from typing import Dict
import os
from omegaconf import DictConfig
import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from utils.print_utils import cyan, red
from utils.distributed_utils import rank_zero_print, rank_zero_only
import numpy
import random
from typing import Literal
from functools import partial
import signal 


def seed_worker(worker_id, worker_seed = None):
    if worker_seed is None:
        worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)

def check_and_append_dataset(dataset_cfg: DictConfig, datasets: list, dataset_cls: torch.utils.data.Dataset, split: str, purpose: Literal["training", "validation", "test"] = "training"):
    rank_zero_print(cyan(f"check_and_append_dataset: {split}, {purpose}"))
    dataset = dataset_cls(dataset_cfg, split=split, purpose=purpose)
    if len(dataset) > 0:
        datasets.append(dataset)
    else:
        rank_zero_print(red(f"|-- {split} is empty (due to no sufficient data with wanted length)"))
    return datasets


def compose_worker_init_fn(dataset, fallback_seed_fn=None, deterministic=False):
    def worker_init_fn(worker_id):
        # Custom SIGTERM handler
        def sigterm_handler(signum, frame):
            print(f"Worker {worker_id} received SIGTERM. Exiting gracefully.")
            os._exit(0) # Clean exit without killing main process

        signal.signal(signal.SIGTERM, sigterm_handler)

        # Call dataset's own worker_init_fn if it exists
        if hasattr(dataset, "worker_init_fn") and callable(dataset.worker_init_fn):
            dataset.worker_init_fn(worker_id)
        # Otherwise call fallback seed fn
        elif fallback_seed_fn is not None:
            fallback_seed_fn(worker_id, deterministic)

    return worker_init_fn


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, root_cfg: DictConfig, compatible_datasets: Dict) -> None:
        super().__init__()
        self.root_cfg = root_cfg
        self.exp_cfg = root_cfg.experiment
        self.compatible_datasets = compatible_datasets
        self.deterministic = self.root_cfg.algorithm.logging.deterministic
        
        
        if self.deterministic is not None:
            rank_zero_only(cyan(f"Deterministic behavior detected, the dataloader will use assigned generator; To ensure reproducibility, please also set the `worker_init_fn`"))
            self.generator = torch.Generator()
            self.generator.manual_seed(self.deterministic)
        else:
            self.generator = None

    def _build_datasets(self, purpose: str) -> torch.utils.data.Dataset:
        if purpose in ["training", "test", "validation"]:
            
            dataset_cls = self.compatible_datasets[(self.root_cfg.dataset._name)]
            splits = dataset_cls.get_splits()
            
            datasets = []
            splits_included = []
            if purpose == "validation":
                for spl in splits:
                    # spl in splits must been built with metadata,  splits_included \in splits
                    if spl in self.exp_cfg.validation.splits_included:
                        splits_included.append(spl)
                        datasets = check_and_append_dataset(self.root_cfg.dataset, datasets, dataset_cls, spl, purpose="validation")

            elif purpose == "training":
                # add the training dataset
                for spl in self.exp_cfg.training.splits_included:
                    splits_included.append(spl)
                    datasets = check_and_append_dataset(self.root_cfg.dataset, datasets, dataset_cls, spl, purpose="training")
            
            valid_cnt = sum([True for dataset in datasets if len(dataset) > 0])
            rank_zero_print(red(f"included {str(splits_included)} datasets in {purpose} dataloader, {valid_cnt} valid datasets"))
            if valid_cnt != len(datasets):
                rank_zero_print(red(f"|-- but only {valid_cnt} dataset(s) is valid and will be used"))
            return datasets
        else:
            raise NotImplementedError(f"purpose '{purpose}' is not implemented")

    @staticmethod
    def _get_shuffle(dataset: torch.utils.data.Dataset, default: bool) -> bool:
        return not isinstance(dataset, torch.utils.data.IterableDataset) and default

    @staticmethod
    def _get_num_workers(num_workers: int) -> int:
        return min(os.cpu_count(), num_workers)

    def _dataloader(self, purpose: str) -> TRAIN_DATALOADERS | EVAL_DATALOADERS:
        """
        Args:
            purpose: str
                The purpose of using dataloader.
                It can be "training", "validation", "test"
        """
        datasets = self._build_datasets(purpose)
        purpose_cfg = self.exp_cfg[purpose]
        
        dataloaders = []
        for dataset in datasets:
            # Custom collate fn:
            collate_fn = None

            dataloaders.append(
                torch.utils.data.DataLoader(
                    dataset,
                    batch_size=purpose_cfg.batch_size,
                    num_workers=self._get_num_workers(purpose_cfg.dataloader.num_workers),
                    shuffle=self._get_shuffle(dataset, purpose_cfg.dataloader.shuffle),
                    persistent_workers=purpose == "training",
                    pin_memory=purpose_cfg.dataloader.pin_memory,
                    prefetch_factor=purpose_cfg.dataloader.prefetch_factor,
                    generator=self.generator, 
                    collate_fn=collate_fn,  # Add the custom collate function here
                    worker_init_fn=compose_worker_init_fn(dataset=dataset, fallback_seed_fn=seed_worker, deterministic=self.deterministic),
                ))
        return dataloaders

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._dataloader("training")

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._dataloader("test")
