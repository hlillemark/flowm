from torch.optim import Optimizer, Adam, AdamW
from transformers import get_scheduler
from lightning.pytorch.utilities import grad_norm
from utils.distributed_utils import rank_zero_print
from utils.print_utils import cyan, red
import torch

class OptimizerMixin:
    
    def configure_optimizers(self):
        default_lr = self.cfg.training.lr  
        lr_map = self.cfg.training.get("lr_map", {})  

        param_groups = []
        already_assigned = set()

        for keyword, lr_config in lr_map.items() if lr_map else []:
            matched_params = [
                param for name, param in getattr(self, self.main_model_prefix).named_parameters() if keyword in name
            ]
            if matched_params:
                param_groups.append({
                    "params": matched_params,
                    "lr": lr_config["lr"],
                })
                already_assigned.update(matched_params)
        other_params = [
            param for param in getattr(self, self.main_model_prefix).parameters() if param not in already_assigned
        ]
        if other_params:
            param_groups.append({
                "params": other_params,
                "lr": default_lr,
            })
        if self.cfg.training.opt_name == "adamw": # with decoupled_weight_decay, AdamW is used
            optimizer = AdamW(param_groups, lr=default_lr, weight_decay=self.cfg.training.weight_decay, betas=self.cfg.training.optimizer_beta)
        elif self.cfg.training.opt_name == "adam":
            optimizer = Adam(
                param_groups,
                lr=default_lr,  
                weight_decay=self.cfg.training.weight_decay,
                betas=self.cfg.training.optimizer_beta,
            )
        else:
            raise ValueError(f"Unknown optimizer name: {self.cfg.training.opt_name}")
        lr_scheduler_config = {
            "scheduler": get_scheduler(
                optimizer=optimizer,
                **self.cfg.training.lr_scheduler,
            ),
            "interval": "step",
            "frequency": 1,
        }

        if self.cfg.load_opt_state:
            if not self.cfg.opt_state_key:
                raise ValueError(f"opt_state_key is not set in the config, while loading optimizer state from {self.cfg.load_opt_state}")
            state_dict = torch.load(self.cfg.load_opt_state, map_location="cpu", weights_only=False)
            if self.cfg.opt_state_key in state_dict:
                state_dict = state_dict[self.cfg.opt_state_key]
            else:
                rank_zero_print(red(f"!!! No optimizer state found in {self.cfg.load_opt_state}, while loading from {self.cfg.opt_state_key} !!!"))
            optimizer.load_state_dict(state_dict)
            rank_zero_print(cyan(f"\n Successfully loaded optimizer state from {self.cfg.load_opt_state}[{self.cfg.opt_state_key}]"))

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler_config,
        }
        
    def log_gradient_stats(self):
        """Log gradient statistics such as the mean or std of norm."""

        with torch.no_grad():
            grad_norms = []
            gpr = []  # gradient-to-parameter ratio
            for param in self.parameters():
                if param.grad is not None:
                    grad_norms.append(torch.norm(param.grad).item())
                    gpr.append(torch.norm(param.grad) / torch.norm(param))
            if len(grad_norms) == 0:
                return
            grad_norms = torch.tensor(grad_norms)
            gpr = torch.tensor(gpr)
            self.log_dict(
                {
                    "train/grad_norm/min": grad_norms.min(),
                    "train/grad_norm/max": grad_norms.max(),
                    "train/grad_norm/std": grad_norms.std(),
                    "train/grad_norm/mean": grad_norms.mean(),
                    "train/grad_norm/median": torch.median(grad_norms),
                    "train/gpr/min": gpr.min(),
                    "train/gpr/max": gpr.max(),
                    "train/gpr/std": gpr.std(),
                    "train/gpr/mean": gpr.mean(),
                    "train/gpr/median": torch.median(gpr),
                }
            )

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        if (
            self.logging_cfg.grad_norm_freq
            and self.global_step % self.logging_cfg.grad_norm_freq == 0
        ):
            norms = grad_norm(getattr(self, self.main_model_prefix), norm_type=2)
            # NOTE: `norms` need not be gathered, as they are already uniform across all devices (DDP)
            self.log_dict(norms)

            if self.logging_cfg.verbose_grad_info:
                # Log gradient stats on configured frequency during training
                self.log_gradient_stats()


    def configure_multiple_optimizers(self):
        # for multiple optimizers, we need to return a list of tuples, each tuple contains an optimizer and a lr_scheduler
        # However, this is not supported by deepspeed 
        default_lr = self.cfg.training.lr  
        lr_map = self.cfg.training.lr_map
        default_lr_scheduler = self.cfg.training.lr_scheduler
        already_assigned = set()

        optimizer_lr_scheduler_tuple = ()
        if lr_map:
            self.automatic_optimization = False
            for keyword, lr_config in lr_map.items():
                matched_params = [
                    param for name, param in getattr(self, self.main_model_prefix).named_parameters() if keyword in name
                ]
                if matched_params:
                    already_assigned.update(matched_params)
                    optimizer = torch.optim.AdamW(
                        matched_params,
                        lr=lr_config["lr"],  
                        weight_decay=self.cfg.training.weight_decay,
                        betas=self.cfg.training.optimizer_beta,
                    )
                    lr_scheduler = get_scheduler(
                        optimizer=optimizer,
                        num_training_steps=self.trainer.estimated_stepping_batches,
                        **lr_config.lr_scheduler,
                    )
                    optimizer_lr_scheduler_tuple += ({
                        "optimizer": optimizer,
                        "lr_scheduler": {
                            "scheduler": lr_scheduler,
                            "interval": "step",
                            "frequency": 1,
                        }
                    },)

        other_params = [
            param for param in getattr(self, self.main_model_prefix).parameters() if param not in already_assigned
        ]
        if other_params:
            optimizer = torch.optim.AdamW(
                other_params,
                lr=default_lr,  
                weight_decay=self.cfg.training.weight_decay,
                betas=self.cfg.training.optimizer_beta,
            )
            lr_scheduler = get_scheduler(
                optimizer=optimizer,
                **default_lr_scheduler,
            )
            optimizer_lr_scheduler_tuple += ({
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "interval": "step",
                    "frequency": 1,
                }
            },)


        # lr_scheduler_config = {
        #     "scheduler": get_scheduler(
        #         optimizer=optimizer,
        #         **self.cfg.training.lr_scheduler,
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }

        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": lr_scheduler_config,
        # }
        return optimizer_lr_scheduler_tuple