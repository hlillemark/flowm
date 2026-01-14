import torch
from torch import Tensor
from typing import Dict, Any
from utils.print_utils import cyan, red
from utils.distributed_utils import rank_zero_print
from utils.torch_utils import freeze_model

class CheckpointMixin:

    # ---------------------------------------------------------------------
    # Checkpoint Utils
    # ---------------------------------------------------------------------

    def _uncompile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict if self.main_model is compiled, to uncompiled."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace(f"{self.main_model_prefix}._orig_mod.", f"{self.main_model_prefix}."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _compile_checkpoint(self, checkpoint: Dict[str, Any]):
        """Converts the state_dict to the format expected by the compiled model."""
        if self.cfg.compile:
            checkpoint["state_dict"] = {
                k.replace(f"{self.main_model_prefix}.", f"{self.main_model_prefix}._orig_mod."): v
                for k, v in checkpoint["state_dict"].items()
            }

    def _should_include_in_checkpoint(self, key: str) -> bool:
        flag = self.main_model_prefix in key
        return flag
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        # 1. (Optionally) uncompile the model's state_dict before saving
        self._uncompile_checkpoint(checkpoint)
        # 2. Only save the meaningful keys defined by self._should_include_in_checkpoint
        # by default, only the model's state_dict is saved and metrics & registered buffes (e.g. diffusion schedule) are not discarded
        state_dict = checkpoint["state_dict"]
        for key in list(state_dict.keys()):
            if not self._should_include_in_checkpoint(key):
                del state_dict[key]

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:

        # 1. (Optionally) compile the model's state_dict before loading
        self._compile_checkpoint(checkpoint)
        # 2. (Optionally) swap the state_dict of the model with the EMA weights for inference
        super().on_load_checkpoint(checkpoint)
        # 3. (Optionally) reset the optimizer states - for fresh finetuning or resuming training
        if "deepspeed" in self.cfg.training.strategy and self.training:
            return
        if "deepspeed" in self.cfg.validation.strategy and not self.training:
            return
        
        if self.cfg.checkpoint.reset_optimizer:
            checkpoint["optimizer_states"] = []

        # 4. Rewrite the state_dict of the checkpoint, only leaving meaningful keys
        # defined by self._should_include_in_checkpoint
        # also print out warnings when the checkpoint does not exactly match the expected format

        new_state_dict = {}
        for key, value in self.state_dict().items():
            if (
                self._should_include_in_checkpoint(key)
                and key in checkpoint["state_dict"]
            ):
                new_state_dict[key] = checkpoint["state_dict"][key]
            else:
                new_state_dict[key] = value

        # print keys that are ignored from the checkpoint
        ignored_keys = [
            key
            for key in checkpoint["state_dict"].keys()
            if not self._should_include_in_checkpoint(key)
        ]
        if ignored_keys:
            rank_zero_print(
                cyan("The following keys are ignored from the checkpoint:"),
                ignored_keys,
            )
        # print keys that are not found in the checkpoint
        missing_keys = [
            key
            for key in self.state_dict().keys()
            if self._should_include_in_checkpoint(key)
            and key not in checkpoint["state_dict"]
        ]
        if missing_keys:
            rank_zero_print(
                cyan("The following keys are not found in the checkpoint:"),
                missing_keys,
            )
            import sys
            print(f"\n\n=== MISSING KEYS ({len(missing_keys)} total) ===", file=sys.stderr)
            for key in missing_keys:
                print(f"  - {key}", file=sys.stderr)
            print("=" * 50 + "\n", file=sys.stderr)
            if self.cfg.checkpoint.strict:
                raise ValueError(
                    f"Found {len(missing_keys)} missing keys in checkpoint. Thus, the checkpoint cannot be loaded. To ignore this error, turn off strict checkpoint loading by setting `algorithm.checkpoint.strict=False`."
                )
            else:
                rank_zero_print(
                    cyan(
                        "Strict checkpoint loading is turned off, so using the initialized value for the missing keys."
                    )
                )
        checkpoint["state_dict"] = new_state_dict


    def _load_ema_weights_to_state_dict(self, checkpoint: Dict[str, Any]) -> None:
        if (
            checkpoint.get("pretrained_ema", False) and len(checkpoint["optimizer_states"]) == 0
        ):
            # NOTE: for lightweight EMA-only ckpts for releasing pretrained models,
            # we already have EMA weights in the state_dict
            rank_zero_print(red("No EMA weights found in the checkpoint, so using the initialized value for the EMA weights."))
            return
        if "ema" not in checkpoint.get("optimizer_states", [{}])[0]:
            rank_zero_print(
                red(
                    "No EMA weights found in the checkpoint, so using the initialized value for the EMA weights."
                )
            )
            return
        ema_weights = checkpoint["optimizer_states"][0]["ema"]
        parameter_keys = [
            f"{self.main_model_prefix}." + k for k, _ in getattr(self, self.main_model_prefix).named_parameters()
        ]
        assert len(parameter_keys) == len(
            ema_weights
        ), "Number of original weights and EMA weights do not match."
        for key, weight in zip(parameter_keys, ema_weights):
            checkpoint["state_dict"][key] = weight
