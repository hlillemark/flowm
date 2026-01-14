"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research 
template [repo](https://github.com/buoyancy99/research-template). 
By its MIT license, you must keep the above sentence in `README.md` 
and the `LICENSE` file to credit the author.
"""

from typing import Literal, Optional, Tuple
import string
import random
from pathlib import Path
from omegaconf import DictConfig
import wandb
from utils.print_utils import cyan
from .huggingface_utils import download_from_hf
from utils.distributed_utils import rank_zero_print, is_rank_zero

from typing import Callable, Dict
import torch

def is_run_id(run_id: str) -> bool:
    """Check if a string is a run ID."""
    return len(run_id) == 8 and run_id.isalnum()


def generate_run_id() -> str:
    """Generate a random 8-character alphanumeric string."""
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(8))


def generate_unexisting_run_id(entity: str, project: str) -> str:
    """Generate a random 8-character alphanumeric string that does not exist in the project."""
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    existing_ids = {run.id for run in runs}
    while True:
        run_id = generate_run_id()
        if run_id not in existing_ids:
            return run_id


def parse_load(load: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse load into run_id and download option.
    (for load=xxxxxxxx in configurations)
    - If load_id is a run_id, return the run_id and None.
    - If load_id is of the form run_id:option, return run_id and option.
    - Otherwise, return None, None.
    """
    split = load.split(":")
    if 1 <= len(split) <= 2 and is_run_id(split[0]):
        return split[0], split[1] if len(split) == 2 else None
    return None, None


def version_to_int(artifact) -> int:
    """Convert versions of the form vX to X. For example, v12 to 12."""
    return int(artifact.version[1:])


def is_existing_run(run_path: str) -> bool:
    """Check if a run exists."""
    api = wandb.Api()
    try:
        _ = api.run(run_path)
        return True
    except wandb.errors.CommError:
        return False
    return False


def has_checkpoint(run_path: str) -> bool:
    """Check if a run has a committed model checkpoint."""
    api = wandb.Api()
    try:
        run = api.run(run_path)
        for artifact in run.logged_artifacts():
            if artifact.type == "model" and artifact.state == "COMMITTED":
                return True
        return False
    except wandb.errors.CommError:
        return False
    return False


def download_checkpoint(
    run_path: str, download_dir: Path, option: Literal["latest", "best"] = "latest"
) -> Path:
    api = wandb.Api()
    run = api.run(run_path)

    # Find the latest saved model checkpoint.
    checkpoint = None
    for artifact in run.logged_artifacts():
        if artifact.type != "model" or artifact.state != "COMMITTED":
            continue

        if option in artifact.aliases or option == artifact.version:
            checkpoint = artifact
            break

    if checkpoint is None:
        print(f"No {option} model checkpoint found in {run_path}.")

    # Download the checkpoint.
    download_dir.mkdir(exist_ok=True, parents=True)
    root = download_dir / run_path
    checkpoint.download(root=root)
    return root / "model.ckpt"


def download_pretrained(
    name: str,
) -> str:
    """
    Download a pretrained model from the DFoT Hugging Face model hub.
    Set is_full to True to download the full model
    (including optimizer states and non-EMA weights).
    """
    prefix, name = name.split(":")
    download_from_hf(filename="config.json")
    return download_from_hf(filename=f"{prefix}_models/{name}")


def is_wandb_run_path(run_path: str) -> bool:
    split = run_path.split("/")
    return len(split) == 3 and is_run_id(split[-1])


def is_hf_path(path: str) -> bool:
    return path.startswith("pretrained:") or path.startswith("full:")


def download_vae_checkpoints(
    cfg: DictConfig,
):
    pretrained_paths = []
    vae = cfg.algorithm.get("vae", None)
    if vae and vae.get("pretrained_path", None):
        pretrained_paths.append(vae.pretrained_path)

    pretrained_path = cfg.algorithm.get("pretrained_path", None)
    if pretrained_path:
        pretrained_paths.append(pretrained_path)

    wandb_pretrained_paths = [
        path for path in pretrained_paths if is_wandb_run_path(path)
    ]
    hf_pretrained_paths = [path for path in pretrained_paths if is_hf_path(path)]

    for path in wandb_pretrained_paths:
        print(cyan("Downloading pretrained VAE from Wandb:"), path)
        download_checkpoint(path, Path("outputs/downloaded"), option="best")

    for path in hf_pretrained_paths:
        print(cyan("Downloading pretrained VAE from Hugging Face:"), path)
        download_pretrained(path)


def wandb_to_local_path(run_path: str) -> Path:
    return Path("outputs/downloaded") / run_path / "model.ckpt"


def smart_load_state_dict(model, state_dict, custom_handlers: Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], verbose=True):
    """
    Load a state dict into a model.
    If the shape of the state dict key does not match the shape of the model key,
    use the custom handler to fix the value.
    Args:
        model: The model to load the state dict into.
        state_dict: The state dict to load.
        custom_handlers: A dictionary of custom handlers for each key.
        verbose: Whether to print verbose output.
    Returns:
        A list of keys that were skipped.
    """
    model_state = model.state_dict()
    loaded_state = {}
    skipped_keys = []

    for k, v in state_dict.items():
        if k not in model_state:
            if verbose:
                rank_zero_print(f"[Missing Key] {k} not in model. Skipping.")
            continue

        model_shape = model_state[k].shape
        if model_shape != v.shape:
            if k in custom_handlers:
                try:
                    fixed_value = custom_handlers[k](v, model_state[k])
                    if verbose:
                        rank_zero_print(f"[Custom Handler] {k}: checkpoint shape {v.shape} -> model shape {fixed_value.shape}")
                    loaded_state[k] = fixed_value
                except Exception as e:
                    rank_zero_print(f"[Handler Failed] {k}: {e}")
                    skipped_keys.append(k)
            else:
                if verbose:
                    rank_zero_print(f"[Shape Mismatch] {k}: checkpoint shape {v.shape}, model shape {model_shape}. Skipping.")
                skipped_keys.append(k)
        else:
            loaded_state[k] = v

    model.load_state_dict(loaded_state, strict=False)
    return skipped_keys


depth_fine_tuning_handlers = {
    "proj_out.weight": lambda ckpt, model: ckpt.repeat(2, 1)[:model.shape[0]],
    "proj_out.bias": lambda ckpt, model: ckpt.repeat(2)[:model.shape[0]],
}
