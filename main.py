"""
This repo is forked from [Boyuan Chen](https://boyuan.space/)'s research
template [repo](https://github.com/buoyancy99/research-template).
By its MIT license, you must keep the above sentence in `README.md`
and the `LICENSE` file to credit the author.

Main file for the project. This will create and run new experiments and load checkpoints from wandb.
Borrowed the wandb code from David Charatan and wandb.ai.
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf
from omegaconf.omegaconf import open_dict
OmegaConf.register_new_resolver("eval", eval)

from utils.print_utils import cyan, bold_red, manually_suppress_warnings
from utils.cluster_utils import submit_slurm_job, snapshot_project
from utils.distributed_utils import rank_zero_print, is_rank_zero
from utils.hydra_utils import unwrap_shortcuts, validate_shortcode_params, validate_tags_param

# Suppress the specific warning about __len__ in IterableDataset
manually_suppress_warnings()
import torch._dynamo
torch._dynamo.config.optimize_ddp = False

torch._dynamo.config.suppress_errors = True
# Set environment variables programmatically


def run_local(cfg: DictConfig):
    """
    Run experiment locally (or on already allocated compute node).

    This function handles:
    1. Setting up wandb logging
    2. Managing checkpoint loading and resuming
    3. Building and running the experiment
    """
    # delay some imports in case they are not needed in non-local envs for submission
    from experiments import build_experiment
    from utils.wandb_utils import OfflineWandbLogger, SpaceEfficientWandbLogger

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    os.environ["WANDB_API_KEY"] = cfg.wandb.api_key

    rank_zero_print(cyan(f"Running on {os.uname().nodename}"))

    # If sbatch, then cluster will be set. Otherwise, running through salloc or on a local node
    # Requires popping these, to prevent Lightning from using ntasks to automatically decide
    # the number of GPUs, since it uses the detection of these env variables to determine 
    # which to use. 
    # lightning, choosing which cluster env 
    #   https://github.com/Lightning-AI/pytorch-lightning/blob/37f559e3bee7572c45331877b66db0ae9210fbea/src/lightning/fabric/connector.py#L380-L392
    # lightning, auto detection of slurm cluster env based on those env variables.
    #   https://github.com/Lightning-AI/pytorch-lightning/blob/37f559e3bee7572c45331877b66db0ae9210fbea/src/lightning/fabric/plugins/environments/slurm.py#L227-L228
    if "cluster" not in cfg:
        os.environ.pop("SLURM_JOB_ID", None)
        os.environ.pop("SLURM_NTASKS", None)
        os.environ.pop("SLURM_NTASKS_PER_NODE", None)

    # Get yaml names
    hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    cfg_choice = OmegaConf.to_container(hydra_cfg.runtime.choices)

    with open_dict(cfg):
        if cfg_choice["experiment"] is not None:
            cfg.experiment._name = cfg_choice["experiment"]
        if cfg_choice["dataset"] is not None:
            cfg.dataset._name = cfg_choice["dataset"]
        if cfg_choice["algorithm"] is not None:
            cfg.algorithm._name = cfg_choice["algorithm"]

    # Set up the output directory.
    output_dir = Path(hydra_cfg.runtime.output_dir)
    if is_rank_zero:
        print(cyan(f"Outputs will be saved to: "), bold_red(output_dir))
        (output_dir.parents[1] / "latest-run").unlink(missing_ok=True)
        (output_dir.parents[1] / "latest-run").symlink_to(
            output_dir, target_is_directory=True
        )

    # Checkpoint loading logic - Multiple sources with priority:
    # 1. Requeued run (if it has checkpoints)
    # 2. Resume wandb run id, with loading from .ckpt local path, or load_model_state.
    # 3. Load from .ckpt local path or load_model_state, without resuming wandb
    # Yet to reimplement: load from wandb run.

    # NOTE: SLURM_RESTART_COUNT is not reliable so we must check for the requeue link directly.
    # To manually do the requeue, if stopped on another server for instance, then pass in 
    # +requeue=True in the config, and the requeue path will point to the outputs/.requeue_links/manual_requeue_location
    job_id = os.environ.get("SLURM_JOB_ID", "doesntexist")
    symlink_path = Path("outputs/.requeue_links") / job_id
    auto_detected_slurm_requeue = symlink_path.exists()
    is_manual_requeue = cfg.get("requeue", False)
    requeue_checkpoint_path = None
    if auto_detected_slurm_requeue or is_manual_requeue:
        rank_zero_print(cyan("Attempting requeue"))
        requeue_error_msg = None
        # job id is for auto requeue, manual requeue is if it is set by the config.
        job_id_or_manual_requeue_location = os.environ.get("SLURM_JOB_ID", "manual_requeue_location")
        symlink_path = Path("outputs/.requeue_links") / job_id_or_manual_requeue_location
        if symlink_path.exists():
            original_output_dir = symlink_path.resolve()
            requeue_meta_path = original_output_dir / "requeue_meta.yaml"
            rank_zero_print(cyan(f"Found requeue metadata path at f{requeue_meta_path}"))
            if requeue_meta_path.exists():
                meta = OmegaConf.load(str(requeue_meta_path))

                # Inject into cfg if resume is not already set
                if "resume" in meta:
                    with open_dict(cfg):
                        cfg.resume = meta["resume"]
                        rank_zero_print(cyan(f"[Requeue] Auto-resuming wandb run: {cfg.resume}"))
                else:
                    rank_zero_print(cyan("Requeue wandb id not found, starting a new one..."))

                # Find checkpoint path from metadata
                if "checkpoint_path" in meta:
                    requeue_checkpoint_path = Path(meta["checkpoint_path"])
                    if not requeue_checkpoint_path.exists():
                        requeue_error_msg = f"Requeue was marked as true, but the requeue checkpoint path does not exist at {requeue_checkpoint_path}"    
                else:
                    requeue_error_msg = f"Requeue was marked as true, but the requeue checkpoint path was not found in the yaml"
            else:
                requeue_error_msg = f"Requeue was marked as true, but the requeue metadata was not found at {requeue_meta_path}"
        else:
            requeue_error_msg = f"Requeue was marked as true, but the requeue symlink was not found at {symlink_path}"
        
        if requeue_error_msg is not None:
            raise RuntimeError(requeue_error_msg)

    # Set up logging with wandb.
    if cfg.wandb.mode != "disabled":
        # If resuming, merge into the existing run on wandb.
        # Resume from requeue takes priority
        resume = cfg.get("resume", None)
        name = (
            f"{cfg.name} ({output_dir.parent.name}/{output_dir.name})"
            if resume is None
            else None
        )

        logger_cls = SpaceEfficientWandbLogger

        offline = cfg.wandb.mode != "online"
        wandb_kwargs = {
            k: v
            for k, v in OmegaConf.to_container(cfg.wandb, resolve=True).items()
            if k != "mode" and k != "api_key"  # won't be passed into wandb.init
        }
        
        # Extract tags from command line arguments if provided
        tags = None
        for arg in sys.argv:
            if arg.startswith("tags=") or arg.startswith("+tags="):
                tags_str = arg.split("=", 1)[1]
                is_valid, error_msg, tags_list = validate_tags_param(tags_str)
                if is_valid:
                    tags = tags_list
                    rank_zero_print(cyan(f"Using wandb tags: {tags}"))
                break
        
        # Add tags to wandb_kwargs if provided
        if tags:
            wandb_kwargs["tags"] = tags
        logger = logger_cls(
            name=name,
            save_dir=str(output_dir),
            offline=offline,
            # log_model="all" if not offline else False,
            log_model=False,
            config=OmegaConf.to_container(cfg),
            id=resume,
            **wandb_kwargs,  # NOTE: this logger will init wandb for us
        )
    else:
        logger = None

    load = cfg.get("load", None)
    checkpoint_path = None
    
    # make sure if load is set, algorithm.load_model_state is not set, and vice versa
    load_model_state_ckpt_path = cfg.algorithm.get("load_model_state", None)
    if load and load_model_state_ckpt_path:
        raise RuntimeError("load, and load_model_state, cannot both be set for a run.")

    # Determine which checkpoint to use
    # requeue takes priority
    if resume and requeue_checkpoint_path:
        rank_zero_print(cyan(f"Now requeueing interrupted run, at wandb id: {resume} and we have the checkpoint at {requeue_checkpoint_path}"))
        checkpoint_path = requeue_checkpoint_path
    else:
        if resume and load:
            rank_zero_print(cyan(f"Resuming from wandb run {resume} and loading full state from local checkpoint path: {load}"))
            checkpoint_path = load
        elif resume and not load:
            if not load_model_state_ckpt_path:
                raise RuntimeError("If resuming a run, must set either load or load_model_state.")
            rank_zero_print(cyan(f"Resuming from wandb run: {resume}, and we are loading weights only from {load_model_state_ckpt_path}"))
            # not setting checkpoint_path because we only load weights with special logic elsewhere
            rank_zero_print(bold_red("Resuming a wandb run and loading weights only is discouraged. Wandb does not overwrite timesteps that came before " +
                                     "so logging results may be inconsistent until the iterations pass the original number. It may also overwrite logged videos"))
        elif not resume and load:
            rank_zero_print(cyan(f"Not resuming wandb run, but still loading full state from local checkpoint path: {load}"))
            checkpoint_path = load
        elif not resume and load_model_state_ckpt_path:
            rank_zero_print(cyan(f"Not resuming wandb run, but still loading weights only state from local checkpoint path: {load_model_state_ckpt_path}"))

    # launch experiment with the determined checkpoint
    # Here we need to make sure the checkpoint path is correct from our saved path
    # the .ckpt includes all optimizer states and epoch etc.
    experiment = build_experiment(cfg, logger, checkpoint_path)
    for task in cfg.experiment.tasks:
        experiment.exec_task(task)


def run_slurm(cfg: DictConfig):
    python_args = " ".join(sys.argv[1:]) + " +_on_compute_node=True"

    # Note: If we are using the duplicated path, make sure this does the right thing still
    project_root = Path.cwd()
    while not (project_root / ".git").exists():
        project_root = project_root.parent
        if project_root == Path("/"):
            raise Exception("Could not find repo directory!")

    # Due to how slurm requeueing works we will need to save a snapshot of our project, using the date time as the unique identifier
    # This will prevent local changes from disrupting jobs if they are requeued.
    
    slurm_log_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{cfg.name}"
    run_dir = project_root  / "outputs" / "temp_proj_snapshot" / slurm_log_name
    excluded_dirs = ["data", "outputs", ".git", "downloaded_checkpoints", "huggingface"]
    snapshot_project(project_root, run_dir, excluded_dirs)
    project_root = run_dir

    # creates the sbatch script, submits, and then saves it to slurm_logs
    slurm_log_dir = submit_slurm_job(
        cfg,
        python_args,
        project_root,
        slurm_log_name=slurm_log_name
    )

    print(
        "Once the job gets allocated and starts running, we will print a command below "
        "for you to trace the errors and outputs: (Ctrl + C to exit without waiting)"
    )
    msg = f"tail -f {slurm_log_dir}/* \n"
    try:
        while not list(slurm_log_dir.glob("*.out")) and not list(
            slurm_log_dir.glob("*.err")
        ):
            time.sleep(1)
        print(cyan("To trace the outputs and errors, run the following command:"), msg)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Exiting...")
        print(
            cyan(
                "To trace the outputs and errors, manually wait for the job to start and run the following command:"
            ),
            msg,
        )


@hydra.main(
    version_base=None,
    config_path="configurations",
    config_name="config",
)
def run(cfg: DictConfig):
    """
    Main entry point for running experiments.
    """

    if "name" not in cfg:
        raise ValueError(
            "must specify a name for the run with command line argument '+name=[name]'"
        )

    if not cfg.wandb.get("entity", None):
        raise ValueError(
            "must specify wandb entity in 'configurations/config.yaml' or with command line"
            " argument 'wandb.entity=[entity]' \n An entity is your wandb user name or group"
            " name. This is used for logging. If you don't have an wandb account, please signup at https://wandb.ai/"
        )

    if cfg.wandb.project is None:
        cfg.wandb.project = str(Path(__file__).parent.name)

    # Validate configuration against template
    # Config validation is ignored for release version. Put it back by uncommenting these lines:
    # if not (cfg.ignore_cfg_validation or os.environ.get("IGNORE_CFG_VALIDATION", False)) and not validate_config_against_template(cfg):
    #     raise ValueError("Configuration validation failed. Please check the output above for details.")

    # Determine whether to run locally or submit to Slurm
    # Cluster does not have package global so it will be auto detected here if it is set
    if "cluster" in cfg and not "_on_compute_node" in cfg:
        # If cluster config exists but we're not on compute node yet, submit Slurm job
        print(
            cyan(
                "Slurm detected, submitting to compute node instead of running locally..."
            )
        )
        run_slurm(cfg)
    else:
        # Either no cluster config or already on compute node, run locally
        run_local(cfg)


if __name__ == "__main__":
    # First validate shortcode parameters
    validate_shortcode_params(sys.argv, config_path="configurations")
    
    # Then unwrap shortcuts
    sys.argv = unwrap_shortcuts(
        sys.argv, config_path="configurations", config_name="config"
    )
    run()  # pylint: disable=no-value-for-parameter
