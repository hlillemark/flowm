"""
utils for submitting to clusters, such as slurm
"""

import os
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
import subprocess
from utils.print_utils import cyan
from typing import Optional

def submit_slurm_job(
    cfg: DictConfig,
    python_args: str,
    project_root: Path,
    slurm_log_name: Optional[str] = None,
):
    if slurm_log_name is None:
        slurm_log_name = f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{cfg.name}"
    log_dir = project_root / "outputs" / "slurm_logs" / slurm_log_name
    log_dir.mkdir(exist_ok=True, parents=True)
    (project_root / "outputs" / "slurm_logs" / "latest").unlink(missing_ok=True)
    (project_root / "outputs" / "slurm_logs" / "latest").symlink_to(log_dir, target_is_directory=True)

    params = dict(name=cfg.name, log_dir=log_dir, project_root=project_root, python_args=python_args)
    params.update(cfg.cluster.params)

    slurm_script = cfg.cluster.launch_template.format(**params)

    slurm_script_path = log_dir / "job.slurm"
    with slurm_script_path.open("w") as f:
        f.write(slurm_script)

    os.system(f"chmod +x {slurm_script_path}")
    os.system(f"sbatch {slurm_script_path}")

    print(f"\n{cyan('script:')} {slurm_script_path}\n{cyan('slurm errors and logs:')} {log_dir}\n")

    return log_dir


def snapshot_project(project_root: Path, run_dir: Path, excluded_dirs: list[str]):
    """
    Copies the contents of project_root into run_dir using rsync, excluding certain directories.
    Creates symlinks to excluded_dirs inside run_dir pointing back to the original ones.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    # Construct rsync command, exclude excluded dirs
    exclude_args = sum([["--exclude", f"{d}/"] for d in excluded_dirs], [])
    rsync_cmd = ["rsync", "-a", *exclude_args, f"{project_root}/", str(run_dir)]

    subprocess.run(rsync_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    # Create symlinks for excluded directories
    for name in excluded_dirs:
        src = project_root / name
        dst = run_dir / name
        if not dst.exists():
            print(f"Symlinking excluded dir: {src} -> {dst}")
            dst.symlink_to(src.resolve(), target_is_directory=src.is_dir())
