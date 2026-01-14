import wandb
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download a W&B artifact checkpoint without running wandb.init()")
    parser.add_argument("--wandbpath", type=str, required=True, help="path to wandb checkpoint, e.g.: your-wandb-path/memory/model-ckpt:v2")
    parser.add_argument("--targetdir", type=str, required=True, help="path to download dir e.g.: projects/checkpoints")

    args = parser.parse_args()

    api = wandb.Api()

    artifact = api.artifact(args.wandbpath, type='model')

    download_path = artifact.download(root=args.targetdir)

    print(f"Artifact downloaded to: {download_path}")

if __name__ == "__main__":
    main()
