import wandb
import hydra
from omegaconf import DictConfig
import argparse
import os
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

"""
python utils/upload_model_to_wandb.py 'load="YOUR_CKPT_PATH"' resume="YOUR_WANDB_ID"
python utils/upload_model_to_wandb.py 'load="checkpoints/epoch=0-step=290000.ckpt"' resume="6l6agd9l"
"""

@hydra.main(
    version_base=None,
    config_path="../configurations",
    config_name="config",
)
def main(cfg: DictConfig):
    ckpt_path = cfg.load
    wandb_id = cfg.resume
    
    # If the model is a deepspeed checkpoint, we need to first convert to pytorch version
    # Make sure to remove the / at the end of the ckpt load path
    if os.path.isdir(ckpt_path):
        # Extract directory name and change extension to .pt
        ckpt_name = os.path.basename(ckpt_path).replace(".ckpt", ".pt")
        new_ckpt_dir = os.path.join("downloaded_checkpoints", wandb_id)
        os.makedirs(new_ckpt_dir, exist_ok=True)
        new_ckpt_path = os.path.join(new_ckpt_dir, ckpt_name)
        
        convert_zero_checkpoint_to_fp32_state_dict(
            ckpt_path,
            new_ckpt_path
        )
        
        ckpt_path = new_ckpt_path
    
    os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
    run = wandb.init(
        entity=cfg.wandb.entity,
        project=cfg.wandb.project,
        # TODO: Uncomment these two lines if you wish to upload to an existing wandb object
        id=wandb_id,
        resume="must"
        
    )

    artifact = wandb.Artifact(
        f"model-ckpt-{wandb_id}",
        type="model",
        description="",#"Checkpoint cogv-matrix pretrained with zero noise applied to cached token, from epoch 8, step 200000",
        metadata={
            "epoch": 8,
            "step": 200000,
        }
    )

    artifact.add_file(ckpt_path)

    run.log_artifact(artifact, aliases=["latest", "best"])

    run.finish()

if __name__ == "__main__":
    main()

