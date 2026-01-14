from enum import Enum


class VideoMetricType(str, Enum):
    LPIPS = "lpips"
    FID = "fid"
    FVD = "fvd"
    RFID = "rfid" # Not Implemented
    MSE = "mse"
    MASKED_MSE = "masked_mse"
    SSIM = "ssim"
    PSNR = "psnr"
    IS = "is"
    REAL_IS = "real_is"
    FVMD = "fvmd"
    VBENCH = "vbench"
    REAL_VBENCH = "real_vbench"
    PER_FRAME_MSE = "per_frame_mse"


class VideoMetricModelType(str, Enum):
    LPIPS = "Lpips"
    INCEPTION_V3 = "InceptionV3"
    I3D = "I3D"
    PIPS = "PIPS"
    CLIP_B_32 = "CLIP_B_32"
    CLIP_L_14 = "CLIP_L_14"
    DINO = "DINO"
    LAION = "LAION"
    MUSIQ = "MUSIQ"
    RAFT = "RAFT"
    AMT_S = "AMT_S"

# Add here to exclude video metrics from wandb log and instead log them manually to a .pt file.
EXCLUDED_VIDEO_METRICS_FROM_WANDB_LOG = ["per_frame_mse"]
