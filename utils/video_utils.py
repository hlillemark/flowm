import os
import torch
import numpy as np
from torch import Tensor
from typing import Union, List, Optional
from torchvision.io import write_video

def write_video_to_file(
    video: Union[Tensor, np.ndarray],
    path: str,
    fps: int = 10,
    quality: int = 7,
    codec: str = "h264"
) -> None:
    """
    Write a video tensor or numpy array to an MP4 file using torchvision.
    
    Args:
        video (Union[Tensor, np.ndarray]): Video frames with shape [T, C, H, W] (PyTorch tensor)
            or [T, H, W, C] (numpy array)
        path (str): Path to save the video file (should end with .mp4)
        fps (int): Frames per second
        quality (int): Video quality (0-10, higher is better)
        codec (str): Video codec to use
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    
    # Convert numpy array to torch tensor if needed
    if isinstance(video, np.ndarray):
        # Convert to torch tensor
        video = torch.from_numpy(video)
        
        # If shape is [T, H, W, C], convert to [T, C, H, W]
        if video.ndim == 4 and video.shape[-1] <= 3:
            video = video.permute(0, 3, 1, 2)
    
    # Handle grayscale videos
    if video.ndim == 3:  # [T, H, W]
        video = video.unsqueeze(1)  # Add channel dimension
    
    # Ensure we have 3 channels for RGB
    if video.shape[1] == 1:
        video = video.repeat(1, 3, 1, 1)
    
    # Normalize to [0, 255] range if float
    if video.dtype == torch.float16 or video.dtype == torch.float32:
        video = (video.clamp(0, 1) * 255).to(torch.uint8)
    
    # Convert from [T, C, H, W] to [T, H, W, C] for torchvision
    video = video.permute(0, 2, 3, 1)
    
    # Set options based on quality
    options = {"crf": str(10 - quality)} if quality is not None else None
    
    # Write video file using torchvision
    write_video(
        path,
        video.cpu(),
        fps=fps,
        video_codec=codec,
        options=options
    )