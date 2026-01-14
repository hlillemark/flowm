from typing import Optional, List, Literal, Dict, Any
import wandb
import numpy as np
import torch
import torch.distributed as dist

import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.animation as animation
from PIL import Image
from pathlib import Path
import imageio
import torchvision
from utils.distributed_utils import rank_zero_print
import cv2

plt.set_loglevel("warning")

from torchmetrics.functional import mean_squared_error, peak_signal_noise_ratio
from torchmetrics.image import (
    PeakSignalNoiseRatio,
)
from torchmetrics.functional import (
    structural_similarity_index_measure,
    universal_image_quality_index,
)
from einops import rearrange
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    FrechetInceptionDistance,
)

def log_video_as_images(
    observation_hats: List[torch.Tensor] | torch.Tensor,
    observation_gt: Optional[torch.Tensor] = None,
    step=0,
    namespace="train",
    prefix="video",
    postfix=[],
    indent=0,
    context_frames=0,
    color=(255, 255, 0),
    logger=None,
    n_frames=None,
    raw_dir=None,
    fps = 20,
    normalize = False,
    indices: List[int] = None,
    format: Literal["pdf", "png"] = "pdf",
    video_metadata: Dict[str, Any] = None,
    rank: int | None = None,
    log_to_logger: bool = True,
):
    """
    Log videos as time-concatenated images.

    Assumptions:
    - Input tensors are in [0, 1] range with shape (B, T, C, H, W).
    - If multiple predictions are provided, only the first one is visualized.
    - Grayscale inputs (C == 1) are repeated to 3 channels for visualization.
    """
    if not logger and log_to_logger:
        logger = wandb

    raw_dir_path = None
    if raw_dir is not None:
        raw_dir_path = Path(raw_dir)
        if rank is not None:
            raw_dir_path = raw_dir_path / f"rank{rank}"

    # Normalize input to a list and choose the first prediction to visualize
    if isinstance(observation_hats, torch.Tensor):
        observation_hats = [observation_hats]
    x = observation_hats[0]

    # Optional index selection before moving to CPU
    if indices is not None:
        x = x[:, indices]
        if observation_gt is not None:
            observation_gt = observation_gt[:, indices]
    else:
        rank_zero_print(f"No indices provided, logging all frames")

    # Move to CPU early to save GPU memory
    x = x.detach().cpu()
    if observation_gt is not None:
        observation_gt = observation_gt.detach().cpu()

    # Validate shape
    if x.dim() != 5:
        raise ValueError(f"Expected (B, T, C, H, W), got {tuple(x.shape)}")

    # Ensure 3 channels for visualization
    if x.shape[2] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
        if observation_gt is not None and observation_gt.shape[2] == 1:
            observation_gt = observation_gt.repeat(1, 1, 3, 1, 1)

    B, _, C, H, W = x.shape

    # Optional per-sample normalization to [0, 1]
    def maybe_normalize(t: torch.Tensor) -> torch.Tensor:
        if not normalize:
            return t.clamp(0, 1)
        t_min = t.amin(dim=(1, 2, 3, 4), keepdim=True)
        t_max = t.amax(dim=(1, 2, 3, 4), keepdim=True)
        denom = torch.where((t_max - t_min) > 0, (t_max - t_min), torch.ones_like(t_max))
        return ((t - t_min) / denom).clamp(0, 1)

    x = maybe_normalize(x)
    if observation_gt is not None:
        observation_gt = maybe_normalize(observation_gt.type_as(x))

    # Convert to uint8 numpy arrays
    x_np = (x.numpy() * 255.0).astype(np.uint8)  # (B, T, C, H, W)
    gt_np = (observation_gt.numpy() * 255.0).astype(np.uint8) if observation_gt is not None else None


    # Save raw frames if requested
    if raw_dir_path is not None:
        for b in range(B):
            for t in range(len(indices)):
                save_dir = raw_dir_path / f"{b + indent}"
                gen_dir = save_dir / namespace / "generated"
                gt_dir = save_dir / namespace / "gt"
                gen_dir.mkdir(parents=True, exist_ok=True)
                gt_dir.mkdir(parents=True, exist_ok=True)

                gen_img = np.transpose(x_np[b, t], (1, 2, 0))  # HWC
                imageio.imwrite(gen_dir / f"frame_{indices[t]}.{format}", gen_img)
                if gt_np is not None:
                    gt_img = np.transpose(gt_np[b, t], (1, 2, 0))  # HWC
                    imageio.imwrite(gt_dir / f"frame_{indices[t]}.{format}", gt_img)

    # Build strips and log
    for b in range(B):
        # add colored borders to each generated frame: yellow for context, red for generated
        gen_frames = []
        for t in range(len(indices)):
            frame = np.transpose(x_np[b, t], (1, 2, 0))  # HWC
            color_rgb = (255, 255, 0) if indices[t] <= context_frames else (255, 0, 0)
            thickness = 2
            h, w = frame.shape[:2]
            bordered = np.empty((h + 2 * thickness, w + 2 * thickness, 3), dtype=frame.dtype)
            bordered[:, :, :] = color_rgb
            bordered[thickness:-thickness, thickness:-thickness, :] = frame
            gen_frames.append(bordered)
        gen_strip = np.concatenate(gen_frames, axis=1)  # concat along width, shape (H, W * T)
        # ensure contiguous, writeable HWC uint8 array for OpenCV
        gen_strip = np.ascontiguousarray(gen_strip)
        if not gen_strip.flags.writeable:
            gen_strip = gen_strip.copy()


        if gt_np is not None:
            # add borders to each GT frame using the same color mapping as generated frames
            gt_frames = []
            for t in range(len(indices)):
                gt_frame = np.transpose(gt_np[b, t], (1, 2, 0))  # HWC
                color_rgb = (255, 255, 0) if indices[t] <= context_frames else (255, 0, 0)
                thickness = 2
                h, w = gt_frame.shape[:2]
                bordered_gt = np.empty((h + 2 * thickness, w + 2 * thickness, 3), dtype=gt_frame.dtype)
                bordered_gt[:, :, :] = color_rgb
                bordered_gt[thickness:-thickness, thickness:-thickness, :] = gt_frame
                gt_frames.append(bordered_gt)
            gt_strip = np.concatenate(gt_frames, axis=1)
            if gen_strip.shape != gt_strip.shape:
                raise ValueError(f"gen_strip.shape {gen_strip.shape} != gt_strip.shape {gt_strip.shape}")
            to_log = np.concatenate([gen_strip, gt_strip], axis=0) # along height
        else:
            to_log = gen_strip
        
        if raw_dir_path is not None:
            imageio.imwrite(raw_dir_path / f"{b + indent}" / namespace / f"concat.{format}", to_log)

        if log_to_logger and logger is not None:
            logger.log(
                {f"{namespace}/{prefix}_{b + indent}": wandb.Image(to_log, caption=f"indices={indices}"),
                "step": step}, # cannot write step as logger.log(.., step=step) because this requires step to be strictly increasing; but wandb logger step will automatically increase per log call by default
                commit=False # don't want to automatically increase the log step of wandb logger
            )


# FIXME: clean up & check this util
def log_video(
    observation_hats: List[torch.Tensor] | torch.Tensor,
    observation_gt: Optional[torch.Tensor] = None,
    step=0,
    namespace="train",
    prefix="video",
    postfix=[],
    captions=[],
    indent=0,
    context_frames=0,
    color=(255, 255, 0),
    logger=None,
    n_frames=None,
    raw_dir=None,
    fps = 20,
    normalize = False,
    video_metadata: Dict[str, Any] = None,
    rank: int | None = None,
    log_to_logger: bool = True,
):
    """
    take in video tensors in range [-1, 1] and log into wandb

    :param observation_gt: ground-truth observation tensor of shape (batch, frame, channel, height, width)
    :param observation_hats: list of predicted observation tensor of shape (batch, frame, channel, height, width)
    :param step: an int indicating the step number
    :param namespace: a string specify a name space this video logging falls under, e.g. train, val
    :param prefix: a string specify a prefix for the video name
    :param postfix: a list of strings specify postfixes for the video name
    :param context_frames: an int indicating how many frames in observation_hat are ground truth given as context
    :param color: a tuple of 3 numbers specifying the color of the border for ground truth frames
    :param logger: optional logger to use. use global wandb if not specified
    :param normalize: whether to normalize the frames to [0, 1], for better visualization
    """
    if not logger and log_to_logger:
        logger = wandb
    if isinstance(observation_hats, torch.Tensor):
        observation_hats = [observation_hats.cpu()]
    
    raw_dir_path = None
    if raw_dir is not None:
        raw_dir_path = Path(raw_dir)
        if rank is not None:
            raw_dir_path = raw_dir_path / f"rank{rank}"
    
    frame_length = observation_hats[0].shape[1]


    if frame_length <= context_frames:
        raise ValueError(f"frame_length {frame_length} must be greater than context_frames {context_frames}, namespace: {namespace}, prefix: {prefix}")

    use_gt = observation_gt is not None
    if observation_gt is None:
        observation_gt = torch.zeros_like(observation_hats[0])
    observation_gt = observation_gt.type_as(observation_hats[0]).cpu()

    # Ensure videos are 3-channel RGB for logging; replicate grayscale (1-channel) to 3 channels
    if observation_hats[0].dim() == 5 and observation_hats[0].shape[2] == 1:
        observation_hats = [x.repeat(1, 1, 3, 1, 1) for x in observation_hats]
        observation_gt = observation_gt.repeat(1, 1, 3, 1, 1)

    if isinstance(context_frames, int):
        context_frames = torch.arange(context_frames, device=observation_gt.device)
    for observation_hat in observation_hats:
        observation_hat[:, context_frames] = observation_gt[:, context_frames]

    if raw_dir_path is not None:
        raw_dir_path.mkdir(parents=True, exist_ok=True)

        for i, (gt, hat) in tqdm(enumerate(zip(observation_gt, observation_hats[0])), total=observation_gt.shape[0], desc="Saving raw videos"):
            (raw_dir_path / f"{i + indent}" / namespace).mkdir(parents=True, exist_ok=True)


            # Calculate PSNR between generated and ground truth videos
            # Convert back to torch tensors for PSNR calculation (range [0, 1])
            n_context_frames = len(context_frames)
            
            # Calculate PSNR for the entire video sequence
            psnr_value_image = PeakSignalNoiseRatio(data_range=1.0)(hat[n_context_frames:], gt[n_context_frames:])
            
            # Save PSNR value as .pt file
            torch.save(psnr_value_image.item(), raw_dir_path / f"{i + indent}" / namespace / "psnr.pt")
            
            # Calculate per-frame MSE for generated frames (excluding context frames)
            # hat and gt have shape (T, C, H, W)
            hat_gen = hat[n_context_frames:]  # (T_gen, C, H, W)
            gt_gen = gt[n_context_frames:]    # (T_gen, C, H, W)
            per_frame_mse = ((hat_gen - gt_gen) ** 2).mean(dim=(1, 2, 3))  # (T_gen,)
            
            # Save per-frame MSE as .pt file
            torch.save(per_frame_mse, raw_dir_path / f"{i + indent}" / namespace / "per_frame_mse.pt")
            # imageio.mimwrite(
            #     (raw_dir_path / f"{i + indent}") / "gen_preview.mp4",
            #     frames,
            #     fps=fps,
            #     macro_block_size=None,
            # )

            # torchvision.io.write_video expects input of shape (T, H, W, C) with dtype uint8 and pixel values in [0, 255]
            # If input is float in [0, 1], it should be scaled to [0, 255] and converted to uint8
            video_np = hat.numpy().transpose(0, 2, 3, 1)  # (T, H, W, C)
            if video_np.dtype != 'uint8':
                # Assume input is float in [0, 1], scale to [0, 255]
                video_np = (video_np * 255).clip(0, 255).astype('uint8')
            gt_np = gt.numpy().transpose(0, 2, 3, 1)
            if gt_np.dtype != 'uint8':
                # Assume input is float in [0, 1], scale to [0, 255]
                gt_np = (gt_np * 255).clip(0, 255).astype('uint8')
            torchvision.io.write_video(str(raw_dir_path / f"{i + indent}" / namespace / "gen_preview.mp4"), video_np, fps=fps)
            torchvision.io.write_video(str(raw_dir_path / f"{i + indent}" / namespace / "gt_preview.mp4"), gt_np, fps=fps)


            if video_metadata is not None:
                try:
                    with open(raw_dir_path / f"{i + indent}" / namespace / "metadata.txt", "w") as f:
                        f.write(f"video_path: {video_metadata['path'][i]} \n")
                        start_frame, end_frame = video_metadata["clip"][0][i].cpu(), video_metadata["clip"][1][i].cpu()
                        f.write(f"start frame: {start_frame} \n")
                        f.write(f"end frame: {end_frame} \n")
                        f.write(f"psnr: {psnr_value_image.item()} \n")
                except Exception as e:
                    print(f"Failed to write metadata: {e} on rank {dist.get_rank() if dist.is_initialized() else 0}")
                    import traceback
                    traceback.print_exc()

            

    # Add red border of 1 pixel width to the context frames
    context_frames, indices = torch.meshgrid(
        context_frames,
        torch.tensor([0, -1], device=observation_gt.device, dtype=torch.long),
        indexing="ij",
    )

    for i, c in enumerate(color):
        c = c / 255.0
        for observation_hat in observation_hats:
            observation_hat[:, context_frames, i, indices, :] = c
            observation_hat[:, context_frames, i, :, indices] = c
        observation_gt[:, :, i, [0, -1], :] = c
        observation_gt[:, :, i, :, [0, -1]] = c
    
    # Move to CPU early to save GPU memory
    observation_hats = [x.cpu() for x in observation_hats]
    if observation_gt is not None:
        observation_gt = observation_gt.cpu()

    # Concatenate first, then normalize for better visual consistency
    video = torch.cat([*observation_hats, observation_gt], -1) if use_gt else torch.cat([*observation_hats], -1)
    
    if normalize:
        # Normalize each video sample individually for better visualization
        video_normalized = []
        for i in range(video.shape[0]):
            vid_sample = video[i]
            vid_min = vid_sample.min()
            vid_max = vid_sample.max()
            if vid_max > vid_min:  # Avoid division by zero
                vid_normalized = (vid_sample - vid_min) / (vid_max - vid_min)
            else:
                vid_normalized = vid_sample
            video_normalized.append(vid_normalized)
        video = torch.stack(video_normalized)
    
    if video.dtype == torch.bfloat16:
        video = video.float()
    
    video = video.detach().cpu().numpy()

    # reshape to original shape
    if n_frames is not None:
        video = rearrange(
            video, "(b n) t c h w -> b (n t) c h w", n=n_frames // video.shape[1]
        )

    video = (np.clip(video, a_min=0.0, a_max=1.0) * 255).astype(np.uint8)
    # video[..., 1:] = video[..., :1]  # remove framestack, only visualize current frame
    n_samples = len(video)

    if raw_dir_path is not None:
        raw_dir_path.mkdir(parents=True, exist_ok=True)
        for i in range(n_samples):
            (raw_dir_path / f"{i + indent}").mkdir(parents=True, exist_ok=True)
            
            torchvision.io.write_video(
                str(raw_dir_path / f"{i + indent}" / namespace / "concatenated.mp4"),
                video[i].transpose(0, 2, 3, 1),
                fps=fps,
            )

    # use wandb directly here since pytorch lightning doesn't support logging videos yet
    if isinstance(captions, str):
        captions = [captions] * n_samples

    for i in range(n_samples):
        name = f"{namespace}/{prefix}_{i + indent}" + (
            f"_{postfix[i]}" if i < len(postfix) else ""
        )
        caption = captions[i] if i < len(captions) else None
        if log_to_logger and logger is not None:
            logger.log(
                {
                    name: wandb.Video(video[i], fps=fps, caption=caption, format="mp4"),
                    "step": step # cannot write step as logger.log(.., step=step) because this requires step to be strictly increasing; but wandb logger step will automatically increase per log call by default
                },
                commit=False # don't want to automatically increase the log step of wandb logger
            )



def get_validation_metrics_for_videos(
    observation_hat,
    observation_gt,
    lpips_model: Optional[LearnedPerceptualImagePatchSimilarity] = None,
    fid_model: Optional[FrechetInceptionDistance] = None,
):
    """
    :param observation_hat: predicted observation tensor of shape (frame, batch, channel, height, width)
    :param observation_gt: ground-truth observation tensor of shape (frame, batch, channel, height, width)
    :param lpips_model: a LearnedPerceptualImagePatchSimilarity object from algorithm.common.metrics
    :param fid_model: a FrechetInceptionDistance object  from algorithm.common.metrics
    :return: a tuple of metrics
    """
    batch, frame, channel, height, width = observation_hat.shape
    output_dict = {}
    # some metrics don't fully support fp16
    if observation_hat.dtype == torch.float16:
        observation_hat = observation_hat.to(torch.float32)
    if observation_gt.dtype == torch.float16:
        observation_gt = observation_gt.to(torch.float32)

    # reshape to (batch * frame, channel, height, width) for image losses
    observation_hat = observation_hat.view(-1, channel, height, width)
    observation_gt = observation_gt.view(-1, channel, height, width)

    output_dict["mse"] = mean_squared_error(observation_hat, observation_gt)
    output_dict["psnr"] = peak_signal_noise_ratio(
        observation_hat, observation_gt, data_range=2.0
    )
    output_dict["ssim"] = structural_similarity_index_measure(
        observation_hat, observation_gt, data_range=2.0
    )
    output_dict["uiqi"] = universal_image_quality_index(observation_hat, observation_gt)
    # operations for LPIPS and FID
    observation_hat = torch.clamp(observation_hat, -1.0, 1.0)
    observation_gt = torch.clamp(observation_gt, -1.0, 1.0)

    if lpips_model is not None:
        lpips_model.update(observation_hat, observation_gt)
        lpips = lpips_model.compute().item()
        # Reset the states of non-functional metrics
        output_dict["lpips"] = lpips
        lpips_model.reset()

    if fid_model is not None:
        observation_hat_uint8 = ((observation_hat + 1.0) / 2 * 255).type(torch.uint8)
        observation_gt_uint8 = ((observation_gt + 1.0) / 2 * 255).type(torch.uint8)
        fid_model.update(observation_gt_uint8, real=True)
        fid_model.update(observation_hat_uint8, real=False)
        fid = fid_model.compute()
        output_dict["fid"] = fid
        # Reset the states of non-functional metrics
        fid_model.reset()

    return output_dict


def is_grid_env(env_id):
    return "maze2d" in env_id or "diagonal2d" in env_id


def get_maze_grid(env_id):
    # import gym
    # maze_string = gym.make(env_id).str_maze_spec
    if "large" in env_id:
        maze_string = "############\\#OOOO#OOOOO#\\#O##O#O#O#O#\\#OOOOOO#OOO#\\#O####O###O#\\#OO#O#OOOOO#\\##O#O#O#O###\\#OO#OOO#OGO#\\############"
    if "medium" in env_id:
        maze_string = "########\\#OO##OO#\\#OO#OOO#\\##OOO###\\#OO#OOO#\\#O#OO#O#\\#OOO#OG#\\########"
    if "umaze" in env_id:
        maze_string = "#####\\#GOO#\\###O#\\#OOO#\\#####"
    lines = maze_string.split("\\")
    grid = [line[1:-1] for line in lines]
    return grid[1:-1]


def get_random_start_goal(env_id, batch_size):
    maze_grid = get_maze_grid(env_id)
    s2i = {"O": 0, "#": 1, "G": 2}
    maze_grid = [[s2i[s] for s in r] for r in maze_grid]
    maze_grid = np.array(maze_grid)
    x, y = np.nonzero(maze_grid == 0)
    indices = np.random.randint(len(x), size=batch_size)
    start = np.stack([x[indices], y[indices]], -1) + 1
    x, y = np.nonzero(maze_grid == 2)
    goal = np.concatenate([x, y], -1)
    goal = np.tile(goal[None, :], (batch_size, 1)) + 1
    return start, goal


def plot_maze_layout(ax, maze_grid):
    ax.clear()

    if maze_grid is not None:
        for i, row in enumerate(maze_grid):
            for j, cell in enumerate(row):
                if cell == "#":
                    square = plt.Rectangle(
                        (i + 0.5, j + 0.5), 1, 1, edgecolor="black", facecolor="black"
                    )
                    ax.add_patch(square)

    ax.set_aspect("equal")
    ax.grid(True, color="white", linewidth=4)
    ax.set_axisbelow(True)
    ax.spines["top"].set_linewidth(4)
    ax.spines["right"].set_linewidth(4)
    ax.spines["bottom"].set_linewidth(4)
    ax.spines["left"].set_linewidth(4)
    ax.set_facecolor("lightgray")
    ax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        left=False,
        right=False,
        labelbottom=False,
        labelleft=False,
    )
    ax.set_xticks(np.arange(0.5, len(maze_grid) + 0.5))
    ax.set_yticks(np.arange(0.5, len(maze_grid[0]) + 0.5))
    ax.set_xlim(0.5, len(maze_grid) + 0.5)
    ax.set_ylim(0.5, len(maze_grid[0]) + 0.5)
    ax.grid(True, color="white", which="minor", linewidth=4)


def plot_start_goal(ax, start_goal: None):
    def draw_star(center, radius, num_points=5, color="black"):
        angles = np.linspace(0.0, 2 * np.pi, num_points, endpoint=False) + 5 * np.pi / (
            2 * num_points
        )
        inner_radius = radius / 2.0

        points = []
        for angle in angles:
            points.extend(
                [
                    center[0] + radius * np.cos(angle),
                    center[1] + radius * np.sin(angle),
                    center[0] + inner_radius * np.cos(angle + np.pi / num_points),
                    center[1] + inner_radius * np.sin(angle + np.pi / num_points),
                ]
            )

        star = plt.Polygon(np.array(points).reshape(-1, 2), color=color)
        ax.add_patch(star)

    start_x, start_y = start_goal[0]
    start_outer_circle = plt.Circle(
        (start_x, start_y), 0.16, facecolor="white", edgecolor="black"
    )
    ax.add_patch(start_outer_circle)
    start_inner_circle = plt.Circle((start_x, start_y), 0.08, color="black")
    ax.add_patch(start_inner_circle)

    goal_x, goal_y = start_goal[1]
    goal_outer_circle = plt.Circle(
        (goal_x, goal_y), 0.16, facecolor="white", edgecolor="black"
    )
    ax.add_patch(goal_outer_circle)
    draw_star((goal_x, goal_y), radius=0.08)


def make_trajectory_images(
    env_id, trajectory, batch_size, start, goal, plot_end_points=True
):
    images = []
    for batch_idx in range(batch_size):
        fig, ax = plt.subplots()
        if is_grid_env(env_id):
            maze_grid = get_maze_grid(env_id)
        else:
            maze_grid = None
        plot_maze_layout(ax, maze_grid)
        ax.scatter(
            trajectory[batch_idx, :, 0],
            trajectory[batch_idx, :, 1],
            c=np.arange(trajectory.shape[1]),
            cmap="Reds",
        )
        if plot_end_points:
            start_goal = (start[batch_idx], goal[batch_idx])
            plot_start_goal(ax, start_goal)
        # plt.title(f"sample_{batch_idx}")
        fig.tight_layout()
        fig.canvas.draw()
        img_shape = fig.canvas.get_width_height()[::-1] + (4,)
        img = (
            np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            .copy()
            .reshape(img_shape)
        )
        images.append(img)

        plt.close()
    return images


def make_convergence_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(
        plan_history, trajectory, goal, open_loop_horizon
    )

    # animate the convergence of the first plan
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[0][frame]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    frames = tqdm(range(len(plan_history[0])), desc="Making convergence animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_convergence.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)
    return filename


def prune_history(plan_history, trajectory, goal, open_loop_horizon):
    dist = np.linalg.norm(
        trajectory[:, :2] - np.array(goal)[None],
        axis=-1,
    )
    reached = dist < 0.2
    if reached.any():
        cap_idx = np.argmax(reached)
        trajectory = trajectory[: cap_idx + open_loop_horizon + 1]
        plan_history = plan_history[: cap_idx // open_loop_horizon + 2]

    pruned_plan_history = []
    for plans in plan_history:
        pruned_plan_history.append([])
        for m in range(len(plans)):
            plan = plans[m]
            pruned_plan_history[-1].append(plan)
        plan = pruned_plan_history[-1][-1]
        dist = np.linalg.norm(plan.numpy()[:, :2] - np.array(goal)[None], axis=-1)
        reached = dist < 0.2
        if reached.any():
            cap_idx = np.argmax(reached) + 1
            pruned_plan_history[-1] = [p[:cap_idx] for p in pruned_plan_history[-1]]
    return trajectory, pruned_plan_history


def make_mpc_animation(
    env_id,
    plan_history,
    trajectory,
    start,
    goal,
    open_loop_horizon,
    namespace,
    interval=100,
    plot_end_points=True,
    batch_idx=0,
):
    # - plan_history: contains for each time step all the MPC predicted plans for each pyramid noise level.
    #                 Structured as a list of length (episode_len // open_loop_horizon), where each
    #                 element corresponds to a control_time_step and stores a list of length pyramid_height,
    #                 where each element is a plan at a different pyramid noise level and stored as a tensor of
    #                 shape (episode_len // open_loop_horizon - control_time_step,
    #                        batch_size, x_stacked_shape)

    # select index and prune history
    start, goal = start[batch_idx], goal[batch_idx]
    trajectory = trajectory[:, batch_idx]
    plan_history = [[pm[:, batch_idx] for pm in pt] for pt in plan_history]
    trajectory, plan_history = prune_history(
        plan_history, trajectory, goal, open_loop_horizon
    )

    # animate the convergence of the plans
    fig, ax = plt.subplots()
    if "large" in env_id:
        fig.set_size_inches(3.5, 5)
    else:
        fig.set_size_inches(3, 3)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1)
    trajectory_colors = np.linspace(0, 1, len(trajectory))

    if is_grid_env(env_id):
        maze_grid = get_maze_grid(env_id)
    else:
        maze_grid = None

    def update(frame):
        control_time_step = 0
        while frame >= 0:
            frame -= len(plan_history[control_time_step])
            control_time_step += 1
        control_time_step -= 1
        m = frame + len(plan_history[control_time_step])
        num_steps_taken = 1 + open_loop_horizon * control_time_step
        plot_maze_layout(ax, maze_grid)

        plan_history_m = plan_history[control_time_step][m]
        plan_history_m = plan_history_m.numpy()
        ax.scatter(
            trajectory[:num_steps_taken, 0],
            trajectory[:num_steps_taken, 1],
            c=trajectory_colors[:num_steps_taken],
            cmap="Blues",
        )
        ax.scatter(
            plan_history_m[:, 0],
            plan_history_m[:, 1],
            c=np.arange(len(plan_history_m))[::-1],
            cmap="Reds",
        )

        if plot_end_points:
            plot_start_goal(ax, (start, goal))

    num_frames = sum([len(p) for p in plan_history])
    frames = tqdm(range(num_frames), desc="Making MPC animation")
    ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval)
    prefix = wandb.run.id if wandb.run is not None else env_id
    filename = f"/tmp/{prefix}_{namespace}_mpc.mp4"
    ani.save(filename, writer="ffmpeg", fps=24)

    return filename


# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import re
from collections import OrderedDict
from datetime import datetime

import numpy as np
import pytz
import torch.distributed as dist
# from mmcv.utils.logging import logger_initialized # FIXME: We actually don't need such logger
from termcolor import colored

from .distributed_utils import is_local_master

logger_initialized = {}


def get_root_logger(
    log_file=None, log_level=logging.INFO, name=colored("[Sana]", attrs=["bold"]), timezone="Asia/Shanghai"
):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.
        name (str): logger name
    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    if log_file is None:
        log_file = "/dev/null"
    logger = get_logger(name=name, log_file=log_file, log_level=log_level, timezone=timezone)
    return logger


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = pytz.timezone(tz) if tz else None

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s


def get_logger(name, log_file=None, log_level=logging.INFO, timezone="UTC"):
    """Initialize and get a logger by name.

    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.

    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        timezone (str): Timezone for the log timestamps.

    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # disable root logger to avoid duplicate logging

    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, "a")
        handlers.append(file_handler)

    formatter = TimezoneFormatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", tz=timezone
    )

    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    # only rank0 for each node will print logs
    log_level = log_level if is_local_master() else logging.ERROR
    logger.setLevel(log_level)

    logger_initialized[name] = True

    return logger


def rename_file_with_creation_time(file_path):

    creation_time = os.path.getctime(file_path)
    creation_time_str = datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d_%H-%M-%S")


    dir_name, file_name = os.path.split(file_path)
    name, ext = os.path.splitext(file_name)
    new_file_name = f"{name}_{creation_time_str}{ext}"
    new_file_path = os.path.join(dir_name, new_file_name)

    os.rename(file_path, new_file_path)
    # print(f"File renamed to: {new_file_path}")
    return new_file_path


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None, tz=None):
        super().__init__(fmt, datefmt)
        self.tz = pytz.timezone(tz) if tz else None

    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            s = dt.isoformat()
        return s


class LogBuffer:
    def __init__(self):
        self.val_history = OrderedDict()
        self.n_history = OrderedDict()
        self.output = OrderedDict()
        self.ready = False

    def clear(self) -> None:
        self.val_history.clear()
        self.n_history.clear()
        self.clear_output()

    def clear_output(self) -> None:
        self.output.clear()
        self.ready = False

    def update(self, vars: dict, count: int = 1) -> None:
        assert isinstance(vars, dict)
        for key, var in vars.items():
            if key not in self.val_history:
                self.val_history[key] = []
                self.n_history[key] = []
            self.val_history[key].append(var)
            self.n_history[key].append(count)

    def average(self, n: int = 0) -> None:
        """Average latest n values or all values."""
        assert n >= 0
        for key in self.val_history:
            values = np.array(self.val_history[key][-n:])
            nums = np.array(self.n_history[key][-n:])
            avg = np.sum(values * nums) / np.sum(nums)
            self.output[key] = avg
        self.ready = True


def tracker(args, result_dict, label="", pattern="epoch_step", metric="FID"):
    if args.report_to == "wandb":
        import wandb

        wandb_name = f"[{args.log_metric}]_{args.name}"
        wandb.init(project=args.tracker_project_name, name=wandb_name, resume="allow", id=wandb_name, tags="metrics")
        run = wandb.run
        if pattern == "step":
            pattern = "sample_steps"
        elif pattern == "epoch_step":
            pattern = "step"
        custom_name = f"custom_{pattern}"
        run.define_metric(custom_name)
        # define which metrics will be plotted against it
        run.define_metric(f"{metric}_{label}", step_metric=custom_name)

        steps = []
        results = []

        def extract_value(regex, exp_name):
            match = re.search(regex, exp_name)
            if match:
                return match.group(1)
            else:
                return "unknown"

        for exp_name, result_value in result_dict.items():
            if pattern == "step":
                regex = r".*step(\d+)_scale.*"
                custom_x = extract_value(regex, exp_name)
            elif pattern == "sample_steps":
                regex = r".*step(\d+)_size.*"
                custom_x = extract_value(regex, exp_name)
            else:
                regex = rf"{pattern}(\d+(\.\d+)?)"
                custom_x = extract_value(regex, exp_name)
                custom_x = 1 if custom_x == "unknown" else custom_x

            assert custom_x != "unknown"
            steps.append(float(custom_x))
            results.append(result_value)

        sorted_data = sorted(zip(steps, results))
        steps, results = zip(*sorted_data)

        for step, result in sorted(zip(steps, results)):
            run.log({f"{metric}_{label}": result, custom_name: step})
    else:
        print(f"{args.report_to} is not supported")
