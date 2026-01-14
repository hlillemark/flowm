import torch
import math

"""
Helper functions for Flowm-based models
"""

def fov_mask_indices(
    shape_HW = (32, 32),
    agent_xy = (32.0//2, 32//2),
    heading_deg = 0.0,
    fov_deg=60.0,
    max_range=None,
    samples_per_edge=3,
    min_coverage=0.0,
    device=None,
):
    """
    Compute a boolean mask and (row, col) indices of pixels that fall inside an agent's FOV.

    Args:
        shape_HW: tuple (H, W) of the map.
        agent_xy: (x, y) agent position in *image pixel coordinates*.
                  Pixels are treated as 1x1 squares centered on integer grid points:
                  pixel (j,i) spans [i-0.5, i+0.5] x [j-0.5, j+0.5].
                  x increases to the right, y increases downward.
        heading_deg: agent heading in degrees. Image convention:
                     0° = right, 90° = down, 180° = left, 270° = up.
        fov_deg: total field-of-view angle in degrees (wedge aperture).
        max_range: optional float. If given, restricts FOV to this radial distance.
        samples_per_edge: sub-pixel grid per dimension (e.g., 3 → 3x3=9 samples per pixel).
                          Use 1 for center-only; ≥3 is good for “partial coverage”.
        min_coverage: include a pixel iff at least this fraction of sub-pixel samples lie inside.
                      For “even just partially”, leave at 0.0 (i.e., any sample inside).
        device: torch device. If None, uses CPU; set to your tensor's device to avoid transfers.

    Returns:
        mask: (H, W) bool tensor — True where the pixel is inside the FOV.
        indices: (N, 2) long tensor — rows are (row=y, col=x) indices of True pixels.
    """
    H, W = shape_HW
    x0, y0 = agent_xy
    if device is None:
        device = torch.device("cpu")

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )

    # Sub-pixel sampling offsets inside each pixel square [-0.5, 0.5] x [-0.5, 0.5]
    if samples_per_edge <= 1:
        sub_off = torch.tensor([0.0], device=device)
        sx = xx[..., None] + sub_off
        sy = yy[..., None] + sub_off
    else:
        step = 1.0 / samples_per_edge
        start = -0.5 + step / 2.0
        off1d = torch.arange(samples_per_edge, device=device) * step + start
        ox, oy = torch.meshgrid(off1d, off1d, indexing="xy")
        ox = ox.reshape(-1)  # (K,)
        oy = oy.reshape(-1)  # (K,)
        sx = xx[..., None] + ox  # (H, W, K)
        sy = yy[..., None] + oy  # (H, W, K)

    # Vector from agent to each sub-sample
    dx = sx - float(x0)
    dy = sy - float(y0)

    # Heading unit vector (image coords: +y downward)
    th = math.radians(float(heading_deg))
    ux = math.cos(th)
    uy = math.sin(th)

    # Angle test via cosine; avoid atan2 for speed and wrap issues
    dist = torch.sqrt(dx * dx + dy * dy).clamp_min(1e-9)
    cos_to_heading = (dx * ux + dy * uy) / dist
    cos_half = math.cos(math.radians(fov_deg) / 2.0)

    inside = cos_to_heading >= cos_half  # within FOV aperture
    if max_range is not None:
        inside = inside & (dist <= float(max_range))

    # Fraction of sub-samples inside each pixel
    frac_inside = inside.float().mean(dim=-1)  # (H, W)

    if min_coverage <= 0.0:
        mask = inside.any(dim=-1)  # any sub-sample inside counts
    else:
        mask = frac_inside >= float(min_coverage)

    indices = torch.nonzero(mask, as_tuple=False)  # (N, 2): (row=y, col=x)
    return mask, indices

