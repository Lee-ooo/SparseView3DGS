"""Self-supervised binocular stereo consistency for sparse-view 3DGS.

The implementation follows Binocular3DGS: translate a training camera along
its camera-right axis, render a virtual right view, and warp that render back
to the left view using disparity d = f * baseline / depth.
"""

import math
import numpy as np
import torch
import torch.nn.functional as F

from scene.cameras import PseudoCamera
from utils.graphics_utils import getWorld2View2


def make_shifted_camera(view, baseline):
    """Return a virtual right camera translated along the source camera's x axis."""
    w2c = getWorld2View2(view.R, view.T)
    c2w = np.linalg.inv(w2c)

    # Camera x is the image-right direction in world coordinates.
    c2w[:3, 3] = c2w[:3, 3] + c2w[:3, 0] * float(baseline)
    shifted_w2c = np.linalg.inv(c2w)

    return PseudoCamera(
        R=shifted_w2c[:3, :3].T,
        T=shifted_w2c[:3, 3],
        FoVx=view.FoVx,
        FoVy=view.FoVy,
        width=view.image_width,
        height=view.image_height,
    )


def binocular_consistency_loss(left_gt, left_pkg, right_pkg, view, baseline,
                               photometric_loss, min_alpha=0.05):
    """Warp the virtual right render into the left view and compare with input RGB.

    The warp is differentiable with respect to both the rendered right image
    and the rendered left depth, so the loss directly guides Gaussian geometry.
    """
    left_depth = left_pkg["depth"][0]
    right_image = right_pkg["render"].unsqueeze(0)
    right_alpha = right_pkg["alpha"].unsqueeze(0)

    height, width = left_depth.shape[-2:]
    device = left_depth.device

    depth = torch.nan_to_num(left_depth, nan=0.0, posinf=0.0, neginf=0.0)
    valid_depth = depth > 1e-4

    # The rendered image uses the same horizontal FOV as the source camera.
    focal = width / (2.0 * math.tan(float(view.FoVx) * 0.5))
    disparity = focal * float(baseline) / depth.clamp_min(1e-4)

    x = torch.arange(width, device=device, dtype=depth.dtype).view(1, -1).expand(height, -1)
    y = torch.arange(height, device=device, dtype=depth.dtype).view(-1, 1).expand(-1, width)
    right_x = x - disparity

    grid_x = 2.0 * right_x / max(width - 1, 1) - 1.0
    grid_y = 2.0 * y / max(height - 1, 1) - 1.0
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    warped_image = F.grid_sample(
        right_image, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0]
    warped_alpha = F.grid_sample(
        right_alpha, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )[0, 0]

    valid = (
        valid_depth
        & (left_pkg["alpha"][0] > min_alpha)
        & (warped_alpha > min_alpha)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
    ).float().unsqueeze(0)

    if valid.sum().item() < 32:
        return left_gt.new_zeros(()), valid

    loss = photometric_loss(warped_image, left_gt, valid=valid)
    return loss, valid

