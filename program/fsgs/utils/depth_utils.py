import os
import sys
from pathlib import Path
import warnings

import torch

midas = None
_midas_unavailable = False
downsampling = 1


def _local_midas():
    """Load an official MiDaS checkpoint shipped under program/weights."""
    weights_root = Path(__file__).resolve().parents[2] / "weights"
    repo = weights_root / "MiDaS-master"
    small_weight = weights_root / "midas_v21_small_256.pt"
    hybrid_weight = weights_root / "dpt_hybrid_384.pt"
    if not repo.is_dir():
        return None
    if small_weight.is_file() and small_weight.stat().st_size > 80_000_000:
        sys.path.insert(0, str(repo))
        from midas.midas_net_custom import MidasNet_small
        return MidasNet_small(
            str(small_weight), features=64, backbone="efficientnet_lite3",
            exportable=True, non_negative=True, blocks={"expand": True}
        )
    if hybrid_weight.is_file() and hybrid_weight.stat().st_size > 400_000_000:
        return torch.hub.load(str(repo), "DPT_Hybrid", source="local", pretrained=False)
    return None


def _load_midas():
    global midas, _midas_unavailable
    if midas is not None or _midas_unavailable:
        return midas
    try:
        midas = _local_midas()
        if midas is None and os.environ.get("SPARSEVIEW_DEPTH_ALLOW_DOWNLOAD", "0") == "1":
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
        if midas is None:
            raise FileNotFoundError(
                "No valid local MiDaS weights found under program/weights"
            )
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        midas.to(device)
        midas.eval()
        for param in midas.parameters():
            param.requires_grad = False
    except Exception as exc:
        _midas_unavailable = True
        warnings.warn(
            "Pretrained monocular depth is unavailable; using a deterministic "
            f"image-intensity proxy ({exc}). Set SPARSEVIEW_DEPTH_ALLOW_DOWNLOAD=1 "
            "or place a MiDaS checkpoint under program/weights.",
            RuntimeWarning,
        )
        midas = None
    return midas

def estimate_depth(img, mode='test'):
    h, w = img.shape[1:3]
    model = _load_midas()
    if model is None:
        # Keep the pipeline runnable without downloading external weights.
        # This is only a fallback; a real FSGS run should use MiDaS depth.
        return (1.0 - img.mean(dim=0)).clamp_min(1e-3)

    norm_img = (img[None] - 0.5) / 0.5
    norm_img = torch.nn.functional.interpolate(
        norm_img,
        size=(384, 512),
        mode="bicubic",
        align_corners=False)

    if mode == 'test':
        with torch.no_grad():
            prediction = model(norm_img)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h//downsampling, w//downsampling),
                mode="bicubic",
                align_corners=False,
            ).squeeze()
    else:
        prediction = model(norm_img)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h//downsampling, w//downsampling),
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction

