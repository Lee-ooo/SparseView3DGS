"""Run the Binocular3DGS-style PDCNet+ dense initializer.

PDCNet+ supplies dense, confidence-filtered correspondences. Camera poses and
intrinsics are intentionally read from the COLMAP model, matching the
reference Binocular3DGS implementation. The resulting PLY replaces the sparse
points3D initialization consumed by CoR-GS.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{description} does not exist: {path}")


def run(dataset: Path, matcher_root: Path, output: Path, python: Path,
        pretrained_model: str, n_views: int, multi_stage_type: str) -> Path:
    require_dir(dataset / "images", "prepared dataset images")
    sparse = dataset / "sparse" / "0"
    require_dir(sparse, "prepared dataset sparse/0")
    require_file(sparse / "cameras.bin", "COLMAP camera poses")
    require_file(sparse / "images.bin", "COLMAP registered image poses")
    require_dir(matcher_root, "PDCNet+ DenseMatching checkout")
    script = matcher_root / "triangulate.py"
    require_file(script, "PDCNet+ triangulation script")
    weights_dir = matcher_root / "pre_trained_models"
    require_dir(weights_dir, "PDCNet+ pre-trained model directory")
    weight = weights_dir / f"PDCNet_plus_{pretrained_model}.pth"
    weight_tar = Path(str(weight) + ".tar")
    if not weight.is_file() and not weight_tar.is_file():
        raise FileNotFoundError(
            "Missing PDCNet+ checkpoint. Put PDCNet_plus_"
            f"{pretrained_model}.pth or .pth.tar in {weights_dir}. "
            "Download it from the Binocular3DGS README link."
        )
    require_file(python, "Python interpreter")

    output.mkdir(parents=True, exist_ok=True)
    args = [str(script), "--network_type", "PDCNet_plus",
            "--pre_trained_model", pretrained_model,
            "--multi_stage_type", multi_stage_type,
            "--data_path", str(dataset), "--n_views", str(n_views),
            "--output_path", str(output), "--dataset_name", "LLFF"]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(matcher_root), old_pythonpath) if part
    )
    print("PDCNet+ dense initialization:", " ".join(args))
    subprocess.run([str(python), *args], cwd=matcher_root, env=env, check=True)

    candidates = sorted(output.glob("*_keypoints_to_3d.ply"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError(f"PDCNet+ did not produce a *_keypoints_to_3d.ply in {output}")
    generated = candidates[-1]
    target = sparse / "points3D.ply"
    shutil.copy2(generated, target)
    print(f"PDCNet+ dense point cloud installed: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--matcher-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path,
                        default=Path(r"C:\Users\leo\.conda\envs\ai\python.exe"))
    parser.add_argument("--pretrained-model", default="megadepth")
    parser.add_argument("--n-views", type=int, default=3)
    parser.add_argument("--multi-stage-type", choices=("d", "h", "ms"), default="h")
    args = parser.parse_args()
    run(args.dataset.resolve(), args.matcher_root.resolve(), args.output.resolve(),
        args.python.resolve(), args.pretrained_model, args.n_views, args.multi_stage_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
