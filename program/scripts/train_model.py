"""Train a SparseView3DGS model.

When --dataset is omitted, the raw images directory is prepared with COLMAP
first.  The default model output is the project-root model/ directory.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_ROOT = PROJECT_ROOT / "program"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_PYTHON = Path(r"C:\Users\leo\.conda\envs\ai\python.exe")


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{description} does not exist: {path}")


def run_python(args: list[str], python: Path, pythonpath: list[Path] | None = None,
               cwd: Path | None = None) -> None:
    require_file(python, "Python interpreter")
    env = os.environ.copy()
    if pythonpath is not None:
        env["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    print("+", python)
    print(" ".join(args))
    subprocess.run([str(python), *args], cwd=cwd, env=env, check=True)


def run_prepare(images: Path, workspace: Path, config: Path, colmap: Path,
                python: Path, recursive_images: bool = False) -> None:
    require_file(colmap, "COLMAP executable")
    prepare_args = [
        "-m", "sparseview3dgs.prepare", "--images", str(images),
        "--workspace", str(workspace), "--config", str(config),
        "--colmap", str(colmap), "--sparse-model", "auto",
    ]
    if recursive_images:
        prepare_args.append("--recursive-images")
    run_python(
        prepare_args,
        python,
        [PROGRAM_ROOT],
        PROJECT_ROOT,
    )


def run_backend(method: str, args: list[str], python: Path) -> None:
    backend_name = "fsgs" if method == "fsgs" else "corgs"
    backend_root = PROGRAM_ROOT / backend_name
    require_dir(backend_root, f"{backend_name} backend directory")
    fsgs_submodules = PROGRAM_ROOT / "fsgs" / "submodules"
    pythonpath = [
        backend_root,
        fsgs_submodules,
        fsgs_submodules / "diff-gaussian-rasterization-confidence",
        fsgs_submodules / "simple-knn",
        backend_root / "submodules" / "diff-gaussian-rasterization-confidence",
        backend_root / "submodules" / "simple-knn",
    ]
    run_python(args, python, pythonpath, backend_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fsgs", "corgs", "corgs_fsgs"),
                        default="corgs_fsgs")
    parser.add_argument("--images", default="data")
    parser.add_argument("--dataset", default="",
                        help="Prepared dataset; omit to prepare from --images first.")
    parser.add_argument("--workspace", default="output/reconstruction/pipeline")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--colmap", default="program/colmap-x64-windows-cuda/bin/colmap.exe")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
    parser.add_argument("--iterations", type=int, default=4000)
    parser.add_argument("--start-checkpoint", default="")
    parser.add_argument("--skip-eval", dest="skip_eval", action="store_true", default=True,
                        help="Skip optional test-view LPIPS evaluation (default; avoids network-dependent VGG weights).")
    parser.add_argument("--eval", dest="skip_eval", action="store_false",
                        help="Enable test-view LPIPS/SSIM evaluation; requires the VGG16 weights to be available.")
    parser.add_argument("--recursive-images", action="store_true",
                        help="Recursively include images below --images; off by default.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    python = resolve_path(args.python)
    images = resolve_path(args.images)
    workspace = resolve_path(args.workspace)
    config = resolve_path(args.config)
    colmap = resolve_path(args.colmap)
    model = resolve_path(args.model_path)

    require_dir(images, "input images directory")
    require_file(config, "configuration file")

    if args.dataset:
        dataset = resolve_path(args.dataset)
    else:
        print(f"No --dataset supplied; preparing training data from: {images}")
        run_prepare(images, workspace, config, colmap, python, args.recursive_images)
        dataset = workspace / "dense"

    require_dir(dataset, "3DGS dataset directory")
    require_dir(dataset / "sparse" / "0", "dataset sparse/0 directory")

    common = [
        "--source_path", str(dataset), "--model_path", str(model),
        "--iterations", str(args.iterations),
        "--test_iterations", "-1" if args.skip_eval else str(args.iterations),
        "--save_iterations", str(args.iterations),
        "--checkpoint_iterations", str(args.iterations),
    ]
    if args.start_checkpoint:
        common += ["--start_checkpoint", str(resolve_path(args.start_checkpoint))]

    if args.method == "fsgs":
        common += ["--images", "images",
                   "--sample_pseudo_interval", "10", "--start_sample_pseudo", "1500",
                   "--end_sample_pseudo", str(args.iterations)]
    else:
        common += [
            "--gaussiansN", "2", "--coreg", "--coprune", "--coprune_interval", "500",
            "--densify_until_iter", "2500", "--sample_pseudo_interval", "10",
            "--start_sample_pseudo", "1500", "--end_sample_pseudo", str(args.iterations),
            "--binocular", "--binocular_start", "2500", "--binocular_interval", "20",
            "--binocular_weight", "0.2", "--binocular_baseline_min", "0.01",
            "--binocular_baseline_max", "0.08", "--pause_checkpoint_interval", "250",
        ]
        if args.method == "corgs_fsgs":
            common += [
                "--fsgs_unpool", "--fsgs_unpool_until", "2000", "--fsgs_unpool_n", "3",
                "--fsgs_depth", "--fsgs_depth_pseudo",
                "--fsgs_depth_weight", "0.05", "--fsgs_depth_pseudo_weight", "0.5",
                "--fsgs_depth_pseudo_interval", "10", "--fsgs_depth_start", "1500",
                "--fsgs_depth_end", str(args.iterations), "--fsgs_depth_final_weight", "0.001",
            ]

    run_backend(args.method, ["train.py", *common], python)
    print(f"Training complete. Method={args.method}")
    print(f"Model directory: {model}")
    print(f"Dataset directory: {dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
