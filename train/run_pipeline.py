"""Explicit pipeline for the retained CoR-GS + Binocular method.

Stages are intentionally separated: image analysis/SfM, PDCNet+ dense
initialization, CoR-GS optimization, Binocular consistency, and rendering.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "train"
CORGS_ROOT = TRAIN_ROOT / "corgs"
SFM_ROOT = TRAIN_ROOT / "sfm"
# 默认实验配置：命令行参数可以覆盖这些值。
DEFAULT_RUN_NAME = "new_reconstruction"
DEFAULT_MODEL_NAME = "new_corgs_4000"
DEFAULT_WORKSPACE = PROJECT_ROOT / "output" / "reconstruction" / DEFAULT_RUN_NAME
DEFAULT_MODEL_DIR = PROJECT_ROOT / "output" / "corgs" / DEFAULT_MODEL_NAME
DEFAULT_ITERATIONS = 4000
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


def run_sfm(images: Path, workspace: Path, config: Path, colmap: Path,
            python: Path, analyze_only: bool = False) -> None:
    require_file(SFM_ROOT / "prepare.py", "SfM preparation program")
    if not analyze_only:
        require_file(colmap, "COLMAP executable")
    args = [str(SFM_ROOT / "prepare.py"), "--images", str(images),
            "--workspace", str(workspace), "--config", str(config)]
    if not analyze_only:
        args += ["--colmap", str(colmap), "--sparse-model", "auto",
                 "--pdcnet-root", str(SFM_ROOT / "dense_matcher"),
                 "--pdcnet-output", str(workspace / "pdcnet"),
                 "--pdcnet-pretrained-model", "megadepth",
                 "--pdcnet-views", "3", "--pdcnet-multi-stage-type", "h"]
    else:
        args.append("--analyze-only")
    run_python(args, python, [SFM_ROOT], PROJECT_ROOT)


def run_pdcnet(dataset: Path, python: Path, output: Path | None = None) -> None:
    """Replace only the 3DGS initializer with official Binocular3DGS PDCNet+."""
    script = SFM_ROOT / "pdcnet_init.py"
    require_file(script, "PDCNet+ initialization program")
    require_dir(dataset / "images", "prepared dataset images")
    require_dir(dataset / "sparse" / "0", "dataset sparse/0 directory")
    output = output or dataset.parent.parent / "pdcnet"
    run_python(
        [str(script), "--dataset", str(dataset),
         "--matcher-root", str(SFM_ROOT / "dense_matcher"),
         "--output", str(output), "--pretrained-model", "megadepth",
         "--n-views", "3", "--multi-stage-type", "h"],
        python, [SFM_ROOT], PROJECT_ROOT,
    )


def ensure_dense_initialization(dataset: Path, python: Path) -> None:
    point_cloud = dataset / "sparse" / "0" / "points3D.ply"
    if not point_cloud.is_file():
        print("PDCNet+ dense initializer: points3D.ply is missing; generating it now.")
        run_pdcnet(dataset, python)

def corgs_runtime_paths() -> list[Path]:
    submodules = CORGS_ROOT / "submodules"
    return [
        CORGS_ROOT,
        TRAIN_ROOT,
        submodules,
        submodules / "diff-gaussian-rasterization-confidence",
        submodules / "simple-knn",
    ]


def verify_corgs_runtime() -> None:
    """Verify the compiled CUDA extensions used by the retained backend."""
    submodules = CORGS_ROOT / "submodules"
    require_dir(submodules, "CoR-GS CUDA runtime")
    required = [
        submodules / "simple-knn" / "simple_knn" / "_C.cp311-win_amd64.pyd",
        submodules / "diff-gaussian-rasterization-confidence" /
        "diff_gaussian_rasterization" / "_C.cp311-win_amd64.pyd",
    ]
    for extension in required:
        require_file(extension, "compiled CoR-GS CUDA extension")
    print("CoR-GS CUDA runtime verified; standalone FSGS training is not retained.")


def run_corgs(args: list[str], python: Path) -> None:
    require_dir(CORGS_ROOT, "CoR-GS backend directory")
    run_python(["train.py", *args], python, corgs_runtime_paths(), CORGS_ROOT)


def build_best_training_args(dataset: Path, model: Path, iterations: int,
                              start_checkpoint: str = "") -> list[str]:
    """Build the retained method as explicit CoR-GS + Binocular stages."""
    args = ["--source_path", str(dataset), "--model_path", str(model),
            "--iterations", str(iterations), "--test_iterations", str(iterations),
            "--save_iterations", str(iterations), "--checkpoint_iterations", str(iterations),
            # CoR-GS stage: two fields, co-regularization and co-pruning.
            "--gaussiansN", "2", "--coreg", "--coprune", "--coprune_interval", "500",
            "--densify_until_iter", "2500", "--sample_pseudo_interval", "10",
            "--start_sample_pseudo", "1500", "--end_sample_pseudo", str(iterations),
            # Binocular stage: virtual right view and stereo consistency loss.
            "--binocular", "--binocular_start", "2500", "--binocular_interval", "20",
            "--binocular_weight", "0.2", "--binocular_baseline_min", "0.01",
            "--binocular_baseline_max", "0.08", "--pause_checkpoint_interval", "250"]
    if start_checkpoint:
        args += ["--start_checkpoint", str(resolve_path(start_checkpoint))]
    return args


def run_render(dataset: Path, model: Path, iteration: int, python: Path,
               views: list[str], render_depth: bool) -> None:
    require_dir(dataset, "3DGS dataset directory")
    require_dir(model, "model output directory")
    args = ["render.py", "--source_path", str(dataset), "--model_path", str(model),
            "--iteration", str(iteration), "--skip_test"]
    if render_depth:
        args.append("--render_depth")
    if views:
        args += ["--view_names", *views]
    run_python(args, python, corgs_runtime_paths(), CORGS_ROOT)


def split_views(values: list[str]) -> list[str]:
    return [part.strip() for value in values for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("analyze", "sfm", "pdcnet", "corgs", "binocular",
                                             "train", "resume", "render", "interactive", "all"),
                        default="analyze")
    parser.add_argument("--images", default="data")
    parser.add_argument("--workspace", default=str(DEFAULT_WORKSPACE))
    parser.add_argument("--dataset", default="")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
    parser.add_argument("--colmap", default="train/sfm/colmap-x64-windows-cuda/bin/colmap.exe")
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--start-checkpoint", default="")
    parser.add_argument("--views", action="append", default=[])
    parser.add_argument("--viewer-output", default="")
    parser.add_argument("--display-scale", type=float, default=0.75)
    parser.add_argument("--render-depth", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    python = resolve_path(args.python)
    images = resolve_path(args.images)
    workspace = resolve_path(args.workspace)
    config = resolve_path(args.config)
    colmap = resolve_path(args.colmap)
    dataset = resolve_path(args.dataset) if args.dataset else workspace / "dense"
    model = resolve_path(args.model_path)
    views = split_views(args.views)
    needs_images = args.stage in {"analyze", "sfm", "all"}
    needs_config = needs_images
    if needs_images:
        require_dir(images, "input images directory")
    if needs_config:
        require_file(config, "configuration file")

    if args.stage == "analyze":
        run_sfm(images, workspace, config, colmap, python, analyze_only=True)
    elif args.stage == "sfm":
        run_sfm(images, workspace, config, colmap, python)
    elif args.stage == "pdcnet":
        require_dir(dataset, "3DGS dataset directory")
        run_pdcnet(dataset, python)
    elif args.stage in {"corgs", "binocular", "train", "resume"}:
        require_dir(dataset, "3DGS dataset directory")
        require_dir(dataset / "sparse" / "0", "dataset sparse/0 directory")
        ensure_dense_initialization(dataset, python)
        verify_corgs_runtime()
        print("CoR-GS stage: enabled")
        print("Binocular stage: enabled")
        run_corgs(build_best_training_args(dataset, model, args.iterations, args.start_checkpoint), python)
    elif args.stage == "render":
        run_render(dataset, model, args.iterations, python, views, args.render_depth)
    elif args.stage == "interactive":
        require_dir(dataset, "3DGS dataset directory")
        require_dir(model, "model output directory")
        viewer = PROJECT_ROOT / "render" / "interactive_viewer.py"
        require_file(viewer, "interactive viewer")
        run_python([str(viewer), "--source_path", str(dataset), "--model_path", str(model),
                    "--iteration", str(args.iterations), "--display_scale", str(args.display_scale)],
                   python, corgs_runtime_paths(), CORGS_ROOT)
    elif args.stage == "all":
        run_sfm(images, workspace, config, colmap, python, analyze_only=True)
        run_sfm(images, workspace, config, colmap, python)
        verify_corgs_runtime()
        require_dir(dataset, "3DGS dataset directory")
        require_dir(dataset / "sparse" / "0", "dataset sparse/0 directory")
        ensure_dense_initialization(dataset, python)
        run_corgs(build_best_training_args(dataset, model, args.iterations), python)
        run_render(dataset, model, args.iterations, python, views, args.render_depth)
    print(f"Finished: Stage={args.stage} Method=CoR-GS+Binocular")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
