"""Render the retained CoR-GS + Binocular model."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "train"
CORGS_ROOT = TRAIN_ROOT / "corgs"
# 与 train/run_pipeline.py 保持一致的默认实验配置。
DEFAULT_RUN_NAME = "new_reconstruction"
DEFAULT_MODEL_NAME = "new_corgs_4000"
DEFAULT_DATASET = PROJECT_ROOT / "output" / "reconstruction" / DEFAULT_RUN_NAME / "dense"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "output" / "corgs" / DEFAULT_MODEL_NAME
DEFAULT_ITERATION = 4000
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


def backend_pythonpath() -> list[Path]:
    submodules = CORGS_ROOT / "submodules"
    return [
        CORGS_ROOT,
        TRAIN_ROOT,
        submodules,
        submodules / "diff-gaussian-rasterization-confidence",
        submodules / "simple-knn",
    ]


def run_backend(args: list[str], python: Path) -> None:
    require_dir(CORGS_ROOT, "CoR-GS backend directory")
    require_file(python, "Python interpreter")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in backend_pythonpath())
    print("+", python)
    print(" ".join(args))
    subprocess.run([str(python), *args], cwd=CORGS_ROOT, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--iteration", type=int, default=DEFAULT_ITERATION)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
    parser.add_argument("--views", action="append", default=[])
    parser.add_argument("--render-depth", action="store_true")
    parser.add_argument("--gaussian-index", type=int, choices=(0, 1), default=0)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-interactive", action="store_true")
    parser.add_argument("--display-scale", type=float, default=0.75)
    args = parser.parse_args()
    python = resolve_path(args.python)
    dataset = resolve_path(args.dataset)
    model = resolve_path(args.model_path)
    require_dir(dataset, "3DGS dataset directory")
    require_dir(model, "model output directory")
    views = [part.strip() for value in args.views for part in value.split(",") if part.strip()]
    use_interactive = args.interactive or (not args.no_interactive and not views and not args.render_depth)
    if use_interactive:
        viewer = PROJECT_ROOT / "render" / "interactive_viewer.py"
        require_file(viewer, "interactive viewer")
        run_backend([str(viewer), "--source_path", str(dataset), "--model_path", str(model),
                     "--iteration", str(args.iteration), "--display_scale", str(args.display_scale)], python)
    else:
        render_args = ["render.py", "--source_path", str(dataset), "--model_path", str(model),
                       "--iteration", str(args.iteration), "--skip_test", "--gaussian_index", str(args.gaussian_index)]
        if args.render_depth:
            render_args.append("--render_depth")
        if views:
            render_args += ["--view_names", *views]
        run_backend(render_args, python)
    print(f"Rendering complete. Method=CoR-GS+Binocular. Model directory: {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
