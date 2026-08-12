"""Render selected views from a trained SparseView3DGS model."""

from __future__ import annotations

import argparse
import sys
import os
import subprocess
from pathlib import Path


DEFAULT_PYTHON = Path(r"C:\Users\leo\.conda\envs\ai\python.exe")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROGRAM_ROOT = PROJECT_ROOT / "program"


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise NotADirectoryError(f"{description} does not exist: {path}")


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
    require_file(python, "Python interpreter")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in pythonpath)
    print("+", python)
    print(" ".join(args))
    subprocess.run([str(python), *args], cwd=backend_root, env=env, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("fsgs", "corgs", "corgs_fsgs"), default="corgs_fsgs")
    parser.add_argument("--dataset", default=str(PROJECT_ROOT / "output" / "reconstruction" / "pipeline" / "dense"))
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "model"))
    parser.add_argument("--iteration", type=int, default=-1,
                        help="Model iteration; -1 automatically uses the latest saved point cloud.")
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.exists() else Path(sys.executable)))
    parser.add_argument("--views", action="append", default=[])
    parser.add_argument("--render-depth", action="store_true")
    parser.add_argument("--gaussian-index", type=int, choices=(0, 1), default=0,
                        help="CoR-GS Gaussian field to render: 0=gs0, 1=gs1")
    parser.add_argument("--interactive", action="store_true",
                        help="Open the free-view interactive renderer.")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Disable the default free-view mode when no named views are supplied.")
    parser.add_argument("--display-scale", type=float, default=0.75)
    args = parser.parse_args()

    python = resolve_path(args.python)
    dataset = resolve_path(args.dataset)
    model = resolve_path(args.model_path)
    require_dir(dataset, "3DGS dataset directory")
    require_dir(model, "model output directory")
    # A bare ``python render.py`` is the common interactive workflow. Named
    # views (or an explicit depth render) retain the deterministic renderer.
    use_interactive = args.interactive or (
        not args.no_interactive and not args.views and not args.render_depth
    )
    if use_interactive:
        viewer = PROGRAM_ROOT / "sparseview3dgs" / "interactive_viewer.py"
        require_file(viewer, "interactive viewer")
        viewer_args = [
            str(viewer), "--source_path", str(dataset), "--model_path", str(model),
            "--iteration", str(args.iteration), "--display_scale", str(args.display_scale),
        ]
        run_backend(args.method, viewer_args, python)
        print(f"Interactive rendering complete. Model directory: {model}")
        return 0

    render_args = ["render.py", "--source_path", str(dataset), "--model_path", str(model),
                   "--iteration", str(args.iteration), "--skip_test"]
    if args.render_depth:
        render_args.append("--render_depth")
    render_args += ["--gaussian_index", str(args.gaussian_index)]
    views = []
    for value in args.views:
        views.extend(part.strip() for part in value.split(",") if part.strip())
    if views:
        render_args += ["--view_names", *views]
    run_backend(args.method, render_args, python)
    print(f"Rendering complete. Model directory: {model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
