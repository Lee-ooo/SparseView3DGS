from __future__ import annotations

import argparse
import json
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(root: Path, recursive: bool = False) -> list[Path]:
    """Collect one image set without mixing sibling datasets by default."""
    candidates = root.rglob("*") if recursive else root.iterdir()
    return sorted(p for p in candidates if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def stage_images(paths: list[Path], source_root: Path, staging: Path) -> None:
    """Copy selected images into a flat ASCII-only COLMAP input directory."""
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(tqdm(paths, desc="Staging images", unit="image"), start=1):
        # COLMAP's Windows binary model can preserve local-codepage bytes for
        # non-ASCII names, while the backends decode image names as UTF-8.
        # Use stable ASCII names in the staged input to keep images.bin portable.
        filename = f"image_{index:06d}{path.suffix.lower()}"
        shutil.copy2(path, staging / filename)


def inspect_images(paths: list[Path], cfg: dict) -> list[dict]:
    rows: list[dict] = []
    for path in tqdm(paths, desc="Analyzing images", unit="image"):
        row = {"path": str(path), "ok": False}
        try:
            with Image.open(path) as im:
                width, height = im.size
                # Pillow handles non-ASCII Windows paths reliably; avoid cv2.imread
                # here because it may fail on Chinese filenames.
                gray = np.asarray(im.convert("L"), dtype=np.uint8)
            blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            row.update({
                "width": width,
                "height": height,
                "blur_score": round(blur, 3),
                "low_resolution": width < cfg["min_image_width"] or height < cfg["min_image_height"],
                "likely_blurry": blur < cfg["blur_threshold"],
                "ok": True,
            })
        except Exception as exc:  # keep the report useful even if one file is corrupt
            row["error"] = repr(exc)
        rows.append(row)
    return rows


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("+", subprocess.list2cmdline(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _read_c_string(handle) -> bytes:
    value = bytearray()
    while True:
        char = handle.read(1)
        if not char or char == b"\x00":
            return bytes(value)
        value.extend(char)


def count_registered_images(model: Path) -> int:
    """Count registered images in a COLMAP model without requiring pycolmap."""
    images_bin = model / "images.bin"
    if images_bin.exists():
        count = 0
        with images_bin.open("rb") as handle:
            count_raw = handle.read(8)
            if len(count_raw) != 8:
                return 0
            image_count = struct.unpack("<Q", count_raw)[0]
            for _ in range(image_count):
                header = handle.read(4 + 4 * 8 + 3 * 8 + 4)
                if len(header) != 4 + 4 * 8 + 3 * 8 + 4:
                    break
                _read_c_string(handle)
                point_count_raw = handle.read(8)
                if len(point_count_raw) != 8:
                    break
                point_count = struct.unpack("<Q", point_count_raw)[0]
                handle.seek(point_count * (8 + 8 + 8), 1)
                count += 1
        return count

    images_txt = model / "images.txt"
    if images_txt.exists():
        count = 0
        for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            # An image header has IMAGE_ID, pose, CAMERA_ID and NAME.
            if len(fields) >= 10:
                try:
                    int(fields[0])
                except ValueError:
                    continue
                count += 1
        return count
    return 0


def select_sparse_model(sparse: Path, requested: str) -> tuple[Path, list[dict]]:
    candidates = []
    for model in sorted(p for p in sparse.iterdir() if p.is_dir()):
        if not ((model / "images.bin").exists() or (model / "images.txt").exists()):
            continue
        registered = count_registered_images(model)
        try:
            numeric_id = int(model.name)
        except ValueError:
            numeric_id = 10**9
        candidates.append({"path": model, "model_id": model.name,
                           "registered_images": registered, "numeric_id": numeric_id})

    if not candidates:
        raise RuntimeError("COLMAP 没有生成包含 images.bin/images.txt 的 sparse 模型。")

    if requested and requested.lower() != "auto":
        requested_path = Path(requested)
        model = requested_path if requested_path.is_absolute() else sparse / requested
        selected = next((item for item in candidates if item["path"].resolve() == model.resolve()), None)
        if selected is None:
            available = ", ".join(item["model_id"] for item in candidates)
            raise RuntimeError(f"指定的 sparse 模型不存在：{model}；可选模型：{available}")
    else:
        selected = max(candidates, key=lambda item: (item["registered_images"], -item["numeric_id"]))

    summary = [
        {key: value for key, value in item.items() if key != "path"}
        for item in sorted(candidates, key=lambda item: item["numeric_id"])
    ]
    return selected["path"], summary


def make_backend_layout(dense: Path) -> None:
    """Keep COLMAP's files and add the sparse/0 layout expected by 3DGS backends."""
    undistorted_sparse = dense / "sparse"
    backend_sparse = undistorted_sparse / "0"
    if not undistorted_sparse.exists():
        raise RuntimeError(f"COLMAP undistorter 没有生成 sparse 目录：{undistorted_sparse}")
    backend_sparse.mkdir(parents=True, exist_ok=True)
    for source in undistorted_sparse.iterdir():
        if source.is_file():
            shutil.copy2(source, backend_sparse / source.name)


def main() -> int:
    parser = argparse.ArgumentParser(description="用 COLMAP 为个人图片生成 3DGS 标准输入")
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path("workspace"))
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--colmap", default=None)
    parser.add_argument("--sparse-model", default="auto",
                        help="COLMAP sparse 模型目录名或 auto；auto 选择注册图片最多的模型。")
    parser.add_argument("--recursive-images", action="store_true",
                        help="Recursively collect images under --images.")
    parser.add_argument("--analyze-only", action="store_true",
                        help="只检查图片并写入 image_quality.json，不运行 COLMAP。")
    parser.add_argument("--pdcnet-root", type=Path, default=None,
                        help="PDCNet+ DenseMatching checkout containing triangulate.py.")
    parser.add_argument("--pdcnet-output", type=Path, default=None,
                        help="Directory for PDCNet+ intermediate dense point clouds.")
    parser.add_argument("--pdcnet-pretrained-model", default=None)
    parser.add_argument("--pdcnet-views", type=int, default=None)
    parser.add_argument("--pdcnet-multi-stage-type", choices=("d", "h", "ms"), default=None)
    parser.add_argument("--skip-pdcnet", action="store_true",
                        help="Only for debugging; do not replace the sparse point cloud.")
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    images = args.images.resolve()
    workspace = args.workspace.resolve()
    if not images.exists():
        raise FileNotFoundError(f"图片目录不存在: {images}")
    paths = collect_images(images, recursive=args.recursive_images)
    if len(paths) < 2:
        raise RuntimeError("至少需要 2 张 JPG/PNG 图片；实际找到的图片不足。")

    workspace.mkdir(parents=True, exist_ok=True)
    reports = workspace / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    rows = inspect_images(paths, cfg)
    (reports / "image_quality.json").write_text(
        json.dumps({"image_count": len(rows), "images": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.analyze_only:
        print(f"完成：已检查 {len(paths)} 张图片，报告：{reports / 'image_quality.json'}")
        return 0

    colmap = args.colmap or cfg.get("colmap_executable", "colmap")
    if shutil.which(colmap) is None and not Path(colmap).exists():
        raise FileNotFoundError("未找到 COLMAP。请安装 COLMAP 并把 colmap.exe 加入 PATH，或用 --colmap 指定完整路径。")

    database = workspace / "database.db"
    sparse = workspace / "sparse"
    dense = workspace / "dense"
    staging = workspace / "input_images"
    database.parent.mkdir(parents=True, exist_ok=True)
    # A repeated default training run must not reuse old database/models.  In
    # particular, stale sparse model IDs can refer to a different image set
    # than the newly generated dense/images directory.
    for database_file in (
        database,
        database.with_name(database.name + "-shm"),
        database.with_name(database.name + "-wal"),
    ):
        if database_file.exists():
            database_file.unlink()
    for generated_dir in (staging, sparse, dense):
        if generated_dir.exists():
            shutil.rmtree(generated_dir)
    stage_images(paths, images, staging)
    # COLMAP's mapper expects its output directory to exist on Windows.
    sparse.mkdir(parents=True, exist_ok=True)
    reader = ["--ImageReader.single_camera", "1" if cfg.get("single_camera", True) else "0",
              "--ImageReader.camera_model", str(cfg.get("camera_model", "OPENCV"))]

    run([colmap, "feature_extractor", "--database_path", str(database), "--image_path", str(staging), *reader])
    matching = cfg.get("matching_method", "exhaustive")
    if matching == "sequential":
        run([colmap, "sequential_matcher", "--database_path", str(database)])
    else:
        run([colmap, "exhaustive_matcher", "--database_path", str(database)])
    run([colmap, "mapper", "--database_path", str(database), "--image_path", str(staging), "--output_path", str(sparse)])
    model, model_summary = select_sparse_model(sparse, args.sparse_model)
    run([colmap, "image_undistorter", "--image_path", str(staging), "--input_path", str(model),
         "--output_path", str(dense), "--output_type", "COLMAP"])
    make_backend_layout(dense)

    pdcnet_point_cloud = None
    pdcnet_enabled = bool(cfg.get("pdcnet_enabled", True)) and not args.skip_pdcnet
    if pdcnet_enabled:
        configured_root = cfg.get("pdcnet_root", str(Path(__file__).resolve().parent / "dense_matcher"))
        pdcnet_root = (args.pdcnet_root or Path(configured_root)).resolve()
        pdcnet_output = (args.pdcnet_output or workspace / "pdcnet").resolve()
        pdcnet_pretrained_model = args.pdcnet_pretrained_model or cfg.get(
            "pdcnet_pretrained_model", "megadepth"
        )
        pdcnet_views = args.pdcnet_views or int(cfg.get("pdcnet_views", 3))
        pdcnet_multi_stage_type = args.pdcnet_multi_stage_type or cfg.get(
            "pdcnet_multi_stage_type", "h"
        )
        pdcnet_script = Path(__file__).resolve().parent / "pdcnet_init.py"
        run([sys.executable, str(pdcnet_script), "--dataset", str(dense),
             "--matcher-root", str(pdcnet_root), "--output", str(pdcnet_output),
             "--pretrained-model", str(pdcnet_pretrained_model),
             "--n-views", str(pdcnet_views),
             "--multi-stage-type", str(pdcnet_multi_stage_type)])
        pdcnet_point_cloud = str(dense / "sparse" / "0" / "points3D.ply")

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "images": str(images),
        "colmap_images": str(staging),
        "workspace": str(workspace),
        "registered_image_minimum": cfg.get("min_registered_images", 4),
        "view_consistency": bool(cfg.get("view_consistency", True)),
        "epipolar_depth_prior": bool(cfg.get("epipolar_depth_prior", False)),
        "depth_model": cfg.get("depth_model", "none"),
        "few_shot_regularization": bool(cfg.get("few_shot_regularization", True)),
        "backend_input": str(dense),
        "initialization": "pdcnet_plus_dense" if pdcnet_point_cloud else "colmap_sparse_debug_only",
        "pdcnet_point_cloud": pdcnet_point_cloud,
        "selected_sparse_model": str(model),
        "sparse_models": model_summary,
    }
    (workspace / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{len(paths)} 张图片已生成 COLMAP 位姿和 PDCNet+ 稠密点云 3DGS 输入：{dense}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



