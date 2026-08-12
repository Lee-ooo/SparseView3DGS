"""Mouse-controlled novel-view renderer for the local FSGS/CoR-GS backends."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import render
from scene import Scene
from scene.cameras import PseudoCamera
from utils.general_utils import safe_state


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length < 1e-8:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return vector / length


def camera_center(camera) -> np.ndarray:
    return camera.camera_center.detach().cpu().numpy().astype(np.float32)


def make_camera(position: np.ndarray, target: np.ndarray, template) -> PseudoCamera:
    """Create a renderer camera from a world-space look-at pose."""
    forward = normalize(target - position)
    up_reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, up_reference))) > 0.98:
        up_reference = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    right = normalize(np.cross(up_reference, forward))
    up = normalize(np.cross(forward, right))
    c2w_rotation = np.stack([right, up, forward], axis=1).astype(np.float32)
    translation = -c2w_rotation.T @ position
    return PseudoCamera(
        R=c2w_rotation,
        T=translation.astype(np.float32),
        FoVx=template.FoVx,
        FoVy=template.FoVy,
        width=template.image_width,
        height=template.image_height,
    )


class ViewerState:
    def __init__(self, position: np.ndarray, target: np.ndarray, scene_extent: float):
        offset = position - target
        self.target = target.copy()
        self.radius = max(float(np.linalg.norm(offset)), scene_extent * 0.25, 0.1)
        self.yaw = math.atan2(float(offset[0]), float(offset[2]))
        self.pitch = math.asin(float(np.clip(offset[1] / self.radius, -0.99, 0.99)))
        self.initial = (self.target.copy(), self.radius, self.yaw, self.pitch)
        self.scene_extent = max(float(scene_extent), 0.1)

    def position(self) -> np.ndarray:
        cos_pitch = math.cos(self.pitch)
        offset = np.array([
            math.sin(self.yaw) * cos_pitch,
            math.sin(self.pitch),
            math.cos(self.yaw) * cos_pitch,
        ], dtype=np.float32)
        return self.target + self.radius * offset

    def reset(self) -> None:
        target, radius, yaw, pitch = self.initial
        self.target = target.copy()
        self.radius = radius
        self.yaw = yaw
        self.pitch = pitch

    def orbit(self, dx: float, dy: float) -> None:
        self.yaw += dx * 0.008
        self.pitch = float(np.clip(self.pitch + dy * 0.008, -1.45, 1.45))

    def zoom(self, direction: float) -> None:
        self.radius *= 0.86 if direction > 0 else 1.16
        self.radius = float(np.clip(self.radius, self.scene_extent * 0.05, self.scene_extent * 30.0))

    def pan(self, dx: float, dy: float) -> None:
        position = self.position()
        forward = normalize(self.target - position)
        up_reference = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = normalize(np.cross(up_reference, forward))
        up = normalize(np.cross(forward, right))
        amount = self.radius * 0.0015
        self.target += (-dx * right + dy * up) * amount


def render_frame(viewer, state: ViewerState, template, gaussians, pipeline, background):
    position = state.position()
    camera = make_camera(position, state.target, template)
    with torch.no_grad():
        package = render(camera, gaussians, pipeline, background)
    rgb = package["render"].clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    bgr = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    return bgr, camera, position


def save_view(output_dir: Path, frame: np.ndarray, camera, state: ViewerState, index: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    image_path = output_dir / f"view_{stamp}_{index:03d}.png"
    cv2.imwrite(str(image_path), frame)
    pose_path = image_path.with_suffix(".json")
    pose = {
        "image": str(image_path),
        "position": state.position().tolist(),
        "target": state.target.tolist(),
        "radius": state.radius,
        "yaw": state.yaw,
        "pitch": state.pitch,
        "R": camera.R.tolist(),
        "T": camera.T.tolist(),
        "FoVx": float(camera.FoVx),
        "FoVy": float(camera.FoVy),
        "width": int(camera.image_width),
        "height": int(camera.image_height),
    }
    pose_path.write_text(json.dumps(pose, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {image_path}")
    print(f"Pose : {pose_path}")
    return image_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive novel-view renderer")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output_dir", default="", type=str)
    parser.add_argument("--display_scale", default=0.75, type=float)
    parser.add_argument("--window", default="SparseView3DGS interactive viewer")
    args = get_combined_args(parser)
    safe_state(args.quiet)

    from gaussian_renderer import GaussianModel

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    gaussians = GaussianModel(args)
    scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
    train_views = scene.getTrainCameras()
    if not train_views:
        raise RuntimeError("模型中没有可用的训练相机。")

    points = gaussians.get_xyz.detach().cpu().numpy()
    lower = np.percentile(points, 5.0, axis=0)
    upper = np.percentile(points, 95.0, axis=0)
    target = ((lower + upper) * 0.5).astype(np.float32)
    centers = np.stack([camera_center(view) for view in train_views])
    initial_position = centers[0]
    scene_extent = float(np.linalg.norm(upper - lower))
    state = ViewerState(initial_position, target, scene_extent)
    template = train_views[0]
    background = torch.tensor(
        [1.0, 1.0, 1.0] if dataset.white_background else [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )
    output_dir = Path(args.output_dir) if args.output_dir else Path(dataset.model_path) / "interactive"
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    mouse = {"mode": None, "x": 0, "y": 0}

    def on_mouse(event, x, y, flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse.update(mode="orbit", x=x, y=y)
        elif event == cv2.EVENT_RBUTTONDOWN:
            mouse.update(mode="pan", x=x, y=y)
        elif event == cv2.EVENT_MOUSEMOVE and mouse["mode"]:
            dx, dy = x - mouse["x"], y - mouse["y"]
            if mouse["mode"] == "orbit":
                state.orbit(dx, dy)
            else:
                state.pan(dx, dy)
            mouse.update(x=x, y=y)
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            mouse["mode"] = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            state.zoom(1.0 if flags > 0 else -1.0)

    cv2.namedWindow(args.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(args.window, int(template.image_width * args.display_scale), int(template.image_height * args.display_scale))
    cv2.setMouseCallback(args.window, on_mouse)
    dirty = True
    frame = None
    camera = None
    save_index = 1
    print("Interactive viewer started.")
    print("Left drag: orbit | Right drag: pan | Wheel: zoom | R: reset | S: save | Q/Esc: quit")
    try:
        while True:
            if dirty or frame is None:
                frame, camera, _ = render_frame(None, state, template, gaussians, pipe, background)
                dirty = False
            display = frame
            if args.display_scale != 1.0:
                display = cv2.resize(frame, None, fx=args.display_scale, fy=args.display_scale, interpolation=cv2.INTER_AREA)
            cv2.putText(display, "L-drag orbit | R-drag pan | Wheel zoom | R reset | S save | Q quit",
                        (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(args.window, display)
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                state.reset()
                dirty = True
            elif key in (ord("+"), ord("=")):
                state.zoom(1.0)
                dirty = True
            elif key in (ord("-"), ord("_")):
                state.zoom(-1.0)
                dirty = True
            elif key in (ord("s"), ord("S")):
                if camera is not None:
                    save_view(output_dir, frame, camera, state, save_index)
                    save_index += 1
            if mouse["mode"] is not None:
                dirty = True
    finally:
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
