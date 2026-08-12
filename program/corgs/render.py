#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import matplotlib.pyplot as plt
import torch
from scene import Scene
import os
from tqdm import tqdm
import numpy as np
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import cv2
import time
from tqdm import tqdm

from utils.graphics_utils import getWorld2View2
from utils.pose_utils import generate_ellipse_path, generate_spiral_path, generate_spiral_path_dtu
from utils.general_utils import vis_depth
import matplotlib.cm as cm



def render_set(model_path, name, iteration, views, gaussians, pipeline, background, args):
    if args.view_names:
        wanted = set(args.view_names)
        available = {view.image_name for view in views}
        missing = sorted(wanted - available)
        if missing:
            raise ValueError(f"Requested view(s) not found: {', '.join(missing)}")
        views = [view for view in views if view.image_name in wanted]
        print(f"Rendering selected views: {', '.join(view.image_name for view in views)}")
    output_name = f"{name}_gs{args.gaussian_index}"
    render_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, output_name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)


    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        render_pkg = render(view, gaussians, pipeline, background)
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, view.image_name + '.png'))
        torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))

        if args.render_depth:
            depth_map = vis_depth(render_pkg['depth'][0].detach().cpu().numpy())
            cv2.imwrite(os.path.join(render_path, view.image_name + '_depth.png'), depth_map)


def render_sets(dataset : ModelParams, pipeline : PipelineParams, args):

    with torch.no_grad():
        gaussians = GaussianModel(args)
        scene = Scene(args, gaussians, load_iteration=args.iteration, shuffle=False)
        if args.gaussian_index == 1:
            second_ply = os.path.join(
                dataset.model_path,
                "point_cloud_gs2",
                "iteration_{}".format(scene.loaded_iter),
                "point_cloud.ply",
            )
            if not os.path.isfile(second_ply):
                raise FileNotFoundError(f"Second Gaussian field not found: {second_ply}")
            gaussians.load_ply(second_ply)
            print(f"Rendering Gaussian field gs1 from {second_ply}")
        else:
            print("Rendering Gaussian field gs0")
        print(f"point number is {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not args.skip_train:
            render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, args)
        if not args.skip_test:
            render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, args)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--fps", default=25, type=int)
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--gaussian_index", type=int, choices=(0, 1), default=0,
                        help="CoR-GS Gaussian field to render: 0=gs0, 1=gs1")
    parser.add_argument("--view_names", nargs="+", default=None,
                        help="只渲染指定的视图名称，例如 view_018 view_026 view_035")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), pipeline.extract(args), args)
