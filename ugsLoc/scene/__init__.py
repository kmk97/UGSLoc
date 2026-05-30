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

import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos


class SceneAnchor:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], dataset_type='7scenes', original_gs=False, eval_train=True):
        self.model_path = args.model_path
        self.source_path = args.source_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.train_cameras = {}
        self.test_cameras = {}

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        callback_result = sceneLoadTypeCallbacks[dataset_type](
            args.source_path, args.model_path, args.images, args.eval)
        if len(callback_result) == 3:
            scene_info, self.gt_dir, self.test_info = callback_result
        else:
            scene_info, self.gt_dir = callback_result
            self.test_info = None

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args, for_eval=eval_train)
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args, for_eval=True)

        if hasattr(self.gaussians, 'set_appearance'):
            self.gaussians.set_appearance(len(scene_info.train_cameras))

        if self.loaded_iter:
            if not original_gs:
                iter_dir = os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}")
                self.gaussians.load_ply_sparse_gaussian(os.path.join(iter_dir, "point_cloud.ply"))
                self.gaussians.load_mlp_checkpoints(iter_dir)
            else:
                self.gaussians.load_ply(os.path.join(
                    self.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
