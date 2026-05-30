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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
import os
from PIL import Image

class VirtualCamera2(nn.Module):
    def __init__(self, colmap_id, uid, R, T, FoVx, FoVy, width, height,scale=1.0, data_device = "cuda", image_name=''):
        super().__init__()
        
        self.uid = uid
        self.colmap_id = colmap_id
        self.projection_matrix = getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy).transpose(0, 1).cuda()
        self.R = R
        self.T = T
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, np.array([0, 0, 0]), 1.0)).transpose(0, 1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = width
        self.image_height = height
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.data_device = data_device

    def load_image(self,main_path):
        image_path = os.path.join(main_path,"images",self.image_name) 
        image_path = os.path.join(main_path,"processed",self.image_name) if not os.path.exists(image_path) else image_path
        image = Image.open(image_path)
        image = image.resize((self.image_width, self.image_height))
        image = np.array(image)
        image = image / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.to(self.data_device)
        # self.original_image = image
        return image


class ParticleCamera(nn.Module):
    def __init__(self, FoVx, FoVy, image_width, image_height,R=None,T=None,c2w=None
                 ):
        super(ParticleCamera, self).__init__()
        self.zfar = 100.0
        self.znear = 0.01
        self.c2w = c2w
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.w2c = np.linalg.inv(c2w)
        self.T = self.w2c[:3, 3]
        self.R = self.c2w[:3, :3]

        self.update_pose()

    def update_pose(self):
        self.world_view_transform = torch.tensor(self.w2c).transpose(0, 1).float().cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
    
    def update_matrices(self,Rel_Pose=None,absolute_pose=None):
        if Rel_Pose is not None:
            if isinstance(Rel_Pose, torch.Tensor):
                Rel_Pose = Rel_Pose.cpu().numpy()

        # Step 1: Update c2w using relative pose
        if Rel_Pose is not None:
            self.c2w = self.c2w @ Rel_Pose
        elif absolute_pose is not None:
            self.c2w = absolute_pose

        # Step 2: Recompute w2c, T, R
        self.w2c = np.linalg.inv(self.c2w)
        self.T = self.w2c[:3, 3]
        self.R = self.c2w[:3, :3]

        # Step 3: Update matrices
        self.update_pose()

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

