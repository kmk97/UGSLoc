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
import sys
from PIL import Image
from typing import NamedTuple

import numpy as np
from plyfile import PlyData, PlyElement

from scene.colmap_loader import (
    read_extrinsics_text, read_intrinsics_text, qvec2rotmat,
    read_extrinsics_binary, read_intrinsics_binary,
    read_points3D_binary, read_points3D_text,
)
from utils.graphics_utils import getWorld2View2, focal2fov
from scene.gaussian_model import BasicPointCloud
import root_file_io as fio


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        dist = np.linalg.norm(cam_centers - avg_cam_center, axis=0, keepdims=True)
        return avg_cam_center.flatten(), np.max(dist)

    cam_centers = []
    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    return {"translate": -center, "radius": diagonal * 1.1}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted PINHOLE/SIMPLE_PINHOLE supported!"

        image_path = os.path.join(images_folder, extr.name)
        if not fio.file_exist(image_path):
            continue
        parts = image_path.split('/')
        image_name = '/'.join(parts[-2:])
        image = Image.open(image_path)
        cam_infos.append(CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name, width=width, height=height,
        ))
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except Exception:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    normals = np.zeros_like(xyz)
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements[:] = list(map(tuple, np.concatenate((xyz, normals, rgb), axis=1)))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)


def read7ScenesSceneInfo(path, model_path, images, eval):
    combo = model_path.split(fio.sep)
    train_path = fio.sep.join(combo[:-1])
    test_path = path

    train_cameras_extrinsic_file = os.path.join(train_path, "sparse/0", "images.bin")
    train_cameras_intrinsic_file = os.path.join(train_path, "sparse/0", "cameras.bin")
    if not (fio.file_exist(train_cameras_extrinsic_file) and fio.file_exist(train_cameras_intrinsic_file)):
        train_cameras_extrinsic_file = os.path.join(train_path, "sparse/0", "images.txt")
        train_cameras_intrinsic_file = os.path.join(train_path, "sparse/0", "cameras.txt")
        if not (fio.file_exist(train_cameras_extrinsic_file) and fio.file_exist(train_cameras_intrinsic_file)):
            print("No pre-trained model detected at", train_path)
            sys.exit()

    train_cam_extrinsics = read_extrinsics_text(train_cameras_extrinsic_file)
    train_cam_intrinsics = read_intrinsics_text(train_cameras_intrinsic_file)
    train_cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=train_cam_extrinsics,
        cam_intrinsics=train_cam_intrinsics,
        images_folder=os.path.join(train_path, images),
    )
    train_cam_infos = sorted(train_cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    test_rgb_files, test_pose_files, test_intrinsic_files = [], [], []
    for file in os.listdir(os.path.join(test_path, "rgb")):
        if file.endswith(".png") or file.endswith(".jpg"):
            test_rgb_files.append(os.path.join(test_path, "rgb", file))
            ext = ".png" if file.endswith(".png") else ".jpg"
            test_pose_files.append(os.path.join(
                test_path, "poses", file.replace(ext, ".txt").replace("color", "pose")))
            test_intrinsic_files.append(os.path.join(
                test_path, "calibration", file.replace(ext, ".txt").replace("color", "calibration")))
    if len(test_rgb_files) == 0:
        print("No test images detected at", test_path)
        sys.exit()

    test_poses, test_intrinsics = [], []
    for pose_file, intrinsic_file in zip(test_pose_files, test_intrinsic_files):
        test_poses.append(np.loadtxt(pose_file))
        test_intrinsics.append(np.loadtxt(intrinsic_file))

    test_info = {
        "test_rgb_files": test_rgb_files,
        "test_poses": test_poses,
        "test_intrinsics": test_intrinsics,
    }
    nerf_normalization = getNerfppNorm(train_cam_infos)
    scene_info = SceneInfo(
        point_cloud=None,
        train_cameras=train_cam_infos,
        test_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=None,
    )
    return scene_info, os.path.join(path, images), test_info


def readCambridgeSceneInfo(path, model_path, images, eval, images_to_read=None):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "processed"
    if os.path.exists(os.path.join(path, "sparse/0", "list_test.txt")):
        with open(os.path.join(path, "sparse/0", "list_test.txt")) as f:
            test_images = [x.strip() for x in f.readlines()]
    elif os.path.exists(os.path.join(path, "dataset_test.txt")):
        with open(os.path.join(path, "dataset_test.txt")) as f:
            test_images = [x.split(" ")[0] for x in f.readlines() if x[0] != '#']
    else:
        test_images = []

    if images_to_read is None:
        images_to_read = None if not eval else test_images

    cam_infos = sorted(
        readColmapCamerasCambridge(
            cam_extrinsics=cam_extrinsics,
            cam_intrinsics=cam_intrinsics,
            images_folder=os.path.join(path, reading_dir),
            images_to_read=images_to_read,
        ),
        key=lambda x: x.image_name,
    )

    if eval:
        train_cam_infos = []
        test_cam_infos = cam_infos
    else:
        train_cam_infos, test_cam_infos = [], []
        for cam_info in cam_infos:
            (test_cam_infos if cam_info.image_name in test_images else train_cam_infos).append(cam_info)

    print(f'test cameras: {len(test_cam_infos)}')
    print(f'train cameras: {len(train_cam_infos)}')

    nerf_normalization = getNerfppNorm(test_cam_infos if len(train_cam_infos) == 0 else train_cam_infos)
    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info, os.path.join(path, "processed")


def readColmapCamerasCambridge(cam_extrinsics, cam_intrinsics, images_folder, images_to_read=None):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        image_name = extr.name
        if images_to_read is not None and image_name not in images_to_read:
            continue

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model in ("PINHOLE", "OPENCV"):
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets supported!"

        image_path = os.path.join(images_folder, extr.name)
        image = Image.open(image_path)
        cam_infos.append(CameraInfo(
            uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
            image_path=image_path, image_name=image_name, width=width, height=height,
        ))
    sys.stdout.write('\n')
    return cam_infos


sceneLoadTypeCallbacks = {
    "cambridge": readCambridgeSceneInfo,
    "7scenes": read7ScenesSceneInfo,
}
