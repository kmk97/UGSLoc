"""Coarse pose refinement (CPR) for UGSLoc — MASt3R matching + PnP."""

from dust3r.inference import inference
from mast3r.fast_nn import fast_reciprocal_NNs
from utils.graphics_utils import fov2focal, qvec2rotmat, focal2fov
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor
import os
import time
from scene.cameras import VirtualCamera2

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x * long_edge_size / S)) for x in img.size)
    return img.resize(new_size, interp)


def tensor_to_dust3r_format_batch(tensor, start_idx=0, img_reso=512, square_ok=False):
    """Convert batch tensor to dust3r image format."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")

    batch_size = tensor.shape[0]
    results = []

    for b in range(batch_size):
        single_tensor = tensor[b]
        tensor_cpu = single_tensor.cpu()
        img_np = tensor_cpu.permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:
            img_np = img_np * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = PIL.Image.fromarray(img_np)

        W1, H1 = img_pil.size
        if img_reso == 224:
            img_pil = _resize_pil_image(img_pil, round(img_reso * max(W1 / H1, H1 / W1)))
        else:
            img_pil = _resize_pil_image(img_pil, img_reso)
        W, H = img_pil.size
        cx, cy = W // 2, H // 2
        if img_reso == 224:
            half = min(cx, cy)
            img_pil = img_pil.crop((cx - half, cy - half, cx + half, cy + half))
        else:
            halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
            if not (square_ok) and W == H:
                halfh = 3 * halfw / 4
            img_pil = img_pil.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        img_normalized = ImgNorm(img_pil)
        results.append(dict(
            img=img_normalized[None],
            true_shape=np.int32([img_pil.size[::-1]]),
            idx=start_idx + b,
            instance=str(start_idx + b),
        ))

    return results


def estimate_pose_cpr_mast3r_batch(
    mast3r_model, device, img_reso, rendered_tensor=None, real_tensor=None,
    rendered_depth_map_batch=None, rendering_cameras=None, real_camera=None,
    original_size=(1080, 1920), reprojection_error=2.5, output_folder=None, viz_matches=0,
    viz_3d_points=False, uncertanity_map=None,
    weighted_refine=False, robust_loss='None',
    uncertanity_threshold=1, new_uncertainty=False,
    alpha=1e2, sigma_min=1e-3, clip_range=1e3,
    use_cropping=False, crop_query=False,
    weighted_epnp=False, max_weighted_pts=512, use_ransac=True,
    uncertainty_only_epnp=False, iterative_pnp=False, iterative_steps=3,
    importance_ransac=False, ransac_importance_iters=1000, ransac_importance_sample_size=6,
    importance_alpha=1.0, lo_refine=True, num_workers=4,
):
    """
    CPR pose refinement using MASt3R (following GS-CPR paper approach).

    Returns:
        list of dicts containing refined pose and matching info for each rendered image
    """
    img1_dict = tensor_to_dust3r_format_batch(rendered_tensor, 0, img_reso)
    img2_dict = tensor_to_dust3r_format_batch(real_tensor, 1, img_reso)

    images_pair_list = [tuple([img1_dict, img2_dict]) for img1_dict, img2_dict in zip(img1_dict, img2_dict)]

    start_time = time.time()
    output = inference(images_pair_list, mast3r_model, device, batch_size=8, verbose=False)
    inference_time = time.time() - start_time
    print(f"MAst3R inference time: {inference_time:.4f} seconds")

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']

    desc1, desc2 = pred1['desc'].detach(), pred2['desc'].detach()

    results = []
    start_time = time.time()

    def process_view(i: int):
        fnn_start_time = time.time()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1[i], desc2[i], subsample_or_initxy1=8,
            device=device, dist='dot', block_size=2**13,
        )
        fnn_time = time.time() - fnn_start_time
        print(f"FNN time: {fnn_time:.4f} seconds")

        H0, W0 = view1['true_shape'][0]
        H1, W1 = view2['true_shape'][0]

        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & \
            (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & \
            (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0, matches_im1 = matches_im0[valid_matches], matches_im1[valid_matches]

        rows = matches_im0[:, 0]
        cols = matches_im0[:, 1]
        match_confidences = pred1['desc_conf'][0][cols, rows]
        matched_conf = match_confidences.mean()
        match_count = len(matches_im0)

        scale_x = original_size[1] / W0.item()
        scale_y = original_size[0] / H0.item()

        for pixel in matches_im0:
            pixel[0] *= scale_x
            pixel[1] *= scale_y
        for pixel in matches_im1:
            pixel[0] *= scale_x
            pixel[1] *= scale_y

        matched_uncertainties = None
        if uncertanity_map is not None:
            unc_map_b = None
            if isinstance(uncertanity_map, (list, tuple)):
                if len(uncertanity_map) > i:
                    unc_map_b = uncertanity_map[i]
            else:
                unc_map_b = uncertanity_map
            if unc_map_b is not None:
                if torch.is_tensor(unc_map_b):
                    unc_map_b = unc_map_b.detach().cpu().numpy()
                vals = []
                H_unc, W_unc = unc_map_b.shape[-2], unc_map_b.shape[-1] if hasattr(unc_map_b, 'shape') else (None, None)
                for (x, y) in matches_im0:
                    xi, yi = int(x), int(y)
                    if H_unc is not None and W_unc is not None and 0 <= yi < H_unc and 0 <= xi < W_unc:
                        vals.append(float(unc_map_b[yi, xi]))
                    else:
                        vals.append(float('inf'))
                matched_uncertainties = np.asarray(vals)

        if viz_matches > 0:
            num_matches = matches_im0.shape[0]
            print(f'Found {num_matches} matches')

            def tensor_to_numpy_img(tensor):
                if tensor.dim() == 4:
                    tensor = tensor.squeeze(0)
                img_np = tensor.cpu().permute(1, 2, 0).numpy()
                if img_np.max() <= 1.0:
                    img_np = img_np * 255
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)
                return img_np

            viz_imgs = [tensor_to_numpy_img(rendered_tensor[i]), tensor_to_numpy_img(real_tensor[i])]

            n_viz = min(viz_matches, num_matches)
            if n_viz > 0:
                match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                viz_matches_rendered = matches_im0[match_idx_to_viz]
                viz_matches_real = matches_im1[match_idx_to_viz]
                viz_confidences = match_confidences[match_idx_to_viz].cpu().numpy()
                conf_min, conf_max = viz_confidences.min(), viz_confidences.max()
                if conf_max > conf_min:
                    viz_confidences_norm = (viz_confidences - conf_min) / (conf_max - conf_min)
                else:
                    viz_confidences_norm = np.ones_like(viz_confidences) * 0.5

                H0, W0, H1, W1 = *viz_imgs[0].shape[:2], *viz_imgs[1].shape[:2]
                img0 = np.pad(viz_imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img1 = np.pad(viz_imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img = np.concatenate((img0, img1), axis=1)

                plt.figure(figsize=(15, 8))
                plt.imshow(img)
                cmap = plt.get_cmap('viridis')
                for j in range(n_viz):
                    (x0, y0), (x1, y1) = viz_matches_rendered[j], viz_matches_real[j]
                    plt.plot([x0, x1 + W0], [y0, y1], '-+',
                             color=cmap(viz_confidences_norm[j]), scalex=False, scaley=False)

                plt.title(f'MASt3R Matches: {n_viz} out of {num_matches} total matches')
                plt.axis('off')

                if output_folder is not None:
                    os.makedirs(output_folder, exist_ok=True)
                    plt.savefig(os.path.join(output_folder, f'mast3r_matches_{i}_{n_viz}.png'),
                                bbox_inches='tight', dpi=150)
                plt.show()
                plt.close()

        rendering_camera = rendering_cameras[i]["cam"]
        FoVx = rendering_camera.FoVx
        FoVy = rendering_camera.FoVy
        width = rendering_camera.image_width
        height = rendering_camera.image_height
        fx = fov2focal(FoVx, width)
        fy = fov2focal(FoVy, height)
        cx = width / 2
        cy = height / 2
        K = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

        initial_pose_c2w = rendering_camera.c2w

        rendered_depth_map = rendered_depth_map_batch[i]
        if rendered_depth_map.dim() == 3:
            rendered_depth_map = rendered_depth_map.squeeze(0)
        if torch.is_tensor(rendered_depth_map):
            rendered_depth_map = rendered_depth_map.cpu().numpy()

        height, width = rendered_depth_map.shape
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

        x_flat = x_coords.flatten()
        y_flat = y_coords.flatten()
        depth_flat = rendered_depth_map.flatten()

        x_normalized = (x_flat - K[0, 2]) / K[0, 0]
        y_normalized = (y_flat - K[1, 2]) / K[1, 1]

        X_camera = depth_flat * x_normalized
        Y_camera = depth_flat * y_normalized
        Z_camera = depth_flat

        points_camera = np.vstack((X_camera, Y_camera, Z_camera, np.ones_like(X_camera)))

        points_world = initial_pose_c2w @ points_camera
        X_world = points_world[0, :]
        Y_world = points_world[1, :]
        Z_world = points_world[2, :]
        points_3D = np.vstack((X_world, Y_world, Z_world))

        scene_coordinates_gs = points_3D.reshape(3, original_size[0], original_size[1])

        points_3D_at_pixels = np.zeros((matches_im0.shape[0], 3))
        for _, (x, y) in enumerate(matches_im0):
            x_int, y_int = int(x), int(y)
            if 0 <= y_int < original_size[0] and 0 <= x_int < original_size[1]:
                points_3D_at_pixels[_] = scene_coordinates_gs[:, y_int, x_int]

        refined_pose_c2w = initial_pose_c2w.copy()
        success = False
        inliers = None

        if matches_im1.shape[0] >= 4:
            initial_rvec, _ = cv2.Rodrigues(initial_pose_c2w[:3, :3].astype(np.float32))
            initial_tvec = initial_pose_c2w[:3, 3].astype(np.float32)

            dist_coeffs = np.array([0, 0, 0, 0], dtype=np.float32)

            each_pnp_start_time = time.time()
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3D_at_pixels.astype(np.float32),
                matches_im1.astype(np.float32),
                K.astype(np.float32),
                dist_coeffs,
                rvec=initial_rvec,
                tvec=initial_tvec,
                useExtrinsicGuess=True,
                reprojectionError=reprojection_error,
                iterationsCount=2000,
                flags=cv2.SOLVEPNP_EPNP,
            )
            each_pnp_time = time.time() - each_pnp_start_time
            print("\033[96m PnP RANSAC time: {:.3f}s\033[0m".format(each_pnp_time))
            if success:
                R, _ = cv2.Rodrigues(rvec)
                trans = -R.T @ tvec

                refined_pose_c2w = np.eye(4)
                refined_pose_c2w[:3, :3] = R.T
                refined_pose_c2w[:3, 3] = trans.reshape(3)

        return {
            'refined_pose_c2w': refined_pose_c2w,
            'initial_pose_c2w': initial_pose_c2w,
            'success': success,
            'num_matches': matches_im0.shape[0],
            'matches_rendered': matches_im0.cpu().numpy() if torch.is_tensor(matches_im0) else matches_im0,
            'matches_real': matches_im1.cpu().numpy() if torch.is_tensor(matches_im1) else matches_im1,
            'points_3d': points_3D_at_pixels if matches_im0.shape[0] >= 4 else None,
            'inliers': inliers if success else None,
            'matched_conf': matched_conf,
            'matched_confidences': match_confidences,
            'matched_uncertainties': matched_uncertainties,
            'match_count': match_count,
        }

    start_pnp_time = time.time()
    if num_workers is None or num_workers <= 0:
        try:
            num_workers = min(desc1.shape[0], os.cpu_count() or 4)
        except Exception:
            num_workers = min(desc1.shape[0], 4)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_view, range(desc1.shape[0])))
    end_pnp_time = time.time()
    print(f"PNP time: {end_pnp_time - start_pnp_time:.4f} seconds")
    return results


def load_poses_from_file(file_path, source_path, test_image_name, pose_estimator='dfnet'):
    camera_intrin_params = [1920, 1080, 1673, 960, 540]
    width = camera_intrin_params[0]
    height = camera_intrin_params[1]
    focal_length_path = source_path + '/calibration/'

    with open(file_path, 'r') as file:
        lines = file.readlines()

    matching_view = None
    if not pose_estimator == 'dfnet':
        test_image_name = test_image_name.replace('/frame', '_frame')

    for line in lines:
        image_name = line.split()[0]

        if image_name == test_image_name:
            qvec = [float(value) for value in line.split()[1:5]]
            tvec = [float(value) for value in line.split()[5:8]]
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array(tvec)

            focal_length = np.loadtxt(focal_length_path + image_name.replace('.png', '.txt').replace('/frame', '_frame'))
            FovY = focal2fov(focal_length * 2.25, height)
            FovX = focal2fov(focal_length * 2.25, width)
            matching_view = VirtualCamera2(
                colmap_id=1, uid=0, R=R, T=T, FoVx=FovX, FoVy=FovY,
                width=width, height=height, image_name=image_name,
            )
            break

    if matching_view is None:
        raise ValueError(f"No matching pose found for test image: {test_image_name}")

    return matching_view


def load_poses_from_file_STD(file_path, source_path, test_image_name, test_cam, pose_estimator='dfnet'):
    camera_intrin_params = [1920, 1080, 1673, 960, 540]
    width = camera_intrin_params[0]
    height = camera_intrin_params[1]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    matching_view = None
    if not pose_estimator == 'dfnet':
        test_image_name = test_image_name.replace('/frame', '_frame')

    for line in lines:
        image_name = line.split()[0]

        if image_name == test_image_name:
            qvec = [float(value) for value in line.split()[1:5]]
            tvec = [float(value) for value in line.split()[5:8]]
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array(tvec)
            FovX = test_cam.FoVx
            FovY = test_cam.FoVy
            matching_view = VirtualCamera2(
                colmap_id=1, uid=0, R=R, T=T, FoVx=FovX, FoVy=FovY,
                width=width, height=height, image_name=image_name,
            )
            break

    if matching_view is None:
        raise ValueError(f"No matching pose found for test image: {test_image_name}")

    return matching_view


def load_poses_from_file_7scenes(file_path, source_path, test_image_name, pose_estimator):
    """7Scenes dataset version."""
    camera_intrin_params = [640, 480, 525.0, 320, 240]
    width = camera_intrin_params[0]
    height = camera_intrin_params[1]

    focal_length_dict = {
        'chess': 526.22,
        'fire': 526.903,
        'heads': 527.745,
        'office': 525.143,
        'pumpkin': 525.647,
        'redkitchen': 525.505,
        'stairs': 525.505,
    }

    scene_name = None
    for scene in focal_length_dict.keys():
        if scene in source_path:
            scene_name = scene
            break

    if scene_name is None:
        focal_length = 525.0
    else:
        focal_length = focal_length_dict[scene_name]

    with open(file_path, 'r') as file:
        lines = file.readlines()

    matching_view = None

    if pose_estimator == 'dfnet':
        test_image_name = test_image_name.replace('-frame', '/frame')
    if pose_estimator == 'mspt':
        test_image_name = f'{scene_name}/' + test_image_name.replace('-frame', '/frame')

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 8:
            continue

        image_name = parts[0]

        if image_name == test_image_name:
            if pose_estimator == 'mspt':
                parts = line.strip().split()
                pos_1_4 = [float(x) for x in parts[1:4]]
                pos_4_8 = [float(x) for x in parts[4:8]]

                pos_4_8_tensor = torch.tensor(pos_4_8)
                pos_4_8_normalized = F.normalize(pos_4_8_tensor.unsqueeze(0), p=2, dim=1).squeeze(0)
                pos_4_8 = pos_4_8_normalized.tolist()

                line = parts[0] + ' ' + ' '.join(map(str, pos_4_8)) + ' ' + ' '.join(map(str, pos_1_4))

            print(f"\033[92mDirect match found: {image_name}\033[0m")
            matching_view = create_camera_from_pose_line_7scenes(line, width, height, focal_length, image_name)
            break

    if matching_view is None:
        print(f"\033[91mNo matching pose found for test image: {test_image_name}\033[0m")
        print(f"\033[91mAvailable images in pose file (first 10):\033[0m")
        with open(file_path, 'r') as file:
            for i, line in enumerate(file.readlines()[:10]):
                parts = line.strip().split()
                if len(parts) >= 1:
                    print(f"  {parts[0]}")
        raise ValueError(f"No matching pose found for test image: {test_image_name}")

    return matching_view


def create_camera_from_pose_line_7scenes(line, width, height, focal_length, image_name):
    """Create camera from pose file line for 7Scenes."""
    parts = line.strip().split()
    qvec = [float(value) for value in parts[1:5]]
    tvec = [float(value) for value in parts[5:8]]
    R = np.transpose(qvec2rotmat(qvec))
    T = np.array(tvec)

    FovY = focal2fov(focal_length, height)
    FovX = focal2fov(focal_length, width)

    return VirtualCamera2(
        colmap_id=1, uid=0, R=R, T=T, FoVx=FovX, FoVy=FovY,
        width=width, height=height, image_name=image_name,
    )


# UGSLoc public CPR API
try:
    from cpr_unc import cpr_unc
except Exception:
    pass

cpr_batch = estimate_pose_cpr_mast3r_batch
load_coarse_pose = load_poses_from_file
load_coarse_pose_std = load_poses_from_file_STD
load_coarse_pose_7s = load_poses_from_file_7scenes
