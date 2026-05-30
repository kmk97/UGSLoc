import os
import datetime
import time
import gc

from argparse import Namespace

import torch
import numpy as np
import random
import cv2
from tqdm import tqdm
import torchvision.transforms.functional as TF
from pytorch_msssim import ssim as torch_ssim
import matplotlib.pyplot as plt

from scene import SceneAnchor
from gaussian_renderer import render_original, modified_render_dof
from gaussian_renderer import GaussianModel
from scene.gaussian_model_original import GaussianModel as GaussianModelOriginal
from scene.cameras import ParticleCamera
from arguments import (
    get_combined_args, ModelParams, PipelineParams, LocalizationParams,
    build_loc_parser, CPR_DEFAULTS, get_appearance_args, LocConfig,
    CAMBRIDGE_CFG, SEVENSCENES_CFG,
)
from loc_utils import combine_3dgs_rotation_translation, trans_t_xyz, rot_phi, rot_theta, rot_psi
from loc_utils import initialize_results_file, save_camera_result, calculate_success_rates, print_success_rates, save_final_results
from loc_utils import _copy_and_perturb_camera
from unc_render import render_unc
from mast3r.model import AsymmetricMASt3R
from cpr import cpr_batch, cpr_unc, load_coarse_pose, load_coarse_pose_std, load_coarse_pose_7s
from scene.nerfh_nff import create_nerf
from dataset_loaders.utils.color import rgb_to_yuv
from utils.graphics_utils import focal2fov

USE_UNCERTAINTY = True


def _load_hessian(cfg: LocConfig, model_path: str, original_gs: bool):
    if not USE_UNCERTAINTY:
        return None
    if original_gs and cfg.hessian_original_gs_file:
        candidates = [cfg.hessian_original_gs_file]
    else:
        candidates = cfg.hessian_files
    for name in candidates:
        path = os.path.join(model_path, name)
        if os.path.exists(path):
            print(f"Loading Hessian color from {path}")
            return torch.load(path)
    raise ValueError(f"No hessian color file found in {model_path} (tried {candidates})")


def _load_image_from_path(image_path, target_size=(640, 480)):
    from PIL import Image

    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    if image.size != target_size:
        print(f"\033[93mImage size {image.size} differs from target {target_size}. Resizing...\033[0m")
        image = image.resize(target_size)
    return torch.from_numpy(np.array(image) / 255.0).permute(2, 0, 1).float()


class UGSLoc:
    def __init__(self, cfg: LocConfig, num_particles, scene, gaussians, pipeline, background,
                 locparam, args, model_path, matcher_model, render_kwargs_test=None,
                 test_cam_index=None, save_dir=None):
        self.cfg = cfg
        self.num_particles = num_particles
        self.scene = scene
        self.gaussians = gaussians
        self.pipeline = pipeline
        self.background = background
        self.localization_params = locparam
        self.filter_iteration = 0
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.main_save_dir = save_dir
        self.args = args
        self.mast3r_model = matcher_model
        self.render_kwargs_test = render_kwargs_test

        np.random.seed(42)
        self.hessian_color = _load_hessian(cfg, model_path, args.original_gs)

        self.save_dir = os.path.join(self.main_save_dir, f'test_cam_{test_cam_index}')
        os.makedirs(self.save_dir, exist_ok=True)

        self.train_cams = scene.train_cameras[1.0].copy()
        self._setup_test_camera(test_cam_index)
        self._load_coarse_pose()
        self.particles = self.initialize_particles()

    def _setup_test_camera(self, test_cam_index):
        if self.cfg.test_source == 'colmap':
            self.test_cams = self.scene.test_cameras[1.0].copy()
            if not self.test_cams:
                raise ValueError("No test cameras available for localization.")
            if test_cam_index is not None:
                print(f"\033[94mtest_cam_index: {test_cam_index}\033[0m")
                self.test_cam = self.test_cams[test_cam_index]
            else:
                self.test_cam = random.choice(self.test_cams)
            self.real_image = self.test_cam.load_image(self.scene.source_path)
            self.img_size = (self.test_cam.image_height, self.test_cam.image_width)
        else:
            self.test_info = self.scene.test_info
            if not self.test_info or not self.test_info.get("test_rgb_files"):
                raise ValueError("No test cameras available for localization.")
            if test_cam_index is not None:
                print(f"\033[94mtest_cam_index: {test_cam_index}\033[0m")
                self.test_cam_index = test_cam_index
            else:
                self.test_cam_index = random.randint(0, len(self.test_info["test_rgb_files"]) - 1)
            self.test_rgb_file = self.test_info["test_rgb_files"][self.test_cam_index]
            self.test_pose = self.test_info["test_poses"][self.test_cam_index]
            self.test_cam = self._create_camera_from_test_info()
            h, w = self.cfg.fixed_img_size
            self.img_size = (h, w)
            self.real_image = _load_image_from_path(self.test_rgb_file, target_size=(w, h))

        self.max_opacity = self.img_size[0] * self.img_size[1]

    def _create_camera_from_test_info(self):
        width, height = 640, 480
        focal_lengths = self.cfg.focal_lengths or {}
        scene_name = getattr(self.args, 'scene_name', None)
        if scene_name not in focal_lengths:
            raise ValueError(f"No focal length found for scene: {scene_name}")
        focal_length = focal_lengths[scene_name]
        print(f"Using focal length for {scene_name}: {focal_length}")
        fx = fy = focal_length
        camera = ParticleCamera(
            FoVx=focal2fov(fx, width),
            FoVy=focal2fov(fy, height),
            image_width=width,
            image_height=height,
            c2w=self.test_pose,
        )
        camera.image_name = os.path.basename(self.test_rgb_file)
        camera.update_pose()
        return camera

    def _load_coarse_pose(self):
        current_path = os.path.dirname(os.path.abspath(__file__))
        coarse_file = self.cfg.coarse_pose_template.format(scene_name=self.args.scene_name)
        coarse_cam_path = os.path.join(
            current_path, "coarse_poses", self.args.pose_estimator,
            self.cfg.coarse_pose_subdir, coarse_file,
        )
        if self.cfg.coarse_pose_loader == 'cambridge':
            try:
                self.coarse_cam = load_coarse_pose(
                    coarse_cam_path, self.scene.source_path, self.test_cam.image_name,
                    pose_estimator=self.args.pose_estimator)
            except Exception:
                self.coarse_cam = load_coarse_pose_std(
                    coarse_cam_path, self.scene.source_path, self.test_cam.image_name,
                    self.test_cam, pose_estimator=self.args.pose_estimator)
        else:
            self.coarse_cam = load_coarse_pose_7s(
                coarse_cam_path, self.scene.source_path, self.test_cam.image_name,
                self.args.pose_estimator)

    def _apply_appearance(self, rendering_rgb, rendering_depth):
        if not self.cfg.use_appearance or self.render_kwargs_test is None:
            return rendering_rgb
        if not hasattr(self, 'gt_image_histogram'):
            yuv = rgb_to_yuv(self.real_image)
            hist = torch.histc(yuv[0], bins=10, min=0., max=1.)
            hist = hist / hist.sum() * 100
            self.gt_image_histogram = torch.round(hist).unsqueeze(0).cuda()
        if not getattr(self.args, 'no_appearance', False):
            network_fn = self.render_kwargs_test.get('network_fn')
            if network_fn is not None:
                appearance_args = self.render_kwargs_test.get('appearance_args', self.args)
                rendering_rgb = network_fn.affine_color_transform(
                    appearance_args, rendering_rgb, self.gt_image_histogram, 1)
        if len(rendering_rgb.shape) != 3:
            rendering_rgb = rendering_rgb.reshape(3, rendering_depth.shape[0], rendering_depth.shape[1])
        return rendering_rgb

    def initialize_particles(self):
        particles = []
        pertub_std = np.array([
            self.args.pertub_std_pos_init, self.args.pertub_std_pos_init, self.args.pertub_std_pos_init,
            self.args.pertub_std_rot_init, self.args.pertub_std_rot_init, self.args.pertub_std_rot_init,
        ])
        translation_noise = np.random.uniform(-1, 1, size=(self.num_particles, 3)) * pertub_std[:3]
        rotation_noise = np.random.uniform(-1, 1, size=(self.num_particles, 3)) * pertub_std[3:]

        start_pose_c2w = combine_3dgs_rotation_translation(self.coarse_cam.R, self.coarse_cam.T)
        self.coarse_cam_error_trans, self.coarse_cam_error_rot = self.compute_pose_errors(
            self.coarse_cam.camera_center.cpu().numpy(), self.coarse_cam.R, self.test_cam)

        for n in range(self.num_particles):
            particle_pose_c2w = (
                trans_t_xyz(translation_noise[n][0], translation_noise[n][1], translation_noise[n][2]) @
                rot_phi(rotation_noise[n][0] / 180. * np.pi) @
                rot_theta(rotation_noise[n][1] / 180. * np.pi) @
                rot_psi(rotation_noise[n][2] / 180. * np.pi) @
                start_pose_c2w
            )
            particles.append({
                "cam": _copy_and_perturb_camera(self.test_cam, particle_pose_c2w),
                "weight": 1.0 / self.num_particles,
                "likelihood": 0.0,
            })
        return particles

    def render_particle_with_nerf_features(self, view):
        if getattr(self.args, 'original_gs', False):
            rendering = render_original(view, self.gaussians, self.pipeline, self.background)
        else:
            rendering = modified_render_dof(view, self.gaussians, self.pipeline, self.background)
        rendering_rgb = rendering["render"]
        rendering_depth = rendering["depth"]
        rendering_rgb = self._apply_appearance(rendering_rgb, rendering_depth)
        return rendering_rgb, rendering_depth

    def render_particle_with_uncertainty(self, view):
        offset = self.gaussians._offset
        N_per_anchor = offset.shape[1]
        to_homo = lambda x: torch.cat([x, torch.ones(x.shape[:-1] + (1,), dtype=x.dtype, device=x.device)], dim=-1)
        xyz_expanded = self.gaussians._anchor.repeat_interleave(N_per_anchor, dim=0)
        gaussian_depths = (to_homo(xyz_expanded) @ view.world_view_transform)[:, 2, None]
        cur_hessian_color = self.hessian_color * gaussian_depths.clamp(min=0)

        rendering_rgb, uncertanity_map, pixel_gaussian_counter, rendering_depth = render_unc(
            view, self.gaussians, self.pipeline, self.background, cur_hessian_color)
        rendering_rgb = self._apply_appearance(rendering_rgb, rendering_depth)
        uncertanity_map = torch.log((uncertanity_map - uncertanity_map.min() + 1e-6) / pixel_gaussian_counter)
        return rendering_rgb, rendering_depth, uncertanity_map, pixel_gaussian_counter

    def _cpr_img_reso(self):
        early, late = self.cfg.img_reso
        if early != late and self.filter_iteration < self.args.num_iterations - 1:
            return early
        return late

    def render_particle_views_and_compute_likelihoods(self):
        real_image_resized = TF.resize(self.real_image, self.img_size).cuda()
        batch_size = self.args.batch_size_arg
        scores = []
        likelihoods = []

        conf_type = "confidence_sum" if self.filter_iteration < self.args.num_iterations - 1 else "SSIM"
        print(f"Confidence type: {conf_type}, Batch size: {batch_size}, Number of particles: {len(self.particles)}")
        real_image_resized_batch = real_image_resized.unsqueeze(0).repeat(batch_size, 1, 1, 1)

        with torch.no_grad():
            for i in tqdm(range(0, len(self.particles), batch_size),
                          desc=f"Rendering Particles (Iteration {self.filter_iteration})"):
                torch.cuda.empty_cache()
                batch_cams = [self.particles[j]["cam"] for j in range(i, min(i + batch_size, len(self.particles)))]

                renderings, depths, uncert_maps = [], [], []
                for cam in batch_cams:
                    if USE_UNCERTAINTY:
                        rendering_rgb, rendering_depth, uncertanity_map, _ = self.render_particle_with_uncertainty(cam)
                        uncert_maps.append(uncertanity_map)
                    else:
                        rendering_rgb, rendering_depth = self.render_particle_with_nerf_features(cam)
                    renderings.append(rendering_rgb)
                    depths.append(rendering_depth)

                renderings = torch.stack(renderings)
                depths = torch.stack(depths)
                rgb_renderings = renderings[:, :3, :, :]

                current_batch_size = renderings.size(0)
                if current_batch_size != batch_size:
                    real_image_resized_batch = real_image_resized.unsqueeze(0).repeat(current_batch_size, 1, 1, 1)

                estimate_pose_fnc = cpr_unc if USE_UNCERTAINTY and self.filter_iteration >= self.args.num_iterations - 1 else cpr_batch
                start_time = time.time()
                pose2to1_batch = estimate_pose_fnc(
                    self.mast3r_model,
                    self.device,
                    img_reso=self._cpr_img_reso(),
                    rendered_tensor=rgb_renderings,
                    real_tensor=real_image_resized_batch,
                    rendered_depth_map_batch=depths,
                    rendering_cameras=self.particles,
                    original_size=self.img_size,
                    reprojection_error=self.cfg.reprojection_error,
                    output_folder=None,
                    viz_matches=0,
                    viz_3d_points=False,
                    uncertanity_map=uncert_maps if USE_UNCERTAINTY else None,
                    **CPR_DEFAULTS,
                )
                print(f"CPR time: {time.time() - start_time:.4f}s")

                for batch_idx, _ in enumerate(renderings):
                    particle_idx = i + batch_idx
                    pose2to1 = pose2to1_batch[particle_idx]
                    self.particles[particle_idx]["cam"].update_matrices(absolute_pose=pose2to1['refined_pose_c2w'])

                    if self.filter_iteration < self.args.num_iterations - 1:
                        conf_score = float(pose2to1['matched_confidences'].sum().detach().cpu().item())
                    else:
                        updated_rendering_rgb, _ = self.render_particle_with_nerf_features(self.particles[particle_idx]["cam"])
                        real_for_ssim = real_image_resized.to(torch.float32).unsqueeze(0)
                        conf_score = torch_ssim(real_for_ssim, updated_rendering_rgb.unsqueeze(0), data_range=1.0).item() ** 4

                    if np.isnan(conf_score) or np.isinf(conf_score) or conf_score <= 0:
                        conf_score = 0.00001
                    scores.append(conf_score)

                scores_array = np.maximum(np.nan_to_num(np.array(scores), nan=0.001, posinf=0.001, neginf=0.001), 1e-6)
                likelihoods.extend(scores_array)
                for batch_idx, score in enumerate(scores):
                    self.particles[i + batch_idx]["likelihood"] = score

                del rgb_renderings, renderings, depths, pose2to1_batch
                if USE_UNCERTAINTY:
                    del uncert_maps
                scores.clear()
                torch.cuda.empty_cache()

        del real_image_resized_batch
        torch.cuda.empty_cache()
        return np.array(likelihoods)

    def resample_particles(self, likelihoods):
        likelihoods /= np.sum(likelihoods)
        indices = np.random.choice(range(self.num_particles), size=self.num_particles, p=likelihoods)
        self.particles = [self.particles[i] for i in indices]
        for particle in self.particles:
            particle["weight"] = 1.0 / self.num_particles
        print(f"resample_particles: {self.filter_iteration}")
        self.apply_motion_noise()

    def apply_motion_noise(self):
        pertub_std = np.array([
            self.args.pertub_std_pos, self.args.pertub_std_pos, self.args.pertub_std_pos,
            self.args.pertub_std_rot, self.args.pertub_std_rot, self.args.pertub_std_rot,
        ])
        translation_noise = np.random.uniform(-1, 1, size=(len(self.particles), 3)) * pertub_std[:3]
        rotation_noise = np.random.uniform(-1, 1, size=(len(self.particles), 3)) * pertub_std[3:]
        print(f'translation_noise={translation_noise[0]}')
        print(f'rotation_noise={rotation_noise[0]}')

        new_particles = []
        for i, particle in enumerate(self.particles):
            particle_pose_c2w = (
                trans_t_xyz(translation_noise[i][0], translation_noise[i][1], translation_noise[i][2]) @
                rot_phi(rotation_noise[i][0] / 180. * np.pi) @
                rot_theta(rotation_noise[i][1] / 180. * np.pi) @
                rot_psi(rotation_noise[i][2] / 180. * np.pi) @
                particle["cam"].c2w
            )
            new_particles.append({
                "cam": _copy_and_perturb_camera(particle["cam"], particle_pose_c2w),
                "weight": particle["weight"],
                "likelihood": particle["likelihood"],
            })
        self.particles = new_particles

    def estimate_pose(self, best_particle=True):
        translations, rotations, weights = [], [], []
        for particle in self.particles:
            translations.append(particle["cam"].camera_center.cpu().numpy())
            rotations.append(particle["cam"].R)
            weights.append(particle["likelihood"])

        translations = np.array(translations)
        rotations = np.array(rotations)
        weights = np.array(weights)
        best = self.particles[np.argmax(weights)]

        if self.localization_params.use_weighted_average:
            estimated_T = np.average(translations, axis=0, weights=weights)
            estimated_R = self._average_rotations(rotations, weights)
        else:
            estimated_T = np.mean(translations, axis=0)
            estimated_R = self._average_rotations(rotations)

        if best_particle:
            return best["cam"].camera_center.cpu().numpy(), best["cam"].R
        return estimated_T, estimated_R

    def _average_rotations(self, rotations, weights=None):
        from scipy.spatial.transform import Rotation as R_scipy

        quaternions = []
        for rot in rotations:
            U, _, Vt = np.linalg.svd(rot)
            if np.linalg.det(U @ Vt) < 0:
                Vt[-1, :] *= -1
            try:
                quaternions.append(R_scipy.from_matrix(U @ Vt).as_quat())
            except Exception as e:
                print(f"Warning: Invalid rotation matrix, using identity: {e}")
                quaternions.append([0, 0, 0, 1])

        quaternions = np.array(quaternions)
        avg_quat = np.average(quaternions, axis=0, weights=weights) if weights is not None else np.mean(quaternions, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)
        try:
            return R_scipy.from_quat(avg_quat).as_matrix()
        except Exception as e:
            print(f"Warning: Failed to convert quaternion to matrix, using identity: {e}")
            return np.eye(3)

    def compute_pose_errors(self, estimated_T, estimated_R, gt_pose):
        trans_error = np.linalg.norm(estimated_T - gt_pose.camera_center.cpu().numpy())
        try:
            r_err = cv2.Rodrigues(np.matmul(estimated_R, gt_pose.R.T))[0]
            rot_error = np.linalg.norm(r_err) * 180 / np.pi
            if np.isnan(rot_error) or np.isinf(rot_error):
                rot_error = 0.0
        except Exception as e:
            print(f"Error computing rotation error: {e}")
            rot_error = 0.0
        return trans_error, rot_error

    def update(self):
        likelihoods = self.render_particle_views_and_compute_likelihoods()
        estimated_T, estimated_R = self.estimate_pose(best_particle=self.args.best_particle)
        trans_error, rot_error = self.compute_pose_errors(estimated_T, estimated_R, self.test_cam)
        if self.filter_iteration < self.args.num_iterations - 1:
            self.resample_particles(likelihoods)
        self.filter_iteration += 1
        return estimated_T, estimated_R, trans_error, rot_error


def _create_gaussians(cfg: LocConfig, dataset: ModelParams, original_gs: bool):
    if original_gs and cfg.supports_original_gs:
        return GaussianModelOriginal(dataset.sh_degree)
    gaussians = GaussianModel(
        dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth,
        dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank,
        dataset.appearance_dim, dataset.ratio, dataset.add_opacity_dist,
        dataset.add_cov_dist, dataset.add_color_dist,
    )
    gaussians.eval()
    return gaussians


def run_loc(cfg: LocConfig, dataset: ModelParams, iteration: int, pipeline: PipelineParams,
            locparam: LocalizationParams, scene_name: str, test_all_cams: bool, shuffle: bool, args: Namespace):
    gaussians = _create_gaussians(cfg, dataset, args.original_gs)
    scene = SceneAnchor(
        dataset, gaussians, load_iteration=iteration, shuffle=False, dataset_type=cfg.dataset_type, original_gs=args.original_gs,
    )

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    render_kwargs_test = None
    appearance_args = None
    if cfg.use_appearance and not getattr(args, 'no_appearance', False):
        appearance_args = get_appearance_args(scene_name, args.ft_path)
        _, render_kwargs_test, _, _, _ = create_nerf(appearance_args)
        if render_kwargs_test is not None and render_kwargs_test.get('network_fn') is not None:
            render_kwargs_test['network_fn'].eval()
            for p in render_kwargs_test['network_fn'].parameters():
                p.requires_grad_(False)
            render_kwargs_test['appearance_args'] = appearance_args

    if cfg.test_source == 'colmap':
        len_test_cams = len(scene.test_cameras[1.0])
    else:
        len_test_cams = len(scene.test_info["test_rgb_files"]) if scene.test_info else 0
    print(f"len_test_cams: {len_test_cams}")

    if test_all_cams:
        test_cam_indices = np.random.permutation(list(range(len_test_cams))) if shuffle else list(range(len_test_cams))
    else:
        test_cam_indices = np.random.randint(0, len_test_cams, size=locparam.num_test_cams)

    if getattr(args, 'test_cams_index', None) is not None:
        test_cam_indices = args.test_cams_index

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_save_dir = os.path.join(args.main_output_dir, args.output_dir, f'{timestamp}_{scene_name}')
    os.makedirs(main_save_dir, exist_ok=True)
    results_file = os.path.join(main_save_dir, 'localization_results.txt')
    initialize_results_file(results_file, timestamp, locparam, args, test_cam_indices)

    print('Loading MAst3r...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mast3r_model = AsymmetricMASt3R.from_pretrained(
        "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric").to(device)
    mast3r_model.eval()
    for p in mast3r_model.parameters():
        p.requires_grad_(False)

    trans_error_list, rot_error_list, time_per_camera_list, coarse_cam_error_list = [], [], [], []
    total_start_time = time.time()

    for test_cam_index in test_cam_indices:
        pf = UGSLoc(
            cfg, args.num_particles, scene, gaussians, pipeline, background, locparam, args,
            dataset.model_path, mast3r_model, render_kwargs_test=render_kwargs_test,
            test_cam_index=test_cam_index, save_dir=main_save_dir,
        )
        start_time = time.time()
        for _ in range(args.num_iterations):
            estimated_T, estimated_R, trans_error, rot_error = pf.update()
            print(f"Iteration {pf.filter_iteration - 1}: Estimated Pose: {estimated_T}")
            print(f"\033[91mTranslation Error: {trans_error:.4f}m, Rotation Error: {rot_error:.4f}°\033[0m")

        trans_error_list.append(trans_error)
        rot_error_list.append(rot_error)
        camera_time = time.time() - start_time
        time_per_camera_list.append(camera_time)
        print(f"Time taken for camera {test_cam_index}: {camera_time:.2f} seconds")
        save_camera_result(results_file, test_cam_index, trans_error, rot_error, camera_time,
                           pf.coarse_cam_error_trans, pf.coarse_cam_error_rot)
        coarse_cam_error_list.append([pf.coarse_cam_error_trans, pf.coarse_cam_error_rot])

        if cfg.test_source == 'colmap':
            del pf
            plt.close('all')
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass

    total_end_time = time.time()
    success_rates = calculate_success_rates(trans_error_list, rot_error_list)
    print_success_rates(trans_error_list, rot_error_list, success_rates)
    save_final_results(results_file, trans_error_list, rot_error_list, time_per_camera_list,
                       total_start_time, total_end_time, success_rates, coarse_cam_error_list)
    print(f"\033[96mResults saved to: {results_file}\033[0m")


def run_loc_main(cfg: LocConfig):
    parser, model, pipeline, localization, _ = build_loc_parser(cfg.description, cfg.loc_params_cls)
    args = get_combined_args(parser)
    run_loc(cfg, model.extract(args), args.iteration, pipeline.extract(args),
            localization.extract(args), args.scene_name, args.test_all_cams, args.shuffle, args)


DATASET_CFGS = {
    'cambridge': CAMBRIDGE_CFG,
    '7scenes': SEVENSCENES_CFG,
}


if __name__ == "__main__":
    import argparse
    import sys

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--dataset', choices=list(DATASET_CFGS), default='cambridge')
    pre_args, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    run_loc_main(DATASET_CFGS[pre_args.dataset])
