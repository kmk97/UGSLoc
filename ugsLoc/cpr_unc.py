from dust3r.utils.image import load_images
from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from mast3r.fast_nn import fast_reciprocal_NNs
from utils.graphics_utils import fov2focal
import torch
import torch.nn.functional as F
import torchvision.transforms as tvf
import numpy as np
import PIL.Image
import cv2
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import os
from mpl_toolkits.mplot3d import Axes3D
from PIL import ImageDraw, ImageFont
import torchvision
import time 
from scene.cameras import VirtualCamera2
from utils.graphics_utils import qvec2rotmat, focal2fov
from transformers import AutoImageProcessor, AutoModel
# from transformers import LoFTRFeatureExtractor, LoFTRModel
# Image normalization used by dust3r
ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

"""
Usage example for 3D point cloud visualization:

# Example 1: Visualize depth map as 3D point cloud
depth_map = torch.randn(480, 640)  # Your depth map
camera_intrinsics = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
camera_pose = np.eye(4)  # Identity pose
visualize_depth_as_3d_points(depth_map, camera_intrinsics, camera_pose)

# Example 2: Use with MASt3R pose estimation
results = estimate_pose_cpr_mast3r(
    mast3r_model, device, img_reso=512,
    rendered_tensor=rendered_img, real_tensor=real_img,
    rendered_depth_map=depth_map, rendering_camera=camera,
    viz_3d_points=True  # Enable 3D visualization
)

# Example 3: Visualize 3D points directly
points_3d = create_3d_points_from_depth(depth_map, camera_intrinsics, camera_pose)
visualize_3d_points(points_3d, output_folder="./output")
"""

import numpy as np
import cv2, os, time, torch
import matplotlib.pyplot as plt

try:
    from scipy.optimize import least_squares
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _resize_pil_image(img, long_edge_size):
    S = max(img.size)
    if S > long_edge_size:
        interp = PIL.Image.LANCZOS
    elif S <= long_edge_size:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*long_edge_size/S)) for x in img.size)
    return img.resize(new_size, interp)

def tensor_to_dust3r_format(tensor, idx, img_reso=512, square_ok=False):
    """Convert tensor to dust3r image format"""
    # Assume tensor is in format [C, H, W] with values in [0, 1]
    if tensor.dim() == 4:  # [B, C, H, W] -> [C, H, W]
        tensor = tensor.squeeze(0)
    
    # Convert to PIL Image for processing
    tensor_cpu = tensor.cpu()
    # Convert from [C, H, W] to [H, W, C] and scale to [0, 255]
    img_np = tensor_cpu.permute(1, 2, 0).numpy()
    if img_np.max() <= 1.0:  # If normalized to [0,1], scale to [0,255]
        img_np = img_np * 255
    img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    img_pil = PIL.Image.fromarray(img_np)
    
    # Apply the exact same resizing logic as load_images
    W1, H1 = img_pil.size
    if img_reso == 224:
        # resize short side to 224 (then crop)
        img_pil = _resize_pil_image(img_pil, round(img_reso * max(W1/H1, H1/W1)))
    else:
        # resize long side to 512
        img_pil = _resize_pil_image(img_pil, img_reso)
    W, H = img_pil.size
    cx, cy = W//2, H//2
    if img_reso == 224:
        half = min(cx, cy)
        img_pil = img_pil.crop((cx-half, cy-half, cx+half, cy+half))
    else:
        halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
        if not (square_ok) and W == H:
            halfh = 3*halfw/4
        img_pil = img_pil.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    # Apply ImgNorm transformation
    img_normalized = ImgNorm(img_pil)
    
    return dict(img=img_normalized[None], true_shape=np.int32([img_pil.size[::-1]]), idx=idx, instance=str(idx))

def tensor_to_dust3r_format_batch_new(tensor, start_idx=0, img_reso=512, square_ok=False):
    """Convert batch tensor to dust3r image format"""
    # Assume tensor is in format [B, C, H, W] with values in [0, 1]
    if tensor.dim() == 3:  # [C, H, W] -> [1, C, H, W]
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")
    
    batch_size = tensor.shape[0]
    results = []
    for b in range(batch_size):
        img = tensor[b]  # [C, H, W]
        if img.dtype != torch.float32:
            img = img.float()
        C, H, W = img.shape
        # Compute resize target while preserving aspect ratio
        if img_reso == 224:
            short_side = min(H, W)
            scale = 224.0 / float(short_side) if short_side > 0 else 1.0
            new_h = max(1, int(round(H * scale)))
            new_w = max(1, int(round(W * scale)))
        else:
            long_side = max(H, W)
            scale = float(img_reso) / float(long_side) if long_side > 0 else 1.0
            new_h = max(1, int(round(H * scale)))
            new_w = max(1, int(round(W * scale)))
        # Resize (bicubic, antialiasing)
        resized = F.interpolate(
            img.unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True
        ).squeeze(0)
        Hr, Wr = resized.shape[1], resized.shape[2]
        cx, cy = Wr // 2, Hr // 2
        if img_reso == 224:
            half = min(cx, cy)
            x1, x2 = cx - half, cx + half
            y1, y2 = cy - half, cy + half
        else:
            halfw = ((2 * cx) // 16) * 8
            halfh = ((2 * cy) // 16) * 8
            if (not square_ok) and (Wr == Hr):
                halfh = int(3 * halfw / 4)
            # clamp to valid range
            halfw = max(1, min(halfw, cx))
            halfh = max(1, min(halfh, cy))
            x1, x2 = cx - halfw, cx + halfw
            y1, y2 = cy - halfh, cy + halfh
        cropped = resized[:, y1:y2, x1:x2]
        # Support inputs in [0,1] or [0,255]
        max_val = float(img.max())
        if max_val <= 1.0 + 1e-6:
            img_normalized = (cropped - 0.5) / 0.5
        else:
            img_normalized = (cropped / 255.0 - 0.5) / 0.5
        img_normalized = img_normalized
        result_dict = dict(
            img=img_normalized[None],
            true_shape=np.int32([ [cropped.shape[1], cropped.shape[2]] ]),
            idx=start_idx + b,
            instance=str(start_idx + b)
        )
        results.append(result_dict)
    return results

def tensor_to_dust3r_format_batch(tensor, start_idx=0, img_reso=512, square_ok=False):
    """Convert batch tensor to dust3r image format"""
    # Assume tensor is in format [B, C, H, W] with values in [0, 1]
    if tensor.dim() == 3:  # [C, H, W] -> [1, C, H, W]
        tensor = tensor.unsqueeze(0)
    elif tensor.dim() != 4:
        raise ValueError(f"Expected tensor with 3 or 4 dimensions, got {tensor.dim()}")
    
    batch_size = tensor.shape[0]
    results = []
    
    for b in range(batch_size):
        # Extract single image from batch
        single_tensor = tensor[b]  # [C, H, W]
        
        # Convert to PIL Image for processing
        tensor_cpu = single_tensor.cpu()
        # Convert from [C, H, W] to [H, W, C] and scale to [0, 255]
        img_np = tensor_cpu.permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:  # If normalized to [0,1], scale to [0,255]
            img_np = img_np * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        img_pil = PIL.Image.fromarray(img_np)
        
        # Apply the exact same resizing logic as load_images
        W1, H1 = img_pil.size
        if img_reso == 224:
            # resize short side to 224 (then crop)
            img_pil = _resize_pil_image(img_pil, round(img_reso * max(W1/H1, H1/W1)))
        else:
            # resize long side to 512
            img_pil = _resize_pil_image(img_pil, img_reso)
        W, H = img_pil.size
        cx, cy = W//2, H//2
        if img_reso == 224:
            half = min(cx, cy)
            img_pil = img_pil.crop((cx-half, cy-half, cx+half, cy+half))
        else:
            halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
            if not (square_ok) and W == H:
                halfh = 3*halfw/4
            img_pil = img_pil.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
        
        # Apply ImgNorm transformation
        img_normalized = ImgNorm(img_pil)
        
        # Create result dict for this image
        result_dict = dict(
            img=img_normalized[None], 
            true_shape=np.int32([img_pil.size[::-1]]), 
            idx=start_idx + b, 
            instance=str(start_idx + b)
        )
        results.append(result_dict)
    
    return results
    
def cpr_unc(
    mast3r_model, device, img_reso, rendered_tensor=None, real_tensor=None, 
    rendered_depth_map_batch=None, rendering_cameras=None, real_camera=None,
    original_size=(1080, 1920), reprojection_error=2.5, output_folder=None,
    viz_matches=0, viz_3d_points=False, uncertanity_map=None,
    weighted_refine=False, robust_loss='soft_l1',uncertanity_threshold=1, new_uncertainty=False,
    alpha=3.0, sigma_min=1e-3, clip_range=1e3, use_cropping=False, crop_query=False,
    weighted_epnp=False, max_weighted_pts=512, use_ransac=True,
    uncertainty_only_epnp=False, iterative_pnp=False, iterative_steps=3,
    importance_ransac=False, ransac_importance_iters=1000, ransac_importance_sample_size=6,
    importance_alpha=1.0, lo_refine=True,
    max_pnp_matches=4096, verbose=True,
    num_workers=4
):
    """
     CPR  pose refinement using MASt3R (following GS-CPR paper approach)
    - RANSAC PnP () +    ()

    Args:
        ...
        uncertanity_map: (H,W)  list[(H,W)] /.   (↑ =  ).
        weighted_refine: True RANSAC  per-match σ   (Scipy ).
        robust_loss: 'huber'/'soft_l1'/'cauchy'  None

    Returns:
        list of dicts (    )
    """
    def _tensor_to_dust3r_format_batch(t, idx, img_reso):
        return tensor_to_dust3r_format_batch_new(t, idx, img_reso)

    def _project_points(Xw, rvec, tvec, K, dist=None):
        x_proj, _ = cv2.projectPoints(Xw, rvec, tvec, K, dist)
        return x_proj.reshape(-1, 2)

    def _chol_inv_sqrt_2x2(cov2x2):
        L = np.linalg.cholesky(cov2x2)
        return np.linalg.inv(L.T)

    def _weighted_residuals(params, Xw, uv, K, dist, sigmas=None, covs=None):
        rvec = params[:3].reshape(3, 1)
        tvec = params[3:].reshape(3, 1)
        uv_hat = _project_points(Xw, rvec, tvec, K, dist)
        res = []
        for k, (e_uv, uhat) in enumerate(zip(uv, uv_hat)):
            e = (e_uv - uhat)  # 2D residual in pixels
            if covs is not None:
                Winvsqrt = _chol_inv_sqrt_2x2(covs[k])
                e = Winvsqrt @ e
            elif sigmas is not None:
                s = max(1e-6, sigmas[k])
                e = e / s
            res.extend(e.tolist())
        return np.array(res)

    def _refine_pnp_weighted(Xw, uv, K, rvec0, tvec0, dist=None, sigmas=None, covs=None, loss='huber'):
        if not _HAS_SCIPY:
            return rvec0, tvec0, None
        x0 = np.hstack([rvec0.ravel(), tvec0.ravel()])
        fun = lambda x: _weighted_residuals(x, Xw, uv, K, dist, sigmas=sigmas, covs=covs)
        opt = least_squares(fun, x0, method='trf', loss=(loss if loss else 'linear'),
                            f_scale=1.0, max_nfev=60) #200)
        rvec = opt.x[:3].reshape(3, 1)
        tvec = opt.x[3:].reshape(3, 1)
        return rvec, tvec, opt

    def _get_uncert_map_for_index(unc_map, idx_b):
        if unc_map is None:
            return None
        if isinstance(unc_map, (list, tuple)):
            return unc_map[idx_b]
        return unc_map

    def _to_numpy_img(tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        img_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
        if img_np.max() <= 1.0:
            img_np = img_np * 255
        return np.clip(img_np, 0, 255).astype(np.uint8)

    def _ensure_numpy_2d(x):
        if torch.is_tensor(x):
            x = x.detach().cpu().squeeze()
            x = x.numpy()
        else:
            x = np.array(x)
        if x.ndim == 3:
            x = x.squeeze()
        return x

    def _find_low_uncertainty_crop_box(unc, crop_h, crop_w):
        # unc is a CUDA tensor, get shape
        H, W = unc.shape
        ch = min(int(crop_h), H)
        cw = min(int(crop_w), W)

        # Create kernel tensor on same device as unc
        kernel = torch.ones((1, 1, ch, cw), device=unc.device) / (ch * cw)

        # Add batch and channel dims for conv2d
        unc_4d = unc.unsqueeze(0).unsqueeze(0)

        # Apply convolution without padding (valid), so output indices map to top-left of window
        avg_valid = torch.nn.functional.conv2d(unc_4d, kernel, padding=0).squeeze()

        # Find min location
        min_idx = torch.argmin(avg_valid.view(-1))
        cy = int((min_idx // avg_valid.shape[1]).item())
        cx = int((min_idx % avg_valid.shape[1]).item())

        # Crop coordinates are the valid top-left indices
        x0 = int(cx)
        y0 = int(cy)

        return x0, y0, cw, ch

    def _crop_CHW(t, box):
        x0, y0, w, h = box
        return t[:, y0:y0+h, x0:x0+w]

    def _solve_pnp_epnp(Xw_s, uv_s, K, dist, rvec0, tvec0):
        try:
            ok, rvec_s, tvec_s = cv2.solvePnP(
                Xw_s, uv_s,
                K, dist,
                rvec=rvec0, tvec=tvec0,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP
            )
            return ok, rvec_s, tvec_s
        except Exception:
            return False, None, None

    def _reproj_errors_sq(Xw_all, uv_all, K, dist, rvec, tvec):
        uv_hat = _project_points(Xw_all, rvec, tvec, K, dist)
        diff = uv_all - uv_hat
        return (diff * diff).sum(axis=1)

    def _ransac_pnp_importance(Xw_all, uv_all, K, dist, rvec0, tvec0,
                               weights=None, iters=1000, sample_size=6,
                               thr=3.0, alpha=1.0, lo=True, confidence=0.99):
        Xw_all = np.asarray(Xw_all, dtype=np.float32)
        uv_all = np.asarray(uv_all, dtype=np.float32)
        K = np.asarray(K, dtype=np.float32)
        dist = np.asarray(dist, dtype=np.float32)
        rvec0 = np.asarray(rvec0, dtype=np.float32)
        tvec0 = np.asarray(tvec0, dtype=np.float32)
        N = Xw_all.shape[0]
        if N < max(4, sample_size):
            return False, None, None, None
        probs = None
        weights_f64 = None
        if weights is not None:
            w = np.asarray(weights, dtype=np.float64).clip(min=0)
            if alpha != 1.0:
                w = np.power(w + 1e-12, alpha)
            s = w.sum()
            probs = w / s if s > 0 else None
            weights_f64 = w

        best_score = -1.0
        best = (None, None, None)
        rng = np.random.default_rng()
        best_inlier_ratio = 0.0
        s_min = sample_size
        thr2 = float(thr) * float(thr)
        for i in range(int(max(1, iters))):
            try:
                if probs is None:
                    samp = rng.choice(N, size=sample_size, replace=False)
                else:
                    samp = rng.choice(N, size=sample_size, replace=False, p=probs)
            except ValueError:
                # fallback uniform
                samp = rng.choice(N, size=sample_size, replace=False)
            ok, rvec_h, tvec_h = _solve_pnp_epnp(Xw_all[samp], uv_all[samp], K, dist, rvec0, tvec0)
            if not ok:
                continue
            errs2 = _reproj_errors_sq(Xw_all, uv_all, K, dist, rvec_h, tvec_h)
            inl = errs2 <= thr2
            inl_sum = int(inl.sum())
            if weights_f64 is not None:
                score = (weights_f64[inl]).sum()
            else:
                score = float(inl_sum)
            if score > best_score and inl_sum >= 4:
                best_score = score
                best = (rvec_h, tvec_h, inl)

                if probs is None:
                    best_inlier_ratio = inl_sum / float(N)
                else:
                    best_inlier_ratio = probs[inl].sum()

            if best_inlier_ratio > 0:
                denom = max(1e-12, 1 - best_inlier_ratio ** s_min)
                N_req = np.log(1 - confidence) / np.log(denom)
                if i >= N_req:
                    break

        if best[0] is None:
            return False, None, None, None

        rvec_b, tvec_b, inl = best
        if lo and inl.sum() >= 4:
            try:
                ok, rvec_b, tvec_b = cv2.solvePnP(
                    Xw_all[inl], uv_all[inl],
                    K, dist,
                    rvec=rvec_b, tvec=tvec_b,
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not ok:
                    return False, None, None, None
            except Exception:
                return False, None, None, None
        inliers_idx = np.where(inl)[0].reshape(-1, 1).astype(np.int32)
        return True, rvec_b, tvec_b, inliers_idx

    B_in = rendered_tensor.shape[0]
    crop_boxes = []
    if use_cropping:
        img1_dict = []
        img2_dict = []
        # Fixed crop size requested: [H, W] = [384, 512]
        crop_h, crop_w = 384, 512
        for b in range(B_in):
            unc_b = uncertanity_map[b] if isinstance(uncertanity_map, (list, tuple)) else uncertanity_map
            # unc_b = _ensure_numpy_2d(unc_b)
            x0, y0, w, h = _find_low_uncertainty_crop_box(unc_b, crop_h, crop_w)  # what does w,h means? crop size
            crop_boxes.append((x0, y0, w, h))
            rend_crop = _crop_CHW(rendered_tensor[b], (x0, y0, w, h))
            img1_dict.append(tensor_to_dust3r_format(rend_crop, b, img_reso))
            # Optionally crop real_tensor (query image) with the SAME crop box
            if crop_query:
                real_crop = _crop_CHW(real_tensor[b], (x0, y0, w, h))
                img2_dict.append(tensor_to_dust3r_format(real_crop, b, img_reso))
            else:
                # Do not crop real_tensor; pass the full real image for MASt3R
                img2_dict.append(tensor_to_dust3r_format(real_tensor[b], b, img_reso))
        images_pair_list = [tuple([a, b]) for a, b in zip(img1_dict, img2_dict)]
    else:
        img1_dict = _tensor_to_dust3r_format_batch(rendered_tensor, 0, img_reso)
        img2_dict = _tensor_to_dust3r_format_batch(real_tensor,     1, img_reso)
        images_pair_list = [tuple([a, b]) for a, b in zip(img1_dict, img2_dict)]

    start_time = time.time()
    with torch.inference_mode():
        output = inference(images_pair_list, mast3r_model, device, batch_size=8, verbose=False)
    inference_time = time.time() - start_time
    if verbose:
        print(f"MASt3R inference time: {inference_time:.4f} seconds")

    view1, pred1 = output['view1'], output['pred1']
    view2, pred2 = output['view2'], output['pred2']
    desc1, desc2 = pred1['desc'].detach(), pred2['desc'].detach()

    results = []
    start_time = time.time()

    B = desc1.shape[0]

    def process_view(b: int):
        t0 = time.time()
        matches_im0, matches_im1 = fast_reciprocal_NNs(
            desc1[b], desc2[b], subsample_or_initxy1=8, device=device, dist='dot', block_size=2**13
        )
        fnn_time = time.time() - t0
        if verbose:
            print(f"FNN time: {fnn_time:.4f} seconds")

        H0, W0 = int(view1['true_shape'][b][0]), int(view1['true_shape'][b][1])
        H1, W1 = int(view2['true_shape'][b][0]), int(view2['true_shape'][b][1])

        valid0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < W0 - 3) & (matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < H0 - 3)
        valid1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < W1 - 3) & (matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < H1 - 3)
        valid = valid0 & valid1
        matches_im0 = matches_im0[valid]
        matches_im1 = matches_im1[valid]
        rows = matches_im0[:, 0].astype(int)
        cols = matches_im0[:, 1].astype(int)
        desc_conf_b = pred1['desc_conf'][b]
        if desc_conf_b.dim() == 3:
            conf_map = desc_conf_b
        else:
            conf_map = desc_conf_b.squeeze(0)
        match_confidences = conf_map[cols, rows]  # (N,)
        matched_conf = match_confidences.mean().item() if match_confidences.numel() > 0 else 0.0
        match_count = len(matches_im0)
        matched_conf_sum = match_confidences.sum().item() if match_confidences.numel() > 0 else 0.0

        if max_pnp_matches is not None and max_pnp_matches > 0 and match_confidences.numel() > max_pnp_matches:
            conf_np = match_confidences.detach().cpu().numpy()
            top_k = max(4, int(max_pnp_matches))
            top_idx = np.argpartition(-conf_np, top_k - 1)[:top_k]
            matches_im0 = matches_im0[top_idx]
            matches_im1 = matches_im1[top_idx]
            match_confidences = match_confidences[top_idx]
            matched_conf = match_confidences.mean().item() if match_confidences.numel() > 0 else 0.0
            match_count = len(matches_im0)
            matched_conf_sum = match_confidences.sum().item() if match_confidences.numel() > 0 else 0.0

        def _to_float_np(x):
            if torch.is_tensor(x): 
                return x.detach().cpu().to(torch.float64).numpy()
            x = np.asarray(x)
            if not np.issubdtype(x.dtype, np.floating):
                x = x.astype(np.float64, copy=True)
            return x

        matches_im0 = _to_float_np(matches_im0)
        matches_im1 = _to_float_np(matches_im1)
        
        if use_cropping:
            x0_c, y0_c, w_c, h_c = crop_boxes[b]
            # When crop size equals MASt3R true_shape (e.g., 384x512), this becomes identity scale
            scale0_x = float(w_c) / float(W0)
            scale0_y = float(h_c) / float(H0)
            matches_im0_full = matches_im0 * np.array([scale0_x, scale0_y], dtype=np.float64)
            matches_im0_full += np.array([x0_c, y0_c], dtype=np.float64)
        else:
            sx0 = original_size[1] / float(W0)
            sy0 = original_size[0] / float(H0)
            matches_im0_full = matches_im0 * np.array([sx0, sy0], dtype=np.float64)

        if use_cropping and crop_query:
            x0_q, y0_q, w_q, h_q = crop_boxes[b]
            scale1_x = float(w_q) / float(W1)
            scale1_y = float(h_q) / float(H1)
            matches_im1_full = matches_im1 * np.array([scale1_x, scale1_y], dtype=np.float64)
            matches_im1_full += np.array([x0_q, y0_q], dtype=np.float64)
        else:
            sx1 = original_size[1] / float(W1)
            sy1 = original_size[0] / float(H1)
            matches_im1_full = matches_im1 * np.array([sx1, sy1], dtype=np.float64)
        
        per_match_sigma = None
        unc_map_b = _get_uncert_map_for_index(uncertanity_map, b)
        if unc_map_b is not None:
            if torch.is_tensor(unc_map_b):
                unc_map_b = unc_map_b.detach().cpu().numpy()
            if matches_im0_full.shape[0] == 0:
                unc_vals = np.zeros((0,), dtype=np.float64)
            else:
                xi = matches_im0_full[:, 0].astype(np.int64)
                yi = matches_im0_full[:, 1].astype(np.int64)
                H_u, W_u = unc_map_b.shape[:2]
                valid = (xi >= 0) & (xi < W_u) & (yi >= 0) & (yi < H_u)
                unc_vals = np.full((matches_im0_full.shape[0],), np.inf, dtype=np.float64)
                unc_vals[valid] = unc_map_b[yi[valid], xi[valid]]
            matched_uncertainties_raw = unc_vals
            matched_uncertainties = unc_vals
            valid_vals = unc_vals[~np.isinf(unc_vals)]
            if not new_uncertainty:
                if len(valid_vals) > 0:
                    min_val = np.min(valid_vals)
                    max_val = np.max(valid_vals)
                    if max_val > min_val:
                        unc_vals_norm = 1/ ((unc_vals - min_val) / (max_val - min_val) + 1e-3)
                    else:
                        unc_vals_norm = np.zeros_like(unc_vals)
                else:
                    unc_vals_norm = np.zeros_like(unc_vals)
            elif (clip_range==1):
                if verbose:
                    print("linear on uncertanity to covariance")
                if len(valid_vals) > 0:
                    lo, hi = np.percentile(valid_vals, [5, 95])
                    unc_vals_norm = (unc_vals - lo) / (hi - lo)
                    unc_vals_norm = np.clip(unc_vals_norm, 0.0, 1.0)
                    unc_vals_norm = 1 + (sigma_min - 1) * unc_vals_norm
                else:
                    unc_vals_norm = np.ones_like(unc_vals)
            else:
                if len(valid_vals) > 0:
                    lo, hi = np.percentile(valid_vals, [5, 95])
                    if hi <= lo:
                        unc_vals_norm = np.zeros_like(unc_vals)
                    else:
                        unc_vals_norm = (unc_vals - lo) / (hi - lo)
                        unc_vals_norm = np.clip(unc_vals_norm, 0.0, 1.0)

                    # sigma_min = 1e-3
                    unc_vals_norm = sigma_min * np.exp(alpha * unc_vals_norm)
                else:
                    unc_vals_norm = np.ones_like(unc_vals)
            
            # Add small epsilon and clip
            
            if unc_vals_norm.size > 0:
                per_match_sigma = np.clip(unc_vals_norm, 1 / clip_range, clip_range) if clip_range != 1 else unc_vals_norm
                if verbose:
                    print(np.nanmin(per_match_sigma), np.nanmax(per_match_sigma), np.median(per_match_sigma))
            else:
                if verbose:
                    print("per_match_sigma is empty.")
                per_match_sigma = np.ones_like(unc_vals_norm)
            # matched_uncertainties_raw = unc_vals
            matched_uncertainties_norm = unc_vals_norm

        if per_match_sigma is None and match_count > 0:
            conf_np = match_confidences.detach().cpu().numpy()
            eps = 1e-3
            per_match_sigma = 1.0 / np.clip(conf_np, eps, None)
            med = np.median(per_match_sigma)
            if med > 0:
                per_match_sigma = per_match_sigma / med
            per_match_sigma = np.clip(per_match_sigma, 1e-3, 1e3)

        rendering_camera = rendering_cameras[b]["cam"]
        FoVx, FoVy = rendering_camera.FoVx, rendering_camera.FoVy
        width, height = rendering_camera.image_width, rendering_camera.image_height
        fx = fov2focal(FoVx, width);  fy = fov2focal(FoVy, height)
        cx, cy = width / 2.0, height / 2.0
        K = np.array([[fx, 0, cx],
                      [0,  fy, cy],
                      [0,   0,  1]], dtype=np.float64)

        initial_pose_c2w = rendering_camera.c2w  # (4,4)
        depth = rendered_depth_map_batch[b]
        if depth.dim() == 3: depth = depth.squeeze(0)
        if torch.is_tensor(depth): depth = depth.detach().cpu().numpy()
        depth = depth.astype(np.float64, copy=False)

        n_m = matches_im0_full.shape[0]
        points_3D_at_pixels = np.zeros((n_m, 3), dtype=np.float64)
        if n_m > 0:
            xi = matches_im0_full[:, 0].astype(np.int64)
            yi = matches_im0_full[:, 1].astype(np.int64)
            H_d, W_d = depth.shape
            valid = (xi >= 0) & (xi < W_d) & (yi >= 0) & (yi < H_d)
            if np.any(valid):
                depth_vals = depth[yi[valid], xi[valid]]
                x_norm = (xi[valid] - cx) / fx
                y_norm = (yi[valid] - cy) / fy
                Xc = depth_vals * x_norm
                Yc = depth_vals * y_norm
                Zc = depth_vals
                pts_cam = np.stack([Xc, Yc, Zc], axis=1)
                R_c2w = initial_pose_c2w[:3, :3]
                t_c2w = initial_pose_c2w[:3, 3]
                pts_w = pts_cam @ R_c2w.T + t_c2w
                points_3D_at_pixels[valid] = pts_w

        refined_pose_c2w = initial_pose_c2w.copy()
        success = False
        rvec, tvec, inliers = None, None, None

        if matches_im1_full.shape[0] >= 4:
            R0 = initial_pose_c2w[:3, :3].astype(np.float64)
            t0 = initial_pose_c2w[:3, 3].astype(np.float64).reshape(3, 1)
            rvec0, _ = cv2.Rodrigues(R0)

            dist_coeffs = np.zeros(4, dtype=np.float64)
            t_pnp0 = time.time()

            if importance_ransac:
                # Use importance-sampling RANSAC with weights from uncertainty (or confidence fallback)
                if 'per_match_sigma' in locals() and per_match_sigma is not None:
                    w_all = 1.0 / (np.asarray(per_match_sigma, dtype=np.float64) ** 2 + 1e-8)
                else:
                    conf_np = match_confidences.detach().cpu().numpy() if match_count > 0 else np.zeros(matches_im1_full.shape[0])
                    w_all = np.asarray(conf_np, dtype=np.float64) + 1e-8
                w_all = np.nan_to_num(w_all, nan=0.0, posinf=0.0, neginf=0.0)
                success, rvec, tvec, inliers = _ransac_pnp_importance(
                    points_3D_at_pixels.astype(np.float64),
                    matches_im1_full.astype(np.float64),
                    K.astype(np.float64), dist_coeffs.astype(np.float64),
                    rvec0.astype(np.float64), t0.astype(np.float64),
                    weights=w_all, iters=ransac_importance_iters,
                    sample_size=ransac_importance_sample_size,
                    thr=float(reprojection_error), alpha=float(importance_alpha),
                    lo=bool(lo_refine)
                )
            elif (not use_ransac) or weighted_epnp:
                # Weighted EPnP path (no RANSAC if use_ransac=False)
                # Build per-match weights strictly from uncertainty if requested
                if ('per_match_sigma' in locals() and per_match_sigma is not None) and (uncertainty_only_epnp or weighted_epnp):
                    w = 1.0 / (np.asarray(per_match_sigma, dtype=np.float64) ** 2 + 1e-8)
                else:
                    # fallback to desc confidence ONLY if not uncertainty_only_epnp
                    if uncertainty_only_epnp:
                        # if no sigma available, treat equally
                        w = np.ones(matches_im1_full.shape[0], dtype=np.float64)
                    else:
                        conf_np = match_confidences.detach().cpu().numpy() if match_count > 0 else np.zeros(matches_im1_full.shape[0])
                        w = np.asarray(conf_np, dtype=np.float64) + 1e-8
                w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
                # Keep same indices length as correspondences
                num_corr = matches_im1_full.shape[0]
                idx_all = np.arange(num_corr)
                # Select top-k by weight
                k = int(min(max_weighted_pts, num_corr))
                if k >= 4:
                    top_idx = np.argpartition(-w, k-1)[:k]
                else:
                    top_idx = idx_all
                Xw_sel = points_3D_at_pixels[top_idx].astype(np.float32)
                uv_sel = matches_im1_full[top_idx].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    Xw_sel, uv_sel, K.astype(np.float32), dist_coeffs.astype(np.float32),
                    rvec=rvec0.astype(np.float32), tvec=t0.astype(np.float32),
                    useExtrinsicGuess=True, flags=cv2.SOLVEPNP_EPNP
                )
                inliers = None
            else:
                print("use cv2.solvePnPRansac")
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    points_3D_at_pixels.astype(np.float32),
                    matches_im1_full.astype(np.float32),
                    K.astype(np.float32),
                    dist_coeffs.astype(np.float32),
                    rvec=rvec0.astype(np.float32),
                    tvec=t0.astype(np.float32),
                    useExtrinsicGuess=True,
                    reprojectionError=float(reprojection_error),
                    iterationsCount=2000,
                    flags=cv2.SOLVEPNP_EPNP
                )
            _ = time.time() - t_pnp0
            if verbose:
                print(f"pnp time: {_}")

            if success:
                # Optional iterative reweighted PnP refinement using uncertainty as weights
                if iterative_pnp and _HAS_SCIPY:
                    if inliers is not None:
                        idx_inl_iter = inliers.ravel().astype(int)
                    else:
                        idx_inl_iter = np.arange(points_3D_at_pixels.shape[0], dtype=np.int64)
                    Xw_iter = points_3D_at_pixels[idx_inl_iter].astype(np.float64)
                    uv_iter = matches_im1_full[idx_inl_iter].astype(np.float64)
                    sig_iter = None
                    if 'per_match_sigma' in locals() and per_match_sigma is not None:
                        sig_all = np.asarray(per_match_sigma, dtype=np.float64)
                        sig_iter = sig_all[idx_inl_iter]
                    for _it in range(max(1, int(iterative_steps))):
                        rvec_ref_i, tvec_ref_i, _ = _refine_pnp_weighted(
                            Xw_iter, uv_iter, K.astype(np.float64),
                            rvec.astype(np.float64), tvec.astype(np.float64),
                            dist=np.zeros(4), sigmas=sig_iter, covs=None, loss=robust_loss
                        )
                        rvec, tvec = rvec_ref_i, tvec_ref_i
                t_weighted_refine = time.time()
                if weighted_refine and _HAS_SCIPY:
                    if inliers is not None:
                        idx_inl = inliers.ravel().astype(int)
                    else:
                        # no RANSAC case: use all or top-k already selected
                        idx_inl = np.arange(points_3D_at_pixels.shape[0], dtype=np.int64)
                    Xw_in = points_3D_at_pixels[idx_inl].astype(np.float64)
                    uv_in = matches_im1_full[idx_inl].astype(np.float64)

                    sigmas = None
                    if per_match_sigma is not None:
                        sigmas = np.asarray(per_match_sigma, dtype=np.float64)[idx_inl]

                    rvec_ref, tvec_ref, _ = _refine_pnp_weighted(
                        Xw_in, uv_in, K.astype(np.float64),
                        rvec.astype(np.float64), tvec.astype(np.float64),
                        dist=np.zeros(4), sigmas=sigmas, covs=None, loss=robust_loss
                    )
                    rvec, tvec = rvec_ref, tvec_ref

                R, _ = cv2.Rodrigues(rvec.astype(np.float64))
                trans = (-R.T @ tvec.reshape(3, 1)).reshape(3)
                refined_pose_c2w = np.eye(4, dtype=np.float64)
                refined_pose_c2w[:3, :3] = R.T
                refined_pose_c2w[:3, 3] = trans
                
                _ = time.time() - t_weighted_refine
                if verbose:
                    print(f"weighted refine time: {_}")
                
        result = {
            'refined_pose_c2w': refined_pose_c2w,
            'initial_pose_c2w': initial_pose_c2w,
            'success': success,
            'num_matches': int(matches_im0_full.shape[0]),
            'matches_rendered': matches_im0_full,
            'matches_real':    matches_im1_full,
            'points_3d': points_3D_at_pixels if matches_im0_full.shape[0] >= 4 else None,
            'inliers': inliers if success else None,
            'conf_map' : desc_conf_b,
            'matched_conf': float(matched_conf),
            'matched_conf_sum': float(matched_conf_sum),
            'matched_confidences': match_confidences,
            'match_count': int(match_count),
            'used_weighted_refine': bool(weighted_refine and _HAS_SCIPY)
        }
        if 'matched_uncertainties_raw' in locals():
            result['matched_uncertainties_raw'] = matched_uncertainties_raw
            result['matched_uncertainties_norm'] = matched_uncertainties_norm
        return result

    # Parallel execution over views
    start_pnp_time = time.time()
    if num_workers is None or num_workers <= 0:
        try:
            num_workers = min(B, os.cpu_count() or 4)
        except Exception:
            num_workers = min(B, 4)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_view, range(B)))
    end_pnp_time = time.time()
    if verbose:
        print(f"PNP time: {end_pnp_time - start_pnp_time:.4f} seconds")

    _ = time.time() - start_time
    return results
