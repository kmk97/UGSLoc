import numpy as np
from scene.cameras import ParticleCamera


def _copy_and_perturb_camera(original_cam, particle_pose_c2w):
    return ParticleCamera(
        c2w=particle_pose_c2w,
        FoVx=original_cam.FoVx,
        FoVy=original_cam.FoVy,
        image_width=original_cam.image_width,
        image_height=original_cam.image_height,
    )


def combine_3dgs_rotation_translation(R_w2c, T_c2w):
    RT_w2c = np.eye(4)
    RT_w2c[:3, :3] = R_w2c.T
    RT_w2c[:3, 3] = T_c2w
    return np.linalg.inv(RT_w2c)


def trans_t_xyz(tx, ty, tz):
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
    ])


rot_psi = lambda phi: np.array([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1],
])

rot_theta = lambda th: np.array([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1],
])

rot_phi = lambda psi: np.array([
    [np.cos(psi), -np.sin(psi), 0, 0],
    [np.sin(psi), np.cos(psi), 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])


def initialize_results_file(results_file, timestamp, locparam, args, test_cam_indices):
    with open(results_file, 'w') as f:
        f.write("=== Particle Filter Localization Results ===\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Number of particles: {args.num_particles}\n")
        f.write(f"Number of iterations per camera: {args.num_iterations}\n")
        f.write(f"Number of test cameras: {len(test_cam_indices)}\n")
        f.write(f"Test camera indices: {test_cam_indices}\n\n")
        f.write("=== Settings ===\n")
        f.write(f"batch_size: {args.batch_size_arg}\n")
        f.write(f"best_particle: {args.best_particle}\n")
        f.write(f"pose_estimator: {args.pose_estimator}\n")
        f.write(f"pertub_std_pos: {args.pertub_std_pos}\n")
        f.write(f"pertub_std_rot: {args.pertub_std_rot}\n\n")
        f.write("=== Individual Camera Results ===\n")


def save_camera_result(results_file, test_cam_index, trans_error, rot_error, camera_time,
                       coarse_cam_error_trans=None, coarse_cam_error_rot=None):
    with open(results_file, 'a') as f:
        if coarse_cam_error_trans is not None and coarse_cam_error_rot is not None:
            f.write(
                f"Camera {test_cam_index}: Translation Error = {trans_error:.4f}m, "
                f"Rotation Error = {rot_error:.4f}°, Time = {camera_time:.2f}s, "
                f"Coarse Cam Error = {coarse_cam_error_trans:.4f}m, {coarse_cam_error_rot:.4f}°\n")
        else:
            f.write(
                f"Camera {test_cam_index}: Translation Error = {trans_error:.4f}m, "
                f"Rotation Error = {rot_error:.4f}°, Time = {camera_time:.2f}s\n")


def calculate_success_rates(trans_error_list, rot_error_list):
    trans_error_array = np.array(trans_error_list)
    rot_error_array = np.array(rot_error_list)
    total_cameras = len(trans_error_list)

    success_5cm = (trans_error_array <= 0.05).sum()
    success_5deg = (rot_error_array <= 5.0).sum()
    success_both_5cm_5deg = ((trans_error_array <= 0.05) & (rot_error_array <= 5.0)).sum()
    success_2cm = (trans_error_array <= 0.02).sum()
    success_2deg = (rot_error_array <= 2.0).sum()
    success_both_2cm_2deg = ((trans_error_array <= 0.02) & (rot_error_array <= 2.0)).sum()
    success_1cm = (trans_error_array <= 0.01).sum()
    success_10cm = (trans_error_array <= 0.10).sum()
    success_1deg = (rot_error_array <= 1.0).sum()
    success_10deg = (rot_error_array <= 10.0).sum()

    return {
        'total_cameras': total_cameras,
        'success_5cm': success_5cm,
        'success_5deg': success_5deg,
        'success_both_5cm_5deg': success_both_5cm_5deg,
        'success_2cm': success_2cm,
        'success_2deg': success_2deg,
        'success_both_2cm_2deg': success_both_2cm_2deg,
        'success_1cm': success_1cm,
        'success_10cm': success_10cm,
        'success_1deg': success_1deg,
        'success_10deg': success_10deg,
        'success_5cm_pct': (success_5cm / total_cameras) * 100,
        'success_5deg_pct': (success_5deg / total_cameras) * 100,
        'success_both_5cm_5deg_pct': (success_both_5cm_5deg / total_cameras) * 100,
        'success_2cm_pct': (success_2cm / total_cameras) * 100,
        'success_2deg_pct': (success_2deg / total_cameras) * 100,
        'success_both_2cm_2deg_pct': (success_both_2cm_2deg / total_cameras) * 100,
        'trans_error_array': trans_error_array,
        'rot_error_array': rot_error_array,
    }


def print_success_rates(trans_error_list, rot_error_list, success_rates):
    print(f"\033[92mTranslation Error: {np.mean(trans_error_list):.4f}m, "
          f"Rotation Error: {np.mean(rot_error_list):.4f}°\033[0m")
    print(f"\033[96mSuccess Rate (≤5cm): {success_rates['success_5cm_pct']:.1f}% "
          f"({success_rates['success_5cm']}/{success_rates['total_cameras']})\033[0m")
    print(f"\033[96mSuccess Rate (≤5°): {success_rates['success_5deg_pct']:.1f}% "
          f"({success_rates['success_5deg']}/{success_rates['total_cameras']})\033[0m")
    print(f"\033[96mSuccess Rate (≤5cm AND ≤5°): {success_rates['success_both_5cm_5deg_pct']:.1f}% "
          f"({success_rates['success_both_5cm_5deg']}/{success_rates['total_cameras']})\033[0m")


def save_final_results(results_file, trans_error_list, rot_error_list, time_per_camera_list,
                       total_start_time, total_end_time, success_rates, coarse_cam_error_list=None):
    with open(results_file, 'a') as f:
        f.write("\n=== Summary Statistics ===\n")
        f.write(f"Mean Translation Error: {np.mean(trans_error_list):.4f}m\n")
        f.write(f"Mean Rotation Error: {np.mean(rot_error_list):.4f}°\n")
        f.write(f"Median Translation Error: {np.median(trans_error_list):.4f}m\n")
        f.write(f"Median Rotation Error: {np.median(rot_error_list):.4f}°\n")
        f.write(f"Std Translation Error: {np.std(trans_error_list):.4f}m\n")
        f.write(f"Std Rotation Error: {np.std(rot_error_list):.4f}°\n")
        f.write(f"Min Translation Error: {np.min(trans_error_list):.4f}m\n")
        f.write(f"Max Translation Error: {np.max(trans_error_list):.4f}m\n")
        f.write(f"Min Rotation Error: {np.min(rot_error_list):.4f}°\n")
        f.write(f"Max Rotation Error: {np.max(rot_error_list):.4f}°\n")

        f.write("\n=== Success Rate Analysis ===\n")
        f.write(f"Total test cameras: {success_rates['total_cameras']}\n\n")
        f.write("Translation Error Success Rates:\n")
        f.write(f"  ≤ 1cm:  {(success_rates['success_1cm'] / success_rates['total_cameras']) * 100:.1f}% "
                f"({success_rates['success_1cm']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 2cm:  {(success_rates['success_2cm'] / success_rates['total_cameras']) * 100:.1f}% "
                f"({success_rates['success_2cm']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 5cm:  {success_rates['success_5cm_pct']:.1f}% "
                f"({success_rates['success_5cm']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 10cm: {(success_rates['success_10cm'] / success_rates['total_cameras']) * 100:.1f}% "
                f"({success_rates['success_10cm']}/{success_rates['total_cameras']})\n\n")
        f.write("Rotation Error Success Rates:\n")
        f.write(f"  ≤ 1°:   {(success_rates['success_1deg'] / success_rates['total_cameras']) * 100:.1f}% "
                f"({success_rates['success_1deg']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 2°:   {(success_rates['success_2deg'] / success_rates['total_cameras']) * 100:.1f}% "
                f"({success_rates['success_2deg']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 5°:   {success_rates['success_5deg_pct']:.1f}% "
                f"({success_rates['success_5deg']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 10°:  {(success_rates['success_10deg'] / success_rates['total_cameras']) * 100:.1f}% "
                f"({success_rates['success_10deg']}/{success_rates['total_cameras']})\n\n")
        f.write("Combined Success Rates:\n")
        f.write(f"  ≤ 2cm AND ≤ 2°: {success_rates['success_both_2cm_2deg_pct']:.1f}% "
                f"({success_rates['success_both_2cm_2deg']}/{success_rates['total_cameras']})\n")
        f.write(f"  ≤ 5cm AND ≤ 5°: {success_rates['success_both_5cm_5deg_pct']:.1f}% "
                f"({success_rates['success_both_5cm_5deg']}/{success_rates['total_cameras']})\n\n")

        f.write("\n=== Timing Statistics ===\n")
        f.write(f"Total execution time: {total_end_time - total_start_time:.2f} seconds\n")
        f.write(f"Mean time per camera: {np.mean(time_per_camera_list):.2f} seconds\n")
        f.write(f"Median time per camera: {np.median(time_per_camera_list):.2f} seconds\n")
        f.write(f"Min time per camera: {np.min(time_per_camera_list):.2f} seconds\n")
        f.write(f"Max time per camera: {np.max(time_per_camera_list):.2f} seconds\n")
        f.write(f"Std time per camera: {np.std(time_per_camera_list):.2f} seconds\n")

        if coarse_cam_error_list is not None:
            coarse_trans = [e[0] for e in coarse_cam_error_list]
            coarse_rot = [e[1] for e in coarse_cam_error_list]
            f.write("\n=== Coarse Camera Error Statistics ===\n")
            f.write(f"Mean Coarse Camera Error: {np.mean(coarse_trans):.4f}m, {np.mean(coarse_rot):.4f}°\n")
            f.write(f"Median Coarse Camera Error: {np.median(coarse_trans):.4f}m, {np.median(coarse_rot):.4f}°\n")
            f.write(f"Std Coarse Camera Error: {np.std(coarse_trans):.4f}m, {np.std(coarse_rot):.4f}°\n")
            f.write(f"Min Coarse Camera Error: {np.min(coarse_trans):.4f}m, {np.min(coarse_rot):.4f}°\n")
            f.write(f"Max Coarse Camera Error: {np.max(coarse_trans):.4f}m, {np.max(coarse_rot):.4f}°\n")
