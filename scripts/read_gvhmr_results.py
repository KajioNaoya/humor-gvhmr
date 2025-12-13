import os
from re import T
import torch
import numpy as np
from humor.utils.transforms import batch_rodrigues


def compute_rotation_matrix_from_points(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Compute 4x4 homogeneous transformation matrix T such that Y = T @ [X; 1] 
    using Umeyama algorithm (extended Kabsch algorithm with translation).
    
    This algorithm computes both rotation and translation, accounting for the fact
    that the origins of the two coordinate systems may not coincide.
    
    Args:
        X: (N, 3) source points
        Y: (N, 3) target points
    
    Returns:
        T: (4, 4) homogeneous transformation matrix
           [[R, t],
            [0, 1]]
           where R is (3, 3) rotation matrix and t is (3,) translation vector
           such that Y = R @ X + t
    """
    assert X.shape == Y.shape, f"X and Y must have the same shape, got {X.shape} and {Y.shape}"
    assert X.shape[1] == 3, f"Points must be 3D, got shape {X.shape}"
    
    N = X.shape[0]
    print(f"Debug: Number of points: {N}")
    print(f"Debug: X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"Debug: X mean: {X.mean(axis=0)}, X std: {X.std(axis=0)}")
    print(f"Debug: Y mean: {Y.mean(axis=0)}, Y std: {Y.std(axis=0)}")
    
    # Check scale factor by comparing distances between adjacent point pairs
    # For rigid transformation, distances should be preserved (ratio = 1.0)
    # For similarity transformation, ratio should be constant (scale factor)
    # If ratio varies, there's a more complex relationship
    print(f"\nDebug: Scale factor verification (distance ratios for adjacent pairs):")
    distance_ratios = []
    for i in range(N - 1):
        X_dist = np.linalg.norm(X[i+1] - X[i])
        Y_dist = np.linalg.norm(Y[i+1] - Y[i])
        if X_dist > 1e-8:  # Avoid division by zero
            ratio = Y_dist / X_dist
            distance_ratios.append(ratio)
            print(f"  Pair {i+1}-{i+2}: X_dist={X_dist:.6f}, Y_dist={Y_dist:.6f}, ratio={ratio:.6f}")
        else:
            print(f"  Pair {i+1}-{i+2}: X_dist={X_dist:.6f} (too small, skipping)")
    
    if len(distance_ratios) > 0:
        distance_ratios = np.array(distance_ratios)
        mean_ratio = np.mean(distance_ratios)
        std_ratio = np.std(distance_ratios)
        min_ratio = np.min(distance_ratios)
        max_ratio = np.max(distance_ratios)
        print(f"\nDebug: Distance ratio statistics:")
        print(f"  Mean ratio: {mean_ratio:.6f}")
        print(f"  Std ratio: {std_ratio:.6f}")
        print(f"  Min ratio: {min_ratio:.6f}")
        print(f"  Max ratio: {max_ratio:.6f}")
        
        if std_ratio < 1e-6:
            print(f"  -> All ratios are approximately {mean_ratio:.6f} (constant scale factor)")
            if abs(mean_ratio - 1.0) < 1e-6:
                print(f"  -> Scale factor is 1.0 (rigid transformation, no scaling)")
            else:
                print(f"  -> Scale factor is {mean_ratio:.6f} (similarity transformation with scaling)")
        elif std_ratio < 0.01:
            print(f"  -> Ratios are approximately constant (mean={mean_ratio:.6f}, std={std_ratio:.6f})")
            print(f"  -> Slight variation may be due to numerical errors or noise")
        else:
            print(f"  -> WARNING: Ratios vary significantly (std={std_ratio:.6f})")
            print(f"  -> This suggests the transformation is NOT a simple rigid or similarity transformation")
            print(f"  -> Possible causes: non-rigid transformation, incorrect point correspondence, or outliers")
    
    # Compute centroids
    X_mean = X.mean(axis=0)  # (3,)
    Y_mean = Y.mean(axis=0)  # (3,)
    
    # Center the points
    X_centered = X - X_mean  # (N, 3)
    Y_centered = Y - Y_mean  # (N, 3)
    
    # Compute covariance matrix (should be divided by N for proper covariance, but not needed for SVD)
    H = X_centered.T @ Y_centered  # (3, 3)
    
    print(f"Debug: Covariance matrix H:\n{H}")
    print(f"Debug: H condition number: {np.linalg.cond(H)}")
    
    # SVD
    U, S, Vt = np.linalg.svd(H, full_matrices=True)
    print(f"Debug: Singular values S: {S}")
    print(f"Debug: S ratio (min/max): {S[-1] / S[0] if S[0] > 0 else 0}")
    
    # Check for degenerate case (rank < 3)
    if S[-1] < 1e-6:
        print("Warning: Third singular value is very small, point cloud may be degenerate or points may not correspond correctly")
    
    # Compute rotation matrix: R = V @ U.T
    # Note: np.linalg.svd returns Vt (V transpose), so V = Vt.T
    R = Vt.T @ U.T  # (3, 3)
    
    # Ensure proper rotation (det(R) = 1, not -1)
    # If det(R) < 0, we need to flip the sign of the last column of V
    det_R = np.linalg.det(R)
    print(f"Debug: det(R) before correction: {det_R}")
    
    if det_R < 0:
        print("Debug: Correcting reflection (det(R) < 0)")
        # Flip the sign of the last row of Vt (which corresponds to the last column of V)
        Vt = Vt.copy()  # Make a copy to avoid modifying the original
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
        det_R = np.linalg.det(R)
        print(f"Debug: det(R) after correction: {det_R}")
    
    # Verify R is a proper rotation matrix
    R_Rt = R @ R.T
    identity_error = np.linalg.norm(R_Rt - np.eye(3))
    print(f"Debug: R @ R.T should be identity, error: {identity_error}")
    
    # Compute translation vector: t = Y_mean - R @ X_mean
    t = Y_mean - R @ X_mean  # (3,)
    
    # Construct 4x4 homogeneous transformation matrix
    T = np.eye(4, dtype=X.dtype)
    T[:3, :3] = R
    T[:3, 3] = t

    # Verify transformation: Y should be approximately equal to R @ X + t
    Y_pred = (R @ X.T).T + t  # (N, 3) - equivalent to X @ R.T + t
    error = Y - Y_pred
    error_norm = np.linalg.norm(error, axis=1)
    mean_error = np.mean(error_norm)
    max_error = np.max(error_norm)
    
    print(f"Debug: Transformation verification:")
    print(f"  Mean error per point: {mean_error:.6f}")
    print(f"  Max error per point: {max_error:.6f}")
    print(f"  Total error norm: {np.linalg.norm(error):.6f}")
    print(f"  R:\n{R}")
    print(f"  t: {t}")
    
    # Check if X and Y might not correspond correctly
    # If the error is very large, it suggests the points don't correspond
    if mean_error > 0.1:
        print(f"WARNING: Large transformation error ({mean_error:.6f}) suggests:")
        print(f"  1. X and Y may not correspond to the same points")
        print(f"  2. The relationship may not be a simple rigid transformation")
        print(f"  3. There may be noise or outliers in the data")
        print(f"\nFirst 3 point pairs:")
        for i in range(min(3, N)):
            print(f"  Point {i}: X={X[i]}, Y={Y[i]}, Y_pred={Y_pred[i]}, error={error_norm[i]:.6f}")
    
    return T


def compute_gvhmr_rotation_matrix(gvhmr_dir: str, start_frame: int = None, end_frame: int = None, num_samples: int = 20) -> np.ndarray:
    """
    Compute 4x4 homogeneous transformation matrix from GVHMR's incam to global coordinate transformation.
    Uses global_orient (rotation) instead of transl (translation) for more robust estimation.
    
    Args:
        gvhmr_dir: Directory containing GVHMR results
        start_frame: Start frame index (inclusive)
        end_frame: End frame index (exclusive)
        num_samples: Number of random samples to use (default: 20)
    
    Returns:
        T_gvhmr_incam2global: (4, 4) homogeneous transformation matrix from incam to global coordinates
           [[R, t],
            [0, 1]]
           where R is (3, 3) rotation matrix and t is (3,) translation vector
           Note: t is set to zero since we only estimate rotation from global_orient
    """
    results_path = os.path.join(gvhmr_dir, "hmr4d_results.pt")
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"hmr4d_results.pt not found in {gvhmr_dir}")
    
    pred = torch.load(results_path, map_location='cpu')
    smpl_params_incam = pred.get("smpl_params_incam", {})
    smpl_params_global = pred.get("smpl_params_global", {})
    
    if "global_orient" not in smpl_params_incam or "global_orient" not in smpl_params_global:
        raise ValueError("Both smpl_params_incam and smpl_params_global must contain 'global_orient'")
    
    global_orient_incam = smpl_params_incam["global_orient"]  # (T, 3) axis-angle
    global_orient_global = smpl_params_global["global_orient"]  # (T, 3) axis-angle
    
    # Convert to numpy if needed
    if isinstance(global_orient_incam, torch.Tensor):
        global_orient_incam = global_orient_incam.numpy()
    if isinstance(global_orient_global, torch.Tensor):
        global_orient_global = global_orient_global.numpy()
    
    global_orient_incam = global_orient_incam.astype(np.float32)  # (T, 3)
    global_orient_global = global_orient_global.astype(np.float32)  # (T, 3)
    
    # Apply frame range if specified
    if start_frame is not None and end_frame is not None:
        global_orient_incam = global_orient_incam[start_frame:end_frame]
        global_orient_global = global_orient_global[start_frame:end_frame]
    
    T = global_orient_incam.shape[0]
    
    # Random sampling
    if T > num_samples:
        # Use fixed seed for reproducibility in debugging
        np.random.seed(42)
        indices = np.random.choice(T, size=num_samples, replace=False)
        indices = np.sort(indices)  # Sort for easier debugging
        print(f"Debug: Using {num_samples} samples from {T} frames")
        print(f"Debug: Sample indices: {indices}")
        global_orient_incam = global_orient_incam[indices]
        global_orient_global = global_orient_global[indices]
    else:
        print(f"Debug: Using all {T} frames")
    
    # Verify that global_orient_incam and global_orient_global have the same shape
    assert global_orient_incam.shape == global_orient_global.shape, \
        f"global_orient_incam shape {global_orient_incam.shape} != global_orient_global shape {global_orient_global.shape}"
    
    N = global_orient_incam.shape[0]
    print(f"Debug: Number of samples: {N}")
    
    # Convert axis-angle to rotation matrices
    # batch_rodrigues expects (N, 3) tensor
    global_orient_incam_t = torch.from_numpy(global_orient_incam)  # (N, 3)
    global_orient_global_t = torch.from_numpy(global_orient_global)  # (N, 3)
    
    R_incam_batch = batch_rodrigues(global_orient_incam_t)  # (N, 3, 3)
    R_global_batch = batch_rodrigues(global_orient_global_t)  # (N, 3, 3)
    
    # Convert back to numpy
    R_incam_batch = R_incam_batch.numpy()  # (N, 3, 3)
    R_global_batch = R_global_batch.numpy()  # (N, 3, 3)
    
    print(f"Debug: Converted {N} axis-angle rotations to rotation matrices")
    
    # Compute R_cam2world for each frame
    # Theory: R_global = R_cam2world @ R_incam
    # Therefore: R_cam2world = R_global @ R_incam.T
    R_cam2world_batch = np.zeros((N, 3, 3), dtype=np.float32)
    for i in range(N):
        R_incam = R_incam_batch[i]  # (3, 3)
        R_global = R_global_batch[i]  # (3, 3)
        R_cam2world = R_global @ R_incam.T  # (3, 3)
        R_cam2world_batch[i] = R_cam2world
    
    print(f"Debug: Computed R_cam2world for {N} frames")
    
    # Verify consistency: all R_cam2world should be approximately the same
    # Compute mean and check variance
    R_cam2world_mean = np.mean(R_cam2world_batch, axis=0)  # (3, 3)
    
    # Compute differences from mean
    differences = R_cam2world_batch - R_cam2world_mean[np.newaxis, :, :]  # (N, 3, 3)
    diff_norms = np.linalg.norm(differences, axis=(1, 2))  # (N,)
    mean_diff = np.mean(diff_norms)
    max_diff = np.max(diff_norms)
    std_diff = np.std(diff_norms)
    
    print(f"\nDebug: Consistency check for R_cam2world:")
    print(f"  Mean difference from mean: {mean_diff:.6f}")
    print(f"  Max difference from mean: {max_diff:.6f}")
    print(f"  Std difference from mean: {std_diff:.6f}")
    
    if mean_diff < 1e-4:
        print(f"  -> All rotations are consistent (mean error < 1e-4)")
    elif mean_diff < 0.01:
        print(f"  -> Rotations are approximately consistent (mean error < 0.01)")
        print(f"  -> Small variations may be due to numerical errors or noise")
    else:
        print(f"  -> WARNING: Rotations are NOT consistent (mean error = {mean_diff:.6f})")
        print(f"  -> This suggests the relationship is not a simple constant rotation")
        print(f"  -> Possible causes: incorrect correspondence, time-varying transformation, or outliers")
    
    # Print individual rotation matrices for first few frames
    print(f"\nDebug: First 3 R_cam2world matrices:")
    for i in range(min(3, N)):
        print(f"  Frame {i}:")
        print(f"    {R_cam2world_batch[i]}")
        diff_from_mean = np.linalg.norm(R_cam2world_batch[i] - R_cam2world_mean)
        print(f"    Difference from mean: {diff_from_mean:.6f}")
    
    # Use mean rotation matrix
    R_cam2world = R_cam2world_mean
    
    # Verify R is a proper rotation matrix
    R_Rt = R_cam2world @ R_cam2world.T
    identity_error = np.linalg.norm(R_Rt - np.eye(3))
    det_R = np.linalg.det(R_cam2world)
    print(f"\nDebug: Final R_cam2world verification:")
    print(f"  R @ R.T should be identity, error: {identity_error:.6f}")
    print(f"  det(R) should be 1.0, got: {det_R:.6f}")
    
    if identity_error > 1e-4 or abs(det_R - 1.0) > 1e-4:
        print(f"  WARNING: R_cam2world is not a proper rotation matrix")
        # Orthonormalize if needed
        U, S, Vt = np.linalg.svd(R_cam2world)
        R_cam2world = U @ Vt
        if np.linalg.det(R_cam2world) < 0:
            Vt[-1, :] *= -1
            R_cam2world = U @ Vt
        print(f"  Orthonormalized R_cam2world")
        print(f"  New det(R): {np.linalg.det(R_cam2world):.6f}")
    
    # Construct 4x4 homogeneous transformation matrix
    # Note: translation is set to zero since we only estimate rotation
    T_transform = np.eye(4, dtype=np.float32)
    T_transform[:3, :3] = R_cam2world
    T_transform[:3, 3] = np.zeros(3, dtype=np.float32)  # t = 0
    
    print(f"\nDebug: Final transformation matrix:")
    print(f"  R:\n{R_cam2world}")
    print(f"  t: {T_transform[:3, 3]} (zero, since only rotation is estimated)")
    
    return T_transform


def load_gvhmr_results(gvhmr_dir, start_frame=None, end_frame=None, fps=30.0):
    """
    GVHMRの結果ファイルを読み込んで、HuMorに渡せる形式に変換する関数

    Args:
        gvhmr_dir (str): GVHMRの結果が格納されたディレクトリ
                         (hmr4d_results.pt, preprocess/vitpose.pt などを含む)
        fps (float, optional): 動画のフレームレート. Defaults to 30.0.
        start_frame (int, optional): 開始フレーム. Defaults to None.
        end_frame (int, optional): 終了フレーム. Defaults to None.
    Returns:
        coco_seq: (T, 17, 3)
        cam_mat: (3, 3)
        beta_ext: (10,)
        pose_body: (T, 21, 3)
        root_orient: (T, 3)
        trans: (T, 3)
    """
    # --- 1. ファイルパスの構築 ---
    results_path = os.path.join(gvhmr_dir, "hmr4d_results.pt")
    vitpose_path = os.path.join(gvhmr_dir, "preprocess/vitpose.pt")

    if not os.path.exists(results_path):
        raise FileNotFoundError(f"hmr4d_results.pt not found in {gvhmr_dir}")
    if not os.path.exists(vitpose_path):
        raise FileNotFoundError(f"vitpose.pt not found in {gvhmr_dir}")

    # --- 2. データのロード ---
    pred = torch.load(results_path, map_location='cpu')
    vitpose = torch.load(vitpose_path, map_location='cpu')
    smpl_params = pred["smpl_params_incam"]

    # keypoints
    coco_seq = vitpose.numpy().astype(np.float32)  # (T, 17, 3)
    
    # intrinsics (assuming the camera is fixed, so we take the mean of the intrinsics)
    cam_mat = torch.mean(pred["K_fullimg"], dim=0).numpy().astype(np.float32)  # (3,3)

    # shape (assuming the shape is fixed, so we take the mean of the shape)
    shape = torch.mean(smpl_params["betas"], dim=0).numpy().astype(np.float32)  # (10,)

    # pose
    pose_body = smpl_params["body_pose"].numpy().astype(np.float32).reshape(-1, 21, 3)  # (T, 21, 3)

    # global orientation
    root_orient = smpl_params["global_orient"].numpy().astype(np.float32).reshape(-1, 3)  # (T, 3)

    # transl (camera frame)
    trans = smpl_params["transl"].numpy().astype(np.float32).reshape(-1, 3)  # (T, 3)

    if start_frame is not None and end_frame is not None:
        coco_seq = coco_seq[start_frame:end_frame, :, :]
        pose_body = pose_body[start_frame:end_frame, :, :]
        root_orient = root_orient[start_frame:end_frame, :]
        trans = trans[start_frame:end_frame, :]

    return coco_seq, cam_mat, shape, pose_body, root_orient, trans