import os
import argparse
from typing import Tuple, Optional

import numpy as np
import torch

from humor.fitting.fitting_utils import (
    DEFAULT_FOCAL_LEN,
    perspective_projection,
)
from humor.body_model.body_model import BodyModel
from humor.body_model.utils import SMPL_JOINTS, smpl_to_openpose
from humor.utils.transforms import batch_rodrigues, rotation_matrix_to_angle_axis

from scripts.read_gvhmr_results import load_gvhmr_results, compute_gvhmr_rotation_matrix
from scripts.demo_mmpose_external_smpl import (
    run_mmpose_halpe_on_video,
    halpe_seq_to_body25_seq,
)
from scripts.temporal_foot_contact_detection import detect_foot_contact


def build_camera_matrix(
    img_h: int, img_w: int, focal_lengths: Optional[float]
) -> np.ndarray:
    if focal_lengths is not None:
        fx = fy = float(focal_lengths)
    else:
        fx, fy = DEFAULT_FOCAL_LEN

    cx = img_w * 0.5
    cy = img_h * 0.5

    cam_mat_np = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return cam_mat_np


def compute_rotation_between_vectors(v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix that rotates v1 to v2.
    
    Args:
        v1: (3,) source vector (normalized)
        v2: (3,) target vector (normalized)
    
    Returns:
        R: (3, 3) rotation matrix such that R @ v1 = v2
    """
    device = v1.device
    
    # Normalize vectors
    v1 = v1 / (v1.norm() + 1e-8)
    v2 = v2 / (v2.norm() + 1e-8)
    
    # Check if vectors are parallel (or anti-parallel)
    dot = torch.dot(v1, v2)
    
    # If vectors are nearly parallel, return identity or 180-degree rotation
    if abs(dot) > 1.0 - 1e-6:
        if dot > 0:
            # Same direction: identity
            return torch.eye(3, device=device, dtype=v1.dtype)
        else:
            # Opposite direction: 180-degree rotation around perpendicular axis
            # Find a perpendicular vector
            if abs(v1[0]) < 0.9:
                perp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=v1.dtype)
            else:
                perp = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=v1.dtype)
            axis = torch.cross(v1, perp)
            axis = axis / (axis.norm() + 1e-8)
            # 180-degree rotation
            angle = torch.tensor(np.pi, device=device, dtype=v1.dtype)
            axis_angle = axis * angle
            return batch_rodrigues(axis_angle.unsqueeze(0))[0]
    
    # Compute rotation axis (perpendicular to both vectors)
    axis = torch.cross(v1, v2)
    axis = axis / (axis.norm() + 1e-8)
    
    # Compute rotation angle
    angle = torch.acos(torch.clamp(dot, min=-1.0, max=1.0))
    
    # Convert axis-angle to rotation matrix
    axis_angle = axis * angle
    R = batch_rodrigues(axis_angle.unsqueeze(0))[0]
    
    return R


def smpl_body25_keypoints(
    body_model: BodyModel,
    betas: torch.Tensor,  # (T, num_betas)
    body_pose: torch.Tensor,  # (T, 21*3)
    root_orient: torch.Tensor,  # (T, 3)
    transl: torch.Tensor,  # (T, 3)
) -> torch.Tensor:
    """
    Run SMPLH and return BODY_25-ordered 3D keypoints in camera coordinates.

    Returns:
        joints3d_op (T, 25, 3)
    """
    device = body_pose.device
    T = body_pose.shape[0]

    smpl_out = body_model(
        betas=betas,
        pose_body=body_pose,
        root_orient=root_orient,
        trans=transl,
        return_dict=True,
    )
    # Jtr: (T, num_joints(+extra), 3)
    joints3d_all = smpl_out["Jtr"]  # (T, J_all, 3)

    # First |SMPL_JOINTS| entries are SMPL body joints, the rest are extra joints
    num_body_joints = len(SMPL_JOINTS)
    body_joints3d = joints3d_all[:, :num_body_joints, :]
    extra_joints3d = joints3d_all[:, num_body_joints:, :]

    joints3d_full = torch.cat([body_joints3d, extra_joints3d], dim=1)

    smpl2op_map = smpl_to_openpose(
        model_type=body_model.model_type,
        use_hands=False,
        use_face=False,
        openpose_format="coco25",
    )

    joints3d_op = joints3d_full[:, smpl2op_map, :]  # (T, 25, 3)
    return joints3d_op


def compute_losses(
    body_model: BodyModel,
    cam_mat_np: np.ndarray,
    betas_batch: torch.Tensor,  # (T, num_betas)
    theta_ext_const: torch.Tensor,  # (T, 21, 3)
    root_orient_const: torch.Tensor,  # (T, 3)
    transl_var: torch.Tensor,  # (T, 3)
    ankle_delta: torch.Tensor,  # (T, 2, 3) [left, right]
    body25_seq: torch.Tensor,  # (T, 25, 3)
    left_contact: torch.Tensor,  # (T,) bool
    right_contact: torch.Tensor,  # (T,) bool
    lambda_2d: float,
    lambda_trans_prior: float,
    lambda_ankle_prior: float,
    lambda_slide: float,
    lambda_plane: float,
    trans_init: torch.Tensor,  # (T, 3)
    R_gvhmr_incam2global: Optional[torch.Tensor] = None,  # (3, 3) optional GVHMR rotation matrix
    lambda_gvhmr_plane_prior: float = 0.0,  # weight for GVHMR plane prior
) -> Tuple[torch.Tensor, dict]:
    """
    Compute objective:
        - 2D keypoint reprojection error (BODY_25)
        - weak L2 prior to GVHMR translation and ankle angles
        - foot sliding suppression during contact
        - contact foot keypoints lying near a single ground plane
    """
    device = transl_var.device
    T = theta_ext_const.shape[0]

    # ------------------------------------------------------------------ #
    # Build current body pose (only ankle joints are changed)
    # ------------------------------------------------------------------ #
    pose_body = theta_ext_const.clone()  # (T, 21, 3)
    left_ankle_idx = SMPL_JOINTS["leftFoot"]
    right_ankle_idx = SMPL_JOINTS["rightFoot"]

    pose_body[:, left_ankle_idx, :] += ankle_delta[:, 0, :]
    pose_body[:, right_ankle_idx, :] += ankle_delta[:, 1, :]

    body_pose_flat = pose_body.reshape(T, -1)  # (T, 21*3)

    # ------------------------------------------------------------------ #
    # 3D BODY_25 keypoints from SMPL
    # ------------------------------------------------------------------ #
    joints3d_op = smpl_body25_keypoints(
        body_model=body_model,
        betas=betas_batch,
        body_pose=body_pose_flat,
        root_orient=root_orient_const,
        transl=transl_var,
    )  # (T, 25, 3)

    # ------------------------------------------------------------------ #
    # 2D reprojection loss
    # ------------------------------------------------------------------ #
    fx = float(cam_mat_np[0, 0])
    fy = float(cam_mat_np[1, 1])
    cx = float(cam_mat_np[0, 2])
    cy = float(cam_mat_np[1, 2])

    T_bs = T
    rotation = torch.eye(3, device=device).unsqueeze(0).expand(T_bs, 3, 3)
    translation = torch.zeros(T_bs, 3, device=device)
    focal = torch.tensor([fx, fy], dtype=torch.float32, device=device).unsqueeze(0)
    focal = focal.expand(T_bs, 2)
    center = torch.tensor([cx, cy], dtype=torch.float32, device=device).unsqueeze(0)
    center = center.expand(T_bs, 2)

    proj_2d = perspective_projection(
        points=joints3d_op,  # (T, 25, 3)
        rotation=rotation,
        translation=translation,
        focal_length=focal,
        camera_center=center,
    )  # (T, 25, 2)

    kp2d_obs = body25_seq[..., :2]  # (T, 25, 2)
    kp2d_conf = body25_seq[..., 2:3]  # (T, 25, 1)

    reproj_err = proj_2d - kp2d_obs
    weight = kp2d_conf**2
    weight_sum = weight.sum().clamp(min=1e-8)
    # Weighted mean reprojection loss over all frames/keypoints
    reproj_loss = (reproj_err**2 * weight).sum() / weight_sum

    # ------------------------------------------------------------------ #
    # Weak priors: translation and ankle corrections
    # ------------------------------------------------------------------ #
    trans_prior_loss = (transl_var - trans_init).pow(2).mean()
    ankle_prior_loss = ankle_delta.pow(2).mean()

    # ------------------------------------------------------------------ #
    # Foot sliding: velocities ~ 0 during contact
    # BODY_25 foot indices (from user):
    #   LBigToe=19, LSmallToe=20, LHeel=21, RBigToe=22, RSmallToe=23, RHeel=24
    # ------------------------------------------------------------------ #
    left_ids = torch.tensor([19, 20, 21], dtype=torch.long, device=device)
    right_ids = torch.tensor([22, 23, 24], dtype=torch.long, device=device)

    left_pos = joints3d_op[:, left_ids, :]  # (T, 3, 3)
    right_pos = joints3d_op[:, right_ids, :]  # (T, 3, 3)

    if T > 1:
        left_vel = left_pos[1:] - left_pos[:-1]  # (T-1, 3, 3)
        right_vel = right_pos[1:] - right_pos[:-1]

        left_mask = left_contact[1:].to(device=device, dtype=torch.float32)  # (T-1,)
        right_mask = right_contact[1:].to(device=device, dtype=torch.float32)

        left_mask = left_mask.view(-1, 1, 1)  # (T-1,1,1)
        right_mask = right_mask.view(-1, 1, 1)

        left_vel_loss = (left_vel.pow(2).sum(dim=-1) * left_mask).sum()
        right_vel_loss = (right_vel.pow(2).sum(dim=-1) * right_mask).sum()

        contact_count = (
            left_mask.sum() * left_ids.numel() + right_mask.sum() * right_ids.numel()
        ).clamp(min=1e-8)

        # Mean velocity penalty over contact frames/joints
        slide_loss = (left_vel_loss + right_vel_loss) / contact_count
    else:
        slide_loss = torch.zeros((), device=device)

    # ------------------------------------------------------------------ #
    # Ground-plane consistency for contact foot keypoints
    # ------------------------------------------------------------------ #
    # Collect contact points (with grad enabled for plane estimation)
    pts_list = []
    if left_contact.any():
        pts_list.append(left_pos[left_contact].reshape(-1, 3))
    if right_contact.any():
        pts_list.append(right_pos[right_contact].reshape(-1, 3))

    if len(pts_list) > 0:
        contact_pts = torch.cat(pts_list, dim=0)  # (M, 3)
    else:
        contact_pts = None

    if contact_pts is not None and contact_pts.shape[0] >= 3:
        X = contact_pts  # Keep gradient flow enabled
        centroid = X.mean(dim=0, keepdim=True)  # (1, 3)
        Xc = X - centroid
        # SVD for best-fit plane normal
        # Xc: (M, 3) -> U, S, Vh with Vh: (3,3)
        # Note: SVD gradients are supported in PyTorch, but may be unstable
        # if singular values are degenerate (close to each other)
        _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
        normal = Vh[-1]  # (3,)
        normal = normal / (normal.norm() + 1e-8)
        d = -torch.dot(normal, centroid.squeeze(0))
    else:
        normal = None
        d = None

    if normal is not None:
        # Use current (grad-enabled) points in plane loss
        all_pts = []
        if left_contact.any():
            all_pts.append(left_pos[left_contact].reshape(-1, 3))
        if right_contact.any():
            all_pts.append(right_pos[right_contact].reshape(-1, 3))
        all_pts = torch.cat(all_pts, dim=0)  # (M, 3)

        dist = all_pts @ normal.to(device) + d.to(device)
        # Mean squared distance to the plane across contact points
        plane_loss = (dist**2).mean()
    else:
        plane_loss = torch.zeros((), device=device)

    # ------------------------------------------------------------------ #
    # GVHMR plane prior: normal should align with GVHMR's (0, 1, 0) in incam
    # ------------------------------------------------------------------ #
    gvhmr_plane_prior_loss = torch.zeros((), device=device)
    if normal is not None and R_gvhmr_incam2global is not None and lambda_gvhmr_plane_prior > 0:
        # GVHMR's world Y-axis (0, 1, 0) rotated to incam coordinates
        world_y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=normal.dtype)
        R_incam2global = R_gvhmr_incam2global.to(device=device, dtype=normal.dtype)
        # R_gvhmr_incam2global rotates from incam to global, so its inverse rotates from global to incam
        R_global2incam = R_incam2global.T  # (3, 3)
        world_y_incam = R_global2incam @ world_y_axis  # (3,)
        world_y_incam = world_y_incam / (world_y_incam.norm() + 1e-8)
        
        # Compute dot product between normal and rotated world Y-axis
        # We want normal to align with world_y_incam, so we minimize (1 - dot(normal, world_y_incam))
        dot_product = torch.abs(torch.dot(normal, world_y_incam))
        gvhmr_plane_prior_loss = 1.0 - dot_product

    # ------------------------------------------------------------------ #
    # Total loss
    # ------------------------------------------------------------------ #
    total_loss = (
        lambda_2d * reproj_loss
        + lambda_trans_prior * trans_prior_loss
        + lambda_ankle_prior * ankle_prior_loss
        + lambda_slide * slide_loss
        + lambda_plane * plane_loss
        + lambda_gvhmr_plane_prior * gvhmr_plane_prior_loss
    )

    stats = {
        "total": total_loss.item(),
        "reproj": reproj_loss.item(),
        "trans_prior": trans_prior_loss.item(),
        "ankle_prior": ankle_prior_loss.item(),
        "slide": slide_loss.item(),
        "plane": plane_loss.item(),
        "gvhmr_plane_prior": gvhmr_plane_prior_loss.item(),
    }
    return total_loss, stats


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Baseline: optimize only SMPL translation and ankle joints "
            "using GVHMR SMPL params, MMPose BODY_25 keypoints, and foot-contact CSV."
        )
    )
    parser.add_argument(
        "--gvhmr-dir",
        type=str,
        default="./data/1029_01",
        help="Directory containing GVHMR results (hmr4d_results.pt, preprocess/vitpose.pt, etc.)",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to RGB video corresponding to GVHMR results.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=900,
        help="Start frame index (inclusive) for optimization.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=1050,
        help="End frame index (exclusive) for optimization.",
    )
    parser.add_argument(
        "--focal-lengths",
        type=float,
        default=None,
        help=(
            "Focal length f in pixels. If not provided, DEFAULT_FOCAL_LEN "
            "is used. fx=fy=f and principal point (cx, cy) is set from "
            "the input video resolution."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for PyTorch (e.g., 'cuda:0' or 'cpu'). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--contact-csv",
        type=str,
        default=None,
        help=(
            "Path to a T x 2 CSV file with foot contact labels per frame "
            "(columns: left_contact, right_contact; values 0 or 1). "
            "If not provided, contact will be auto-detected from MMPose results."
        ),
    )
    parser.add_argument(
        "--contact-side",
        type=str,
        default="both",
        choices=["left", "right", "both"],
        help=(
            "Which foot(s) to detect contact for when --contact-csv is not provided. "
            "Options: 'left' (left foot only), 'right' (right foot only), 'both' (both feet). "
            "For the non-detected foot, contact will be set to False (always floating)."
        ),
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="smplx",
        choices=["smplh", "smplx"],
        help="Body model type to use for the baseline optimization.",
    )
    args = parser.parse_args()
    
    # Validate arguments
    if args.contact_csv is None and args.contact_side not in ["left", "right", "both"]:
        raise ValueError(
            "When --contact-csv is not provided, --contact-side must be one of: 'left', 'right', 'both'"
        )

    # Device
    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )
    print(f"Using device: {device}")

    model_type = args.model_type.lower()

    # ------------------------------------------------------------------ #
    # 1) Load GVHMR results (SMPL parameters etc.)
    # ------------------------------------------------------------------ #
    (
        _coco_unused,
        _cam_mat_gvhmr,
        beta_ext_np,
        theta_ext_np,
        root_orient_np,
        trans_np,
    ) = load_gvhmr_results(
        gvhmr_dir=args.gvhmr_dir,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )

    n_timestep = theta_ext_np.shape[0]
    print(f"Number of frames (GVHMR): {n_timestep}")

    # ------------------------------------------------------------------ #
    # 2) Run MMPose (Halpe) and convert to BODY_25
    #     → 一度推定したら動画ディレクトリに npz でキャッシュして再利用
    # ------------------------------------------------------------------ #
    DET_CONFIG = "./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco.py"
    DET_CHECKPOINT = "./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    POSE_CONFIG = "./checkpoints/mmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
    POSE_CHECKPOINT = "./checkpoints/mmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"

    # キャッシュファイル名（元動画と同じディレクトリ）
    video_dir = os.path.dirname(os.path.abspath(args.video_path))
    video_stem = os.path.splitext(os.path.basename(args.video_path))[0]
    end_tag = args.end_frame if args.end_frame is not None else "end"
    mmpose_cache_path = os.path.join(
        video_dir,
        f"{video_stem}_mmpose_halpe_{args.start_frame}_{end_tag}.npz",
    )

    if os.path.exists(mmpose_cache_path):
        print(f"Loading cached MMPose result from {mmpose_cache_path}")
        cache = np.load(mmpose_cache_path)
        halpe_seq = cache["halpe_seq"].astype(np.float32)
        img_h = int(cache["img_h"])
        img_w = int(cache["img_w"])
    else:
        print(f"No MMPose cache found. Running MMPose and saving to {mmpose_cache_path}")
        halpe_seq, (img_h, img_w) = run_mmpose_halpe_on_video(
            pose_config=POSE_CONFIG,
            pose_checkpoint=POSE_CHECKPOINT,
            video_path=args.video_path,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            device=str(device),
            det_config=DET_CONFIG,
            det_checkpoint=DET_CHECKPOINT,
            det_score_thr=0.5,
        )
        os.makedirs(video_dir, exist_ok=True)
        np.savez(
            mmpose_cache_path,
            halpe_seq=halpe_seq.astype(np.float32),
            img_h=np.int32(img_h),
            img_w=np.int32(img_w),
        )

    if halpe_seq.shape[0] != n_timestep:
        raise RuntimeError(
            f"Mismatch between GVHMR frames ({n_timestep}) and MMPose frames "
            f"({halpe_seq.shape[0]}). Please check start/end frame settings."
        )

    body25_seq_np = halpe_seq_to_body25_seq(halpe_seq)  # (T, 25, 3)

    # ------------------------------------------------------------------ #
    # 3) Camera intrinsics
    #     - focal-lengths が指定されていれば動画解像度と f から生成
    #     - 未指定なら GVHMR の K_fullimg 由来の行列（load_gvhmr_results の戻り値）
    # ------------------------------------------------------------------ #
    if args.focal_lengths is not None:
        cam_mat_np = build_camera_matrix(
            img_h=img_h,
            img_w=img_w,
            focal_lengths=args.focal_lengths,
        )
    else:
        # load_gvhmr_results が pred['K_fullimg'] を時間平均したものを返している
        cam_mat_np = _cam_mat_gvhmr

    # ------------------------------------------------------------------ #
    # 4) Load or detect foot contact
    # ------------------------------------------------------------------ #
    if args.contact_csv is not None:
        # Load from CSV file
        print("Loading foot contact from CSV file...")
        contact_labels = np.loadtxt(args.contact_csv, delimiter=",")
        if contact_labels.ndim != 2 or contact_labels.shape[1] != 2:
            raise ValueError(
                f"Expected contact CSV with shape (T, 2), got {contact_labels.shape}. "
                "Columns should be [left_contact, right_contact] with values 0 or 1."
            )
        if contact_labels.shape[0] != n_timestep:
            raise ValueError(
                "Mismatch between GVHMR frames and contact labels: "
                f"GVHMR={n_timestep}, contact_csv={contact_labels.shape[0]}."
            )
        left_contact_np = contact_labels[:, 0].astype(bool)
        right_contact_np = contact_labels[:, 1].astype(bool)
    else:
        # Auto-detect contact from MMPose results
        print(f"Auto-detecting foot contact from MMPose results (contact_side={args.contact_side})...")
        
        # Ensure halpe_seq is available (should be loaded from cache or computed above)
        if 'halpe_seq' not in locals():
            raise RuntimeError("halpe_seq not available. This should not happen.")
        
        # Detect contact using temporal_foot_contact_detection
        contact_labels_auto = detect_foot_contact(
            mmpose_keypoints=halpe_seq,
            n_MA=10,
            threshold_percentile=0.25,
            n_consecutive=10
        )  # (T, 2)
        
        # Apply contact_side setting
        if args.contact_side == "left":
            # Left foot: use auto-detection, Right foot: always False
            left_contact_np = contact_labels_auto[:, 0].astype(bool)
            right_contact_np = np.zeros(n_timestep, dtype=bool)
        elif args.contact_side == "right":
            # Right foot: use auto-detection, Left foot: always False
            left_contact_np = np.zeros(n_timestep, dtype=bool)
            right_contact_np = contact_labels_auto[:, 1].astype(bool)
        else:  # "both"
            # Both feet: use auto-detection for both
            left_contact_np = contact_labels_auto[:, 0].astype(bool)
            right_contact_np = contact_labels_auto[:, 1].astype(bool)
        
        print(f"Auto-detected contact: Left={left_contact_np.sum()}/{n_timestep}, Right={right_contact_np.sum()}/{n_timestep}")

    # ------------------------------------------------------------------ #
    # 5) Prepare tensors for optimization
    # ------------------------------------------------------------------ #
    T_seq = n_timestep

    theta_ext_const = torch.tensor(theta_ext_np, dtype=torch.float32, device=device)  # (T, 21, 3)
    root_orient_const = torch.tensor(root_orient_np, dtype=torch.float32, device=device)  # (T, 3)
    trans_init = torch.tensor(trans_np, dtype=torch.float32, device=device)  # (T, 3)

    # ------------------------------------------------------------------ #
    # 5.5) Choose body model (SMPLH or SMPLX) and betas dimension
    # ------------------------------------------------------------------ #
    if model_type == "smplx":
        smpl_model_path = "./body_models/smplx/SMPLX_NEUTRAL.npz"
        # SMPLX は通常 10 次元 shapecoeff
        num_betas_model = 10
    else:
        smpl_model_path = "./body_models/smplh/neutral/model.npz"
        # 既存 HuMoR コードと同様に 16 次元 beta を使う
        num_betas_model = 16

    if not os.path.exists(smpl_model_path):
        raise FileNotFoundError(f"Body model not found at: {smpl_model_path}")

    # betas: external estimate is (10,), copy into the first components
    beta_ext = torch.tensor(beta_ext_np, dtype=torch.float32, device=device)  # (10,)
    betas_batch = torch.zeros(T_seq, num_betas_model, dtype=torch.float32, device=device)
    betas_batch[:, : beta_ext.shape[0]] = beta_ext.unsqueeze(0).expand(T_seq, -1)

    body25_seq = torch.tensor(body25_seq_np, dtype=torch.float32, device=device)  # (T, 25, 3)
    left_contact = torch.tensor(left_contact_np, dtype=torch.bool, device=device)
    right_contact = torch.tensor(right_contact_np, dtype=torch.bool, device=device)

    # Optimization variables
    transl_var = trans_init.clone().detach().requires_grad_(True)  # (T, 3)
    ankle_delta = torch.zeros(T_seq, 2, 3, dtype=torch.float32, device=device, requires_grad=True)

    # ------------------------------------------------------------------ #
    # 6) Body model
    # ------------------------------------------------------------------ #
    body_model = BodyModel(
        bm_path=smpl_model_path,
        num_betas=num_betas_model,
        batch_size=T_seq,
        use_vtx_selector=True,
        model_type=model_type,
    ).to(device)

    # ------------------------------------------------------------------ #
    # 6.5) Compute GVHMR rotation matrix for plane prior
    # ------------------------------------------------------------------ #
    try:
        T_gvhmr_incam2global_np = compute_gvhmr_rotation_matrix(
            gvhmr_dir=args.gvhmr_dir,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            num_samples=20,
        )
        T_gvhmr_incam2global = torch.tensor(
            T_gvhmr_incam2global_np, dtype=torch.float32, device=device
        )  # (4, 4)
        # Extract rotation matrix (3x3) from homogeneous transformation matrix
        R_gvhmr_incam2global = T_gvhmr_incam2global[:3, :3]  # (3, 3)
        t_gvhmr_incam2global = T_gvhmr_incam2global[:3, 3]  # (3,) translation vector
        print(f"Computed GVHMR transformation matrix from incam to global")
        print(f"Rotation matrix:\n{R_gvhmr_incam2global}")
        print(f"Translation vector: {t_gvhmr_incam2global}")
    except Exception as e:
        print(f"Warning: Could not compute GVHMR rotation matrix: {e}")
        print("GVHMR plane prior will be disabled.")
        R_gvhmr_incam2global = None

    # ------------------------------------------------------------------ #
    # 7) Loss weights
    # ------------------------------------------------------------------ #
    lambda_2d = 0.025
    lambda_trans_prior = 10.0
    lambda_ankle_prior = 1.0
    lambda_slide = 1.0
    lambda_plane = 1000.0
    lambda_gvhmr_plane_prior = 10.0 # 100.0 if R_gvhmr_incam2global is not None else 0.0

    # ------------------------------------------------------------------ #
    # 8) Optimization (LBFGS)
    # ------------------------------------------------------------------ #
    params = [transl_var, ankle_delta]
    lr = 1.0
    lbfgs_max_iter = 20
    num_outer_iters = 60

    optimizer = torch.optim.LBFGS(
        params,
        max_iter=lbfgs_max_iter,
        lr=lr,
        line_search_fn="strong_wolfe",
    )

    last_loss_val = None
    rel_tol = 1e-3
    patience = 5
    no_improve = 0

    for it in range(num_outer_iters):

        def closure():
            optimizer.zero_grad()
            loss, stats = compute_losses(
                body_model=body_model,
                cam_mat_np=cam_mat_np,
                betas_batch=betas_batch,
                theta_ext_const=theta_ext_const,
                root_orient_const=root_orient_const,
                transl_var=transl_var,
                ankle_delta=ankle_delta,
                body25_seq=body25_seq,
                left_contact=left_contact,
                right_contact=right_contact,
                lambda_2d=lambda_2d,
                lambda_trans_prior=lambda_trans_prior,
                lambda_ankle_prior=lambda_ankle_prior,
                lambda_slide=lambda_slide,
                lambda_plane=lambda_plane,
                trans_init=trans_init,
                R_gvhmr_incam2global=R_gvhmr_incam2global,
                lambda_gvhmr_plane_prior=lambda_gvhmr_plane_prior,
            )
            loss.backward()
            print(
                f"[Iter {it}] "
                f"total={stats['total']:.4e}, "
                f"reproj={stats['reproj']:.4e}, "
                f"trans_prior={stats['trans_prior']:.4e}, "
                f"ankle_prior={stats['ankle_prior']:.4e}, "
                f"slide={stats['slide']:.4e}, "
                f"plane={stats['plane']:.4e}, "
                f"gvhmr_plane_prior={stats['gvhmr_plane_prior']:.4e}"
            )
            return loss

        loss_val = optimizer.step(closure).item()

        if last_loss_val is not None:
            improvement = last_loss_val - loss_val
            if improvement <= 0.0:
                no_improve += 1
            else:
                rel_dec = improvement / max(1.0, abs(last_loss_val))
                if rel_dec < rel_tol:
                    no_improve += 1
                else:
                    no_improve = 0
            if no_improve >= patience:
                print(f"Early stopping at outer iter {it} with loss {loss_val:.6f}")
                break
        last_loss_val = loss_val

    print("Optimization finished.")

    # ------------------------------------------------------------------ #
    # 8.5) Compute ground plane and camera-to-world rotation
    # ------------------------------------------------------------------ #
    # Recompute final joints3d_op with optimized parameters
    left_ankle_idx = SMPL_JOINTS["leftFoot"]
    right_ankle_idx = SMPL_JOINTS["rightFoot"]
    pose_body_final = theta_ext_const.clone()  # (T, 21, 3)
    pose_body_final[:, left_ankle_idx, :] += ankle_delta[:, 0, :]
    pose_body_final[:, right_ankle_idx, :] += ankle_delta[:, 1, :]
    body_pose_flat_final = pose_body_final.reshape(T_seq, -1)  # (T, 21*3)

    joints3d_op_final = smpl_body25_keypoints(
        body_model=body_model,
        betas=betas_batch,
        body_pose=body_pose_flat_final,
        root_orient=root_orient_const,
        transl=transl_var,
    )  # (T, 25, 3)

    # Extract foot keypoints (BODY_25 indices: LBigToe=19, LSmallToe=20, LHeel=21, RBigToe=22, RSmallToe=23, RHeel=24)
    left_ids = torch.tensor([19, 20, 21], dtype=torch.long, device=device)
    right_ids = torch.tensor([22, 23, 24], dtype=torch.long, device=device)
    left_pos_final = joints3d_op_final[:, left_ids, :]  # (T, 3, 3)
    right_pos_final = joints3d_op_final[:, right_ids, :]  # (T, 3, 3)

    # Collect contact points for plane estimation
    pts_list = []
    if left_contact.any():
        pts_list.append(left_pos_final[left_contact].reshape(-1, 3))
    if right_contact.any():
        pts_list.append(right_pos_final[right_contact].reshape(-1, 3))

    R_cam2world = None
    if len(pts_list) > 0:
        contact_pts = torch.cat(pts_list, dim=0)  # (M, 3)
        
        if contact_pts.shape[0] >= 3:
            # Compute plane using SVD
            X = contact_pts.detach()
            centroid = X.mean(dim=0, keepdim=True)  # (1, 3)
            Xc = X - centroid
            _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
            normal_cam = Vh[-1]  # (3,) - plane normal in camera coordinates
            normal_cam = normal_cam / (normal_cam.norm() + 1e-8)
            
            # Determine world Y-axis direction based on normal's Y component
            # If normal[1] < 0, then -normal points upward (world Y+)
            if normal_cam[1] > 0:
                world_y_dir = -normal_cam
            else:
                world_y_dir = normal_cam
            
            # World coordinate system Y-axis
            world_y_axis = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=normal_cam.dtype)
            
            # Compute rotation matrix from camera to world coordinates
            R_cam2world = compute_rotation_between_vectors(world_y_dir, world_y_axis)
            print(f"Computed camera-to-world rotation matrix")
            print(f"Ground plane normal (camera): {normal_cam.cpu().numpy()}")
            print(f"World Y direction (camera): {world_y_dir.cpu().numpy()}")
        else:
            print("Warning: Not enough contact points to estimate ground plane. Skipping global coordinate conversion.")
    else:
        print("Warning: No contact points found. Skipping global coordinate conversion.")

    # ------------------------------------------------------------------ #
    # 9) Save updated SMPL params in GVHMR format
    # ------------------------------------------------------------------ #
    gvhmr_results_path = os.path.join(args.gvhmr_dir, "hmr4d_results.pt")
    if not os.path.exists(gvhmr_results_path):
        raise FileNotFoundError(f"GVHMR results file not found at: {gvhmr_results_path}")

    gvhmr_pred = torch.load(gvhmr_results_path, map_location="cpu")
    smpl_params_incam = gvhmr_pred.get("smpl_params_incam", {})

    T_opt = T_seq

    # betas: (T, 10) – repeat original external shape over time and keep first 10
    betas_10 = beta_ext_np.astype(np.float32)
    if betas_10.ndim == 1:
        betas_10 = np.repeat(betas_10[None, :], T_opt, axis=0)
    betas_10_t = torch.tensor(betas_10, dtype=torch.float32)

    # body_pose: (T, 63) – recompute from optimized ankles
    left_ankle_idx = SMPL_JOINTS["leftFoot"]
    right_ankle_idx = SMPL_JOINTS["rightFoot"]
    theta_out = theta_ext_const.detach().cpu().clone()  # (T, 21, 3)
    theta_out[:, left_ankle_idx, :] += ankle_delta.detach().cpu()[:, 0, :]
    theta_out[:, right_ankle_idx, :] += ankle_delta.detach().cpu()[:, 1, :]
    pose_body_out = theta_out.reshape(T_opt, -1)

    global_orient_out = root_orient_const.detach().cpu().reshape(T_opt, 3)
    transl_out = transl_var.detach().cpu().reshape(T_opt, 3)

    smpl_params_incam["betas"] = betas_10_t
    smpl_params_incam["body_pose"] = pose_body_out
    smpl_params_incam["global_orient"] = global_orient_out
    smpl_params_incam["transl"] = transl_out

    gvhmr_pred["smpl_params_incam"] = smpl_params_incam

    # Intrinsics: (T, 3, 3)
    cam_mat_full = torch.tensor(
        cam_mat_np, dtype=torch.float32
    ).unsqueeze(0).repeat(T_opt, 1, 1)
    gvhmr_pred["K_fullimg"] = cam_mat_full

    # ------------------------------------------------------------------ #
    # 10) Compute and save smpl_params_global
    # ------------------------------------------------------------------ #
    smpl_params_global = gvhmr_pred.get("smpl_params_global", {})
    
    if R_cam2world is not None:
        # Convert global_orient from axis-angle to rotation matrix
        global_orient_mat = batch_rodrigues(global_orient_out.reshape(-1, 3))  # (T, 3, 3)
        
        # Apply camera-to-world rotation: R_world = R_cam2world @ R_cam @ R_cam2world.T
        # This is the correct way to transform a rotation matrix between coordinate systems
        R_cam2world_expanded = R_cam2world.unsqueeze(0).expand(T_opt, 3, 3)  # (T, 3, 3)
        global_orient_world_mat = torch.matmul(R_cam2world_expanded, global_orient_mat)  # (T, 3, 3)
        
        # Convert back to axis-angle
        global_orient_world = rotation_matrix_to_angle_axis(
            global_orient_world_mat.reshape(-1, 3, 3)
        ).reshape(T_opt, 3)  # (T, 3)
        
        # Transform translation: t_world = R_cam2world @ t_cam
        transl_world = torch.matmul(
            R_cam2world_expanded,
            transl_out.unsqueeze(-1)
        ).squeeze(-1)  # (T, 3)
        
        # Center X and Z coordinates (set mean to 0)
        # Y coordinate (height) is preserved
        transl_world_mean = transl_world.mean(dim=0)  # (3,)
        transl_world_offset = torch.zeros_like(transl_world_mean)
        transl_world_offset[0] = -transl_world_mean[0]  # X offset
        transl_world_offset[2] = -transl_world_mean[2]  # Z offset
        # Y offset is 0 (preserve height)
        transl_world = transl_world + transl_world_offset.unsqueeze(0)  # (T, 3)
        
        smpl_params_global["betas"] = betas_10_t
        smpl_params_global["body_pose"] = pose_body_out
        smpl_params_global["global_orient"] = global_orient_world
        smpl_params_global["transl"] = transl_world
        
        print("Computed smpl_params_global using ground plane-based coordinate transformation")
        print(f"  Centered X and Z coordinates (offset: X={transl_world_offset[0]:.6f}, Z={transl_world_offset[2]:.6f})")
    else:
        # If no rotation matrix computed, use incam values (fallback)
        # Center X and Z coordinates (set mean to 0)
        # Y coordinate (height) is preserved
        transl_out_mean = transl_out.mean(dim=0)  # (3,)
        transl_out_offset = torch.zeros_like(transl_out_mean)
        transl_out_offset[0] = -transl_out_mean[0]  # X offset
        transl_out_offset[2] = -transl_out_mean[2]  # Z offset
        # Y offset is 0 (preserve height)
        transl_out_centered = transl_out + transl_out_offset.unsqueeze(0)  # (T, 3)
        
        smpl_params_global["betas"] = betas_10_t
        smpl_params_global["body_pose"] = pose_body_out
        smpl_params_global["global_orient"] = global_orient_out
        smpl_params_global["transl"] = transl_out_centered
        print("Warning: Using incam values for smpl_params_global (no ground plane estimated)")
        print(f"  Centered X and Z coordinates (offset: X={transl_out_offset[0]:.6f}, Z={transl_out_offset[2]:.6f})")
    
    gvhmr_pred["smpl_params_global"] = smpl_params_global

    out_pt_path = os.path.join(args.gvhmr_dir, "baseline_foot_correction.pt")
    torch.save(gvhmr_pred, out_pt_path)
    print(f"Saved baseline optimization results in GVHMR format to {out_pt_path}")


if __name__ == "__main__":
    main()


