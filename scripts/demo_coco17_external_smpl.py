import os
import numpy as np
import torch

from tools.convert_keypoints_to_openpose import convert_coco_seq_to_body25

from humor.fitting.motion_optimizer import MotionOptimizer, J_BODY
from humor.fitting.fitting_utils import NSTAGES, DEFAULT_FOCAL_LEN, load_vposer
from humor.models.humor_model import HumorModel
from humor.body_model.body_model import BodyModel
from humor.datasets.rgb_dataset import DEFAULT_GROUND
from humor.utils.torch import load_state
from humor.utils.logging import Logger, mkdir
import datetime

from scripts.read_gvhmr_results import load_gvhmr_results
from scripts.imu import compute_contacts_from_imu

def build_dummy_inputs(n_timestep=30, image_size=(1080, 1080)):
    """
    Build dummy COCO17 keypoints, camera intrinsics, and external SMPL parameters.
    """
    T = n_timestep

    # Dummy COCO17 keypoints (x, y in image coordinates, confidence in [0,1])
    h, w = image_size
    coco_seq = np.zeros((T, 17, 3), dtype=np.float32)
    # random positions inside the image
    coco_seq[:, :, 0] = np.random.uniform(0, w, size=(T, 17))
    coco_seq[:, :, 1] = np.random.uniform(0, h, size=(T, 17))
    coco_seq[:, :, 2] = np.random.uniform(0.5, 1.0, size=(T, 17))  # reasonably confident

    # Camera intrinsics (simple pinhole, centered principal point)
    fx, fy = DEFAULT_FOCAL_LEN
    cam_mat = np.array(
        [
            [fx, 0.0, w / 2.0],
            [0.0, fy, h / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    # External SMPL parameters: beta (10,), theta (T, 21, 3)
    beta_ext = np.random.randn(10).astype(np.float32) * 0.03
    theta_ext = np.random.randn(T, 21, 3).astype(np.float32) * 0.1

    return coco_seq, cam_mat, beta_ext, theta_ext


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Early stopping settings for LBFGS
    REL_LOSS_TOL = 1e-3
    REL_LOSS_PATIENCE = 5

    log_dir = "./out/demo_logs"
    mkdir(log_dir)
    log_path = os.path.join(log_dir, f"fit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    Logger.init(log_path)

    # Paths (assumes running from repo root)
    smplh_model_path = "./body_models/smplh/neutral/model.npz"
    vposer_path = "./body_models/vposer_v1_0"
    humor_ckpt = "./checkpoints/humor/best_model.pth"
    init_motion_prior_dir = "./checkpoints/init_state_prior_gmm"

    # Build inputs from GVHMR results
    coco_seq, cam_mat_np, beta_ext_np, theta_ext_np, root_orient_np, trans_np = load_gvhmr_results(
        gvhmr_dir="./data/1029_01", start_frame=900, end_frame=1050
    )
    n_timestep = theta_ext_np.shape[0]

    # --------------------------------------------------
    # IMU-based foot contact detection (for future use in optimization)
    # --------------------------------------------------
    data_fps = 30.0
    left_contact, right_contact = compute_contacts_from_imu(
        T=n_timestep,
        fps=data_fps,
    )

    print("coco_seq:", coco_seq[:3, :, :])
    print("cam_mat_np:", cam_mat_np)
    print("beta_ext_np:", beta_ext_np)
    print("theta_ext_np:", theta_ext_np[:3, :, :])
    print("root_orient_np:", root_orient_np[:3, :])
    print("trans_np:", trans_np[:3, :])
    print("left_contact:", left_contact)
    print("right_contact:", right_contact)
    input()

    # Convert COCO17 -> BODY_25
    body25_seq = convert_coco_seq_to_body25(coco_seq)  # (T, 25, 3)

    # Wrap into observed/gt dicts as used by MotionOptimizer.run
    B, T = 1, n_timestep
    observed_data = {}
    gt_data = {}

    # joints2d: (B, T, 25, 3)
    observed_data["joints2d"] = torch.tensor(
        body25_seq[None, ...], dtype=torch.float32, device=device
    )

    # floor_plane: (B, 4) using default ground plane
    observed_data["floor_plane"] = torch.tensor(
        np.array(DEFAULT_GROUND, dtype=np.float32)[None, :], device=device
    )

    # seq_interval: (B, 2) = [0, T)
    observed_data["seq_interval"] = torch.tensor(
        [[0, T]], dtype=torch.int32, device=device
    )

    # External SMPL pose observations:
    theta_ext = torch.tensor(theta_ext_np, dtype=torch.float32, device=device)  # (T, 21, 3)
    observed_data["smpl_pose_obs"] = theta_ext.reshape(B, T, 63)

    # External betas: (B, 1, 10) so it plays nicely with any temporal slicing
    betas_obs = torch.tensor(
        beta_ext_np[None, None, :], dtype=torch.float32, device=device
    )
    observed_data["smpl_betas_obs"] = betas_obs

    # Camera intrinsics
    cam_mat = torch.tensor(cam_mat_np[None, ...], dtype=torch.float32, device=device)
    gt_data["cam_matx"] = cam_mat
    gt_data["name"] = ["demo_seq"]

    # Loss weights: simple example taking inspiration from fit_rgb_demo configs
    # Format: 3 stages [stage1, stage2, stage3]
    loss_weights = {
        "joints2d": [0.001, 0.001, 0.001],
        "joints3d": [0.0, 0.0, 0.0],
        "joints3d_rollout": [0.0, 0.0, 0.0],
        "verts3d": [0.0, 0.0, 0.0],
        "points3d": [0.0, 0.0, 0.0],
        "pose_prior": [0.04, 0.04, 0.0],
        "shape_prior": [0.05, 0.05, 0.05],
        "motion_prior": [0.0, 0.0, 0.075],
        "init_motion_prior": [0.0, 0.0, 0.075],
        "joint_consistency": [0.0, 0.0, 100.0],
        "bone_length": [0.0, 0.0, 2000.0],
        "joints3d_smooth": [100.0, 100.0, 0.0],
        "contact_vel": [0.0, 0.0, 100.0],
        "contact_height": [0.0, 0.0, 10.0],
        "floor_reg": [0.0, 0.0, 0.167],
        "rgb_overlap_consist": [0.0, 0.0, 0.0],
        # new term: encourage agreement with external SMPL pose/betas
        "smpl_param_obs": [0.0, 0.1, 0.1],
    }

    all_stage_loss_weights = []
    for sidx in range(NSTAGES):
        stage_loss_weights = {k: v[sidx] for k, v in loss_weights.items()}
        all_stage_loss_weights.append(stage_loss_weights)

    # Load pose prior (VPoser)
    pose_prior, _ = load_vposer(vposer_path)
    pose_prior = pose_prior.to(device)
    pose_prior.eval()

    # Load HuMoR motion prior
    motion_prior = HumorModel(
        in_rot_rep="mat",
        out_rot_rep="aa",
        latent_size=48,
        model_data_config="smpl+joints+contacts",
        steps_in=1,
    ).to(device)
    load_state(humor_ckpt, motion_prior, map_location=device)
    motion_prior.eval()

    # Load initial state prior GMM
    init_motion_prior = {}
    gmm_path = os.path.join(init_motion_prior_dir, "prior_gmm.npz")
    if os.path.exists(gmm_path):
        gmm_res = np.load(gmm_path)
        gmm_weights = torch.tensor(gmm_res["weights"], dtype=torch.float32, device=device)
        gmm_means = torch.tensor(gmm_res["means"], dtype=torch.float32, device=device)
        gmm_covs = torch.tensor(gmm_res["covariances"], dtype=torch.float32, device=device)
        init_motion_prior["gmm"] = (gmm_weights, gmm_means, gmm_covs)
    else:
        init_motion_prior = None

    # Body model (SMPL+H)
    body_model = BodyModel(
        bm_path=smplh_model_path,
        num_betas=16,
        batch_size=B * T,
        use_vtx_selector=True,
    ).to(device)

    # Basic optimizer settings
    num_iters = [30, 80, 70]
    lr = 1.0
    lbfgs_max_iter = 20

    # Create MotionOptimizer
    optimizer = MotionOptimizer(
        device,
        body_model,
        num_betas=16,
        batch_size=B,
        seq_len=T,
        observed_modalities=list(observed_data.keys()),
        loss_weights=all_stage_loss_weights,
        pose_prior=pose_prior,
        motion_prior=motion_prior,
        init_motion_prior=init_motion_prior,
        optim_floor=True,
        camera_matrix=cam_mat,
        robust_loss_type="bisquare",
        robust_tuning_const=4.6851,
        joint2d_sigma=100.0,
        stage3_tune_init_state=True,
        stage3_tune_init_num_frames=15,
        stage3_tune_init_freeze_start=30,
        stage3_tune_init_freeze_end=55,
        stage3_contact_refine_only=False,
        use_chamfer=False,
        im_dim=(image_size := (1080, 1080)),
    )

    # --------------------------------------------------
    # Provide IMU-based foot contacts to the optimizer
    # (used to overwrite per-frame left/right foot contact probabilities)
    # --------------------------------------------------
    optimizer.ext_left_contact = torch.tensor(
        left_contact[None, :], dtype=torch.bool, device=device
    )
    optimizer.ext_right_contact = torch.tensor(
        right_contact[None, :], dtype=torch.bool, device=device
    )

    # --------------------------------------------------
    # Initialize SMPL betas & local body pose from external observations
    # + initialize root_orient & trans from GVHMR
    # --------------------------------------------------
    with torch.no_grad():
        # shape (betas): copy first 10 components from external estimate
        # betas_obs: (B, 1, 10) -> (B, 10)
        ext_betas_10 = betas_obs[:, 0, :]  # (B, 10)
        num_beta = ext_betas_10.shape[-1]
        optimizer.betas[:, :num_beta] = ext_betas_10

        # body pose: build (B, T, J_BODY*3) from external local joint angles
        B_opt, T_opt = B, T
        body_pose = torch.zeros(
            (B_opt, T_opt, J_BODY * 3),
            dtype=torch.float32,
            device=device,
        )
        # theta_ext: (T, 21, 3) -> (B, T, 21, 3)
        theta_ext_b = theta_ext.unsqueeze(0)
        num_joints_obs = theta_ext_b.shape[2]
        num_joints_fill = min(num_joints_obs, J_BODY)
        body_pose[:, :, : num_joints_fill * 3] = theta_ext_b[
            :, :, :num_joints_fill, :
        ].reshape(B_opt, T_opt, num_joints_fill * 3)

        # encode into VPoser latent space
        optimizer.latent_pose = optimizer.pose2latent(body_pose)

        # root_orient / trans: (T, 3) from GVHMR -> (B, T, 3) as optimizer initialization
        root_orient_ext = torch.tensor(root_orient_np, dtype=torch.float32, device=device)
        trans_ext = torch.tensor(trans_np, dtype=torch.float32, device=device)
        optimizer.root_orient[:, :, :] = root_orient_ext.unsqueeze(0)
        optimizer.trans[:, :, :] = trans_ext.unsqueeze(0)

    # Run optimization
    optim_result, per_stage_outputs = optimizer.run(
        observed_data,
        data_fps=data_fps,
        lr=lr,
        num_iter=num_iters,
        lbfgs_max_iter=lbfgs_max_iter,
        stages_res_out=None,
        fit_gender="neutral",
        rel_loss_tol=REL_LOSS_TOL,
        rel_loss_patience=REL_LOSS_PATIENCE,
    )

    trans = optim_result["trans"]  # (B, T, 3) in camera frame
    pose_body = optim_result["pose_body"]  # (B, T, J_body*3)
    root_orient = optim_result["root_orient"]  # (B, T, 3) in camera frame
    betas = optim_result["betas"]  # (B, num_betas)

    # Also get motion in the prior (ground) frame where the floor is y=0, if available.
    # When optim_floor=True, stage3 outputs include:
    #   'prior_trans'      : (B, T, 3) in ground/prior coordinates
    #   'prior_root_orient': (B, T, 3) in ground/prior coordinates
    prior_trans = None
    prior_root_orient = None
    if "stage3" in per_stage_outputs:
        stage3 = per_stage_outputs["stage3"]
        if "prior_trans" in stage3 and "prior_root_orient" in stage3:
            prior_trans = stage3["prior_trans"]
            prior_root_orient = stage3["prior_root_orient"]

    # Save to .npz file
    # NOTE: stage3 の prior 系変数は grad 付きなので detach() してから .numpy() する
    np.savez(
        "optim_result_vars.npz",
        trans=trans.detach().cpu().numpy(),
        trans_prior=prior_trans.detach().cpu().numpy(),
        pose_body=pose_body.detach().cpu().numpy(),
        root_orient=root_orient.detach().cpu().numpy(),
        root_orient_prior=prior_root_orient.detach().cpu().numpy(),
        betas=betas.detach().cpu().numpy(),
    )
    print("Saved trans, pose_body, betas to optim_result_vars.npz")

    print("Optimization finished.")
    print(f"trans shape: {tuple(trans.shape)}")
    print(f"pose_body shape: {tuple(pose_body.shape)}")
    print(f"betas shape: {tuple(betas.shape)}")
    print("First frame trans:", trans[0, 0].cpu().numpy())
    print("First 5 betas:", betas[0, :5].cpu().numpy())


if __name__ == "__main__":
    main()


