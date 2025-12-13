import os
import datetime
from typing import List, Tuple, Optional

import cv2
import numpy as np
import torch

from humor.fitting.motion_optimizer import MotionOptimizer, J_BODY
from humor.fitting.fitting_utils import (
    NSTAGES,
    DEFAULT_FOCAL_LEN,
    load_vposer,
)
from humor.models.humor_model import HumorModel
from humor.body_model.body_model import BodyModel
from humor.datasets.rgb_dataset import DEFAULT_GROUND
from humor.utils.torch import load_state
from humor.utils.logging import Logger, mkdir

from scripts.read_gvhmr_results import load_gvhmr_results
from scripts.imu import compute_contacts_from_imu
from mmpose.apis import init_model as mm_init_model
from mmpose.apis import inference_topdown as mm_inference_topdown
from mmdet.apis import init_detector as mm_init_detector
from mmdet.apis import inference_detector as mm_inference_detector
from mmengine.registry import init_default_scope


def halpe_to_body25_single(halpe_keypoints: np.ndarray) -> np.ndarray:
    """
    Convert a single-frame Halpe / COCO-WholeBody keypoint array to BODY_25.

    We assume that the incoming keypoints follow the common COCO-WholeBody /
    Halpe ordering where the *first 23* entries are:

        0: nose
        1: left_eye
        2: right_eye
        3: left_ear
        4: right_ear
        5: left_shoulder
        6: right_shoulder
        7: left_elbow
        8: right_elbow
        9: left_wrist
        10: right_wrist
        11: left_hip
        12: right_hip
        13: left_knee
        14: right_knee
        15: left_ankle
        16: right_ankle
        17: left_big_toe
        18: left_small_toe
        19: left_heel
        20: right_big_toe
        21: right_small_toe
        22: right_heel

    and each entry is (x, y, score).

    Returns:
        body25 (25, 3): x, y, confidence in OpenPose BODY_25 order.
    """
    if halpe_keypoints.ndim != 2 or halpe_keypoints.shape[1] != 3:
        raise ValueError(
            f"Expected halpe_keypoints with shape (K, 3), got {halpe_keypoints.shape}"
        )

    if halpe_keypoints.shape[0] < 23:
        raise ValueError(
            "Halpe / whole-body keypoints must contain at least 23 joints "
            "(body + feet) for this converter."
        )

    body25 = np.zeros((25, 3), dtype=np.float32)

    # Direct one-to-one mappings (Halpe index -> BODY_25 index)
    # reference: https://github.com/open-mmlab/mmpose/blob/main/configs/_base_/datasets/halpe26.py
    mapping = {
        0: 0,   # nose -> Nose
        5: 5,   # left_shoulder -> LShoulder
        6: 2,   # right_shoulder -> RShoulder
        7: 6,   # left_elbow -> LElbow
        8: 3,   # right_elbow -> RElbow
        9: 7,   # left_wrist -> LWrist
        10: 4,  # right_wrist -> RWrist
        11: 12, # left_hip -> LHip
        12: 9,  # right_hip -> RHip
        13: 13, # left_knee -> LKnee
        14: 10, # right_knee -> RKnee
        15: 14, # left_ankle -> LAnkle
        16: 11, # right_ankle -> RAnkle
        1: 16,  # left_eye -> LEye
        2: 15,  # right_eye -> REye
        3: 18,  # left_ear -> LEar
        4: 17,  # right_ear -> REar
        20: 19, # left_big_toe  -> LBigToe
        22: 20, # left_small_toe -> LSmallToe
        24: 21, # left_heel -> LHeel
        21: 22, # right_big_toe -> RBigToe
        23: 23, # right_small_toe -> RSmallToe
        25: 24, # right_heel -> RHeel

        18: 1, # neck -> Neck
        19: 8, # hip -> MidHip
    }

    for h_idx, b_idx in mapping.items():
        body25[b_idx] = halpe_keypoints[h_idx]

    # Neck (1): midpoint of shoulders
    # l_shoulder = halpe_keypoints[5]
    # r_shoulder = halpe_keypoints[6]
    # if l_shoulder[2] > 0.0 and r_shoulder[2] > 0.0:
    #     body25[1, :2] = (l_shoulder[:2] + r_shoulder[:2]) * 0.5
    #     body25[1, 2] = (l_shoulder[2] + r_shoulder[2]) * 0.5

    # # MidHip (8): midpoint of hips
    # l_hip = halpe_keypoints[11]
    # r_hip = halpe_keypoints[12]
    # if l_hip[2] > 0.0 and r_hip[2] > 0.0:
    #     body25[8, :2] = (l_hip[:2] + r_hip[:2]) * 0.5
    #     body25[8, 2] = (l_hip[2] + r_hip[2]) * 0.5

    return body25


def halpe_seq_to_body25_seq(halpe_seq: np.ndarray) -> np.ndarray:
    """
    Convert a sequence of Halpe / COCO-WholeBody keypoints to BODY_25.

    Args:
        halpe_seq (T, K, 3): per-frame keypoints in Halpe / whole-body order.

    Returns:
        body25_seq (T, 25, 3)
    """
    halpe_seq = np.asarray(halpe_seq, dtype=np.float32)
    if halpe_seq.ndim != 3 or halpe_seq.shape[2] != 3:
        raise ValueError(
            f"Expected halpe_seq with shape (T, K, 3), got {halpe_seq.shape}"
        )

    T = halpe_seq.shape[0]
    body25_seq = np.zeros((T, 25, 3), dtype=np.float32)
    for t in range(T):
        body25_seq[t] = halpe_to_body25_single(halpe_seq[t])
    return body25_seq


def _select_person_bbox_from_mmdet_result(
    det_result,
    score_thr: float = 0.5,
) -> np.ndarray:
    """
    Extract the highest-scoring person bbox from an MMDetection result.

    Supports both legacy MMDetection outputs (list of ndarrays) and the
    newer DetDataSample-style outputs where bboxes live in
    ``result.pred_instances.bboxes``.

    Returns:
        bboxes (N, 4) in xyxy format. For our single-person case, this will
        be either shape (1, 4) or (0, 4) if no bbox passes the threshold.
    """
    # New-style DetDataSample (MMDetection >= 3.x)
    # if hasattr(det_result, "pred_instances"):
    pred_instances = det_result.pred_instances
    if not hasattr(pred_instances, "bboxes"):
        return np.zeros((0, 4), dtype=np.float32)

    bboxes = pred_instances.bboxes
    scores = getattr(pred_instances, "scores", None)
    labels = getattr(pred_instances, "labels", None)

    if scores is None:
        return np.zeros((0, 4), dtype=np.float32)

    # Move to CPU numpy for convenience
    bboxes = bboxes.cpu().numpy()
    scores = scores.cpu().numpy()
    if labels is not None:
        labels = labels.cpu().numpy()

    # Filter to person class if labels are available (COCO person id = 0)
    if labels is not None:
        person_mask = labels == 0
        bboxes = bboxes[person_mask]
        scores = scores[person_mask]

    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    score_mask = scores >= score_thr
    bboxes = bboxes[score_mask]
    scores = scores[score_mask]
    if bboxes.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    best_idx = int(np.argmax(scores))
    return bboxes[best_idx : best_idx + 1].astype(np.float32)

    # Legacy-style outputs (list of ndarrays, one per class)
    # if isinstance(det_result, tuple):
    #     det_result = det_result[0]

    # if isinstance(det_result, list) and len(det_result) > 0:
    #     # Assume COCO-style ordering: index 0 is person
    #     person_dets = det_result[0]
    #     if person_dets is None or person_dets.size == 0:
    #         return np.zeros((0, 4), dtype=np.float32)

    #     # person_dets: (num_dets, 5) -> [x1, y1, x2, y2, score]
    #     bboxes = person_dets[:, :4]
    #     scores = person_dets[:, 4]

    #     score_mask = scores >= score_thr
    #     bboxes = bboxes[score_mask]
    #     scores = scores[score_mask]
    #     if bboxes.size == 0:
    #         return np.zeros((0, 4), dtype=np.float32)

    #     best_idx = int(np.argmax(scores))
    #     return bboxes[best_idx : best_idx + 1].astype(np.float32)

    # return np.zeros((0, 4), dtype=np.float32)


def run_mmpose_halpe_on_video(
    pose_config: str,
    pose_checkpoint: str,
    video_path: str,
    start_frame: int,
    end_frame: int,
    device: Optional[str] = None,
    det_config: Optional[str] = None,
    det_checkpoint: Optional[str] = None,
    det_score_thr: float = 0.5,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Run an MMPose whole-body model (Halpe / COCO-WholeBody) on a segment of a video.

    This function expects a *top-down* whole-body model whose output keypoints
    follow the Halpe / COCO-WholeBody ordering described in ``halpe_to_body25_single``.

    Args:
        pose_config: Path to MMPose config (.py).
        pose_checkpoint: Path to MMPose checkpoint (.pth) or URL.
        video_path: Path to an RGB video file.
        start_frame: Inclusive frame index (0-based) corresponding to GVHMR frame.
        end_frame: Exclusive frame index.
        device: Device string for MMPose model ("cuda:0", "cpu", ...). If None,
                chooses CUDA if available.
        det_config: Optional path to an MMDetection config (.py) for the
            person detector. If provided together with ``det_checkpoint``, a
            detector will be run per-frame to estimate the person bbox.
        det_checkpoint: Optional path or URL to an MMDetection checkpoint (.pth).
        det_score_thr: Detection score threshold for selecting the person bbox.

    Returns:
        halpe_seq (T, K, 3): detected keypoints per frame.
        image_size (H, W): size of the processed frames.
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise RuntimeError(f"Video appears to have no frames: {video_path}")

    if end_frame is None or end_frame > total_frames:
        end_frame = total_frames

    if start_frame < 0 or start_frame >= end_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}, "
            f"total_frames={total_frames}"
        )

    # Initialize MMPose model (MMPose >= 1.x new API)
    pose_model = mm_init_model(pose_config, pose_checkpoint, device=device)

    # Optional: initialize person detector (MMDetection).
    det_model = None
    if det_config is not None and det_checkpoint is not None:
        # Ensure MMDetection's scope is active while building the detector
        init_default_scope("mmdet")
        det_model = mm_init_detector(det_config, det_checkpoint, device=device)

    halpe_list: List[np.ndarray] = []
    img_h, img_w = None, None
    num_kpts: Optional[int] = None

    frame_idx = 0
    while frame_idx < end_frame:
        ret, img = cap.read()
        if not ret:
            break
        if frame_idx >= start_frame:
            img_h, img_w = img.shape[:2]

            # ------------------------------------------------------------------
            # 1) (Optional) run person detector to get bbox in the frame
            # ------------------------------------------------------------------
            bboxes = None
            if det_model is not None:
                # MMDetection transforms (e.g., PackDetInputs) live under the
                # "mmdet" scope, so make sure it is active before calling
                # inference_detector. Otherwise, MMEngine might try to look for
                # these transforms in the "mmpose" scope and fail.
                init_default_scope("mmdet")
                det_result = mm_inference_detector(det_model, img)
                bboxes = _select_person_bbox_from_mmdet_result(
                    det_result, score_thr=det_score_thr
                )
                if bboxes.shape[0] == 0:
                    # No valid detection for this frame -> return an all-zero
                    # keypoint array consistent with other frames.
                    if num_kpts is None:
                        num_kpts = 133
                    halpe_list.append(np.zeros((num_kpts, 3), dtype=np.float32))
                    frame_idx += 1
                    continue
            print("bboxes:", bboxes)

            # ------------------------------------------------------------------
            # 2) Run MMPose top-down pose estimator inside the bbox
            # ------------------------------------------------------------------
            # Switch to the "mmpose" scope so that the MMPose transforms
            # and models are correctly resolved inside inference_topdown.
            init_default_scope("mmpose")
            pose_results = mm_inference_topdown(
                pose_model,
                img,
                bboxes=bboxes,
            )

            if pose_results:
                # pose_results is a list of PoseDataSample
                data_sample = pose_results[0]
                pred_instances = data_sample.pred_instances
                # (N_instances, K, 2) and (N_instances, K)
                keypoints = pred_instances.keypoints[0].astype(np.float32)
                scores = pred_instances.keypoint_scores[0].astype(np.float32)
                kpts = np.concatenate(
                    [keypoints, scores[..., None]], axis=-1
                )  # (K, 3)
                if num_kpts is None:
                    num_kpts = kpts.shape[0]
                halpe_list.append(kpts)
            else:
                if num_kpts is None:
                    num_kpts = 133
                halpe_list.append(np.zeros((num_kpts, 3), dtype=np.float32))

        frame_idx += 1

    cap.release()

    expected_frames = end_frame - start_frame
    if len(halpe_list) != expected_frames:
        raise RuntimeError(
            f"Expected {expected_frames} frames from video but got {len(halpe_list)}. "
            "Please check start/end frame indices."
        )

    halpe_seq = np.stack(halpe_list, axis=0)
    return halpe_seq, (img_h, img_w)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Demo: optimize HuMoR motion using GVHMR SMPL params + "
            "MMPose (Halpe / whole-body) 2D keypoints mapped to BODY_25."
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
            "Focal length f in pixels. If not provided, HuMoR's DEFAULT_FOCAL_LEN "
            "is used. fx=fy=f and the principal point (cx, cy) is set from the "
            "input video resolution."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for PyTorch / MMPose (e.g., 'cuda:0' or 'cpu'). "
        "Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--contact-csv",
        type=str,
        default=None,
        help=(
            "Optional path to a T x 2 CSV file with manual foot contact labels for "
            "each frame (columns: left_contact, right_contact; values 0 or 1). If "
            "provided, these labels override IMU-based contact detection."
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    # DET_CONFIG = "./checkpoints/mmdet/sparse_rcnn_r50_fpn_1x_coco.py"
    # DET_CHECKPOINT = "./checkpoints/mmdet/sparse_rcnn_r50_fpn_1x_coco_20201222_214453-dc79b137.pth"
    DET_CONFIG = "./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco.py"
    DET_CHECKPOINT = "./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
    POSE_CONFIG = "./checkpoints/mmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
    POSE_CHECKPOINT = "./checkpoints/mmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"

    # Early stopping settings for LBFGS
    REL_LOSS_TOL = 1e-3
    REL_LOSS_PATIENCE = 5

    log_dir = "./out/demo_logs"
    mkdir(log_dir)
    log_path = os.path.join(
        log_dir, f"fit_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    Logger.init(log_path)

    # Paths (assumes running from repo root)
    smplh_model_path = "./body_models/smplh/neutral/model.npz"
    vposer_path = "./body_models/vposer_v1_0"
    humor_ckpt = "./checkpoints/humor/best_model.pth"
    init_motion_prior_dir = "./checkpoints/init_state_prior_gmm"

    # ------------------------------------------------------------------
    # 1) Load GVHMR results (SMPL parameters, camera intrinsics, etc.)
    #    We ignore the VitPose-based COCO keypoints returned here.
    # ------------------------------------------------------------------
    (
        _coco_seq_unused,
        cam_mat_np,
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

    # ------------------------------------------------------------------
    # 2) Run MMPose whole-body (Halpe / COCO-WholeBody) on RGB video
    # ------------------------------------------------------------------
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

    if halpe_seq.shape[0] != n_timestep:
        raise RuntimeError(
            f"Mismatch between GVHMR frames ({n_timestep}) and MMPose frames "
            f"({halpe_seq.shape[0]}). Please check start/end frame settings."
        )

    # Convert Halpe / whole-body -> BODY_25
    body25_seq = halpe_seq_to_body25_seq(halpe_seq)  # (T, 25, 3)

    # ------------------------------------------------------------------
    # 2.5) Build / override camera intrinsics from video resolution and focal length
    # ------------------------------------------------------------------
    if args.focal_lengths is not None:
        fx = fy = float(args.focal_lengths)
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

    # ------------------------------------------------------------------
    # 3) Foot contact detection
    # ------------------------------------------------------------------
    data_fps = 30.0
    if args.contact_csv is not None:
        # --------------------------------------------------------------
        # 3a) Manual contact labels from CSV (T x 2: left, right; 0/1)
        # --------------------------------------------------------------
        contact_labels = np.loadtxt(args.contact_csv, delimiter=",")

        if contact_labels.ndim != 2 or contact_labels.shape[1] != 2:
            raise ValueError(
                f"Expected contact CSV with shape (T, 2), got {contact_labels.shape}. "
                "Columns should be [left_contact, right_contact] with values 0 or 1."
            )

        if contact_labels.shape[0] != n_timestep:
            raise ValueError(
                "Mismatch between number of frames in GVHMR results and contact "
                f"labels: GVHMR={n_timestep}, contact_csv={contact_labels.shape[0]}."
            )

        left_contact = contact_labels[:, 0].astype(bool)
        right_contact = contact_labels[:, 1].astype(bool)
    else:
        # --------------------------------------------------------------
        # 3b) Default: IMU-based contact detection
        # --------------------------------------------------------------
        left_contact, right_contact = compute_contacts_from_imu(
            T=n_timestep,
            fps=data_fps,
        )

    print("cam_mat_np:", cam_mat_np)
    print("beta_ext_np:", beta_ext_np)
    print("theta_ext_np:", theta_ext_np[:3, :, :])
    print("root_orient_np:", root_orient_np[:3, :])
    print("trans_np:", trans_np[:3, :])
    print("halpe_seq (first 3 frames):", halpe_seq[:3, :23, :])
    print("body25_seq (first 3 frames):", body25_seq[:3, :, :])

    # ------------------------------------------------------------------
    # 4) Wrap into observed/gt dicts as used by MotionOptimizer.run
    # ------------------------------------------------------------------
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
    theta_ext = torch.tensor(theta_ext_np, dtype=torch.float32, device=device)
    observed_data["smpl_pose_obs"] = theta_ext.reshape(B, T, 63)

    # External betas: (B, 1, 10) so it plays nicely with any temporal slicing
    betas_obs = torch.tensor(
        beta_ext_np[None, None, :], dtype=torch.float32, device=device
    )
    observed_data["smpl_betas_obs"] = betas_obs

    # Camera intrinsics
    cam_mat = torch.tensor(cam_mat_np[None, ...], dtype=torch.float32, device=device)
    gt_data["cam_matx"] = cam_mat
    gt_data["name"] = ["demo_seq_mmpose"]

    # ------------------------------------------------------------------
    # 5) Loss weights (similar to demo_coco17_external_smpl.py)
    # ------------------------------------------------------------------
    loss_weights = {
        "joints2d": [0.001, 0.001, 0.001],
        "joints3d": [0.0, 0.0, 0.0],
        "joints3d_rollout": [0.0, 0.0, 0.0],
        "verts3d": [0.0, 0.0, 0.0],
        "points3d": [0.0, 0.0, 0.0],
        "pose_prior": [0.04, 0.04, 0.0],
        "shape_prior": [0.05, 0.05, 0.05],
        "motion_prior": [0.0, 0.0, 0.01], #"motion_prior": [0.0, 0.0, 0.075],
        "init_motion_prior": [0.0, 0.0, 0.01], #"init_motion_prior": [0.0, 0.0, 0.075],
        "joint_consistency": [0.0, 0.0, 100.0],
        "bone_length": [0.0, 0.0, 2000.0],
        "joints3d_smooth": [100.0, 100.0, 0.0],
        "contact_vel": [0.0, 0.0, 100.0],
        "contact_height": [0.0, 0.0, 10.0],
        "floor_reg": [0.0, 0.0, 0.167],
        "rgb_overlap_consist": [0.0, 0.0, 0.0],
        # Encourage agreement with external SMPL pose/betas
        "smpl_param_obs": [0.0, 0.01, 0.01],
    }

    all_stage_loss_weights = []
    for sidx in range(NSTAGES):
        stage_loss_weights = {k: v[sidx] for k, v in loss_weights.items()}
        all_stage_loss_weights.append(stage_loss_weights)

    # ------------------------------------------------------------------
    # 6) Load priors and body model
    # ------------------------------------------------------------------
    pose_prior, _ = load_vposer(vposer_path)
    pose_prior = pose_prior.to(device)
    pose_prior.eval()

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
        gmm_weights = torch.tensor(
            gmm_res["weights"], dtype=torch.float32, device=device
        )
        gmm_means = torch.tensor(
            gmm_res["means"], dtype=torch.float32, device=device
        )
        gmm_covs = torch.tensor(
            gmm_res["covariances"], dtype=torch.float32, device=device
        )
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

    # ------------------------------------------------------------------
    # 7) Create MotionOptimizer
    # ------------------------------------------------------------------
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
        im_dim=(img_h, img_w),
    )

    # Provide IMU-based foot contacts to the optimizer
    optimizer.ext_left_contact = torch.tensor(
        left_contact[None, :], dtype=torch.bool, device=device
    )
    optimizer.ext_right_contact = torch.tensor(
        right_contact[None, :], dtype=torch.bool, device=device
    )

    # ------------------------------------------------------------------
    # 8) Initialize SMPL betas & local body pose from external observations
    #    + initialize root_orient & trans from GVHMR
    # ------------------------------------------------------------------
    with torch.no_grad():
        # shape (betas): copy first 10 components from external estimate
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
        theta_ext_b = theta_ext.unsqueeze(0)  # (1, T, 21, 3)
        num_joints_obs = theta_ext_b.shape[2]
        num_joints_fill = min(num_joints_obs, J_BODY)
        body_pose[:, :, : num_joints_fill * 3] = theta_ext_b[
            :, :, :num_joints_fill, :
        ].reshape(B_opt, T_opt, num_joints_fill * 3)

        # encode into VPoser latent space
        optimizer.latent_pose = optimizer.pose2latent(body_pose)

        # root_orient / trans: (T, 3) from GVHMR -> (B, T, 3) as optimizer initialization
        root_orient_ext = torch.tensor(
            root_orient_np, dtype=torch.float32, device=device
        )
        trans_ext = torch.tensor(trans_np, dtype=torch.float32, device=device)
        optimizer.root_orient[:, :, :] = root_orient_ext.unsqueeze(0)
        optimizer.trans[:, :, :] = trans_ext.unsqueeze(0)

    # ------------------------------------------------------------------
    # 9) Run optimization
    # ------------------------------------------------------------------
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
    prior_trans = None
    prior_root_orient = None
    if "stage3" in per_stage_outputs:
        stage3 = per_stage_outputs["stage3"]
        if "prior_trans" in stage3 and "prior_root_orient" in stage3:
            prior_trans = stage3["prior_trans"]
            prior_root_orient = stage3["prior_root_orient"]

    # --------------------------------------------------------------
    # 9.5) Save results in GVHMR hmr4d_results.pt-style format
    # --------------------------------------------------------------
    # We load the original GVHMR results dictionary, replace the
    # SMPL parameters and intrinsics with the optimized ones, and
    # save to a new .pt file.
    gvhmr_results_path = os.path.join(args.gvhmr_dir, "hmr4d_results.pt")
    if not os.path.exists(gvhmr_results_path):
        raise FileNotFoundError(
            f"GVHMR results file not found at: {gvhmr_results_path}"
        )

    gvhmr_pred = torch.load(gvhmr_results_path, map_location="cpu")
    smpl_params_incam = gvhmr_pred.get("smpl_params_incam", {})

    T_opt = T

    # betas: (T, 10) – repeat optimized shape over time and keep
    # only the first 10 components for compatibility.
    betas_10 = (
        betas[0, :10].detach().cpu().unsqueeze(0).repeat(T_opt, 1)
    )  # (T, 10)

    # body_pose: (T, J_BODY * 3) in axis-angle, matching GVHMR.
    body_pose_flat = (
        pose_body[0].detach().cpu().view(T_opt, J_BODY * 3)
    )  # (T, 21*3)

    # global orientation and translation in camera frame: (T, 3)
    global_orient_out = root_orient[0].detach().cpu().view(T_opt, 3)
    transl_out = trans[0].detach().cpu().view(T_opt, 3)

    smpl_params_incam["betas"] = betas_10
    smpl_params_incam["body_pose"] = body_pose_flat
    smpl_params_incam["global_orient"] = global_orient_out
    smpl_params_incam["transl"] = transl_out
    gvhmr_pred["smpl_params_incam"] = smpl_params_incam

    # Intrinsics: (T, 3, 3) – repeat the optimized camera matrix
    # over time to match GVHMR's K_fullimg.
    cam_mat_full = cam_mat.detach().cpu().repeat(T_opt, 1, 1)
    gvhmr_pred["K_fullimg"] = cam_mat_full

    smpl_params_global = gvhmr_pred.get("smpl_params_global", {})
    smpl_params_global["betas"] = betas_10
    smpl_params_global["body_pose"] = body_pose_flat
    
    # Get translation and center X and Z coordinates (set mean to 0)
    prior_trans_global = prior_trans[0].detach().cpu().view(T_opt, 3)  # (T, 3)
    prior_trans_mean = prior_trans_global.mean(dim=0)  # (3,)
    prior_trans_offset = torch.zeros_like(prior_trans_mean)
    prior_trans_offset[0] = -prior_trans_mean[0]  # X offset
    prior_trans_offset[2] = -prior_trans_mean[2]  # Z offset
    # Y offset is 0 (preserve height)
    prior_trans_centered = prior_trans_global + prior_trans_offset.unsqueeze(0)  # (T, 3)
    
    smpl_params_global["transl"] = prior_trans_centered
    smpl_params_global["global_orient"] = prior_root_orient[0].detach().cpu().view(T_opt, 3)
    gvhmr_pred["smpl_params_global"] = smpl_params_global

    out_pt_path = os.path.join(args.gvhmr_dir, "humor_result.pt")
    torch.save(gvhmr_pred, out_pt_path)
    print(f"Saved optimization results in GVHMR format to {out_pt_path}")

    print("Optimization finished.")
    print(f"trans shape: {tuple(trans.shape)}")
    print(f"pose_body shape: {tuple(pose_body.shape)}")
    print(f"betas shape: {tuple(betas.shape)}")
    print("First frame trans:", trans[0, 0].detach().cpu().numpy())
    print("First 5 betas:", betas[0, :5].detach().cpu().numpy())


if __name__ == "__main__":
    main()


