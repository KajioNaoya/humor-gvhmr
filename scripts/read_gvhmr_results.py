import os
from re import T
import torch
import numpy as np

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
        theta_ext: (T, 24, 3)
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

    if start_frame is not None and end_frame is not None:
        coco_seq = coco_seq[start_frame:end_frame, :, :]
        pose_body = pose_body[start_frame:end_frame, :, :]

    return coco_seq, cam_mat, shape, pose_body