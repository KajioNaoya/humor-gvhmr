import os
import argparse

import numpy as np
import cv2
import torch

from humor.body_model.body_model import BodyModel
from humor.body_model.utils import SMPL_JOINTS, smpl_to_openpose
from humor.viz.utils import create_video
from scripts.demo_mmpose_external_smpl import halpe_seq_to_body25_seq


def smpl_body25_keypoints(
    body_model: BodyModel,
    betas: torch.Tensor,        # (T, num_betas)
    body_pose: torch.Tensor,    # (T, 21*3)
    root_orient: torch.Tensor,  # (T, 3)
    transl: torch.Tensor,       # (T, 3)
) -> torch.Tensor:
    """
    BodyModel (SMPLH/SMPLX) から BODY_25 順の 3D キーポイントを返す。

    Returns:
        joints3d_op: (T, 25, 3)
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


def render_debug_overlay(
    video_path: str,
    out_dir: str,
    K: np.ndarray,             # (3,3)
    verts_seq: np.ndarray,     # (T, Nv, 3) カメラ座標系
    mmpose_body25: np.ndarray, # (T, 25, 3) [x,y,score] in original image coords
    smpl_body25_3d: np.ndarray,# (T, 25, 3) camera 3D
    resize_scale: float = 0.5,
    fps: float = 30.0,
):
    """
    動画に以下を重ねて書き出す:
      - SMPL メッシュのワイヤーフレーム
      - MMPose BODY_25 2D キーポイント (赤)
      - SMPL から投影した BODY_25 2D キーポイント (青)
    """
    if verts_seq.ndim != 3:
        raise ValueError(f"verts_seq must be (T, V, 3), got {verts_seq.shape}")
    T = verts_seq.shape[0]

    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_frames = []
    while len(bg_frames) < T:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        if resize_scale is not None and resize_scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(orig_w * resize_scale), int(orig_h * resize_scale)),
                interpolation=cv2.INTER_AREA,
            )
        bg_frames.append(frame_bgr)
    cap.release()

    if len(bg_frames) == 0:
        raise RuntimeError("No frames read from video; cannot create overlay.")

    seq_len = min(T, len(bg_frames))
    verts_seq = verts_seq[:seq_len]
    bg_frames = bg_frames[:seq_len]
    mmpose_body25 = mmpose_body25[:seq_len]
    smpl_body25_3d = smpl_body25_3d[:seq_len]

    # カメラ行列をリサイズに合わせてスケール
    H_new, W_new = bg_frames[0].shape[:2]
    scale_x = W_new / float(orig_w)
    scale_y = H_new / float(orig_h)
    s = 0.5 * (scale_x + scale_y)

    fx = float(K[0, 0]) * s
    fy = float(K[1, 1]) * s
    cx = float(K[0, 2]) * s
    cy = float(K[1, 2]) * s

    # エッジ集合（メッシュワイヤーフレーム用）を faces から作るのは高コストなので、
    # ここでは簡易に「頂点インデックスを間引いて点列として描画」でもよいが、
    # 顔インデックスがない場合もあるので線分は描かず、点群とキーポイントに限定する実装にする。
    # （もし faces が欲しければ別途 pt から読み取って渡すように拡張可能）

    for t in range(seq_len):
        frame = bg_frames[t].copy()
        verts = verts_seq[t]  # (Nv, 3)

        z = verts[:, 2]
        valid = z > 1e-6
        x = verts[:, 0]
        y = verts[:, 1]

        u = fx * x / z + cx
        v = fy * y / z + cy
        H, W = frame.shape[:2]

        # メッシュ頂点（ワイヤーフレームではなく、粗な点群として可視化）
        step = max(1, verts.shape[0] // 2000)  # 軽量化のため間引き
        for i in range(0, verts.shape[0], step):
            if not valid[i]:
                continue
            ui, vi = int(u[i]), int(v[i])
            if 0 <= ui < W and 0 <= vi < H:
                cv2.circle(frame, (ui, vi), 1, (0, 255, 0), -1)  # 緑

        # MMPose BODY_25 2D（赤）
        kp_m = mmpose_body25[t]  # (25,3) [x,y,score] in original resolution
        for j in range(25):
            x2d, y2d, conf = kp_m[j]
            if conf <= 0.0:
                continue
            u_r = int(x2d * s)
            v_r = int(y2d * s)
            if 0 <= u_r < W and 0 <= v_r < H:
                cv2.circle(frame, (u_r, v_r), 3, (0, 0, 255), -1)  # 赤

        # SMPL BODY_25 3Dキーポイントを 2D に投影（青）
        kp_s_3d = smpl_body25_3d[t]  # (25,3)
        X = kp_s_3d[:, 0]
        Y = kp_s_3d[:, 1]
        Z = kp_s_3d[:, 2]
        valid_z = Z > 1e-6
        u_b = fx * X / Z + cx
        v_b = fy * Y / Z + cy
        for j in range(25):
            if not valid_z[j]:
                continue
            ub = int(u_b[j])
            vb = int(v_b[j])
            if 0 <= ub < W and 0 <= vb < H:
                cv2.circle(frame, (ub, vb), 3, (255, 0, 0), -1)  # 青

        out_fname = os.path.join(out_dir, f"frame_{t:08d}.png")
        cv2.imwrite(out_fname, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    out_video_path = os.path.join(out_dir, "overlay_debug.mp4")
    create_video(os.path.join(out_dir, "frame_%08d.png"), out_video_path, fps)
    print(f"Saved debug overlay video to {out_video_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Debug visualization: overlay SMPL mesh + MMPose BODY_25 + SMPL BODY_25 projection."
    )
    parser.add_argument(
        "--result-pt",
        type=str,
        required=True,
        help="GVHMR-style pt file (e.g., baseline_foot_correction.pt).",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the original RGB video.",
    )
    parser.add_argument(
        "--mmpose-npz",
        type=str,
        required=True,
        help="Path to cached MMPose Halpe npz (e.g., 0_input_video_mmpose_halpe_0_619.npz).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for debug overlay frames and video.",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="smplx",
        choices=["smplh", "smplx"],
        help="Body model type for SMPL projection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for SMPL evaluation (e.g., cpu or cuda:0).",
    )
    parser.add_argument(
        "--overlay-scale",
        type=float,
        default=0.5,
        help="Resize scale for overlay video (1.0=original resolution).",
    )

    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1) Load GVHMR-style result pt
    pred = torch.load(args.result_pt, map_location="cpu")
    smpl_incam = pred.get("smpl_params_incam", None)
    K_full = pred.get("K_fullimg", None)

    if smpl_incam is None or K_full is None:
        raise KeyError("result-pt must contain 'smpl_params_incam' and 'K_fullimg'.")

    betas = smpl_incam["betas"].detach().cpu().numpy()          # (T,10) or (10,)
    body_pose = smpl_incam["body_pose"].detach().cpu().numpy()  # (T,63)
    global_orient = smpl_incam["global_orient"].detach().cpu().numpy()  # (T,3)
    transl = smpl_incam["transl"].detach().cpu().numpy()        # (T,3)

    if isinstance(K_full, torch.Tensor):
        K_first = K_full[0].detach().cpu().numpy()
    else:
        K_first = np.asarray(K_full[0])
    print("Using K_first:", K_first)

    # 2) Load MMPose Halpe npz and convert to BODY_25
    cache = np.load(args.mmpose_npz)
    halpe_seq = cache["halpe_seq"].astype(np.float32)  # (T, K, 3)
    body25_mmpose = halpe_seq_to_body25_seq(halpe_seq) # (T,25,3)

    # 3) Align lengths
    T_smpl = body_pose.shape[0]
    T_mmp = body25_mmpose.shape[0]
    T = min(T_smpl, T_mmp)
    if T_smpl != T_mmp:
        print(f"Warning: SMPL T={T_smpl}, MMPose T={T_mmp}. Using first T={T} frames.")
    betas = betas[:T]
    body_pose = body_pose[:T]
    global_orient = global_orient[:T]
    transl = transl[:T]
    body25_mmpose = body25_mmpose[:T]

    # 4) Normalize betas to (T, num_betas_model)
    betas_t = torch.from_numpy(betas).float().to(device)
    body_pose_t = torch.from_numpy(body_pose).float().to(device)
    global_orient_t = torch.from_numpy(global_orient).float().to(device)
    transl_t = torch.from_numpy(transl).float().to(device)

    # 5) Build body model
    model_type = args.model_type.lower()
    if model_type == "smplx":
        bm_path = "body_models/smplx/SMPLX_NEUTRAL.npz"
        num_betas_model = 10
    else:
        bm_path = "body_models/smplh/neutral/model.npz"
        num_betas_model = 16

    if not os.path.exists(bm_path):
        raise FileNotFoundError(f"Body model not found at {bm_path}")

    betas_model = torch.zeros(T, num_betas_model, dtype=torch.float32, device=device)
    b_src = betas_t
    if b_src.ndim == 1:
        b_src = b_src.unsqueeze(0).expand(T, -1)
    elif b_src.shape[0] != T:
        b_src = b_src[:1, :].expand(T, -1)
    betas_model[:, :b_src.shape[1]] = b_src[:, :betas_model.shape[1]]

    body_model = BodyModel(
        bm_path=bm_path,
        num_betas=num_betas_model,
        batch_size=T,
        use_vtx_selector=True,  # ← ここを True に
        model_type=model_type,
    ).to(device)

    # 6) Compute SMPL vertices in camera frame
    body_pose_flat_t = body_pose_t.reshape(T, -1)  # (T,63)
    smpl_out = body_model(
        betas=betas_model,
        pose_body=body_pose_flat_t,
        root_orient=global_orient_t,
        trans=transl_t,
        return_dict=True,
    )
    verts_seq = smpl_out["v"].detach().cpu().numpy()  # (T, Nv, 3)

    # 7) Compute SMPL BODY_25 3D keypoints
    joints3d_op = smpl_body25_keypoints(
        body_model,
        betas_model,
        body_pose_flat_t,
        global_orient_t,
        transl_t,
    )  # (T,25,3)
    joints3d_op_np = joints3d_op.detach().cpu().numpy()

    # 8) Render debug overlay video
    render_debug_overlay(
        video_path=args.video_path,
        out_dir=args.out_dir,
        K=K_first,
        verts_seq=verts_seq,
        mmpose_body25=body25_mmpose,
        smpl_body25_3d=joints3d_op_np,
        resize_scale=args.overlay_scale,
        fps=30.0,
    )


if __name__ == "__main__":
    main()