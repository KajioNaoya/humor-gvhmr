import os
import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import trimesh
import cv2

from humor.body_model.body_model import BodyModel
from humor.viz.mesh_viewer import MeshViewer
from humor.viz.utils import create_video


def visualize_result(
    betas,
    body_pose,
    global_orient,
    transl,
    model_type="smplx",
    use_meshviewer=True,
    return_vertices=False,
    fps=30,
):
    """
    betas: (n, 10) or (10,)
    body_pose: (n, 63)
    global_orient: (n, 3)
    transl: (n, 3)
    model_type: "smplh" or "smplx"
    """
    # 入力をテンソルに変換
    if isinstance(betas, np.ndarray):
        betas = torch.from_numpy(betas).float()
    if isinstance(body_pose, np.ndarray):
        body_pose = torch.from_numpy(body_pose).float()
    if isinstance(global_orient, np.ndarray):
        global_orient = torch.from_numpy(global_orient).float()
    if isinstance(transl, np.ndarray):
        transl = torch.from_numpy(transl).float()
    
    # 次元を調整
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)
    if body_pose.dim() == 1:
        body_pose = body_pose.unsqueeze(0)
    if global_orient.dim() == 1:
        global_orient = global_orient.unsqueeze(0)
    if transl.dim() == 1:
        transl = transl.unsqueeze(0)
    
    num_frames = betas.shape[0]

    # BodyModel 用モデルパスと beta 次元
    model_type = model_type.lower()
    if model_type == "smplx":
        bm_path = "body_models/smplx/SMPLX_NEUTRAL.npz"
        num_betas_model = 10
    elif model_type == "smplh":
        bm_path = "body_models/smplh/neutral/model.npz"
        num_betas_model = 16
    else:
        raise ValueError(f"Invalid model type for visualization: {model_type}")

    if not os.path.exists(bm_path):
        raise FileNotFoundError(f"Body model not found at: {bm_path}")

    device = torch.device("cpu")

    # betas を BodyModel の num_betas に合わせてパディング／切り詰め
    betas = betas.to(device)
    body_pose = body_pose.to(device)
    global_orient = global_orient.to(device)
    transl = transl.to(device)

    betas_model = torch.zeros(num_frames, num_betas_model, dtype=torch.float32, device=device)
    b_src = betas
    if b_src.shape[0] != num_frames:
        # 時間次元が合わない場合は先頭フレームを複製
        b_src = b_src[:1, :].expand(num_frames, -1)
    max_copy = min(b_src.shape[1], num_betas_model)
    betas_model[:, :max_copy] = b_src[:, :max_copy]

    # BodyModel を構築
    body_model = BodyModel(
        bm_path=bm_path,
        num_betas=num_betas_model,
        batch_size=num_frames,
        use_vtx_selector=True,   # joints 用の追加頂点も含める
        model_type=model_type,
    ).to(device)

    # SMPL 頂点・faces を計算
    body_pose_flat = body_pose.reshape(num_frames, -1)  # (T,63)
    smpl_out = body_model(
        betas=betas_model,
        pose_body=body_pose_flat,
        root_orient=global_orient,
        trans=transl,
        return_dict=True,
    )
    all_vertices_np = smpl_out["v"].detach().cpu().numpy()   # (T, Nv, 3)
    faces = smpl_out["f"].detach().cpu().numpy()             # (F, 3)

    all_vertices = [v for v in all_vertices_np]

    # 3D描画（MeshViewerを使用してメッシュとして表示）
    if use_meshviewer and faces is not None:
        all_trimesh = [trimesh.Trimesh(vertices=v, faces=faces, process=False) for v in all_vertices]
        mv = MeshViewer(
            width=1080,
            height=1080,
            use_offscreen=False,
            follow_camera=True,
            camera_intrinsics=None,
        )
        mv.add_mesh_seq(all_trimesh)
        mv.add_ground()
        mv.animate(fps=fps)

    if return_vertices:
        return all_vertices, faces


def visualize_with_meshviewer(all_vertices, faces, fps=30):
    """HuMoR の MeshViewer を用いてメッシュとして描画する."""
    if faces is None:
        print("No faces found for SMPL model; cannot create mesh.")
        return

    # trimesh.Trimesh のシーケンスを作成
    mesh_seq = []
    for verts in all_vertices:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        mesh_seq.append(mesh)

    # インタラクティブビューア（Raymond ライティング & 地面付き）
    mv = MeshViewer(
        width=1080,
        height=1080,
        use_offscreen=False,
        follow_camera=True,
        camera_intrinsics=None,
    )
    mv.add_mesh_seq(mesh_seq)
    mv.add_ground()
    mv.animate(fps=fps)


def render_overlay_mesh_on_video(
    all_vertices,
    faces,
    video_path,
    K,
    out_dir,
    fps=30,
    resize_scale=0.5,
):
    """
    元動画に SMPL メッシュを重ねてレンダリングし，動画として保存する。

    all_vertices: List[np.ndarray]  (T, N, 3) のリスト（カメラ座標系）
    faces: (F, 3)
    video_path: 元動画パス
    K: (3, 3) カメラ内部パラメータ行列
    out_dir: 連番画像と動画の出力ディレクトリ
    fps: 出力動画のフレームレート
    """
    if faces is None:
        print("No faces provided; cannot overlay mesh.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    # 元動画の解像度（カメラ行列 K はこの解像度に対応している想定）
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bg_frames = []
    target_len = len(all_vertices)
    while len(bg_frames) < target_len:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # 解像度を縮小してメモリ使用量を抑える
        if resize_scale is not None and resize_scale != 1.0:
            frame_bgr = cv2.resize(
                frame_bgr,
                (int(orig_w * resize_scale), int(orig_h * resize_scale)),
                interpolation=cv2.INTER_AREA,
            )

        # BGR のまま uint8 で保持（この上に 2D 投影メッシュを描画）
        bg_frames.append(frame_bgr)
    cap.release()

    if len(bg_frames) == 0:
        raise RuntimeError("No frames read from video; cannot create overlay.")

    seq_len = min(len(bg_frames), len(all_vertices))
    bg_frames = bg_frames[:seq_len]
    verts_seq = all_vertices[:seq_len]

    # 画像を縮小した場合はカメラ内部パラメータも同じ比率でスケーリング
    H_new, W_new = bg_frames[0].shape[:2]
    scale_x = W_new / float(orig_w)
    scale_y = H_new / float(orig_h)
    # 基本的には等方スケーリングを想定
    s = (scale_x + scale_y) * 0.5

    fx = float(K[0, 0]) * s
    fy = float(K[1, 1]) * s
    cx = float(K[0, 2]) * s
    cy = float(K[1, 2]) * s

    # エッジリスト（face から重複のない辺集合を作る）
    faces_np = np.asarray(faces, dtype=np.int32)
    edges = set()
    for f in faces_np:
        i, j, k = int(f[0]), int(f[1]), int(f[2])
        edges.add(tuple(sorted((i, j))))
        edges.add(tuple(sorted((j, k))))
        edges.add(tuple(sorted((k, i))))
    edges = list(edges)

    os.makedirs(out_dir, exist_ok=True)

    # 各フレームごとに 2D 投影したメッシュのワイヤーフレームを重ね描き
    for t in range(seq_len):
        frame = bg_frames[t].copy()
        verts = verts_seq[t]

        # 深度が正の頂点のみ投影
        z = verts[:, 2]
        valid = z > 1e-6
        x = verts[:, 0]
        y = verts[:, 1]

        u = fx * x / z + cx
        v = fy * y / z + cy

        H, W = frame.shape[:2]

        for (i, j) in edges:
            if not (valid[i] and valid[j]):
                continue
            ui, vi = int(u[i]), int(v[i])
            uj, vj = int(u[j]), int(v[j])
            if not (0 <= ui < W and 0 <= vi < H and 0 <= uj < W and 0 <= vj < H):
                continue
            cv2.line(frame, (ui, vi), (uj, vj), (0, 255, 0), 1, lineType=cv2.LINE_AA)

        out_fname = os.path.join(out_dir, f"frame_{t:08d}.png")
        cv2.imwrite(out_fname, frame, [cv2.IMWRITE_PNG_COMPRESSION, 9])

    # ffmpeg を用いて動画化
    out_video_path = os.path.join(out_dir, "overlay.mp4")
    create_video(os.path.join(out_dir, "frame_%08d.png"), out_video_path, fps)
    print(f"Saved overlay video to {out_video_path}")

# def visualize_with_matplotlib(all_vertices, num_frames):
#     """Matplotlibを使用して3D描画（点群として表示、Z軸が鉛直方向、Z=0が地面）"""
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
    
#     # 最初のフレームを表示
#     current_frame = 0
    
#     # 全フレームの座標範囲を計算（地面平面のサイズ決定用）
#     # 頂点座標は (X, Y, Z) とし，Z を「高さ」とみなす（Z=0 が地面）
#     all_coords = np.concatenate(all_vertices, axis=0)
#     x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
#     y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
#     z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
#     # 地面平面（Z=0）の範囲を少し広げる
#     ground_margin = max(x_max - x_min, y_max - y_min) * 0.2
#     ground_x_range = np.linspace(x_min - ground_margin, x_max + ground_margin, 20)
#     ground_y_range = np.linspace(y_min - ground_margin, y_max + ground_margin, 20)
#     ground_x, ground_y = np.meshgrid(ground_x_range, ground_y_range)
#     ground_z = np.zeros_like(ground_x)  # Z=0 の平面
    
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')
    
#     def update_plot():
#         ax.clear()
#         vertices = all_vertices[current_frame]
        
#         # 頂点を10個おきに描画（軽量化）
#         sampled_vertices = vertices[::10]
        
#         # 点群として描画（Z軸が鉛直方向になるように、X, Y, Z の順で描画）
#         ax.scatter(
#             sampled_vertices[:, 0],  # X（横方向）
#             sampled_vertices[:, 1],  # Y（奥行き方向）
#             sampled_vertices[:, 2],  # Z（鉛直方向）
#             s=2,  # 点のサイズ
#             c='lightblue',
#             alpha=0.6,
#             edgecolors='none'
#         )
        
#         # Z=0 に半透明な地面の平面を描画
#         ax.plot_surface(
#             ground_x, ground_y, ground_z,
#             alpha=0.3,
#             color='gray',
#             shade=True
#         )
        
#         # 座標軸を設定（Z軸が鉛直方向）
#         ax.set_xlabel('X')
#         ax.set_ylabel('Y')
#         ax.set_zlabel('Z (Vertical)')
#         ax.set_title(f'FGO Result Visualization (Frame {current_frame+1}/{num_frames}) - Point Cloud (Z=0 ground)')
        
#         # 軸の範囲を設定
#         max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
#         mid_x = (x_max + x_min) * 0.5
#         mid_y = (y_max + y_min) * 0.5
#         mid_z = (z_max + z_min) * 0.5
#         ax.set_xlim(mid_x - max_range, mid_x + max_range)
#         ax.set_ylim(mid_y - max_range, mid_y + max_range)
#         ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
#         ax.set_box_aspect([1, 1, 1])
#         plt.draw()
    
#     def on_key(event):
#         nonlocal current_frame
#         if event.key == 'n' or event.key == 'N':
#             current_frame = (current_frame + 1) % num_frames
#             update_plot()
#         elif event.key == 'p' or event.key == 'P':
#             current_frame = (current_frame - 1) % num_frames
#             update_plot()
    
#     fig.canvas.mpl_connect('key_press_event', on_key)
#     update_plot()
    
#     print("Controls:")
#     print("  N: Next frame")
#     print("  P: Previous frame")
#     print("  Close window to exit")
    
#     plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Visualize HuMoR / GVHMR SMPL results from humor_result.pt "
            "in world coordinates (smpl_params_global)."
        )
    )
    parser.add_argument(
        "--result-pt",
        type=str,
        default=None,
        help="Path to humor_result.pt (GVHMR-style dict with smpl_params_global).",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="smplx",
        choices=["smpl", "smplx"],
        help="Body model type for visualization.",
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="元のRGB動画へのパス（指定するとオーバーレイ動画を出力）",
    )
    parser.add_argument(
        "--overlay-out-dir",
        type=str,
        default=None,
        help="SMPLメッシュを重ねた動画／フレームの出力先ディレクトリ",
    )
    parser.add_argument(
        "--overlay-scale",
        type=float,
        default=0.5,
        help=(
            "オーバーレイ動画の解像度縮小率（1.0: 元解像度, 0.5: 縦横1/2）。"
            "メモリ節約のため Docker などでは 0.5 推奨。"
        ),
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="MeshViewer のインタラクティブな 3D ウィンドウ表示を無効化し、オーバーレイ動画の出力のみにする",
    )

    args = parser.parse_args()
    
    # --------------------------------------------------------------
    # Load from HuMoR-style GVHMR .pt (humor_result.pt)
    # --------------------------------------------------------------
    pt_path = args.result_pt
    pred = torch.load(pt_path, map_location="cpu")
    smpl_params = pred["smpl_params_global"]

    # betas: (T, 10) or (10,) etc.
    betas = smpl_params["betas"].detach().cpu().numpy()

    # body_pose: (T, 63) = (T, 21*3) in axis-angle, as written by demo_mmpose_external_smpl.py
    body_pose = smpl_params["body_pose"].detach().cpu().numpy()

    # global orientation & translation
    global_orient = smpl_params["global_orient"].detach().cpu().numpy()
    transl = smpl_params["transl"].detach().cpu().numpy()

    # Ensure consistent time dimension
    num_frames = body_pose.shape[0]

    # Normalize betas to (num_frames, num_betas)
    if betas.ndim == 1:
        betas = np.repeat(betas[None, :], num_frames, axis=0)
    elif betas.shape[0] != num_frames:
        betas = np.repeat(betas[:1, :], num_frames, axis=0)

    # Sanity print
    print("Loaded from PT:", pt_path)
    print("betas shape:", betas.shape)
    print("body_pose shape:", body_pose.shape)
    print("global_orient shape:", global_orient.shape)
    print("transl shape:", transl.shape)

    # DISPLAY が無い環境（Docker 等）では自動的に MeshViewer GUI を無効化
    has_display = bool(os.environ.get("DISPLAY"))
    use_meshviewer = has_display and (not args.no_viewer)
    if not has_display and not args.no_viewer:
        print(
            "環境変数 DISPLAY が見つからないため、MeshViewer の GUI 表示はスキップします。"
            "（--no-viewer と同等の動作）"
        )

    # 3D: MeshViewer でメッシュとして表示（ワールド座標系：smpl_params_global）
    all_vertices_global, faces = visualize_result(
        betas,
        body_pose,
        global_orient,
        transl,
        model_type=args.model_type,
        use_meshviewer=use_meshviewer,
        return_vertices=True,
        fps=30,
    )

    # --------------------------------------------------------------
    # 元動画に SMPL メッシュをオーバーレイして可視化（カメラ座標系）
    # --------------------------------------------------------------
    if args.video_path is not None:
        smpl_params_incam = pred.get("smpl_params_incam", None)
        K_full = pred.get("K_fullimg", None)

        if smpl_params_incam is None or K_full is None:
            print(
                "humor_result.pt に smpl_params_incam または K_fullimg が無いため、"
                "動画オーバーレイはスキップします。"
            )
        else:
            betas_in = smpl_params_incam["betas"].detach().cpu().numpy()
            body_pose_in = smpl_params_incam["body_pose"].detach().cpu().numpy()
            global_orient_in = (
                smpl_params_incam["global_orient"].detach().cpu().numpy()
            )
            transl_in = smpl_params_incam["transl"].detach().cpu().numpy()

            num_frames_in = body_pose_in.shape[0]

            # betas の時間次元整形（global と同様）
            if betas_in.ndim == 1:
                betas_in = np.repeat(betas_in[None, :], num_frames_in, axis=0)
            elif betas_in.shape[0] != num_frames_in:
                betas_in = np.repeat(betas_in[:1, :], num_frames_in, axis=0)

            all_vertices_incam, _ = visualize_result(
                betas_in,
                body_pose_in,
                global_orient_in,
                transl_in,
                model_type=args.model_type,
                use_meshviewer=False,
                return_vertices=True,
                fps=30,
            )

            # K_fullimg: (T, 3, 3) 想定。最初のフレームを代表として使用。
            if isinstance(K_full, torch.Tensor):
                K_first = K_full[0].detach().cpu().numpy()
            else:
                K_first = np.asarray(K_full[0])
            print("K_first: ", K_first)

            overlay_out_dir = (
                args.overlay_out_dir
                if args.overlay_out_dir is not None
                else os.path.join(os.path.dirname(pt_path), "overlay_render")
            )

            render_overlay_mesh_on_video(
                all_vertices_incam,
                faces,
                args.video_path,
                K_first,
                overlay_out_dir,
                fps=30,
                resize_scale=args.overlay_scale,
            )