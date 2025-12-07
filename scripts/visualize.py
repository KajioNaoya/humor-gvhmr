import smplx
import numpy as np
import torch

def visualize_result(betas, body_pose, global_orient, transl, model_type="smpl"):
    """
    betas: (n, 10) or (10,)
    body_pose: (n, 63)
    global_orient: (n, 3)
    transl: (n, 3)
    model_type: "smpl" or "smplx"
    model_path: Path to SMPL model file
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
    
    # モデルパスの設定
    if model_type == "smpl":
        print("SMPL model is not supported")
        # model_path = "SMPL_models/SMPL_NEUTRAL.pkl"
    elif model_type == "smplx":
        model_path = "body_models/smplx/SMPLX_NEUTRAL.npz"
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    # SMPLモデルをsmplx.createで作成
    model = smplx.create(
        model_path=model_path,
        model_type=model_type,
        create_global_orient=True,
        create_body_pose=True,
        create_betas=True,
        create_transl=True,
        batch_size=1,
        device="cpu"
    )
    model.eval()

    # モデルが期待するベータ数（shape parameters）の次元に合わせる
    # SMPL/SMPL-X の shapecoeff 数と、最適化結果のベータ次元数が異なると
    # smplx.lbs.blend_shapes 内の einsum で次元不一致エラーが出るため、
    # ここでモデル側の次元に揃える（多い分は切り捨て、足りない分は 0 埋め）。
    num_betas_model = model.betas.shape[1]
    if betas.shape[1] > num_betas_model:
        # 余分な次元は先頭から使用し、残りは捨てる
        betas = betas[:, :num_betas_model]
    elif betas.shape[1] < num_betas_model:
        # 足りない次元は 0 でパディング
        pad = betas.new_zeros((betas.shape[0], num_betas_model - betas.shape[1]))
        betas = torch.cat([betas, pad], dim=1)

    # 各フレームのメッシュを生成
    all_vertices = []
    for i in range(num_frames):
        output = model(
            betas=betas[i:i+1],
            body_pose=body_pose[i:i+1],
            global_orient=global_orient[i:i+1],
            transl=transl[i:i+1],
            return_verts=True
        )
        all_vertices.append(output.vertices.squeeze(0).detach().cpu().numpy())
    
    # 3D描画（Matplotlibを使用、点群として表示）
    visualize_with_matplotlib(all_vertices, num_frames)

def visualize_with_matplotlib(all_vertices, num_frames):
    """Matplotlibを使用して3D描画（点群として表示、Z軸が鉛直方向、Z=0が地面）"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 最初のフレームを表示
    current_frame = 0
    
    # 全フレームの座標範囲を計算（地面平面のサイズ決定用）
    # 頂点座標は (X, Y, Z) とし，Z を「高さ」とみなす（Z=0 が地面）
    all_coords = np.concatenate(all_vertices, axis=0)
    x_min, x_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    y_min, y_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    z_min, z_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    # 地面平面（Z=0）の範囲を少し広げる
    ground_margin = max(x_max - x_min, y_max - y_min) * 0.2
    ground_x_range = np.linspace(x_min - ground_margin, x_max + ground_margin, 20)
    ground_y_range = np.linspace(y_min - ground_margin, y_max + ground_margin, 20)
    ground_x, ground_y = np.meshgrid(ground_x_range, ground_y_range)
    ground_z = np.zeros_like(ground_x)  # Z=0 の平面
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    def update_plot():
        ax.clear()
        vertices = all_vertices[current_frame]
        
        # 頂点を10個おきに描画（軽量化）
        sampled_vertices = vertices[::10]
        
        # 点群として描画（Z軸が鉛直方向になるように、X, Y, Z の順で描画）
        ax.scatter(
            sampled_vertices[:, 0],  # X（横方向）
            sampled_vertices[:, 1],  # Y（奥行き方向）
            sampled_vertices[:, 2],  # Z（鉛直方向）
            s=2,  # 点のサイズ
            c='lightblue',
            alpha=0.6,
            edgecolors='none'
        )
        
        # Z=0 に半透明な地面の平面を描画
        ax.plot_surface(
            ground_x, ground_y, ground_z,
            alpha=0.3,
            color='gray',
            shade=True
        )
        
        # 座標軸を設定（Z軸が鉛直方向）
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z (Vertical)')
        ax.set_title(f'FGO Result Visualization (Frame {current_frame+1}/{num_frames}) - Point Cloud (Z=0 ground)')
        
        # 軸の範囲を設定
        max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
        mid_x = (x_max + x_min) * 0.5
        mid_y = (y_max + y_min) * 0.5
        mid_z = (z_max + z_min) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        ax.set_box_aspect([1, 1, 1])
        plt.draw()
    
    def on_key(event):
        nonlocal current_frame
        if event.key == 'n' or event.key == 'N':
            current_frame = (current_frame + 1) % num_frames
            update_plot()
        elif event.key == 'p' or event.key == 'P':
            current_frame = (current_frame - 1) % num_frames
            update_plot()
    
    fig.canvas.mpl_connect('key_press_event', on_key)
    update_plot()
    
    print("Controls:")
    print("  N: Next frame")
    print("  P: Previous frame")
    print("  Close window to exit")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    import os

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

    visualize_result(
        betas, body_pose, global_orient, transl, model_type=args.model_type
    )