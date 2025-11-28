# HuMoR Tools

このディレクトリには、HuMoR を使用する際に役立つユーティリティスクリプトが含まれています。

## convert_keypoints_to_openpose.py

他の姿勢推定ツール（VitPose、MMPose、MediaPipe など）から出力された 2D キーポイントを、HuMoR が必要とする OpenPose BODY_25 形式に変換するツールです。

### OpenPose BODY_25 フォーマット

OpenPose BODY_25 は 25 個のキーポイントを持ちます：

```
0: Nose          8: MidHip        16: LEye
1: Neck          9: RHip          17: REar
2: RShoulder    10: RKnee         18: LEar
3: RElbow       11: RAnkle        19: LBigToe
4: RWrist       12: LHip          20: LSmallToe
5: LShoulder    13: LKnee         21: LHeel
6: LElbow       14: LAnkle        22: RBigToe
7: LWrist       15: REye          23: RSmallToe
                                  24: RHeel
```

### 使用方法

#### コマンドラインから使用する場合

```bash
python tools/convert_keypoints_to_openpose.py \
    --input ./input_keypoints \
    --output ./output_keypoints \
    --video-name hiphop_clip1 \
    --format coco
```

引数：

- `--input`: 入力キーポイントファイルのディレクトリ
- `--output`: 出力する OpenPose 形式 JSON のディレクトリ
- `--video-name`: 出力ファイル名のプレフィックス（デフォルト: "video"）
- `--format`: 入力キーポイントの形式（現在は"coco"をサポート）

#### Python スクリプト内で使用する場合

```python
import numpy as np
from tools.convert_keypoints_to_openpose import save_openpose_json, convert_coco_to_body25

# 例1: BODY_25形式のキーポイントを直接保存
keypoints = np.array([[x0, y0], [x1, y1], ..., [x24, y24]])  # (25, 2)
confidences = np.array([c0, c1, ..., c24])  # (25,)
save_openpose_json(keypoints, confidences, "output_000000000000_keypoints.json")

# 例2: COCO形式からBODY_25形式に変換
coco_keypoints = np.array([[x0, y0], ..., [x16, y16]])  # (17, 2)
coco_confidences = np.array([c0, ..., c16])  # (17,)
body25_keypoints, body25_confidences = convert_coco_to_body25(
    coco_keypoints, coco_confidences
)
save_openpose_json(body25_keypoints, body25_confidences, "output.json")
```

### 出力ファイル形式

出力される JSON ファイルは以下の命名規則に従います：

```
{video_name}_000000000000_keypoints.json
{video_name}_000000000001_keypoints.json
{video_name}_000000000002_keypoints.json
...
```

各 JSON ファイルの構造：

```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": [-1],
      "pose_keypoints_2d": [x0, y0, c0, x1, y1, c1, ..., x24, y24, c24],
      "face_keypoints_2d": [],
      "hand_left_keypoints_2d": [],
      "hand_right_keypoints_2d": [],
      "pose_keypoints_3d": [],
      "face_keypoints_3d": [],
      "hand_left_keypoints_3d": [],
      "hand_right_keypoints_3d": []
    }
  ]
}
```

### カスタマイズ

このスクリプトは基本的な COCO 形式からの変換をサポートしています。他の形式（VitPose、MMPose など）を使用する場合は、`process_video_keypoints`関数を修正して、適切なキーポイントの読み込みと変換を実装してください。

### VitPose からの変換例

VitPose を使用している場合の変換例：

```python
import json
import numpy as np
from tools.convert_keypoints_to_openpose import save_openpose_json, convert_coco_to_body25

# VitPoseの出力を読み込む（VitPoseはCOCO形式を出力することが多い）
with open('vitpose_output.json', 'r') as f:
    vitpose_data = json.load(f)

for frame_idx, detection in enumerate(vitpose_data):
    # VitPoseのキーポイント形式に応じて調整
    keypoints = np.array(detection['keypoints']).reshape(-1, 3)
    coco_keypoints = keypoints[:17, :2]  # x, y
    coco_confidences = keypoints[:17, 2]  # confidence

    # COCO -> BODY_25 変換
    body25_keypoints, body25_confidences = convert_coco_to_body25(
        coco_keypoints, coco_confidences
    )

    # OpenPose形式で保存
    output_file = f"video_{frame_idx:012d}_keypoints.json"
    save_openpose_json(body25_keypoints, body25_confidences, output_file)
```

## HuMoR での使用

変換したキーポイントを HuMoR で使用するには：

```bash
# Docker内で実行
python humor/fitting/run_fitting.py \
    --data-path ./data/rgb_demo/hiphop_clip1.mp4 \
    --data-type RGB \
    --op-keypts ./output_keypoints \
    --smpl ./body_models/smplh/male/model.npz \
    --init-motion-prior ./checkpoints/init_state_prior_gmm \
    --humor ./checkpoints/humor/best_model.pth \
    --out ./out/rgb_demo_custom_keypoints \
    --save-results

# または設定ファイルを使用
python humor/fitting/run_fitting.py @./configs/fit_rgb_demo_no_openpose.cfg
```

## トラブルシューティング

### キーポイントの座標系

- OpenPose はピクセル座標（非正規化）を使用します
- 座標は画像の左上を原点(0, 0)とします
- 正規化された座標を使用している場合は、画像サイズで乗算してください

### 信頼度スコア

- 信頼度は 0 から 1 の範囲で指定してください
- キーポイントが検出されない場合は、座標を(0, 0)、信頼度を 0 に設定してください

### フレームレート

- HuMoR は 30fps を想定しています
- 異なるフレームレートの動画を使用する場合は、キーポイントを 30fps にリサンプリングしてください
