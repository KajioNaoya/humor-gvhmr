# HuMoR Windows セットアップガイド

このガイドでは、Windows 11 環境で Docker を使用して HuMoR を実行するための詳細な手順を説明します。

## 前提条件

- Windows 11
- Docker Desktop for Windows がインストール済み
- 約 10GB 以上の空きディスク容量

## セットアップ手順

### 1. 必要なモデルとデータのダウンロード

HuMoR を実行するには、以下のファイルを手動でダウンロードする必要があります。

#### 1.1 SMPL+H Body Model のダウンロード

SMPL+H は人体の形状・姿勢モデルで、**必須**です。

1. https://mano.is.tue.mpg.de/ にアクセス
2. アカウントを作成してログイン
3. "Downloads" ページに移動
4. "Extended SMPL+H model (used in AMASS)" をダウンロード（`smplh.tar.xz`）
5. ダウンロードしたファイルを解凍：

   ```powershell
   # PowerShellで実行（または7-Zipなどの解凍ツールを使用）
   # まずbody_models/smplhディレクトリを作成
   mkdir body_models\smplh

   # tar.xzファイルを解凍（7-Zipなどを使用）
   # 解凍後、以下のようなディレクトリ構造になることを確認：
   # body_models/
   #   └── smplh/
   #       ├── male/
   #       │   └── model.npz
   #       ├── female/
   #       │   └── model.npz
   #       └── neutral/
   #           └── model.npz
   ```

#### 1.2 VPoser v1.0 のダウンロード

VPoser は姿勢の事前分布モデルで、RGB 動画へのフィッティングに**必須**です。

1. https://smpl-x.is.tue.mpg.de/ にアクセス
2. アカウントを作成してログイン
3. "Download" ページに移動
4. "VPoser: Variational Human Pose Prior" セクションで "Download VPoser v1.0 - CVPR'19" をクリック
   - **重要**: v2.0 ではなく、必ず v1.0 をダウンロードしてください
5. ダウンロードした `vposer_v1_0.zip` を解凍：
   ```powershell
   # body_modelsディレクトリに解凍
   # 解凍後、以下のディレクトリ構造になることを確認：
   # body_models/
   #   └── vposer_v1_0/
   #       ├── snapshots/
   #       ├── V02_05.yaml
   #       └── ...
   ```

#### 1.3 HuMoR 学習済みモデルのダウンロード

HuMoR の学習済みチェックポイント（約 215MB）をダウンロードします。

**方法 1: PowerShell を使用**

```powershell
# プロジェクトルートで実行
Invoke-WebRequest -Uri "http://download.cs.stanford.edu/orion/humor/checkpoints.zip" -OutFile "checkpoints.zip"

# 解凍（7-Zipまたは標準のExpand-Archiveを使用）
Expand-Archive -Path checkpoints.zip -DestinationPath .

# checkpoints.zipを削除（オプション）
Remove-Item checkpoints.zip
```

**方法 2: ブラウザでダウンロード**

1. http://download.cs.stanford.edu/orion/humor/checkpoints.zip をブラウザで開く
2. ダウンロード後、プロジェクトルートに解凍
3. 以下のディレクトリ構造になることを確認：
   ```
   checkpoints/
   ├── humor/
   │   └── best_model.pth
   ├── humor_qual/
   │   └── best_model.pth
   └── init_state_prior_gmm/
       └── ...
   ```

### 2. ディレクトリ構造の確認

すべてのダウンロードが完了したら、以下のディレクトリ構造になっているか確認してください：

```
humor/
├── body_models/
│   ├── smplh/
│   │   ├── male/
│   │   │   └── model.npz
│   │   ├── female/
│   │   │   └── model.npz
│   │   └── neutral/
│   │       └── model.npz
│   └── vposer_v1_0/
│       ├── snapshots/
│       └── V02_05.yaml
├── checkpoints/
│   ├── humor/
│   │   └── best_model.pth
│   ├── humor_qual/
│   │   └── best_model.pth
│   └── init_state_prior_gmm/
├── data/
├── humor/
├── configs/
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### 3. Docker イメージのビルド

```powershell
# プロジェクトルートで実行
docker-compose build
```

ビルドには 10-20 分程度かかります（初回のみ）。

### 4. Docker コンテナの起動

```powershell
# コンテナを起動してbashシェルに入る
docker-compose run --rm humor
```

これで Docker コンテナ内の bash シェルが起動します。

### 5. デモの実行

#### 5.1 OpenPose なしで実行する方法

OpenPose のセットアップは複雑で時間がかかるため、以下の 2 つの代替方法があります：

**方法 A: 事前計算された 2D キーポイントを使用（推奨）**

別の姿勢推定ツール（VitPose、MMPose、MediaPipe など）で 2D キーポイントを検出し、OpenPose BODY_25 形式の JSON に変換して使用します。

```bash
# コンテナ内で実行
python humor/fitting/run_fitting.py \
    --data-path ./data/rgb_demo/hiphop_clip1.mp4 \
    --data-type RGB \
    --op-keypts ./data/rgb_demo/keypoints/ \
    --smpl ./body_models/smplh/male/model.npz \
    --init-motion-prior ./checkpoints/init_state_prior_gmm \
    --humor ./checkpoints/humor/best_model.pth \
    --out ./out/rgb_demo_no_openpose \
    --save-results \
    --save-stages-results
```

**方法 B: 設定ファイルを使用**

```bash
# コンテナ内で実行（OpenPoseの行をコメントアウトした設定ファイルを使用）
python humor/fitting/run_fitting.py @./configs/fit_rgb_demo_no_split.cfg
```

注: デフォルトの設定ファイルは OpenPose を実行しようとするため、`--op-keypts`を指定するか、設定ファイルの`--openpose`行を削除する必要があります。

#### 5.2 結果の可視化

```bash
# コンテナ内で実行
python humor/fitting/viz_fitting_rgb.py \
    --results ./out/rgb_demo_no_split/results_out \
    --out ./out/rgb_demo_no_split/viz_out \
    --viz-prior-frame
```

結果は `./out/rgb_demo_no_split/viz_out/` ディレクトリに保存されます。

### 6. 独自の動画での実行

独自の動画を使用する場合：

1. 動画ファイルを `data/` ディレクトリに配置
2. 姿勢推定ツールで 2D キーポイントを検出
3. キーポイントを OpenPose BODY_25 形式の JSON に変換（下記参照）
4. `--data-path` と `--op-keypts` を指定して実行

## OpenPose BODY_25 形式への変換

OpenPose を使用せずに他の姿勢推定ツールを使用する場合、その出力を OpenPose BODY_25 形式に変換する必要があります。

### BODY_25 スケルトン定義

OpenPose BODY_25 は 25 個のキーポイントを持ちます：

```
0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip,
9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
15: REye, 16: LEye, 17: REar, 18: LEar,
19: LBigToe, 20: LSmallToe, 21: LHeel,
22: RBigToe, 23: RSmallToe, 24: RHeel
```

### JSON 形式

各フレームごとに、以下の形式の JSON ファイルを作成します：

```json
{
  "version": 1.3,
  "people": [
    {
      "person_id": [-1],
      "pose_keypoints_2d": [
        x0, y0, c0, x1, y1, c1, ..., x24, y24, c24
      ],
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

- `xi, yi`: ピクセル座標（非正規化）
- `ci`: 信頼度スコア（0-1）

### 変換スクリプトの例

`tools/convert_keypoints_to_openpose.py` として保存：

```python
import json
import numpy as np
import os

def convert_to_openpose_body25(keypoints, confidences, output_path):
    """
    他の形式のキーポイントをOpenPose BODY_25形式に変換

    Args:
        keypoints: shape (25, 2) のnumpy配列 [[x, y], ...]
        confidences: shape (25,) のnumpy配列 [c0, c1, ...]
        output_path: 出力JSONファイルのパス
    """
    # OpenPose形式: [x0, y0, c0, x1, y1, c1, ...]
    pose_keypoints_2d = []
    for i in range(25):
        pose_keypoints_2d.extend([
            float(keypoints[i, 0]),
            float(keypoints[i, 1]),
            float(confidences[i])
        ])

    openpose_data = {
        "version": 1.3,
        "people": [
            {
                "person_id": [-1],
                "pose_keypoints_2d": pose_keypoints_2d,
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

    with open(output_path, 'w') as f:
        json.dump(openpose_data, f)

# 使用例
# keypoints_from_vitpose = np.array([[x0, y0], [x1, y1], ...])  # (25, 2)
# confidences_from_vitpose = np.array([c0, c1, ...])  # (25,)
# convert_to_openpose_body25(keypoints_from_vitpose, confidences_from_vitpose,
#                            "output_000000000000_keypoints.json")
```

### ファイル命名規則

OpenPose の出力 JSON ファイルは以下の命名規則に従います：

```
{video_name}_000000000000_keypoints.json
{video_name}_000000000001_keypoints.json
{video_name}_000000000002_keypoints.json
...
```

例：`hiphop_clip1_000000000000_keypoints.json`

## トラブルシューティング

### エラー: "SMPL+H model not found"

`body_models/smplh/` ディレクトリが正しく配置されているか確認してください。

### エラー: "VPoser not found"

`body_models/vposer_v1_0/` ディレクトリが正しく配置されているか確認してください。

### エラー: "Checkpoint not found"

`checkpoints/humor/best_model.pth` が存在するか確認してください。

### Docker 起動時の権限エラー

Docker Desktop for Windows が管理者権限で実行されているか確認してください。

### レンダリングエラー

CPU 環境では一部のレンダリング機能が制限される場合があります。`PYOPENGL_PLATFORM=osmesa` 環境変数が設定されているか確認してください（Dockerfile で自動設定済み）。

## 参考情報

- HuMoR 公式リポジトリ: https://github.com/davrempe/humor
- HuMoR プロジェクトページ: https://geometry.stanford.edu/projects/humor/
- SMPL+H: https://mano.is.tue.mpg.de/
- VPoser: https://github.com/nghorbani/human_body_prior

## CPU 環境での性能に関する注意

このセットアップは CPU 版の PyTorch を使用しています。GPU 版に比べて以下の点に注意してください：

- 処理速度が大幅に遅くなります（数十倍〜数百倍）
- メモリ使用量が増加する可能性があります
- 長時間の動画処理には不向きです

本格的な使用には、NVIDIA GPU と CUDA 環境の使用を強く推奨します。

## OpenPose のビルド（オプション）

OpenPose を自分でビルドしたい場合は、以下の手順を実行してください。

### 前提条件

- Dockerfile の編集が必要
- ビルドには 30 分〜1 時間以上かかります
- 十分なディスク容量（追加で 5GB 以上）が必要

### 手順

1. **Dockerfile を編集**

`Dockerfile`を開き、以下のセクションのコメントアウトを解除します：

```dockerfile
# ============================================================================
# OpenPose dependencies (OPTIONAL - uncomment to build OpenPose)
# ============================================================================
```

コメントを解除する行：

- OpenPose 依存関係のインストール（cmake, libopencv-dev など）
- OpenPose のクローンとビルド
- PYTHONPATH の設定

2. **Docker イメージを再ビルド**

```powershell
docker-compose build --no-cache
```

⚠️ ビルドには非常に時間がかかります（30 分〜1 時間以上）

3. **OpenPose の確認**

コンテナ内で以下のコマンドを実行して、OpenPose が正しくビルドされたか確認：

```bash
ls -la /workspace/external/openpose/build/
./external/openpose/build/examples/openpose/openpose.bin --help
```

4. **HuMoR で OpenPose を使用**

OpenPose がビルドされている場合、デフォルトの設定ファイルが使用できます：

```bash
python humor/fitting/run_fitting.py @./configs/fit_rgb_demo_no_split.cfg
```

### OpenPose ビルド時の注意事項

- **CPU 版のみ**: この Dockerfile は CPU 版 OpenPose をビルドします。GPU 版は別途 CUDA 環境が必要です。
- **ビルド時間**: CPU のスペックによっては 1 時間以上かかることがあります。
- **メモリ**: ビルド中に 4GB 以上のメモリを使用する可能性があります。
- **モデルの自動ダウンロード**: BODY_25 モデルは自動でダウンロードされますが、手や顔のモデルは無効化されています。

### トラブルシューティング：OpenPose ビルド

#### CMake エラー

→ OpenCV 関連のライブラリが不足している可能性があります。Dockerfile の依存関係セクションを確認してください。

#### メモリ不足エラー

→ `make -j$(nproc)` の部分を `make -j2` に変更して並列ビルド数を減らしてください。

#### ビルドが非常に遅い

→ これは正常です。OpenPose は大規模なプロジェクトで、CPU 版のビルドには時間がかかります。
