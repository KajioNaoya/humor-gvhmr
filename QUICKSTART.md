# HuMoR クイックスタートガイド

このガイドでは、Windows 環境で Docker を使用して HuMoR を最短で実行する手順を説明します。

## 前提条件の確認

- [ ] Docker Desktop for Windows がインストール済み
- [ ] 約 10GB 以上の空きディスク容量がある
- [ ] インターネット接続がある

## セットアップ手順（10 ステップ）

### ステップ 1: リポジトリの準備

すでにこのリポジトリをクローンしている場合は、そのディレクトリに移動してください。

```powershell
cd c:\Users\<your-username>\Documents\GitHub\humor
```

### ステップ 2: 必要なディレクトリを作成

```powershell
mkdir body_models\smplh
mkdir body_models\vposer_v1_0
mkdir checkpoints
```

### ステップ 3: SMPL+H Body Model をダウンロード

1. ブラウザで https://mano.is.tue.mpg.de/ を開く
2. アカウントを作成してログイン
3. "Downloads" → "Extended SMPL+H model (used in AMASS)" をダウンロード
4. `smplh.tar.xz` を `body_models/` に配置
5. 解凍して `body_models/smplh/` に以下の構造を作成：
   ```
   body_models/smplh/
   ├── male/model.npz
   ├── female/model.npz
   └── neutral/model.npz
   ```

### ステップ 4: VPoser v1.0 をダウンロード

1. ブラウザで https://smpl-x.is.tue.mpg.de/ を開く
2. アカウントを作成してログイン
3. "Download" → "VPoser v1.0" をダウンロード（**v2.0 ではない**）
4. `vposer_v1_0.zip` を解凍して `body_models/vposer_v1_0/` に配置

### ステップ 5: 学習済みモデルをダウンロード

```powershell
# PowerShellで実行
Invoke-WebRequest -Uri "http://download.cs.stanford.edu/orion/humor/checkpoints.zip" -OutFile "checkpoints.zip"
Expand-Archive -Path checkpoints.zip -DestinationPath .
Remove-Item checkpoints.zip
```

または、ブラウザでダウンロード：

- http://download.cs.stanford.edu/orion/humor/checkpoints.zip
- プロジェクトルートに解凍

### ステップ 6: ディレクトリ構造を確認

以下の構造になっているか確認：

```
humor/
├── body_models/
│   ├── smplh/
│   │   ├── male/model.npz
│   │   ├── female/model.npz
│   │   └── neutral/model.npz
│   └── vposer_v1_0/
│       └── (VPoserのファイル群)
├── checkpoints/
│   ├── humor/best_model.pth
│   ├── humor_qual/best_model.pth
│   └── init_state_prior_gmm/
├── Dockerfile
├── docker-compose.yml
└── ...
```

### ステップ 7: Docker イメージをビルド

```powershell
docker-compose build
```

⏱️ 初回は 10-20 分程度かかります。コーヒーブレイクの時間です ☕

### ステップ 8: Docker コンテナを起動

```powershell
docker-compose run --rm humor
```

これで Docker コンテナ内の bash シェルが起動します。

### ステップ 9: デモを実行（コンテナ内）

コンテナ内で以下のコマンドを実行：

```bash
# OpenPoseなしでの実行（事前計算されたキーポイントが必要）
python humor/fitting/run_fitting.py @./configs/fit_rgb_demo_no_openpose.cfg

# または、OpenPoseの行をコメントアウトした設定ファイルで実行
# configs/fit_rgb_demo_no_split.cfg を編集して --openpose の行を削除してから：
# python humor/fitting/run_fitting.py @./configs/fit_rgb_demo_no_split.cfg
```

**注意**: デフォルトの設定ファイル（`fit_rgb_demo_no_split.cfg`）は OpenPose を実行しようとします。OpenPose がインストールされていない場合は、以下のいずれかを行ってください：

1. 事前計算されたキーポイントを使用する（`--op-keypts`フラグ）
2. OpenPose の行をコメントアウトする
3. `fit_rgb_demo_no_openpose.cfg`を使用する

### ステップ 10: 結果を確認

処理が完了したら、`out/` ディレクトリに結果が保存されます：

```
out/
└── rgb_demo_no_split/
    └── results_out/
        └── final_results/
            └── stage3_results.npz
```

結果を可視化するには（コンテナ内）：

```bash
python humor/fitting/viz_fitting_rgb.py \
    --results ./out/rgb_demo_no_split/results_out \
    --out ./out/rgb_demo_no_split/viz_out \
    --viz-prior-frame
```

## OpenPose なしでの実行方法

HuMoR は OpenPose がなくても実行できます。以下の 2 つの方法があります：

### 方法 1: 事前計算されたキーポイントを使用

他の姿勢推定ツール（VitPose、MMPose など）で 2D キーポイントを検出し、OpenPose BODY_25 形式に変換して使用します。

```bash
# キーポイント変換ツールを使用
python tools/convert_keypoints_to_openpose.py \
    --input ./your_keypoints \
    --output ./data/rgb_demo/keypoints \
    --video-name hiphop_clip1 \
    --format coco

# HuMoRを実行
python humor/fitting/run_fitting.py \
    --data-path ./data/rgb_demo/hiphop_clip1.mp4 \
    --data-type RGB \
    --op-keypts ./data/rgb_demo/keypoints \
    --smpl ./body_models/smplh/male/model.npz \
    --init-motion-prior ./checkpoints/init_state_prior_gmm \
    --humor ./checkpoints/humor/best_model.pth \
    --out ./out/rgb_demo_custom \
    --save-results
```

### 方法 2: 3D データでテスト

2D キーポイントが不要な 3D データでテストすることもできます：

```bash
# AMASS 3Dデータへのフィッティング（学習用データは不要）
# ただし、このモードではAMASSデータのダウンロードと処理が必要
```

## トラブルシューティング

### エラー: "SMPL+H model not found"

→ `body_models/smplh/male/model.npz` が存在するか確認してください。

### エラー: "VPoser not found"

→ `body_models/vposer_v1_0/` ディレクトリが存在するか確認してください。

### エラー: "OpenPose not found"

→ OpenPose なしで実行するには、`--op-keypts`フラグを使用するか、設定ファイルから`--openpose`行を削除してください。

### CPU 環境での実行が遅い

→ これは正常です。CPU 版 PyTorch は GPU 版より大幅に遅くなります。長い動画の処理には不向きです。

## OpenPose をビルドしたい場合（上級者向け）

OpenPose を自分でビルドする場合：

1. `Dockerfile`内の OpenPose 関連のコメントを解除
2. `docker-compose build --no-cache` で再ビルド（30 分〜1 時間以上かかります）
3. 詳細は [SETUP_WINDOWS.md](./SETUP_WINDOWS.md) の「OpenPose のビルド」セクションを参照

⚠️ **注意**: OpenPose のビルドは時間とリソースを大量に消費します。まずは事前計算されたキーポイントを使用することを推奨します。

## 次のステップ

- 詳細なセットアップ情報: [SETUP_WINDOWS.md](./SETUP_WINDOWS.md)
- キーポイント変換ツール: [tools/README.md](./tools/README.md)
- メインドキュメント: [README.md](./README.md)

## 独自の動画で実行

1. 動画を `data/` ディレクトリに配置
2. VitPose や MMPose で 2D キーポイントを検出
3. `tools/convert_keypoints_to_openpose.py` で OpenPose 形式に変換
4. `--data-path` と `--op-keypts` を指定して HuMoR を実行

詳細は [SETUP_WINDOWS.md](./SETUP_WINDOWS.md) を参照してください。

---

質問や問題がある場合は、GitHub の Issues で報告してください。
