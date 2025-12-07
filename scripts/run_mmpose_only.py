from demo_mmpose_external_smpl import run_mmpose_halpe_on_video, halpe_seq_to_body25_seq
import cv2

POSE_CONFIG = "./checkpoints/mmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py"
POSE_CHECKPOINT = "./checkpoints/mmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth"

video_path = "./data/1207_01/20251207_01.mp4"
output_video_path = "./data/1207_01/mmpose_result.mp4"

halpe_seq, (img_h, img_w) = run_mmpose_halpe_on_video(
    pose_config=POSE_CONFIG,
    pose_checkpoint=POSE_CHECKPOINT,
    video_path=video_path,
    start_frame=0,
    end_frame=None,
    device="cpu",
)
body25_seq = halpe_seq_to_body25_seq(halpe_seq)

print(halpe_seq.shape)
print(img_h, img_w)

# save the result as a video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
raw_video = cv2.VideoCapture(video_path)
out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (img_w, img_h))

frame_idx = 0
while True:
    ret, frame = raw_video.read()
    if not ret:
        break
    
    keypoints = body25_seq[frame_idx]
    for key_id, keypoint in enumerate(keypoints):
        x, y, score = keypoint
        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{key_id}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    frame_idx += 1
    out.write(frame)

out.release()
raw_video.release()
