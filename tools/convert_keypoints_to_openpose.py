"""
Convert keypoints from various pose estimation tools to OpenPose BODY_25 format.

This script provides utilities to convert keypoints from different pose estimation
frameworks (e.g., VitPose, MMPose, MediaPipe) to the OpenPose BODY_25 format
required by HuMoR.

OpenPose BODY_25 skeleton (25 keypoints):
0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip,
9: RHip, 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
15: REye, 16: LEye, 17: REar, 18: LEar,
19: LBigToe, 20: LSmallToe, 21: LHeel,
22: RBigToe, 23: RSmallToe, 24: RHeel

Usage:
    python convert_keypoints_to_openpose.py --input <input_dir> --output <output_dir> --format <format>
"""

import json
import numpy as np
import os
import argparse
from pathlib import Path


def save_openpose_json(keypoints, confidences, output_path, frame_name=""):
    """
    Save keypoints in OpenPose BODY_25 JSON format.
    
    Args:
        keypoints: numpy array of shape (25, 2) with [x, y] coordinates
        confidences: numpy array of shape (25,) with confidence scores
        output_path: path to save the JSON file
        frame_name: optional frame name for the filename
    """
    # OpenPose format: [x0, y0, c0, x1, y1, c1, ...]
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
        json.dump(openpose_data, f, indent=2)


def convert_coco_to_body25(coco_keypoints, coco_confidences):
    """
    Convert COCO format (17 keypoints) to OpenPose BODY_25 format (25 keypoints).
    
    COCO keypoints (17):
    0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    Args:
        coco_keypoints: numpy array of shape (17, 2)
        coco_confidences: numpy array of shape (17,)
        
    Returns:
        body25_keypoints: numpy array of shape (25, 2)
        body25_confidences: numpy array of shape (25,)
    """
    body25_keypoints = np.zeros((25, 2), dtype=np.float32)
    body25_confidences = np.zeros(25, dtype=np.float32)
    
    # Mapping from COCO to BODY_25 indices
    coco_to_body25 = {
        0: 0,   # nose -> Nose
        1: 16,  # left_eye -> LEye
        2: 15,  # right_eye -> REye
        3: 18,  # left_ear -> LEar
        4: 17,  # right_ear -> REar
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
    }
    
    for coco_idx, body25_idx in coco_to_body25.items():
        body25_keypoints[body25_idx] = coco_keypoints[coco_idx]
        body25_confidences[body25_idx] = coco_confidences[coco_idx]
    
    # Compute Neck (1) as midpoint of shoulders
    if coco_confidences[5] > 0 and coco_confidences[6] > 0:
        body25_keypoints[1] = (coco_keypoints[5] + coco_keypoints[6]) / 2
        body25_confidences[1] = (coco_confidences[5] + coco_confidences[6]) / 2
    
    # Compute MidHip (8) as midpoint of hips
    if coco_confidences[11] > 0 and coco_confidences[12] > 0:
        body25_keypoints[8] = (coco_keypoints[11] + coco_keypoints[12]) / 2
        body25_confidences[8] = (coco_confidences[11] + coco_confidences[12]) / 2
    
    # Foot keypoints (19-24) are not in COCO, set to zero
    # These can be estimated or left as zero
    
    return body25_keypoints, body25_confidences


def convert_coco_seq_to_body25(coco_seq):
    """
    Convert a sequence of COCO keypoints to OpenPose BODY_25 format.

    This is a convenience wrapper around ``convert_coco_to_body25`` for
    in-memory arrays.

    Args:
        coco_seq: numpy array of shape (T, 17, 3)
                  per-frame COCO keypoints [x, y, confidence].

    Returns:
        body25_seq: numpy array of shape (T, 25, 3)
                    per-frame BODY_25 keypoints [x, y, confidence].
    """
    coco_seq = np.asarray(coco_seq, dtype=np.float32)
    if coco_seq.ndim != 3 or coco_seq.shape[1] != 17 or coco_seq.shape[2] != 3:
        raise ValueError(
            f"Expected coco_seq with shape (T, 17, 3), but got {coco_seq.shape}"
        )

    T = coco_seq.shape[0]
    body25_keypoints_seq = np.zeros((T, 25, 2), dtype=np.float32)
    body25_conf_seq = np.zeros((T, 25), dtype=np.float32)

    for t in range(T):
        coco_frame = coco_seq[t]
        coco_keypoints = coco_frame[:, :2]
        coco_confidences = coco_frame[:, 2]
        body25_keypoints, body25_confidences = convert_coco_to_body25(
            coco_keypoints, coco_confidences
        )
        body25_keypoints_seq[t] = body25_keypoints
        body25_conf_seq[t] = body25_confidences

    body25_seq = np.concatenate(
        [body25_keypoints_seq, body25_conf_seq[..., None]], axis=-1
    )
    return body25_seq


def process_video_keypoints(input_dir, output_dir, video_name="video", input_format="coco"):
    """
    Process keypoints from a directory and convert to OpenPose format.
    
    Args:
        input_dir: directory containing input keypoint files
        output_dir: directory to save OpenPose JSON files
        video_name: name prefix for output files
        input_format: format of input keypoints ("coco", "vitpose", etc.)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # This is a template - you need to implement the actual loading based on your input format
    input_files = sorted(Path(input_dir).glob("*.json"))
    
    for frame_idx, input_file in enumerate(input_files):
        # Load your keypoints here
        # Example for COCO format:
        with open(input_file, 'r') as f:
            data = json.load(f)
            # Adjust this based on your actual input format
            # This is just an example structure
            if "keypoints" in data:
                # Assuming shape (17, 3) for COCO [x, y, confidence]
                coco_data = np.array(data["keypoints"]).reshape(-1, 3)
                coco_keypoints = coco_data[:, :2]
                coco_confidences = coco_data[:, 2]
                
                # Convert to BODY_25
                body25_keypoints, body25_confidences = convert_coco_to_body25(
                    coco_keypoints, coco_confidences
                )
                
                # Save in OpenPose format
                output_filename = f"{video_name}_{frame_idx:012d}_keypoints.json"
                output_path = os.path.join(output_dir, output_filename)
                save_openpose_json(body25_keypoints, body25_confidences, output_path, video_name)
                
                print(f"Processed frame {frame_idx}: {output_filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert keypoints to OpenPose BODY_25 format"
    )
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input directory containing keypoint files"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for OpenPose JSON files"
    )
    parser.add_argument(
        "--video-name", type=str, default="video",
        help="Video name prefix for output files"
    )
    parser.add_argument(
        "--format", type=str, default="coco",
        choices=["coco", "custom"],
        help="Input keypoint format"
    )
    
    args = parser.parse_args()
    
    print(f"Converting keypoints from {args.input} to {args.output}")
    print(f"Input format: {args.format}")
    
    process_video_keypoints(
        args.input,
        args.output,
        args.video_name,
        args.format
    )
    
    print("Conversion complete!")


if __name__ == "__main__":
    # Example usage for programmatic access:
    # from convert_keypoints_to_openpose import save_openpose_json
    # keypoints = np.random.rand(25, 2) * 100  # Example: 25 keypoints with x,y in [0, 100]
    # confidences = np.random.rand(25)  # Example: confidence scores
    # save_openpose_json(keypoints, confidences, "output_000000000000_keypoints.json")
    
    main()

