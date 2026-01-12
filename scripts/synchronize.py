import argparse
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import cv2

try:
    from scipy.signal import find_peaks, butter, filtfilt, medfilt
except Exception:  # pragma: no cover
    # scipy is expected in this repo; keep a clear error later if missing
    find_peaks = None  # type: ignore[assignment]
    butter = None  # type: ignore[assignment]
    filtfilt = None  # type: ignore[assignment]
    medfilt = None  # type: ignore[assignment]

from scripts.imu import read_imu_orphe
from scripts.demo_mmpose_external_smpl import (
    run_mmpose_halpe_on_video,
    halpe_seq_to_body25_seq,
)


@dataclass(frozen=True)
class Peak1D:
    t: float
    value: float
    idx: int


def _read_video_info(video_path: str) -> Tuple[float, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, total_frames


def _nanmean2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise nanmean of two 1D arrays with equal length."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError(f"Expected 1D arrays with same shape, got a={a.shape}, b={b.shape}")
    out = np.empty_like(a, dtype=np.float64)
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    both_nan = a_nan & b_nan
    out[both_nan] = np.nan
    out[~both_nan] = np.where(
        a_nan[~both_nan],
        b[~both_nan],
        np.where(b_nan[~both_nan], a[~both_nan], 0.5 * (a[~both_nan] + b[~both_nan])),
    )
    return out


def _local_maxima_indices(x: np.ndarray) -> np.ndarray:
    """Fallback peak finder: strict local maxima indices for 1D array."""
    if x.size < 3:
        return np.empty((0,), dtype=np.int64)
    return np.where((x[1:-1] > x[0:-2]) & (x[1:-1] >= x[2:]))[0] + 1


def _find_peaks_1d(
    x: np.ndarray,
    *,
    distance: Optional[int] = None,
    prominence: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={x.shape}")

    if find_peaks is None:
        idx = _local_maxima_indices(x)
        props: Dict[str, Any] = {"peak_heights": x[idx] if idx.size > 0 else np.empty((0,), dtype=np.float64)}
        return idx.astype(np.int64), props

    kwargs: Dict[str, Any] = {}
    if distance is not None:
        kwargs["distance"] = int(distance)
    if prominence is not None:
        kwargs["prominence"] = float(prominence)

    idx, props = find_peaks(x, **kwargs)
    if "peak_heights" not in props:
        props = dict(props)
        props["peak_heights"] = x[idx] if idx.size > 0 else np.empty((0,), dtype=np.float64)
    return idx.astype(np.int64), props


def _pick_top2_peaks_by_height(
    t: np.ndarray,
    x: np.ndarray,
    peak_idx: np.ndarray,
    peak_heights: np.ndarray,
) -> Tuple[Peak1D, Peak1D]:
    if peak_idx.size < 2:
        raise ValueError(f"Need >=2 peaks, got {peak_idx.size}")

    order = np.argsort(peak_heights)[::-1]  # descending by height
    i0 = int(peak_idx[int(order[0])])
    i1 = int(peak_idx[int(order[1])])
    p0 = Peak1D(t=float(t[i0]), value=float(x[i0]), idx=i0)
    p1 = Peak1D(t=float(t[i1]), value=float(x[i1]), idx=i1)
    if p1.t < p0.t:
        return p1, p0
    return p0, p1


def _camera_two_jump_peaks(
    *,
    video_path: str,
    calib_start_frame: int,
    calib_end_frame: int,
    pose_config: str,
    pose_checkpoint: str,
    device: str,
    det_config: Optional[str],
    det_checkpoint: Optional[str],
    det_score_thr: float,
    cam_peak_distance_frames: int,
    cam_peak_prominence: Optional[float],
    heel_score_thr: float,
) -> Tuple[Peak1D, Peak1D, float, float]:
    """
    Detect two jump apex peaks from heel Y in a calibration segment.

    Returns:
      (cam1, cam2, dt_cam, fps)
    where cam1/cam2.t are in *camera time* with frame 0 at t=0, i.e. t = frame_idx / fps.
    """
    fps, total_frames = _read_video_info(video_path)
    if total_frames <= 0:
        raise RuntimeError(f"Video appears to have no frames: {video_path}")

    start_f = int(calib_start_frame)
    end_f_incl = int(calib_end_frame)
    if start_f < 0 or end_f_incl < start_f:
        raise ValueError(f"Invalid calib range: start={start_f}, end={end_f_incl}")
    if end_f_incl >= total_frames:
        raise ValueError(f"calib_end_frame={end_f_incl} out of range (total_frames={total_frames})")

    end_f_excl = end_f_incl + 1
    halpe_seq, _img_size = run_mmpose_halpe_on_video(
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        video_path=video_path,
        start_frame=start_f,
        end_frame=end_f_excl,
        device=device,
        det_config=det_config,
        det_checkpoint=det_checkpoint,
        det_score_thr=float(det_score_thr),
    )
    if halpe_seq.shape[0] < 3:
        raise ValueError("Too few frames processed in the calib segment; please widen the segment.")

    body25 = halpe_seq_to_body25_seq(halpe_seq)  # (Tseg, 25, 3)
    # BODY_25 heel indices (OpenPose): LHeel=21, RHeel=24
    l_y = body25[:, 21, 1].astype(np.float64)
    r_y = body25[:, 24, 1].astype(np.float64)
    l_s = body25[:, 21, 2].astype(np.float64)
    r_s = body25[:, 24, 2].astype(np.float64)
    l_y = np.where(l_s >= float(heel_score_thr), l_y, np.nan)
    r_y = np.where(r_s >= float(heel_score_thr), r_y, np.nan)

    y_mean = _nanmean2(l_y, r_y)

    # Build camera-time timestamps (frame0 -> t=0)
    frame_idx = (start_f + np.arange(y_mean.shape[0], dtype=np.float64)).astype(np.float64)
    t = frame_idx / float(fps)

    # y is pixel coords (down is positive). Jump apex -> smallest y -> peak of (-y)
    x = -y_mean
    valid = ~np.isnan(x)
    if int(valid.sum()) < 3:
        raise ValueError("Heel Y is NaN for most frames in the segment; cannot detect jumps.")

    # For peak finding, replace NaNs with very small values so they won't become peaks
    x_filled = x.copy()
    x_filled[~valid] = np.nanmin(x[valid]) - 1.0

    peak_idx, props = _find_peaks_1d(
        x_filled,
        distance=int(cam_peak_distance_frames) if cam_peak_distance_frames > 0 else None,
        prominence=cam_peak_prominence,
    )
    peak_heights = np.asarray(props.get("peak_heights", x_filled[peak_idx]), dtype=np.float64)

    if peak_idx.size < 2:
        # Fallback: pick the two best (largest -y_mean) points (not necessarily local peaks)
        sort_idx = np.argsort(x_filled)[::-1]
        i0 = int(sort_idx[0])
        i1 = int(sort_idx[1]) if sort_idx.size > 1 else int(sort_idx[0])
        p0 = Peak1D(t=float(t[i0]), value=float(y_mean[i0]), idx=i0)
        p1 = Peak1D(t=float(t[i1]), value=float(y_mean[i1]), idx=i1)
        cam1, cam2 = (p1, p0) if p1.t < p0.t else (p0, p1)
    else:
        cam1_x, cam2_x = _pick_top2_peaks_by_height(t, x_filled, peak_idx, peak_heights)
        cam1 = Peak1D(t=cam1_x.t, value=float(y_mean[cam1_x.idx]), idx=cam1_x.idx)
        cam2 = Peak1D(t=cam2_x.t, value=float(y_mean[cam2_x.idx]), idx=cam2_x.idx)

    dt_cam = float(cam2.t - cam1.t)
    if dt_cam <= 0:
        raise ValueError(f"Invalid camera dt_cam={dt_cam}. Check calib segment / peak selection.")

    # Debug
    print("=== Camera jump detection ===")
    print(f"[camera] segment_frames: {start_f}..{end_f_incl} (inclusive)")
    print(f"[camera] fps: {fps}")
    print(f"[camera] selected_peaks: (t1={cam1.t:.6f}s, y_mean={cam1.value:.3f}), (t2={cam2.t:.6f}s, y_mean={cam2.value:.3f})")
    print(f"[camera] dt_cam_sec: {dt_cam:.6f}")
    print(f"[camera] all_peak_candidates_count: {int(peak_idx.size)}")
    if peak_idx.size > 0:
        cand_ts = t[peak_idx]
        cand_vals = y_mean[peak_idx]
        n = int(min(30, peak_idx.size))
        print(f"[camera] peak_candidates_t_sec(head): {cand_ts[:n]}")
        print(f"[camera] peak_candidates_y_mean(head): {cand_vals[:n]}")

    return cam1, cam2, dt_cam, float(fps)


def _estimate_fs_from_timestamps(t: np.ndarray) -> float:
    t = np.asarray(t, dtype=np.float64)
    if t.size < 3:
        return 0.0
    dt = np.diff(t)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return 0.0
    return float(1.0 / np.median(dt))


def _robust_mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    # 1.4826 scales MAD to be comparable to std for normal dist
    return float(1.4826 * mad)


def _robust_imu_peaks_acc_z(
    t: np.ndarray,
    acc_z: np.ndarray,
    *,
    baseline_sec: float = 1.0,
    lowpass_hz: float = 20.0,
    min_distance_sec: float = 0.20,
    k_prom: float = 6.0,
    k_height: float = 6.0,
    polarity: str = "positive",  # positive | negative | abs
    distance_samples_override: Optional[int] = None,
    prominence_override: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any], np.ndarray, float, Dict[str, float]]:
    """
    Robust peak detection for IMU acc_z.

    Steps:
      1) Remove slow trend by subtracting running median (medfilt).
      2) Low-pass filter to suppress high-frequency noise.
      3) Use MAD-based thresholds for `height` and `prominence`, plus a minimum peak distance.
    """
    if find_peaks is None or butter is None or filtfilt is None or medfilt is None:
        raise ImportError("scipy is required for IMU peak detection (scipy.signal)")

    t = np.asarray(t, dtype=np.float64)
    x = np.asarray(acc_z, dtype=np.float64)
    if t.ndim != 1 or x.ndim != 1 or t.shape[0] != x.shape[0]:
        raise ValueError(f"Expected 1D arrays with same length, got t={t.shape}, acc_z={x.shape}")

    fs = _estimate_fs_from_timestamps(t)
    if fs <= 0:
        fs = 0.0

    # 1) baseline removal with median filter
    y = x.copy()
    if baseline_sec is not None and float(baseline_sec) > 0 and fs > 0:
        k = int(round(float(baseline_sec) * fs))
        k = max(3, k | 1)  # odd, >=3
        try:
            base = medfilt(y, kernel_size=int(k))
            y = y - base
        except Exception:
            y = y - float(np.median(y))
    else:
        y = y - float(np.median(y))

    # 2) low-pass filter
    if lowpass_hz is not None and float(lowpass_hz) > 0 and fs > 0:
        wn = min(0.999, float(lowpass_hz) / (0.5 * fs))
        wn = max(wn, 1e-6)
        b, a = butter(4, wn, btype="low", analog=False)
        min_len = 3 * (max(len(a), len(b)) - 1)
        if y.size > min_len:
            y = filtfilt(b, a, y)

    # 3) choose polarity
    pol = str(polarity).lower().strip()
    if pol == "positive":
        z = y
    elif pol == "negative":
        z = -y
    elif pol == "abs":
        z = np.abs(y)
    else:
        raise ValueError(f"Unknown polarity='{polarity}'. Expected: positive, negative, abs")

    # thresholds (MAD-based)
    z_med = float(np.median(z))
    z_mad = _robust_mad(z) + 1e-12
    height = float(z_med + float(k_height) * z_mad)
    prominence = float(float(k_prom) * z_mad)

    # min distance in samples
    if distance_samples_override is not None:
        distance = int(distance_samples_override)
    else:
        if fs > 0:
            distance = int(round(float(min_distance_sec) * fs))
        else:
            distance = 1
    distance = max(1, distance)

    # allow explicit override for prominence (useful for tuning)
    if prominence_override is not None:
        prominence = float(prominence_override)

    idx, props = find_peaks(z, distance=distance, prominence=prominence, height=height)
    props = dict(props)
    if "peak_heights" not in props:
        props["peak_heights"] = z[idx] if idx.size > 0 else np.empty((0,), dtype=np.float64)

    dbg = {
        "fs_hz": float(fs),
        "baseline_sec": float(baseline_sec),
        "lowpass_hz": float(lowpass_hz),
        "min_distance_sec": float(min_distance_sec),
        "distance_samples": float(distance),
        "mad": float(z_mad),
        "height_thr": float(height),
        "prominence_thr": float(prominence),
    }
    return idx.astype(np.int64), props, y.astype(np.float64), float(fs), dbg


def _imu_select_jump_quadruple(
    *,
    foot_data: np.ndarray,
    dt_cam: float,
    tolerance_ratio: float,
    imu_peak_distance_samples: Optional[int],
    imu_peak_prominence: Optional[float],
    label: str,
    imu_baseline_sec: float,
    imu_lowpass_hz: float,
    imu_min_distance_sec: float,
    imu_k_prom: float,
    imu_k_height: float,
    imu_polarity: str,
    flight_time_min_sec: float,
    flight_time_max_sec: float,
) -> Tuple[Peak1D, Peak1D, Peak1D, Peak1D]:
    if foot_data.shape[0] == 0:
        raise ValueError(f"No IMU data for {label}")

    t = foot_data[:, 0].astype(np.float64)
    acc_z = foot_data[:, 6].astype(np.float64)

    peak_idx, props, detrended, _fs, dbg = _robust_imu_peaks_acc_z(
        t=t,
        acc_z=acc_z,
        baseline_sec=float(imu_baseline_sec),
        lowpass_hz=float(imu_lowpass_hz),
        min_distance_sec=float(imu_min_distance_sec),
        k_prom=float(imu_k_prom),
        k_height=float(imu_k_height),
        polarity=str(imu_polarity),
        distance_samples_override=imu_peak_distance_samples,
        prominence_override=imu_peak_prominence,
    )
    peak_heights = np.asarray(props.get("peak_heights", np.empty((0,), dtype=np.float64)), dtype=np.float64)

    # Debug: list detected peaks
    print(f"=== IMU jump detection ({label}) ===")
    print(
        f"[imu:{label}] fs_hz={dbg['fs_hz']:.3f}, baseline_sec={dbg['baseline_sec']:.3f}, lowpass_hz={dbg['lowpass_hz']:.3f}, "
        f"min_distance_sec={dbg['min_distance_sec']:.3f} (distance_samples={int(dbg['distance_samples'])})"
    )
    print(
        f"[imu:{label}] thresholds: height_thr={dbg['height_thr']:.6f}, prominence_thr={dbg['prominence_thr']:.6f}, mad={dbg['mad']:.6f}"
    )
    print(f"[imu:{label}] peaks_count: {int(peak_idx.size)}")
    if peak_idx.size > 0:
        peak_ts = t[peak_idx]
        n = int(min(150, peak_idx.size))
        print(f"[imu:{label}] peaks_t_sec(head): {peak_ts[:n]}")
        print(f"[imu:{label}] peaks_acc_z_raw(head): {acc_z[peak_idx][:n]}")
        print(f"[imu:{label}] peaks_signal_detrended(head): {detrended[peak_idx][:n]}")
        print(f"[imu:{label}] peaks_height_used(head): {peak_heights[:n]}")

    if peak_idx.size < 4:
        raise ValueError(f"Need >=4 IMU peaks for {label}, got {peak_idx.size}. Consider adjusting prominence/distance.")

    dt_min = float(dt_cam) * (1.0 - float(tolerance_ratio))
    dt_max = float(dt_cam) * (1.0 + float(tolerance_ratio))
    if dt_min <= 0:
        raise ValueError(f"Invalid dt_cam={dt_cam} or tolerance_ratio={tolerance_ratio}")

    ft_min = float(flight_time_min_sec)
    ft_max = float(flight_time_max_sec)
    if ft_min <= 0 or ft_max <= 0 or ft_max < ft_min:
        raise ValueError(f"Invalid flight time range: [{ft_min}, {ft_max}]")

    candidates: List[Tuple[int, int, int, int, float, float, float, float]] = []
    # tuple: (i1,i2,i3,i4, dt12, dt13, dt34, score)
    for k in range(int(peak_idx.size) - 3):
        i1 = int(peak_idx[k + 0])
        i2 = int(peak_idx[k + 1])
        i3 = int(peak_idx[k + 2])
        i4 = int(peak_idx[k + 3])

        t1 = float(t[i1])
        t2 = float(t[i2])
        t3 = float(t[i3])
        t4 = float(t[i4])

        dt12 = t2 - t1
        dt13 = t3 - t1
        dt34 = t4 - t3

        if not (ft_min <= dt12 <= ft_max):
            continue
        if not (dt_min <= dt13 <= dt_max):
            continue
        if not (ft_min <= dt34 <= ft_max):
            continue

        # score: prefer smaller dt13 error, then larger total peak heights
        h1 = float(peak_heights[k + 0]) if peak_heights.size > k + 0 else 0.0
        h2 = float(peak_heights[k + 1]) if peak_heights.size > k + 1 else 0.0
        h3 = float(peak_heights[k + 2]) if peak_heights.size > k + 2 else 0.0
        h4 = float(peak_heights[k + 3]) if peak_heights.size > k + 3 else 0.0
        sum_h = h1 + h2 + h3 + h4
        norm_err = abs(dt13 - float(dt_cam)) / max(1e-6, float(dt_cam))
        score = float(sum_h - 100.0 * norm_err)
        candidates.append((i1, i2, i3, i4, float(dt12), float(dt13), float(dt34), float(score)))

    print(f"[imu:{label}] dt_cam_sec={dt_cam:.6f}, tol={tolerance_ratio:.3f} => dt13_range=[{dt_min:.6f}, {dt_max:.6f}]")
    print(f"[imu:{label}] flight_time_range=[{ft_min:.3f}, {ft_max:.3f}]")
    print(f"[imu:{label}] matched_quadruples(4 consecutive peaks): {len(candidates)}")

    if len(candidates) == 0:
        raise ValueError(
            f"No 4-consecutive-peak quadruple matched (dt12/dt13/dt34) for {label}. "
            f"Try relaxing flight time range or dt tolerance."
        )

    best = max(candidates, key=lambda x: x[7])
    i1, i2, i3, i4, dt12, dt13, dt34, score = best
    p1 = Peak1D(t=float(t[i1]), value=float(acc_z[i1]), idx=int(i1))
    p2 = Peak1D(t=float(t[i2]), value=float(acc_z[i2]), idx=int(i2))
    p3 = Peak1D(t=float(t[i3]), value=float(acc_z[i3]), idx=int(i3))
    p4 = Peak1D(t=float(t[i4]), value=float(acc_z[i4]), idx=int(i4))
    print(
        f"[imu:{label}] selected_quad: "
        f"(t1={p1.t:.6f}, t2={p2.t:.6f}, t3={p3.t:.6f}, t4={p4.t:.6f}, "
        f"dt12={dt12:.3f}, dt13={dt13:.3f}, dt34={dt34:.3f}, score={score:.3f})"
    )
    return p1, p2, p3, p4


def _write_offsets_txt_minimal(out_path: str, *, left_offset: float, right_offset: float) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    lines = [
        f"left_imu_offset: {left_offset:.6f}",
        f"right_imu_offset: {right_offset:.6f}",
        "back_imu_offset: 0.000000",
        "",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def estimate_offsets_by_two_jumps(
    *,
    video_path: str,
    calib_start_frame: int,
    calib_end_frame: int,
    imu_csv: str,
    pose_config: str = "./checkpoints/mmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
    pose_checkpoint: str = "./checkpoints/mmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth",
    device: str = "cpu",
    det_config: Optional[str] = "./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco.py",
    det_checkpoint: Optional[str] = "./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
    det_score_thr: float = 0.5,
    cam_peak_distance_frames: int = 10,
    cam_peak_prominence: Optional[float] = None,
    heel_score_thr: float = 0.1,
    tolerance_ratio: float = 0.10,
    imu_peak_distance_samples: Optional[int] = None,
    imu_peak_prominence: Optional[float] = None,
    imu_baseline_sec: float = 1.0,
    imu_lowpass_hz: float = 20.0,
    imu_min_distance_sec: float = 0.20,
    imu_k_prom: float = 6.0,
    imu_k_height: float = 6.0,
    imu_polarity: str = "positive",
    imu_flight_time_min_sec: float = 0.3,
    imu_flight_time_max_sec: float = 0.7,
) -> Tuple[float, float]:
    """
    Estimate camera-based left/right IMU start-time offsets using two jumps.

    Offset definition matches `scripts.imu.read_imu_orphe`:
        imu_time_in_camera = imu_time_local + offset_sec
    where camera frame 0 corresponds to t=0 sec.

    Returns:
        (left_offset_sec, right_offset_sec)
    """
    cam1, cam2, dt_cam, _fps = _camera_two_jump_peaks(
        video_path=video_path,
        calib_start_frame=calib_start_frame,
        calib_end_frame=calib_end_frame,
        pose_config=pose_config,
        pose_checkpoint=pose_checkpoint,
        device=device,
        det_config=det_config,
        det_checkpoint=det_checkpoint,
        det_score_thr=det_score_thr,
        cam_peak_distance_frames=cam_peak_distance_frames,
        cam_peak_prominence=cam_peak_prominence,
        heel_score_thr=heel_score_thr,
    )

    # IMU: read with offset=0 to get IMU-local normalized timeline per foot
    left, right = read_imu_orphe(imu_csv, left_imu_offset=0.0, right_imu_offset=0.0)
    left_p1, left_p2, left_p3, left_p4 = _imu_select_jump_quadruple(
        foot_data=left,
        dt_cam=dt_cam,
        tolerance_ratio=tolerance_ratio,
        imu_peak_distance_samples=imu_peak_distance_samples,
        imu_peak_prominence=imu_peak_prominence,
        label="left",
        imu_baseline_sec=imu_baseline_sec,
        imu_lowpass_hz=imu_lowpass_hz,
        imu_min_distance_sec=imu_min_distance_sec,
        imu_k_prom=imu_k_prom,
        imu_k_height=imu_k_height,
        imu_polarity=imu_polarity,
        flight_time_min_sec=imu_flight_time_min_sec,
        flight_time_max_sec=imu_flight_time_max_sec,
    )
    right_p1, right_p2, right_p3, right_p4 = _imu_select_jump_quadruple(
        foot_data=right,
        dt_cam=dt_cam,
        tolerance_ratio=tolerance_ratio,
        imu_peak_distance_samples=imu_peak_distance_samples,
        imu_peak_prominence=imu_peak_prominence,
        label="right",
        imu_baseline_sec=imu_baseline_sec,
        imu_lowpass_hz=imu_lowpass_hz,
        imu_min_distance_sec=imu_min_distance_sec,
        imu_k_prom=imu_k_prom,
        imu_k_height=imu_k_height,
        imu_polarity=imu_polarity,
        flight_time_min_sec=imu_flight_time_min_sec,
        flight_time_max_sec=imu_flight_time_max_sec,
    )

    cam_apex1 = float(cam1.t)
    cam_apex2 = float(cam2.t)

    left_mid12 = 0.5 * (float(left_p1.t) + float(left_p2.t))
    left_mid34 = 0.5 * (float(left_p3.t) + float(left_p4.t))
    left_offset = 0.5 * ((cam_apex1 - left_mid12) + (cam_apex2 - left_mid34))

    right_mid12 = 0.5 * (float(right_p1.t) + float(right_p2.t))
    right_mid34 = 0.5 * (float(right_p3.t) + float(right_p4.t))
    right_offset = 0.5 * ((cam_apex1 - right_mid12) + (cam_apex2 - right_mid34))

    print("=== Offsets (camera-based) ===")
    print(f"[offset] camera_apex_t_sec: t1={cam_apex1:.6f}, t2={cam_apex2:.6f}")
    print(f"[offset] left_offset_sec: {left_offset:.6f}")
    print(f"[offset] right_offset_sec: {right_offset:.6f}")

    return float(left_offset), float(right_offset)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gvhmr-dir", type=str, required=True)
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--calib-start-frame", type=int, required=True)
    parser.add_argument("--calib-end-frame", type=int, required=True)
    parser.add_argument("--imu-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    # MMPose options (defaults aligned with repo scripts)
    parser.add_argument(
        "--pose-config",
        default="./checkpoints/mmpose/rtmpose-m_8xb512-700e_body8-halpe26-256x192.py",
    )
    parser.add_argument(
        "--pose-checkpoint",
        default="./checkpoints/mmpose/rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.pth",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--det-config",
        default="./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco.py",
        help="MMDetection config used by mmpose top-down pipeline",
    )
    parser.add_argument(
        "--det-checkpoint",
        default="./checkpoints/mmdet/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
        help="MMDetection checkpoint used by mmpose top-down pipeline",
    )
    parser.add_argument("--det-score-thr", type=float, default=0.5)

    # Camera peak detection knobs
    parser.add_argument("--cam-peak-distance-frames", type=int, default=10)
    parser.add_argument("--cam-peak-prominence", type=float, default=None)
    parser.add_argument("--heel-score-thr", type=float, default=0.1)

    # IMU peak / matching knobs
    parser.add_argument("--tolerance-ratio", type=float, default=0.10)
    parser.add_argument("--imu-peak-distance-samples", type=int, default=None)
    parser.add_argument("--imu-peak-prominence", type=float, default=None)
    parser.add_argument("--imu-baseline-sec", type=float, default=1.0)
    parser.add_argument("--imu-lowpass-hz", type=float, default=20.0)
    parser.add_argument("--imu-min-distance-sec", type=float, default=0.20)
    parser.add_argument("--imu-k-prom", type=float, default=6.0)
    parser.add_argument("--imu-k-height", type=float, default=6.0)
    parser.add_argument("--imu-polarity", type=str, default="positive")
    parser.add_argument("--imu-flight-time-min-sec", type=float, default=0.3)
    parser.add_argument("--imu-flight-time-max-sec", type=float, default=0.7)

    parser.add_argument(
        "--out-offsets-txt",
        default=None,
        help="If set, write offsets_cam.txt compatible output to this path; otherwise uses out-dir/offsets_cam.txt",
    )

    args = parser.parse_args()

    left_offset, right_offset = estimate_offsets_by_two_jumps(
        video_path=args.video_path,
        calib_start_frame=args.calib_start_frame,
        calib_end_frame=args.calib_end_frame,
        imu_csv=args.imu_csv,
        pose_config=args.pose_config,
        pose_checkpoint=args.pose_checkpoint,
        device=args.device,
        det_config=args.det_config,
        det_checkpoint=args.det_checkpoint,
        det_score_thr=args.det_score_thr,
        cam_peak_distance_frames=args.cam_peak_distance_frames,
        cam_peak_prominence=args.cam_peak_prominence,
        heel_score_thr=args.heel_score_thr,
        tolerance_ratio=args.tolerance_ratio,
        imu_peak_distance_samples=args.imu_peak_distance_samples,
        imu_peak_prominence=args.imu_peak_prominence,
        imu_baseline_sec=args.imu_baseline_sec,
        imu_lowpass_hz=args.imu_lowpass_hz,
        imu_min_distance_sec=args.imu_min_distance_sec,
        imu_k_prom=args.imu_k_prom,
        imu_k_height=args.imu_k_height,
        imu_polarity=args.imu_polarity,
        imu_flight_time_min_sec=args.imu_flight_time_min_sec,
        imu_flight_time_max_sec=args.imu_flight_time_max_sec,
    )

    out_path = args.out_offsets_txt or os.path.join(args.out_dir, "offsets_cam.txt")
    _write_offsets_txt_minimal(out_path, left_offset=float(left_offset), right_offset=float(right_offset))
    print(f"[offset] wrote: {out_path}")


if __name__ == "__main__":
    main()