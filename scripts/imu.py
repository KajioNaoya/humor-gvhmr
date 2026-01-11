import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Any, Optional
from scipy.signal import butter, filtfilt


def read_imu_orphe(csv_path: str, left_imu_offset: float = 0.0, right_imu_offset: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Orphe IMUのCSVファイルを読み込み、左足と右足のデータを分けて返す。
    カメラの最初フレームを0秒とした時刻同期を行う。
    
    Args:
        csv_path (str): CSVファイルのパス
        left_imu_offset (float): 左足IMUの最初の点がカメラの最初フレームから何秒後か（秒）
        right_imu_offset (float): 右足IMUの最初の点がカメラの最初フレームから何秒後か（秒）
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (left_foot_data, right_foot_data)
            - left_foot_data: 左足のIMUデータ [timestamp, gx, gy, gz, ax, ay, az]
            - right_foot_data: 右足のIMUデータ [timestamp, gx, gy, gz, ax, ay, az]
    """

    G = 9.80665

    # CSVファイルを読み込み
    df = pd.read_csv(csv_path)

    # 単位の変換 タイムスタンプ: ms -> s
    df['timestamp'] = df['timestamp'] / 1000.0

    # 単位の変換 加速度: g -> m/s^2, 角速度: deg/s -> rad/s
    df['acc_x'] = df['acc_x'] * G
    df['acc_y'] = df['acc_y'] * G
    df['acc_z'] = df['acc_z'] * G
    df['gyro_x'] = df['gyro_x'] * np.pi / 180.0
    df['gyro_y'] = df['gyro_y'] * np.pi / 180.0
    df['gyro_z'] = df['gyro_z'] * np.pi / 180.0
    
    # 左足と右足のデータを分離
    left_data = df[df['foot'] == 'left'].copy()
    right_data = df[df['foot'] == 'right'].copy()
    
    # タイムスタンプでソート
    if len(left_data) > 0:
        left_data = left_data.sort_values('timestamp').reset_index(drop=True)
    
    if len(right_data) > 0:
        right_data = right_data.sort_values('timestamp').reset_index(drop=True)

    # タイムスタンプの重複があった場合は、数値列のみ平均を取って1つにする
    numeric_cols = ['gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
    if len(left_data) > 0:
        left_data = left_data.groupby('timestamp', as_index=False)[numeric_cols].mean()
    if len(right_data) > 0:
        right_data = right_data.groupby('timestamp', as_index=False)[numeric_cols].mean()
    
    # カメラ基準の時刻同期（カメラの最初フレームを0秒とする）
    if len(left_data) > 0:
        # 左足の最初のタイムスタンプを基準に、カメラ基準の時刻に変換
        left_start_time = left_data['timestamp'].iloc[0]
        left_data['timestamp'] = left_data['timestamp'] - left_start_time + left_imu_offset
    
    if len(right_data) > 0:
        # 右足の最初のタイムスタンプを基準に、カメラ基準の時刻に変換
        right_start_time = right_data['timestamp'].iloc[0]
        right_data['timestamp'] = right_data['timestamp'] - right_start_time + right_imu_offset
    
    # 必要な列を選択してnumpy配列に変換
    # 列の順序: timestamp, gyro_x, gyro_y, gyro_z, acc_x, acc_y, acc_z
    left_columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
    right_columns = ['timestamp', 'gyro_x', 'gyro_y', 'gyro_z', 'acc_x', 'acc_y', 'acc_z']
    
    left_foot_data = left_data[left_columns].values if len(left_data) > 0 else np.empty((0, 7))
    right_foot_data = right_data[right_columns].values if len(right_data) > 0 else np.empty((0, 7))
    
    return left_foot_data, right_foot_data


def lowpass_imu_data(imu_data: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    """
    IMUデータ(各行 [timestamp, gx, gy, gz, ax, ay, az]) に対して、
    ゼロ位相のバターワースローパスフィルタ(scipy.signal.filtfilt)を適用する。

    Args:
        imu_data (np.ndarray): 形状 (N, 7) のIMUデータ。列は [t, gx, gy, gz, ax, ay, az]。
        fs (float): サンプリング周波数(Hz)。
        cutoff_hz (float): ローパスフィルタのカットオフ周波数(Hz)。
        order (int, optional): フィルタ次数。デフォルトは4。

    Returns:
        np.ndarray: フィルタ適用後のIMUデータ (同形状)。
    """

    nyquist = 0.5 * fs
    wn = cutoff_hz / nyquist
    # Wn は (0, 1) にある必要がある
    wn = max(min(wn, 0.999), 1e-6)

    b, a = butter(order, wn, btype='low', analog=False)

    # filtfilt のパディングに必要な最小長
    min_len = 3 * (max(len(a), len(b)) - 1)
    if imu_data.shape[0] <= min_len:
        # データ長が短すぎる場合はフィルタをスキップ
        return imu_data

    timestamps = imu_data[:, 0:1]
    signals = imu_data[:, 1:]

    # チャンネルごとに同時にフィルタ (axis=0 が時間軸)
    filtered_signals = filtfilt(b, a, signals, axis=0)

    return np.hstack((timestamps, filtered_signals))


def _detect_stance_for_one_foot(
    foot_data: np.ndarray,
    window_size: int = 20,
    gyro_threshold: float = 0.5,
    acc_var_threshold: float = 3.0 # 0.5
) -> List[Tuple[float, float]]:
    """
    Detects stance phases for a single foot using a threshold-based algorithm.

    A time point is considered to be in a stance phase if both the magnitude of the
    angular velocity and the variance of the acceleration within a surrounding
    window are below their respective thresholds.

    Args:
        foot_data (np.ndarray): IMU data for a single foot.
            Expected columns: [timestamp, gx, gy, gz, ax, ay, az]
        window_size (int): The size of the moving window for calculating variance.
        gyro_threshold (float): The threshold for the angular velocity magnitude (in rad/s).
        acc_var_threshold (float): The threshold for the acceleration variance (in m/s^2).

    Returns:
        List[Tuple[float, float]]: A list of tuples, where each tuple represents
                                   the start and end timestamp of a stance phase.
    """
    if foot_data.shape[0] < window_size:
        return []

    timestamps = foot_data[:, 0]
    gyro_data = foot_data[:, 1:4]
    acc_data = foot_data[:, 4:7]

    # Calculate the magnitude of the angular velocity
    gyro_mag = np.linalg.norm(gyro_data, axis=1)

    # Use pandas for efficient moving window variance calculation
    acc_df = pd.DataFrame(acc_data)
    acc_var = acc_df.rolling(window=window_size, center=True).var().sum(axis=1)

    # Identify stance frames based on thresholds
    is_stance = (gyro_mag < gyro_threshold) & (acc_var < acc_var_threshold)

    # Find the start and end indices of continuous stance phases
    stance_indices = np.where(is_stance)[0]
    if len(stance_indices) == 0:
        return []

    stance_periods = []
    start_idx = stance_indices[0]

    for i in range(1, len(stance_indices)):
        if stance_indices[i] > stance_indices[i-1] + 1:
            # End of a stance period
            end_idx = stance_indices[i-1]
            if end_idx > start_idx:
                stance_periods.append((float(timestamps[start_idx]), float(timestamps[end_idx])))
            # Start of a new stance period
            start_idx = stance_indices[i]

    # Add the last stance period
    end_idx = stance_indices[-1]
    if end_idx > start_idx:
        stance_periods.append((float(timestamps[start_idx]), float(timestamps[end_idx])))

    return stance_periods


def detect_foot_stance(
    left_foot_data: np.ndarray,
    right_foot_data: np.ndarray
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
    """
    Detect stance phases for left and right feet based on IMU data.

    This function applies a threshold-based stance detection algorithm to the IMU
    data of each foot.

    Args:
        left_foot_data (np.ndarray): IMU data for the left foot.
            Columns: [timestamp, gx, gy, gz, ax, ay, az]
        right_foot_data (np.ndarray): IMU data for the right foot.
            Columns: [timestamp, gx, gy, gz, ax, ay, az]

    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        - left_stance_periods: List of (stance_start_time, stance_end_time) for the left foot.
        - right_stance_periods: List of (stance_start_time, stance_end_time) for the right foot.
    """
    left_stance_periods = _detect_stance_for_one_foot(left_foot_data)
    right_stance_periods = _detect_stance_for_one_foot(right_foot_data)

    return left_stance_periods, right_stance_periods


def _stance_periods_to_contacts(
    stance_periods: List[Tuple[float, float]], frame_times: np.ndarray
) -> np.ndarray:
    """
    Convert stance periods (start_time, end_time) into a boolean contact array
    aligned with the given frame timestamps.

    Args:
        stance_periods (List[Tuple[float, float]]): List of stance intervals.
        frame_times (np.ndarray): 1D array of frame timestamps (seconds).

    Returns:
        np.ndarray: Boolean array of shape (T,) where True indicates contact.
    """
    contact = np.zeros_like(frame_times, dtype=bool)

    if len(stance_periods) == 0 or frame_times.size == 0:
        return contact

    for start_t, end_t in stance_periods:
        # Mark frames whose timestamps fall within the stance interval
        in_interval = (frame_times >= start_t) & (frame_times <= end_t)
        contact |= in_interval

    return contact


def compute_contacts_from_imu(
    T: int,
    fps: float,
    csv_path: str = "./data/1029_01/raw_sensor_data.csv",
    left_imu_offset: float = 0.0,
    right_imu_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read IMU data and compute per-frame left/right foot contact flags.

    This helper is intended to be called from demo scripts that already know
    the number of GVHMR frames T and the frame rate fps.

    Args:
        T (int): Number of frames in the GVHMR sequence.
        fps (float): Frame rate of the GVHMR sequence (frames per second).
        csv_path (str): Path to the IMU CSV file.
        left_imu_offset (float): Time offset (s) for the left IMU w.r.t. camera frame 0.
        right_imu_offset (float): Time offset (s) for the right IMU w.r.t. camera frame 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - left_contact:  Boolean array of shape (T,) for the left foot.
            - right_contact: Boolean array of shape (T,) for the right foot.
    """
    left_contact, right_contact, _left_valid, _right_valid = _compute_contacts_and_coverage_from_imu(
        T=T,
        fps=fps,
        csv_path=csv_path,
        left_imu_offset=left_imu_offset,
        right_imu_offset=right_imu_offset,
    )
    return left_contact, right_contact


def _compute_contacts_and_coverage_from_imu(
    T: int,
    fps: float,
    csv_path: str,
    left_imu_offset: float,
    right_imu_offset: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Like `compute_contacts_from_imu`, but also returns per-frame validity masks indicating
    where IMU samples exist (per foot) in camera-time.

    Validity is computed from the IMU timestamp coverage after applying the provided offsets:
        valid[t] = True iff frame_time[t] is within [min(imu_t), max(imu_t)]
    for each foot independently.

    Returns:
        left_contact: (T,) bool
        right_contact: (T,) bool
        left_valid: (T,) bool
        right_valid: (T,) bool
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    # 1) Read IMU data and synchronize timestamps to the camera frame.
    left_foot_data, right_foot_data = read_imu_orphe(
        csv_path, left_imu_offset=left_imu_offset, right_imu_offset=right_imu_offset
    )

    # 2) Detect stance periods (in seconds) for each foot.
    left_stance_periods, right_stance_periods = detect_foot_stance(left_foot_data, right_foot_data)

    # 3) Build GVHMR frame timestamps, assuming frame 0 is at t=0.
    frame_times = np.arange(T, dtype=np.float32) / float(fps)

    # 4) Convert stance intervals into per-frame contact flags.
    left_contact = _stance_periods_to_contacts(left_stance_periods, frame_times)
    right_contact = _stance_periods_to_contacts(right_stance_periods, frame_times)

    # 5) Compute per-frame validity (coverage) based on IMU timestamp range.
    if left_foot_data.shape[0] > 0:
        left_t0 = float(np.min(left_foot_data[:, 0]))
        left_t1 = float(np.max(left_foot_data[:, 0]))
        left_valid = (frame_times >= left_t0) & (frame_times <= left_t1)
    else:
        left_valid = np.zeros((T,), dtype=bool)

    if right_foot_data.shape[0] > 0:
        right_t0 = float(np.min(right_foot_data[:, 0]))
        right_t1 = float(np.max(right_foot_data[:, 0]))
        right_valid = (frame_times >= right_t0) & (frame_times <= right_t1)
    else:
        right_valid = np.zeros((T,), dtype=bool)

    return left_contact, right_contact, left_valid.astype(bool), right_valid.astype(bool)


def _shift_bool_with_mask(arr: np.ndarray, shift: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Shift a boolean array by an integer number of frames with a validity mask.

    We define the shifted sequence as:
        out[t] = arr[t - shift] when (t - shift) is within [0, T-1]
        otherwise out[t] is undefined (we set False) and mask[t]=False.

    Args:
        arr: (T,) boolean array.
        shift: integer shift in frames. Positive shift means "use earlier arr indices".

    Returns:
        out: (T,) shifted boolean array (undefined entries filled with False)
        valid_mask: (T,) boolean mask indicating entries that are valid (in-range)
    """
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={arr.shape}")

    T = arr.shape[0]
    idx_src = np.arange(T, dtype=np.int64) - int(shift)
    valid = (idx_src >= 0) & (idx_src < T)
    out = np.zeros(T, dtype=bool)
    out[valid] = arr[idx_src[valid]]
    return out, valid


def _confusion_from_bools(
    a: np.ndarray, b: np.ndarray, mask: np.ndarray
) -> Tuple[int, int, int, int, int]:
    """
    Compute confusion counts over mask for boolean arrays.

    Returns:
        (tp, tn, fp, fn, n)
    """
    if a.shape != b.shape or a.shape != mask.shape:
        raise ValueError(
            f"Shape mismatch: a={a.shape}, b={b.shape}, mask={mask.shape}"
        )
    m = mask.astype(bool)
    if m.sum() == 0:
        return 0, 0, 0, 0, 0

    aa = a[m].astype(bool)
    bb = b[m].astype(bool)
    tp = int(np.sum(aa & bb))
    tn = int(np.sum((~aa) & (~bb)))
    fp = int(np.sum((~aa) & bb))
    fn = int(np.sum(aa & (~bb)))
    n = int(aa.shape[0])
    return tp, tn, fp, fn, n


def _score_contacts(
    cam: np.ndarray, imu: np.ndarray, mask: np.ndarray, score: str
) -> float:
    tp, tn, fp, fn, n = _confusion_from_bools(cam, imu, mask)
    if n <= 0:
        return -1.0  # invalid / no overlap

    score = score.lower().strip()

    if score == "accuracy":
        return float((tp + tn) / max(1, n))

    if score == "balanced_acc":
        tpr_den = tp + fn
        tnr_den = tn + fp
        tpr = tp / tpr_den if tpr_den > 0 else 0.0
        tnr = tn / tnr_den if tnr_den > 0 else 0.0
        return float(0.5 * (tpr + tnr))

    if score == "mcc":
        # Matthews correlation coefficient
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom <= 0:
            return 0.0
        return float((tp * tn - fp * fn) / np.sqrt(denom))

    raise ValueError(f"Unknown score='{score}'. Expected one of: mcc, balanced_acc, accuracy")


def save_contact_band_png(
    contact: np.ndarray,
    times_sec: np.ndarray,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    """
    Save a 1D contact sequence as a horizontal band plot.

    Color:
      - contact=True  -> red
      - contact=False -> gray

    Args:
        contact: (T,) boolean array
        times_sec: (T,) timestamps in seconds (monotonic recommended)
        out_path: output PNG path
        title: optional figure title
    """
    contact = np.asarray(contact).astype(bool)
    times_sec = np.asarray(times_sec).astype(np.float64)
    if contact.ndim != 1 or times_sec.ndim != 1 or contact.shape[0] != times_sec.shape[0]:
        raise ValueError(
            f"Expected contact and times_sec as 1D arrays with same length, got "
            f"contact={contact.shape}, times_sec={times_sec.shape}"
        )
    if contact.shape[0] == 0:
        raise ValueError("Empty contact sequence")

    try:
        import matplotlib

        matplotlib.use("Agg")  # headless backend
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        # Build a 2D image: 1xT
        img = contact.astype(np.int32)[None, :]  # (1, T)
        cmap = ListedColormap([(0.75, 0.75, 0.75), (1.0, 0.0, 0.0)])  # gray, red

        t0 = float(times_sec[0])
        t1 = float(times_sec[-1])
        # Ensure non-zero extent even for single frame
        if times_sec.shape[0] == 1:
            t1 = t0 + 1e-3

        fig = plt.figure(figsize=(10, 1.3))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            img,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap,
            vmin=0,
            vmax=1,
            extent=[t0, t1, 0, 1],
        )
        ax.set_yticks([])
        ax.set_xlabel("time (sec)")
        if title:
            ax.set_title(title)
        ax.grid(False)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    except Exception as e:
        # Debug plotting should not break callers
        print(f"Warning: failed to save contact band plot to {out_path}: {e}")


def _contacts_from_stance_periods_at_times(
    stance_periods: List[Tuple[float, float]], query_times: np.ndarray
) -> np.ndarray:
    """
    Evaluate stance/contact boolean values at arbitrary query times (seconds).

    This is faster than looping all intervals for every candidate offset by using
    a vectorized search over sorted interval start times.

    Args:
        stance_periods: list of (start_t, end_t) in seconds, in the IMU time base.
        query_times: (T,) array of times in the same time base as stance_periods.

    Returns:
        contact: (T,) boolean array where True indicates stance/contact.
    """
    if query_times.ndim != 1:
        raise ValueError(f"Expected 1D query_times, got shape={query_times.shape}")

    if len(stance_periods) == 0:
        return np.zeros_like(query_times, dtype=bool)

    # Sort intervals by start time
    starts = np.asarray([s for s, _e in stance_periods], dtype=np.float64)
    ends = np.asarray([_e for _s, _e in stance_periods], dtype=np.float64)
    order = np.argsort(starts)
    starts = starts[order]
    ends = ends[order]

    t = query_times.astype(np.float64)
    # idx = index of the last interval with start <= t
    idx = np.searchsorted(starts, t, side="right") - 1
    in_range = idx >= 0
    contact = np.zeros_like(t, dtype=bool)
    contact[in_range] = t[in_range] <= ends[idx[in_range]]
    return contact


def _grid_search_best_offset(
    cam_contact: np.ndarray,
    frame_times: np.ndarray,
    stance_periods: List[Tuple[float, float]],
    imu_t0: float,
    imu_t1: float,
    fps: float,
    max_offset_sec: float,
    coarse_step_sec: float,
    refine_window_sec: float,
    score: str,
    debug_plot_path: Optional[str] = None,
    debug_plot_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Search for the best offset (seconds) that maximizes agreement between camera contact
    and IMU stance-derived contact over the *full IMU timeline*.

    Model:
        camera_time = imu_time + offset
        => imu_time = camera_time - offset

    Therefore, for a candidate offset, we evaluate IMU contact at times:
        imu_query_times = frame_times - offset
    and score over frames whose imu_query_times are within [imu_t0, imu_t1].
    """
    T = int(cam_contact.shape[0])
    if frame_times.shape != (T,):
        raise ValueError(f"Expected frame_times shape (T,), got {frame_times.shape}, T={T}")

    if coarse_step_sec <= 0:
        raise ValueError(f"coarse_step_sec must be positive, got {coarse_step_sec}")

    # Build coarse offset grid
    n_steps = int(np.floor((2.0 * float(max_offset_sec)) / float(coarse_step_sec))) + 1
    offsets_coarse = (-float(max_offset_sec)) + np.arange(n_steps, dtype=np.float64) * float(coarse_step_sec)

    best_off = 0.0
    best_score = -1e9
    best_overlap = -1

    coarse_scores: List[float] = []

    # Coarse evaluation
    for off in offsets_coarse:
        imu_q = frame_times.astype(np.float64) - float(off)
        cov = (imu_q >= float(imu_t0)) & (imu_q <= float(imu_t1))
        if int(cov.sum()) == 0:
            sc = -1.0
            overlap = 0
        else:
            imu_contact = _contacts_from_stance_periods_at_times(stance_periods, imu_q)
            sc = _score_contacts(cam_contact, imu_contact, cov, score=score)
            overlap = int(cov.sum())

        coarse_scores.append(float(sc))

        if (sc > best_score) or (sc == best_score and overlap > best_overlap):
            best_score = float(sc)
            best_off = float(off)
            best_overlap = overlap

    # Optional debug plot (coarse only)
    if debug_plot_path is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless backend
            import matplotlib.pyplot as plt

            xs = offsets_coarse
            ys = np.asarray(coarse_scores, dtype=np.float64)

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(xs, ys, linewidth=1.5)
            ax.axvline(best_off, linestyle="--", linewidth=1.0)
            ax.set_xlabel("offset (sec)")
            ax.set_ylabel(f"score ({score})")
            if debug_plot_title:
                ax.set_title(debug_plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(debug_plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"Warning: failed to save sync debug plot to {debug_plot_path}: {e}")

    # Refine around best coarse offset at 1-frame resolution
    refine_win_frames = int(round(float(refine_window_sec) * float(fps)))
    if refine_win_frames > 0:
        refine_offsets = best_off + (np.arange(-refine_win_frames, refine_win_frames + 1, dtype=np.float64) / float(fps))
        for off in refine_offsets:
            imu_q = frame_times.astype(np.float64) - float(off)
            cov = (imu_q >= float(imu_t0)) & (imu_q <= float(imu_t1))
            if int(cov.sum()) == 0:
                sc = -1.0
                overlap = 0
            else:
                imu_contact = _contacts_from_stance_periods_at_times(stance_periods, imu_q)
                sc = _score_contacts(cam_contact, imu_contact, cov, score=score)
                overlap = int(cov.sum())

            if (sc > best_score) or (sc == best_score and overlap > best_overlap):
                best_score = float(sc)
                best_off = float(off)
                best_overlap = overlap

    best_shift_frames = int(round(best_off * float(fps)))

    return {
        "best_offset_sec": float(best_off),
        "best_shift_frames": int(best_shift_frames),
        "best_score": float(best_score),
        "best_overlap_frames": int(best_overlap),
        "coarse_offsets_sec": offsets_coarse.astype(np.float64),
        "coarse_scores": np.asarray(coarse_scores, dtype=np.float64),
    }


def _grid_search_best_shift(
    cam_contact: np.ndarray,
    imu_contact0: np.ndarray,
    imu_valid0: Optional[np.ndarray],
    fps: float,
    max_offset_sec: float,
    coarse_step_sec: float,
    refine_window_sec: float,
    score: str,
    debug_plot_path: Optional[str] = None,
    debug_plot_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Find the integer frame shift that maximizes agreement between cam_contact and imu_contact0.

    Returns a dict including best_shift_frames, best_offset_sec, and best_score.
    """
    T = int(cam_contact.shape[0])
    if imu_contact0.shape[0] != T:
        raise ValueError(f"Expected imu_contact0 length {T}, got {imu_contact0.shape[0]}")
    if imu_valid0 is not None and imu_valid0.shape[0] != T:
        raise ValueError(f"Expected imu_valid0 length {T}, got {imu_valid0.shape[0]}")

    max_shift = int(round(float(max_offset_sec) * float(fps)))
    coarse_step_frames = int(round(float(coarse_step_sec) * float(fps)))
    coarse_step_frames = max(1, coarse_step_frames)

    shifts_coarse = np.arange(-max_shift, max_shift + 1, coarse_step_frames, dtype=np.int64)

    best_shift = 0
    best_score = -1e9
    best_overlap = -1

    coarse_offsets_sec: List[float] = []
    coarse_scores: List[float] = []

    # Coarse search
    for s in shifts_coarse:
        imu_s, valid = _shift_bool_with_mask(imu_contact0, int(s))
        if imu_valid0 is not None:
            imu_cov_s, _ = _shift_bool_with_mask(imu_valid0.astype(bool), int(s))
            score_mask = valid & imu_cov_s
        else:
            score_mask = valid

        sc = _score_contacts(cam_contact, imu_s, score_mask, score=score)
        overlap = int(score_mask.sum())

        coarse_offsets_sec.append(float(s) / float(fps))
        coarse_scores.append(float(sc))

        # Tie-breaker: prefer more overlap if scores equal-ish
        if (sc > best_score) or (sc == best_score and overlap > best_overlap):
            best_score = sc
            best_shift = int(s)
            best_overlap = overlap

    # Optional debug plot (coarse only)
    if debug_plot_path is not None:
        try:
            import matplotlib

            matplotlib.use("Agg")  # headless backend
            import matplotlib.pyplot as plt

            xs = np.asarray(coarse_offsets_sec, dtype=np.float64)
            ys = np.asarray(coarse_scores, dtype=np.float64)

            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(xs, ys, linewidth=1.5)
            ax.axvline(float(best_shift) / float(fps), linestyle="--", linewidth=1.0)
            ax.set_xlabel("offset (sec)")
            ax.set_ylabel(f"score ({score})")
            if debug_plot_title:
                ax.set_title(debug_plot_title)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(debug_plot_path, dpi=150)
            plt.close(fig)
        except Exception as e:
            # Debug plotting should never break sync estimation
            print(f"Warning: failed to save sync debug plot to {debug_plot_path}: {e}")

    # Refine around best coarse shift at 1-frame resolution
    refine_win = int(round(float(refine_window_sec) * float(fps)))
    refine_start = best_shift - refine_win
    refine_end = best_shift + refine_win

    for s in range(refine_start, refine_end + 1):
        imu_s, valid = _shift_bool_with_mask(imu_contact0, int(s))
        if imu_valid0 is not None:
            imu_cov_s, _ = _shift_bool_with_mask(imu_valid0.astype(bool), int(s))
            score_mask = valid & imu_cov_s
        else:
            score_mask = valid

        sc = _score_contacts(cam_contact, imu_s, score_mask, score=score)
        overlap = int(score_mask.sum())
        if (sc > best_score) or (sc == best_score and overlap > best_overlap):
            best_score = sc
            best_shift = int(s)
            best_overlap = overlap

    return {
        "best_shift_frames": best_shift,
        "best_offset_sec": float(best_shift) / float(fps),
        "best_score": float(best_score),
        "best_overlap_frames": int(best_overlap),
    }


def estimate_imu_offsets_from_contacts(
    cam_left_contact: np.ndarray,
    cam_right_contact: np.ndarray,
    T: int,
    fps: float,
    imu_csv_path: str,
    max_offset_sec: float = 200.0,
    coarse_step_sec: float = 1.0,
    refine_window_sec: float = 2.0,
    score: str = "mcc",
    estimate_left: bool = True,
    estimate_right: bool = True,
    debug_plot_left_path: Optional[str] = None,
    debug_plot_right_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Estimate per-foot camera↔IMU start-time offsets by maximizing agreement between
    camera-based and IMU-based contact signals.

    Definition matches `read_imu_orphe` / `compute_contacts_from_imu`:\n
      - offset_sec is the time (in seconds) at which the first IMU sample occurs,
        relative to camera frame 0 at t=0.\n
      - Using an offset is equivalent to shifting the IMU contact sequence in frames:\n
            imu_contact_offset[t] = imu_contact0[t - shift]\n
        where shift = round(offset_sec * fps).

    Notes:
    - IMU stance detection is run **once** to produce imu_contact0 at offset=0.
      The search then evaluates different offsets by shifting in frame space.\n
    - Frames that fall outside the IMU-valid range after shifting are excluded from scoring.

    Returns a dict with left/right offsets in seconds plus best scores and shifts.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")

    cam_left = np.asarray(cam_left_contact).astype(bool)
    cam_right = np.asarray(cam_right_contact).astype(bool)
    if cam_left.shape != (T,) or cam_right.shape != (T,):
        raise ValueError(
            f"Expected cam contact shapes (T,), got left={cam_left.shape}, right={cam_right.shape}, T={T}"
        )

    # Read IMU once (timestamps become IMU-local time starting at ~0 sec for each foot)
    left_foot_data, right_foot_data = read_imu_orphe(
        imu_csv_path, left_imu_offset=0.0, right_imu_offset=0.0
    )
    left_stance_periods, right_stance_periods = detect_foot_stance(left_foot_data, right_foot_data)
    frame_times = (np.arange(T, dtype=np.float64) / float(fps)).astype(np.float64)

    # IMU coverage (seconds) per foot
    left_imu_t0 = float(np.min(left_foot_data[:, 0])) if left_foot_data.shape[0] > 0 else None
    left_imu_t1 = float(np.max(left_foot_data[:, 0])) if left_foot_data.shape[0] > 0 else None
    right_imu_t0 = float(np.min(right_foot_data[:, 0])) if right_foot_data.shape[0] > 0 else None
    right_imu_t1 = float(np.max(right_foot_data[:, 0])) if right_foot_data.shape[0] > 0 else None

    result: Dict[str, Any] = {
        "fps": float(fps),
        "T": int(T),
        "camera_duration_sec": float(frame_times[-1] - frame_times[0]) if T > 1 else 0.0,
        "score": str(score),
        "left_imu_duration_sec": float(left_imu_t1 - left_imu_t0) if left_imu_t0 is not None else 0.0,
        "right_imu_duration_sec": float(right_imu_t1 - right_imu_t0) if right_imu_t0 is not None else 0.0,
    }

    if estimate_left:
        if left_imu_t0 is None or left_imu_t1 is None:
            left_res = None
        else:
            left_res = _grid_search_best_offset(
                cam_contact=cam_left,
                frame_times=frame_times,
                stance_periods=left_stance_periods,
                imu_t0=float(left_imu_t0),
                imu_t1=float(left_imu_t1),
                fps=float(fps),
                max_offset_sec=float(max_offset_sec),
                coarse_step_sec=float(coarse_step_sec),
                refine_window_sec=float(refine_window_sec),
                score=str(score),
                debug_plot_path=debug_plot_left_path,
                debug_plot_title="IMU sync coarse search (left)",
            )

        if left_res is None:
            result.update(
                {
                    "left_offset_sec": 0.0,
                    "left_best_shift_frames": 0,
                    "left_best_score": None,
                    "left_best_overlap_frames": 0,
                }
            )
        else:
            result.update(
                {
                    "left_offset_sec": float(left_res["best_offset_sec"]),
                    "left_best_shift_frames": int(left_res["best_shift_frames"]),
                    "left_best_score": float(left_res["best_score"]),
                    "left_best_overlap_frames": int(left_res["best_overlap_frames"]),
                }
            )
    else:
        result.update(
            {
                "left_offset_sec": 0.0,
                "left_best_shift_frames": 0,
                "left_best_score": None,
                "left_best_overlap_frames": 0,
            }
        )

    if estimate_right:
        if right_imu_t0 is None or right_imu_t1 is None:
            right_res = None
        else:
            right_res = _grid_search_best_offset(
                cam_contact=cam_right,
                frame_times=frame_times,
                stance_periods=right_stance_periods,
                imu_t0=float(right_imu_t0),
                imu_t1=float(right_imu_t1),
                fps=float(fps),
                max_offset_sec=float(max_offset_sec),
                coarse_step_sec=float(coarse_step_sec),
                refine_window_sec=float(refine_window_sec),
                score=str(score),
                debug_plot_path=debug_plot_right_path,
                debug_plot_title="IMU sync coarse search (right)",
            )

        if right_res is None:
            result.update(
                {
                    "right_offset_sec": 0.0,
                    "right_best_shift_frames": 0,
                    "right_best_score": None,
                    "right_best_overlap_frames": 0,
                }
            )
        else:
            result.update(
                {
                    "right_offset_sec": float(right_res["best_offset_sec"]),
                    "right_best_shift_frames": int(right_res["best_shift_frames"]),
                    "right_best_score": float(right_res["best_score"]),
                    "right_best_overlap_frames": int(right_res["best_overlap_frames"]),
                }
            )
    else:
        result.update(
            {
                "right_offset_sec": 0.0,
                "right_best_shift_frames": 0,
                "right_best_score": None,
                "right_best_overlap_frames": 0,
            }
        )

    return result