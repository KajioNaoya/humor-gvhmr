import numpy as np
import pandas as pd
from typing import Tuple, List
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
    left_imu_offset: float = -100.0,
    right_imu_offset: float = -104.55,
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
    # 1) Read IMU data and synchronize timestamps to the camera frame.
    left_foot_data, right_foot_data = read_imu_orphe(
        csv_path, left_imu_offset=left_imu_offset, right_imu_offset=right_imu_offset
    )

    # 2) Detect stance periods (in seconds) for each foot.
    left_stance_periods, right_stance_periods = detect_foot_stance(
        left_foot_data, right_foot_data
    )

    # 3) Build GVHMR frame timestamps, assuming frame 0 is at t=0.
    frame_times = np.arange(T, dtype=np.float32) / float(fps)

    # 4) Convert stance intervals into per-frame contact flags.
    left_contact = _stance_periods_to_contacts(left_stance_periods, frame_times)
    right_contact = _stance_periods_to_contacts(right_stance_periods, frame_times)

    return left_contact, right_contact