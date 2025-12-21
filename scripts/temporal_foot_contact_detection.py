import numpy as np
from scipy import interpolate

from scripts.demo_mmpose_external_smpl import halpe_seq_to_body25_seq


# BODY_25 indices for heel
LHEEL_BODY25_IDX = 21
RHEEL_BODY25_IDX = 24


def filter_and_interpolate_keypoints(keypoints: np.ndarray, threshold: float = 5.0) -> np.ndarray:
    """
    Filter out undetected keypoints (absolute value <= threshold) and interpolate.
    
    Args:
        keypoints: (T, 2) array of keypoint positions [x, y]
        threshold: Threshold for detecting missing keypoints (default: 5.0 pixels)
    
    Returns:
        interpolated_keypoints: (T, 2) array with interpolated values
    """
    T = keypoints.shape[0]
    interpolated = keypoints.copy().astype(np.float64)
    
    # Mark undetected points (absolute value <= threshold) as NaN
    mask = np.abs(keypoints) <= threshold
    interpolated[mask] = np.nan
    
    # Interpolate each dimension separately
    frames = np.arange(T)
    for dim in range(2):
        values = interpolated[:, dim]
        valid_mask = ~np.isnan(values)
        
        if np.sum(valid_mask) > 1:
            # Use linear interpolation
            valid_frames = frames[valid_mask]
            valid_values = values[valid_mask]
            
            # Interpolate only if there are valid points
            if len(valid_frames) > 0:
                # Use scipy's interp1d for interpolation
                if len(valid_frames) == 1:
                    # If only one valid point, fill all with that value
                    interpolated[:, dim] = valid_values[0]
                else:
                    # Linear interpolation
                    f = interpolate.interp1d(
                        valid_frames, 
                        valid_values, 
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    interpolated[:, dim] = f(frames)
        elif np.sum(valid_mask) == 1:
            # Only one valid point, fill all with that value
            interpolated[:, dim] = values[valid_mask][0]
        # If no valid points, keep NaN (shouldn't happen in practice)
    
    return interpolated


def compute_moving_average(data: np.ndarray, window_size: int = 10) -> np.ndarray:
    """
    Compute moving average of data using centered window.
    
    Args:
        data: (T,) array of data
        window_size: Size of the moving average window
    
    Returns:
        moving_avg: (T,) array of moving averages (centered at each time point)
    """
    T = len(data)
    moving_avg = np.zeros_like(data)
    
    if T == 0:
        return moving_avg
    
    # Use centered window: for each time t, average over [t - window_size//2, t + window_size//2]
    half_window = window_size // 2
    
    # Pad both ends: beginning with first value, end with last value
    padded_data = np.concatenate([
        [data[0]] * half_window,
        data,
        [data[-1]] * half_window
    ])
    
    # Compute moving average using convolution
    kernel = np.ones(window_size) / window_size
    moving_avg = np.convolve(padded_data, kernel, mode='valid')
    
    # The result should have length T (since we padded by half_window on both sides)
    # and the convolution with mode='valid' will give us the centered averages
    if len(moving_avg) == T:
        return moving_avg
    elif len(moving_avg) > T:
        # Should not happen, but handle it
        return moving_avg[:T]
    else:
        # Should not happen, but handle it
        return np.pad(moving_avg, (0, T - len(moving_avg)), mode='edge')


def filter_short_contacts(contact: np.ndarray, n_consecutive: int) -> np.ndarray:
    """
    Filter out contact periods shorter than n_consecutive frames.
    
    Args:
        contact: (T,) boolean array indicating contact
        n_consecutive: Minimum number of consecutive frames for valid contact
    
    Returns:
        filtered_contact: (T,) boolean array with short contacts removed
    """
    T = len(contact)
    filtered = contact.copy()
    
    # Find contact regions
    in_contact = False
    start_frame = 0
    
    for i in range(T):
        if contact[i] and not in_contact:
            # Start of contact region
            start_frame = i
            in_contact = True
        elif not contact[i] and in_contact:
            # End of contact region
            end_frame = i - 1
            contact_length = end_frame - start_frame + 1
            
            # If contact period is shorter than n_consecutive, mark as non-contact
            if contact_length < n_consecutive:
                filtered[start_frame:end_frame + 1] = False
            
            in_contact = False
    
    # Handle case where contact continues to the end
    if in_contact:
        end_frame = T - 1
        contact_length = end_frame - start_frame + 1
        if contact_length < n_consecutive:
            filtered[start_frame:end_frame + 1] = False
    
    return filtered


def detect_foot_contact(
    mmpose_keypoints: np.ndarray,
    n_MA: int = 10,
    threshold_percentile: float = 0.25,
    n_consecutive: int = 10
) -> np.ndarray:
    """
    Detect foot contact from MMPose keypoints using heel VY (vertical velocity).
    
    Args:
        mmpose_keypoints: (T, K, 3) numpy array - MMPose Halpe format output [x, y, confidence]
        n_MA: int (default=10) - Moving average window size
        threshold_percentile: float (default=0.2) - Percentile for VY threshold (0.0-1.0)
        n_consecutive: int (default=10) - Minimum consecutive frames for valid contact
    
    Returns:
        contact_labels: (T, 2) numpy array - [left_contact, right_contact] as 0/1
    """
    # Validate input
    if mmpose_keypoints.ndim != 3 or mmpose_keypoints.shape[2] != 3:
        raise ValueError(
            f"Expected mmpose_keypoints with shape (T, K, 3), got {mmpose_keypoints.shape}"
        )
    
    T = mmpose_keypoints.shape[0]
    
    # 1. Convert Halpe to BODY_25 format
    body25_seq = halpe_seq_to_body25_seq(mmpose_keypoints)  # (T, 25, 3)
    
    # 2. Extract heel Y positions
    left_heel_y = body25_seq[:, LHEEL_BODY25_IDX, 1]  # (T,)
    right_heel_y = body25_seq[:, RHEEL_BODY25_IDX, 1]  # (T,)
    
    # Reshape to (T, 1) for filter_and_interpolate_keypoints compatibility
    # (function expects (T, 2) but we only need Y coordinate)
    left_heel_pos = np.stack([np.zeros(T), left_heel_y], axis=1)  # (T, 2) [dummy_x, y]
    right_heel_pos = np.stack([np.zeros(T), right_heel_y], axis=1)  # (T, 2) [dummy_x, y]
    
    # 3. Filter and interpolate missing detections (5px threshold)
    left_heel_interp = filter_and_interpolate_keypoints(left_heel_pos, threshold=5.0)
    right_heel_interp = filter_and_interpolate_keypoints(right_heel_pos, threshold=5.0)
    
    # Extract Y coordinates after interpolation
    left_heel_y_interp = left_heel_interp[:, 1]  # (T,)
    right_heel_y_interp = right_heel_interp[:, 1]  # (T,)
    
    # 4. Compute VY (vertical velocity) as difference
    left_vy = np.zeros(T, dtype=np.float64)
    right_vy = np.zeros(T, dtype=np.float64)
    
    if T > 1:
        left_vy[1:] = left_heel_y_interp[1:] - left_heel_y_interp[:-1]
        right_vy[1:] = right_heel_y_interp[1:] - right_heel_y_interp[:-1]
    
    # 5. Apply moving average to reduce noise
    left_vy_ma = compute_moving_average(left_vy, window_size=n_MA)
    right_vy_ma = compute_moving_average(right_vy, window_size=n_MA)
    
    # 6. Threshold-based contact detection
    # Compute absolute values
    left_vy_abs = np.abs(left_vy_ma)
    right_vy_abs = np.abs(right_vy_ma)
    
    # Compute threshold as percentile
    # threshold_percentile=0.2 means we use the 80th percentile (top 20% of values)
    left_threshold = np.percentile(left_vy_abs, (1.0 - threshold_percentile) * 100)
    right_threshold = np.percentile(right_vy_abs, (1.0 - threshold_percentile) * 100)
    
    # Contact when absolute VY is below threshold
    left_contact = left_vy_abs < left_threshold
    right_contact = right_vy_abs < right_threshold
    
    # 7. Filter short contact periods
    left_contact_filtered = filter_short_contacts(left_contact, n_consecutive)
    right_contact_filtered = filter_short_contacts(right_contact, n_consecutive)
    
    # Convert to 0/1 format
    contact_labels = np.zeros((T, 2), dtype=np.int32)
    contact_labels[:, 0] = left_contact_filtered.astype(np.int32)
    contact_labels[:, 1] = right_contact_filtered.astype(np.int32)
    
    return contact_labels

