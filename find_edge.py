import cv2
import numpy as np
import matplotlib.pyplot as plt

# optional imports (scipy faster if present)
try:
    from scipy.ndimage import gaussian_filter1d
except Exception:
    gaussian_filter1d = None
try:
    from scipy.signal import find_peaks
except Exception:
    find_peaks = None

def detect_edge_simple(
    img_bgr,
    side='left',                 # 'left' or 'right' (which side is background)
    expected_x=None,             # optional prior column index (int)
    band_width_px=None,          # if expected_x set, explicit band width in px
    band_width_frac=0.2,         # else band width fraction of image width (centered on expected_x)
    search_width_frac=0.4,       # fallback side-based band fraction (from background side)
    smooth_sigma=8.0,
    peak_frac=0.35,             # threshold for peak height relative to band max
    peak_select='first',        # 'first' = first from bg->fabric; 'highest' = highest peak in band
    expand_px=0                 # expansion margin for ROI mask
):
    """
    Returns (x_edge:int, mask:uint8 0/255, diagnostics:dict)
    diagnostics contains arrays and peak info for plotting/tuning.
    """
    H,W = img_bgr.shape[:2]
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # column-wise robust summary (median)
    a_col = np.median(lab[:,:,1], axis=0)
    b_col = np.median(lab[:,:,2], axis=0)

    # chroma magnitude (no bg subtraction)
    col = np.sqrt(a_col**2 + b_col**2)

    # smoothing (prefer scipy's gaussian_filter1d)
    if gaussian_filter1d is not None:
        col_smooth = gaussian_filter1d(col, sigma=smooth_sigma, mode='nearest')
    else:
        # small gaussian conv fallback
        radius = max(1, int(round(3.0 * smooth_sigma)))
        xs = np.arange(-radius, radius+1, dtype=np.float32)
        k = np.exp(-(xs*xs)/(2.0*smooth_sigma*smooth_sigma)); k /= k.sum()
        col_smooth = np.convolve(col, k, mode='same')

    grad = np.abs(np.gradient(col_smooth))

    # determine search band: expected_x-centered preferred, else side-based
    if expected_x is not None:
        if band_width_px is not None:
            half = int(max(1, round(band_width_px / 2)))
        else:
            half = int(max(1, round((band_width_frac * W) / 2)))
        start = max(0, int(expected_x) - half)
        end = min(W-1, int(expected_x) + half)
    else:
        if side == 'left':
            start = 0
            end = min(W-1, int(W * search_width_frac))
        else:
            end = W-1
            start = max(0, W - int(W * search_width_frac))

    if end <= start:
        start = max(0, start-1); end = min(W-1, end+1)

    band = grad[start:end+1]
    # dynamic height threshold for peaks
    height_thresh = max(float(band.max() * peak_frac), float(band.mean() + 0.5 * band.std()))

    # detect peaks (prefer scipy)
    peaks = np.array([], dtype=int)
    peak_heights = {}
    try:
        if find_peaks is not None:
            peaks_all, props = find_peaks(band, height=height_thresh)
            peaks = np.array(peaks_all, dtype=int)
            peak_heights = {int(p): float(props['peak_heights'][i]) for i,p in enumerate(peaks)}
    except Exception:
        peaks = np.array([], dtype=int)

    # fallback if no peaks found by find_peaks
    if peaks.size == 0:
        cand = np.where(band >= height_thresh)[0]
        if cand.size:
            peaks = cand
            for p in peaks: peak_heights[int(p)] = float(band[p])
        else:
            # fallback to argmax in band
            arg = int(band.argmax())
            peaks = np.array([arg], dtype=int)
            peak_heights[int(arg)] = float(band[arg])

    # choose peak according to peak_select and side
    if peak_select == 'highest':
        # pick peak with max height
        best_idx_in_band = int(max(peaks, key=lambda p: float(band[p])))
    else:  # 'first' or anything else -> first from background side
        if side == 'left':
            best_idx_in_band = int(peaks.min())
        else:
            best_idx_in_band = int(peaks.max())

    x_edge = start + best_idx_in_band

    # build mask (fabric on opposite side of bg)
    if side == 'left':
        x0 = max(0, x_edge - expand_px); x1 = W-1
    else:
        x0 = 0; x1 = min(W-1, x_edge + expand_px)

    mask = np.zeros((H,W), dtype=np.uint8)
    mask[:, x0:x1+1] = 255

    diagnostics = {
        'a_col': a_col, 'b_col': b_col, 'col': col, 'col_smooth': col_smooth,
        'grad': grad, 'start': start, 'end': end, 'band': band,
        'peaks': peaks.tolist(), 'peak_heights': {int(k): float(v) for k,v in peak_heights.items()},
        'height_thresh': float(height_thresh), 'x_edge': int(x_edge)
    }
    return int(x_edge), mask, diagnostics
