"""
debug_psnr.py

Computes PSNR/SSIM with every method across all frames of two videos,
then prints per-method mean ± std so we can see exactly where numbers differ.

Usage:
    python debug_psnr.py --sr path/to/SR_raw.mp4 --gt path/to/original.mp4
"""

import argparse
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from skimage.util import img_as_float
from basicsr.metrics import calculate_metric


# ---------------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------------

def method_basicsr(sr_bgr, gt_bgr):
    metric_data = {'img': sr_bgr, 'img2': gt_bgr}
    psnr = calculate_metric(metric_data, {'type': 'calculate_psnr', 'crop_border': 2, 'test_y_channel': True})
    ssim = calculate_metric(metric_data, {'type': 'calculate_ssim', 'crop_border': 2, 'test_y_channel': True})
    return psnr, ssim


def method_skimage_uint8_rgb(sr_bgr, gt_bgr):
    psnr = sk_psnr(gt_bgr, sr_bgr, data_range=255)
    ssim = sk_ssim(gt_bgr, sr_bgr, channel_axis=2, data_range=255)
    return psnr, ssim


def method_img_as_float(sr_bgr, gt_bgr):
    """Your calculate_ssim_psnr — float image but data_range=255 passed to SSIM."""
    i1 = img_as_float(sr_bgr)
    i2 = img_as_float(gt_bgr)
    psnr = sk_psnr(i1, i2)
    ssim = sk_ssim(i1, i2, channel_axis=2, data_range=255)
    return psnr, ssim


def method_img_as_float_fixed(sr_bgr, gt_bgr):
    """Same but data_range=1.0 to match float image range."""
    i1 = img_as_float(sr_bgr)
    i2 = img_as_float(gt_bgr)
    psnr = sk_psnr(i1, i2)
    ssim = sk_ssim(i1, i2, channel_axis=2, data_range=1.0)
    return psnr, ssim


def method_y_channel_crop2(sr_bgr, gt_bgr):
    """Manual Y channel + crop_border=2 — should match basicsr exactly."""
    sr_c = sr_bgr[2:-2, 2:-2]
    gt_c = gt_bgr[2:-2, 2:-2]
    sr_y = _bgr_to_y(sr_c)
    gt_y = _bgr_to_y(gt_c)
    psnr = sk_psnr(gt_y, sr_y, data_range=255)
    ssim = sk_ssim(gt_y, sr_y, data_range=255)
    return psnr, ssim


METHODS = [
    ("basicsr  (Y ch, crop=2)",          method_basicsr),
    ("skimage  uint8 RGB, no crop",       method_skimage_uint8_rgb),
    ("img_as_float  (data_range=255 bug)",method_img_as_float),
    ("img_as_float  (data_range=1.0 fix)",method_img_as_float_fixed),
    ("Y channel + crop_border=2",         method_y_channel_crop2),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bgr_to_y(img_bgr):
    rgb = img_bgr[:, :, ::-1].astype(np.float32)
    return (rgb @ np.array([65.481, 128.553, 24.966], dtype=np.float32)) / 255.0 + 16.0


def _read_frames(path):
    frames = []
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sr', required=True, help='Path to SR video (raw/lossless)')
    parser.add_argument('--gt', required=True, help='Path to GT video (original HR)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Only process this many frames (default: all)')
    args = parser.parse_args()

    print(f"Reading SR:  {args.sr}")
    sr_frames = _read_frames(args.sr)
    print(f"Reading GT:  {args.gt}")
    gt_frames = _read_frames(args.gt)

    n = min(len(sr_frames), len(gt_frames))
    if args.max_frames:
        n = min(n, args.max_frames)
    print(f"Comparing {n} frames\n")

    # Accumulate per-method results
    results = {name: {'psnr': [], 'ssim': []} for name, _ in METHODS}

    for i in range(n):
        sr = sr_frames[i]
        gt = gt_frames[i]
        if sr.shape != gt.shape:
            gt = cv2.resize(gt, (sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)
        for name, fn in METHODS:
            psnr, ssim = fn(sr, gt)
            results[name]['psnr'].append(psnr)
            results[name]['ssim'].append(ssim)

        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{n} frames...")

    # Print summary table
    print(f"\n{'Method':<42} {'PSNR mean':>10} {'±':>2} {'std':>6}   {'SSIM mean':>10} {'±':>2} {'std':>6}")
    print("-" * 85)
    for name, _ in METHODS:
        p = results[name]['psnr']
        s = results[name]['ssim']
        print(f"  {name:<40} {np.mean(p):>10.4f}  {np.std(p):>6.4f}   {np.mean(s):>10.4f}  {np.std(s):>6.4f}")


if __name__ == '__main__':
    main()