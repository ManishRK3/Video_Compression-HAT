import argparse
import csv
import logging
import os
import subprocess
import sys
import tempfile
from os import path as osp

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim
from skimage.util import img_as_float
from tqdm import tqdm

import hat.archs
import hat.data
import hat.models

from hat.data import build_dataloader, build_dataset
from hat.models import build_model
from hat.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, tensor2img
from hat.utils.options import dict2str, parse_options


# ---------------------------------------------------------------------------
# Model summary
# ---------------------------------------------------------------------------

def _print_model_summary(model, logger, bench_size=320):
    try:
        from thop import profile, clever_format
    except ImportError:
        logger.warning('thop not installed — run `pip install thop` for GMacs profiling')
        return

    net = model.net_g
    device = next(net.parameters()).device
    dummy = torch.randn(1, 3, bench_size, bench_size).to(device)

    # Scale from square benchmark to actual 360p LQ area (640x360)
    scale_to_360p = (640 * 360) / (bench_size * bench_size)

    net.eval()
    with torch.no_grad():
        macs, params = profile(net, inputs=(dummy,), verbose=False)

    _, params_str = clever_format([macs, params], '%.2f')
    logger.info('=== Model Summary ===')
    logger.info(f'Architecture    : {net.__class__.__name__}')
    logger.info(f'Benchmark input : {bench_size}x{bench_size}')
    logger.info(f'GMacs ({bench_size}x{bench_size})  : {macs / 1e9:.1f} G')
    logger.info(f'GMacs (~360p LQ): {macs * scale_to_360p / 1e9:.1f} G  (scaled from {bench_size}x{bench_size})')
    logger.info(f'Params          : {params / 1e6:.2f} M  ({params_str})')
    logger.info(str(net))


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

class _FfmpegWriter:
    """Wraps an ffmpeg subprocess. Accepts BGR uint8 numpy frames via .append_data()."""
    def __init__(self, path, w, h, fps, codec, crf):
        cmd = [
            'ffmpeg', '-y', '-loglevel', 'error',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', 'pipe:0',
            '-vcodec', codec, '-preset', 'fast', '-crf', str(crf),
            path,
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def append_data(self, frame_bgr):
        self._proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        self._proc.stdin.close()
        self._proc.wait()


def _open_writer(path, w, h, fps, crf=20):
    return _FfmpegWriter(path, w, h, fps, codec='libx264', crf=crf)


def _open_writer_nvenc(path, w, h, fps, crf=23):
    return _FfmpegWriter(path, w, h, fps, codec='h264_nvenc', crf=crf)


def _get_video_info(path):
    cap = cv2.VideoCapture(path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, n


def _iter_frames(path):
    """Yield BGR uint8 frames sequentially — keeps VideoCapture open, no per-frame seeking."""
    cap = cv2.VideoCapture(path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _calculate_ssim_psnr(img1_bgr, img2_bgr):
    img1 = img_as_float(img1_bgr)
    img2 = img_as_float(img2_bgr)
    ssim = sk_ssim(img1, img2, channel_axis=2, data_range=1.0)
    psnr = sk_psnr(img1, img2, data_range=1.0)
    return ssim, psnr


def _to_lpips_tensor(img_bgr, device):
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


def _frame_metrics(sr_bgr, gt_bgr, lpips_fn, device):
    ssim, psnr = _calculate_ssim_psnr(sr_bgr, gt_bgr)
    lp = lpips_fn(_to_lpips_tensor(sr_bgr, device),
                  _to_lpips_tensor(gt_bgr, device)).item() if lpips_fn is not None else 0.0
    return {'psnr': psnr, 'ssim': ssim, 'lpips': lp}


def _average(records):
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

_FIELDS = ['video',
           'psnr_raw', 'ssim_raw', 'lpips_raw',
           'psnr_h264', 'ssim_h264', 'lpips_h264']


def _fmt(metrics):
    return {k: f"{v:.4f}" for k, v in metrics.items()}


def _make_row(video_name, raw, h264):
    return {'video': video_name,
            **{f"{k}_raw":  v for k, v in _fmt(raw).items()},
            **{f"{k}_h264": v for k, v in _fmt(h264).items()}}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _bgr_to_tensor(bgr, device):
    """BGR uint8 HWC → float32 (1,3,H,W) tensor on device."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)


def _run_sr(model, lq_tensor, opt):
    """Run SR on a single LQ tensor. Returns SR as BGR uint8 numpy."""
    model.feed_data({'lq': lq_tensor})
    model.pre_process()
    if 'tile' in opt:
        model.tile_process()
    else:
        model.process()
    model.post_process()
    sr_bgr = tensor2img([model.get_current_visuals()['result']])
    del model.lq, model.output
    if hasattr(model, 'gt'):
        del model.gt
    return sr_bgr


# ---------------------------------------------------------------------------
# Per-video pipeline
# ---------------------------------------------------------------------------

def _process_video(video_path, model, opt, save_dir,
                   lr_h, lr_w, hr_h, hr_w,
                   lpips_fn, device,
                   write_raw, write_h264, skip_metrics, logger,
                   crop_to_640=False):
    video_name = osp.splitext(osp.basename(video_path))[0]
    vid_w, vid_h, fps, n_frames = _get_video_info(video_path)

    # When crop_to_640 is enabled, override dimensions and define a crop function.
    # Portrait 720p (720×H) → crop width to 640 (remove right 80px).
    # Landscape 720p (W×720) → crop height to 640 (remove bottom 80px).
    if crop_to_640:
        if vid_w == 720:
            scale = hr_h // lr_h
            hr_w = 640
            lr_w = 640 // scale
        elif vid_h == 720:
            scale = hr_h // lr_h
            hr_h = 640
            lr_h = 640 // scale

    def _crop(frame):
        if crop_to_640:
            if vid_w == 720:
                return frame[:, :640, :]
            if vid_h == 720:
                return frame[:640, :, :]
        return frame

    logger.info(f"\n{'='*60}")
    logger.info(f"Video: {video_name}  ({n_frames} frames @ {fps:.2f}fps)")
    if crop_to_640 and (vid_w == 720 or vid_h == 720):
        logger.info(f"  720p crop applied → HR: {hr_h}×{hr_w}  LQ: {lr_h}×{lr_w}")

    # --- Pass 1: raw LR → SR, and build compressed LR for H264 pass ---
    raw_path      = osp.join(save_dir, f"{video_name}_SR_raw.mp4")
    sr_raw_writer = _open_writer(raw_path, hr_w, hr_h, fps) if write_raw else None
    lr_writer     = None
    tmp_lr_path   = None
    if write_h264:
        tmp_lr_path = tempfile.NamedTemporaryFile(delete=False, suffix='_lr.mp4').name
        lr_writer   = _open_writer_nvenc(tmp_lr_path, lr_w, lr_h, fps)

    raw_records = []
    for bgr in tqdm(_iter_frames(video_path), total=n_frames,
                    desc=f'  raw  {video_name[:30]}', unit='frame', leave=False):
        src    = _crop(bgr)
        lq_bgr = cv2.resize(src, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        gt_bgr = cv2.resize(src, (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

        sr_bgr = _run_sr(model, _bgr_to_tensor(lq_bgr, device), opt)

        if write_raw:
            sr_raw_writer.append_data(sr_bgr)
        if write_h264:
            lr_writer.append_data(lq_bgr)
        if write_raw and not skip_metrics:
            raw_records.append(_frame_metrics(sr_bgr, gt_bgr, lpips_fn, device))

    if write_raw:
        sr_raw_writer.close()
        logger.info(f"  Saved raw SR : {raw_path}")
    if write_h264:
        lr_writer.close()

    # Free CUDA cache once per video rather than per frame
    torch.cuda.empty_cache()

    raw_m = _average(raw_records) if raw_records else {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
    if write_raw and not skip_metrics:
        logger.info(f"  Raw  — PSNR: {raw_m['psnr']:.4f}  SSIM: {raw_m['ssim']:.4f}  LPIPS: {raw_m['lpips']:.4f}")

    # --- Pass 2: compressed LR → SR ---
    h264_m = {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
    if write_h264:
        h264_path      = osp.join(save_dir, f"{video_name}_SR_h264.mp4")
        sr_h264_writer = _open_writer(h264_path, hr_w, hr_h, fps)
        h264_records   = []
        orig_iter      = _iter_frames(video_path)

        for lr_bgr in tqdm(_iter_frames(tmp_lr_path), total=n_frames,
                           desc=f'  h264 {video_name[:30]}', unit='frame', leave=False):
            orig_bgr = next(orig_iter, None)
            gt_bgr   = cv2.resize(_crop(orig_bgr), (hr_w, hr_h), interpolation=cv2.INTER_CUBIC)

            # nvenc may adjust dimensions to even numbers; correct if needed
            if lr_bgr.shape[:2] != (lr_h, lr_w):
                lr_bgr = cv2.resize(lr_bgr, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)

            sr_bgr = _run_sr(model, _bgr_to_tensor(lr_bgr, device), opt)
            sr_h264_writer.append_data(sr_bgr)
            if not skip_metrics:
                h264_records.append(_frame_metrics(sr_bgr, gt_bgr, lpips_fn, device))

        sr_h264_writer.close()
        os.remove(tmp_lr_path)
        torch.cuda.empty_cache()

        h264_m = _average(h264_records) if h264_records else {'psnr': 0.0, 'ssim': 0.0, 'lpips': 0.0}
        logger.info(f"  Saved H264 SR: {h264_path}")
        if not skip_metrics:
            logger.info(f"  H264 — PSNR: {h264_m['psnr']:.4f}  SSIM: {h264_m['ssim']:.4f}  LPIPS: {h264_m['lpips']:.4f}")

    return raw_m, h264_m


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def test_video_pipeline(root_path, write_raw=True, write_h264=False, skip_metrics=False, compute_lpips=True, crop_to_640=False):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--weights', type=str, default=None,
                     help='Path to a .pth checkpoint to override pretrain_network_g in the config')
    pre.add_argument('--video', type=str, default=None,
                     help='Process only this video (stem without extension)')
    pre.add_argument('--print_summary', action='store_true',
                     help='Print model architecture, parameter count, and GMacs before inference')
    pre.add_argument('--summary_size', type=int, default=320,
                     help='Square benchmark input size for GMacs profiling (must be divisible by window_size=16, default: 320)')
    pre.add_argument('--no_lpips', action='store_true',
                     help='Skip LPIPS computation (still computes PSNR and SSIM)')
    extra, remaining = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    opt, _ = parse_options(root_path, is_train=False)

    if extra.weights is not None:
        opt['path']['pretrain_network_g'] = osp.abspath(extra.weights)

    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='hat', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_model(opt)

    if extra.print_summary:
        _print_model_summary(model, logger, bench_size=extra.summary_size)

    # Load LPIPS only when metrics are needed — avoids occupying GPU memory otherwise
    lpips_fn = None
    if not skip_metrics and compute_lpips:
        import lpips as _lpips
        lpips_fn = _lpips.LPIPS(net='alex').to(device).eval()

    for _, dataset_opt in sorted(opt['datasets'].items()):
        if dataset_opt.get('type') != 'VideoPairedDataset':
            test_set    = build_dataset(dataset_opt)
            test_loader = build_dataloader(
                test_set, dataset_opt,
                num_gpu=opt['num_gpu'], dist=opt['dist'],
                sampler=None, seed=opt['manual_seed']
            )
            model.validation(test_loader, current_iter=opt['name'],
                             tb_logger=None, save_img=opt['val']['save_img'])
            continue

        dataset_name = dataset_opt['name']
        save_dir     = osp.join(opt['path']['visualization'], dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        csv_path     = osp.join(save_dir, 'metrics.csv')

        scale  = dataset_opt.get('scale', 2)
        hr_h   = dataset_opt.get('hr_height', 360)
        hr_w   = dataset_opt.get('hr_width', 640)
        lr_h, lr_w = hr_h // scale, hr_w // scale

        video_dir   = dataset_opt['dataroot_hr']
        video_paths = sorted(
            osp.join(video_dir, f) for f in os.listdir(video_dir)
            if f.endswith(('.mp4', '.avi'))
        )

        logger.info(f"Dataset: {dataset_name}  |  {len(video_paths)} videos")
        logger.info(f"HR: {hr_h}x{hr_w}  LQ: {lr_h}x{lr_w}")

        all_rows = []
        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=_FIELDS)
            csv_writer.writeheader()

            for video_path in video_paths:
                video_name = osp.splitext(osp.basename(video_path))[0]
                if extra.video is not None and video_name != extra.video:
                    continue

                raw_m, h264_m = _process_video(
                    video_path, model, opt, save_dir,
                    lr_h, lr_w, hr_h, hr_w,
                    lpips_fn, device,
                    write_raw, write_h264, skip_metrics, logger,
                    crop_to_640=crop_to_640,
                )

                if not skip_metrics:
                    row = _make_row(video_name, raw_m, h264_m)
                    csv_writer.writerow(row)
                    f.flush()
                    all_rows.append(row)

            if not skip_metrics and all_rows:
                mean_row = {'video': 'MEAN'}
                for field in _FIELDS[1:]:
                    mean_row[field] = f"{np.mean([float(r[field]) for r in all_rows]):.4f}"
                csv_writer.writerow(mean_row)
                logger.info(f"\n{'='*60}")
                logger.info(f"MEAN Raw  — PSNR: {mean_row['psnr_raw']}  SSIM: {mean_row['ssim_raw']}  LPIPS: {mean_row['lpips_raw']}")
                if write_h264:
                    logger.info(f"MEAN H264 — PSNR: {mean_row['psnr_h264']}  SSIM: {mean_row['ssim_h264']}  LPIPS: {mean_row['lpips_h264']}")
                logger.info(f"Metrics saved to: {csv_path}")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_video_pipeline(root_path, write_raw=True, write_h264=False, skip_metrics=False, compute_lpips=False, crop_to_640=False)
