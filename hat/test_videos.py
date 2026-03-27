import logging
import os
import csv
import subprocess
import tempfile
import numpy as np
import torch
from collections import defaultdict
from os import path as osp
from tqdm import tqdm

import lpips
from skimage.metrics import peak_signal_noise_ratio as sk_psnr, structural_similarity as sk_ssim
from skimage.util import img_as_float

import hat.archs
import hat.data
import hat.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, tensor2img
from basicsr.utils.options import dict2str, parse_options


# ---------------------------------------------------------------------------
# Video I/O — ffmpeg subprocess, no imageio dependency
# ---------------------------------------------------------------------------

class _FfmpegWriter:
    """
    Wraps an ffmpeg subprocess. Accepts BGR uint8 numpy frames via .append_data().
    ffmpeg receives raw BGR on stdin and encodes to the output path.
    """
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
    """libx264, crf=20 — matches DVR pipeline default."""
    return _FfmpegWriter(path, w, h, fps, codec='libx264', crf=crf)


def _open_writer_nvenc(path, w, h, fps, crf=23):
    """h264_nvenc GPU encoder — used for the H264 compression case."""
    return _FfmpegWriter(path, w, h, fps, codec='h264_nvenc', crf=crf)


def _read_frames(path, w, h):
    """Decode a video and yield BGR uint8 frames."""
    cmd = [
        'ffmpeg', '-loglevel', 'error',
        '-i', path,
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1',
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    while True:
        raw = proc.stdout.read(frame_bytes)
        if len(raw) < frame_bytes:
            break
        yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
    proc.stdout.close()
    proc.wait()


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _calculate_ssim_psnr(img1_bgr, img2_bgr):
    img1 = img_as_float(img1_bgr)
    img2 = img_as_float(img2_bgr)
    ssim = sk_ssim(img1, img2, channel_axis=2, data_range=255)
    psnr = sk_psnr(img1, img2)
    return ssim, psnr


def _to_lpips_tensor(img_bgr, device):
    rgb = img_bgr[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).unsqueeze(0).to(device)


def _frame_metrics(sr_bgr, gt_bgr, lpips_fn, device):
    ssim, psnr = _calculate_ssim_psnr(sr_bgr, gt_bgr)
    lp = lpips_fn(_to_lpips_tensor(sr_bgr, device),
                  _to_lpips_tensor(gt_bgr, device)).item()
    return {'psnr': psnr, 'ssim': ssim, 'lpips': lp}


def _average(records):
    keys = records[0].keys()
    return {k: float(np.mean([r[k] for r in records])) for k in keys}


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _run_inference(model, frames_data, opt, desc='inference'):
    """
    Run HAT on a list of val_data dicts.
    Returns (sr_frames, gt_frames) as lists of uint8 BGR numpy arrays.
    """
    sr_frames, gt_frames = [], []
    for val_data in tqdm(frames_data, desc=f'  {desc}', leave=False, unit='frame'):
        model.feed_data(val_data)
        model.pre_process()
        if 'tile' in opt:
            model.tile_process()
        else:
            model.process()
        model.post_process()

        visuals = model.get_current_visuals()
        sr_frames.append(tensor2img([visuals['result']]))  # BGR uint8
        gt_frames.append(tensor2img([visuals['gt']]))

        del model.lq, model.output, model.gt
        torch.cuda.empty_cache()

    return sr_frames, gt_frames


def _compress_lr_frames(frames_data, fps, crf=20):
    """
    DVR approach: compress LR frames through H264 (nvenc) and decode back,
    then replace 'lq' in each val_data with the compressed version.

    Simulates real deployment where the input has been H264-compressed
    before the SR model sees it.
    """
    # Extract LR frames as BGR uint8
    lr_frames = []
    for val_data in frames_data:
        lq = val_data['lq'][0]  # (C, H, W), float [0,1], BGR
        bgr = (lq.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        lr_frames.append(bgr)

    h, w = lr_frames[0].shape[:2]
    tmp_path = tempfile.mktemp(suffix='_lr_h264.mp4')

    # Encode LR frames to temp H264 file
    writer = _open_writer_nvenc(tmp_path, w, h, fps, crf=crf)
    for frame in lr_frames:
        writer.append_data(frame)
    writer.close()

    # Decode back and replace lq in val_data
    compressed_frames_data = []
    for val_data, decoded_bgr in zip(frames_data, _read_frames(tmp_path, w, h)):
        compressed_lq = torch.from_numpy(
            decoded_bgr.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)

        new_val_data = {k: v for k, v in val_data.items()}
        new_val_data['lq'] = compressed_lq
        compressed_frames_data.append(new_val_data)

    os.remove(tmp_path)
    return compressed_frames_data


# ---------------------------------------------------------------------------
# Video writing + metric measurement
# ---------------------------------------------------------------------------

def _write_and_measure(sr_frames, gt_frames, out_path, fps, lpips_fn, device):
    """
    Write SR frames with libx264 (crf=20).
    Metrics computed from in-memory frames — no decode round-trip needed.
    """
    h, w = sr_frames[0].shape[:2]
    writer = _open_writer(out_path, w, h, fps, crf=20)
    records = []
    for sr, gt in zip(sr_frames, gt_frames):
        writer.append_data(sr)
        records.append(_frame_metrics(sr, gt, lpips_fn, device))
    writer.close()
    return _average(records)


# ---------------------------------------------------------------------------
# CSV
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
# Main pipeline
# ---------------------------------------------------------------------------

def test_video_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()

    model = build_model(opt)

    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt,
            num_gpu=opt['num_gpu'], dist=opt['dist'],
            sampler=None, seed=opt['manual_seed']
        )

        if test_set.__class__.__name__ != 'VideoPairedDataset':
            model.validation(test_loader, current_iter=opt['name'],
                             tb_logger=None, save_img=opt['val']['save_img'])
            continue

        dataset_name = dataset_opt['name']
        save_dir     = osp.join(opt['path']['visualization'], dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        csv_path     = osp.join(save_dir, 'metrics.csv')

        logger.info(f"Dataset: {dataset_name}  |  {len(test_set)} frames total")

        # Group frames by video — preserves per-video ordering
        video_frames = defaultdict(list)
        for val_data in test_loader:
            video_frames[val_data['video_name'][0]].append(val_data)

        all_rows = []

        with open(csv_path, 'w', newline='') as f:
            csv_writer = csv.DictWriter(f, fieldnames=_FIELDS)
            csv_writer.writeheader()

            for video_name, frames_data in video_frames.items():
                fps = test_set.video_fps.get(video_name, 30)
                logger.info(f"\n{'='*60}")
                logger.info(f"Video: {video_name}  ({len(frames_data)} frames @ {fps:.2f}fps)")

                # Case 1: Raw LR -> SR
                logger.info("  [1/2] Raw: LR -> SR model")
                sr_frames, gt_frames = _run_inference(model, frames_data, opt, desc='raw inference')
                raw_path = osp.join(save_dir, f"{video_name}_SR_raw.mp4")
                raw_m    = _write_and_measure(sr_frames, gt_frames, raw_path, fps, lpips_fn, device)
                logger.info(f"  Raw  — PSNR: {raw_m['psnr']:.4f}  SSIM: {raw_m['ssim']:.4f}  LPIPS: {raw_m['lpips']:.4f}")
                logger.info(f"  Saved: {raw_path}")

                # Case 2: LR -> H264 compress -> SR (DVR approach)
                logger.info("  [2/2] H264: LR -> H264 compress -> SR model")
                compressed_frames_data = _compress_lr_frames(frames_data, fps, crf=20)
                sr_frames_h264, _ = _run_inference(model, compressed_frames_data, opt, desc='h264 inference')
                h264_path = osp.join(save_dir, f"{video_name}_SR_h264.mp4")
                h264_m    = _write_and_measure(sr_frames_h264, gt_frames, h264_path, fps, lpips_fn, device)
                logger.info(f"  H264 — PSNR: {h264_m['psnr']:.4f}  SSIM: {h264_m['ssim']:.4f}  LPIPS: {h264_m['lpips']:.4f}")
                logger.info(f"  Saved: {h264_path}")

                row = _make_row(video_name, raw_m, h264_m)
                csv_writer.writerow(row)
                f.flush()
                all_rows.append(row)

            if all_rows:
                mean_row = {'video': 'MEAN'}
                for field in _FIELDS[1:]:
                    mean_row[field] = f"{np.mean([float(r[field]) for r in all_rows]):.4f}"
                csv_writer.writerow(mean_row)
                logger.info(f"\n{'='*60}")
                logger.info(f"MEAN Raw  — PSNR: {mean_row['psnr_raw']}  SSIM: {mean_row['ssim_raw']}  LPIPS: {mean_row['lpips_raw']}")
                logger.info(f"MEAN H264 — PSNR: {mean_row['psnr_h264']}  SSIM: {mean_row['ssim_h264']}  LPIPS: {mean_row['lpips_h264']}")
                logger.info(f"Metrics saved to: {csv_path}")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_video_pipeline(root_path)