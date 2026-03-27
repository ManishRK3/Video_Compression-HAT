import logging
import os
import cv2
import torch
from collections import defaultdict
from os import path as osp

import hat.archs
import hat.data
import hat.models

from basicsr.data import build_dataloader, build_dataset
from basicsr.models import build_model
from basicsr.utils import get_env_info, get_root_logger, get_time_str, make_exp_dirs, tensor2img
from basicsr.utils.options import dict2str, parse_options


def test_video_pipeline(root_path):
    opt, _ = parse_options(root_path, is_train=False)
    torch.backends.cudnn.benchmark = True

    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'], f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    model = build_model(opt)

    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = build_dataset(dataset_opt)
        test_loader = build_dataloader(
            test_set, dataset_opt,
            num_gpu=opt['num_gpu'], dist=opt['dist'],
            sampler=None, seed=opt['manual_seed']
        )
        logger.info(f"Dataset: {dataset_opt['name']} — {len(test_set)} frames")

        is_video_dataset = test_set.__class__.__name__ == 'VideoPairedDataset'

        if is_video_dataset:
            _run_video_inference(model, test_loader, opt, logger)
        else:
            model.validation(
                test_loader,
                current_iter=opt['name'],
                tb_logger=None,
                save_img=opt['val']['save_img']
            )


def _run_video_inference(model, dataloader, opt, logger):
    """
    Run SISR model frame-by-frame on a video dataset and reassemble outputs
    into per-video MP4 files.

    Frames are written incrementally (no full-video buffering in memory).
    This assumes the dataloader is ordered (shuffle=False), which is standard
    for test pipelines. If you need random-order support, switch to the
    collect-then-sort approach at the cost of higher memory usage.
    """
    dataset_name = dataloader.dataset.opt['name']
    save_dir = osp.join(opt['path']['visualization'], dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    writers = {}   # video_name -> cv2.VideoWriter
    frame_counts = defaultdict(int)
    video_fps = dataloader.dataset.video_fps  # populated at dataset init

    for val_data in dataloader:
        video_name = val_data['video_name'][0]
        frame_idx = int(val_data['frame_idx'][0])
        expected = frame_counts[video_name]
        assert frame_idx == expected, (
            f"{video_name}: expected frame {expected}, got {frame_idx}. "
            "Ensure dataset is ordered and shuffle=False."
        )
        model.feed_data(val_data)
        model.pre_process()
        if 'tile' in opt:
            model.tile_process()
        else:
            model.process()
        model.post_process()

        visuals = model.get_current_visuals()
        sr_img = tensor2img([visuals['result']])  # HWC, BGR uint8

        # Lazily create a writer the first time we see each video
        if video_name not in writers:
            h, w = sr_img.shape[:2]
            fps = video_fps.get(video_name, 30)
            video_path = osp.join(save_dir, f"{video_name}_SR.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writers[video_name] = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
            logger.info(f"Started writing: {video_path}  ({w}x{h} @ {fps}fps)")

        writers[video_name].write(sr_img)
        frame_counts[video_name] += 1

        # Free GPU memory each frame — important for long videos
        del model.lq, model.output
        if hasattr(model, 'gt'):
            del model.gt
        torch.cuda.empty_cache()

    for video_name, writer in writers.items():
        writer.release()
        out_path = osp.join(save_dir, f"{video_name}_SR.mp4")
        logger.info(f"Saved {out_path} ({frame_counts[video_name]} frames)")


if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_video_pipeline(root_path)
