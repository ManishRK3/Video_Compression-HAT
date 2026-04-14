import torch
from torch.nn import functional as F

from hat.utils.registry import MODEL_REGISTRY
from hat.models.sr_model import SRModel
from hat.metrics import calculate_metric
from hat.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp

import cv2
import os


@MODEL_REGISTRY.register()
class HATModel(SRModel):

    def pre_process(self):
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)

    def tile_process(self):
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']

                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                if hasattr(self, 'net_g_ema'):
                    self.net_g_ema.eval()
                    with torch.no_grad():
                        output_tile = self.net_g_ema(input_tile)
                else:
                    self.net_g.eval()
                    with torch.no_grad():
                        output_tile = self.net_g(input_tile)

                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale,
                                  0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']

        is_video_dataset = dataloader.dataset.__class__.__name__ == 'VideoPairedDataset'
        video_writers = {} if is_video_dataset else None

        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)

        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()

        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):

            if 'video_name' in val_data:
                img_name = f"{val_data['video_name'][0]}_{int(val_data['frame_idx'][0])}"
            else:
                img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)

            self.pre_process()
            if 'tile' in self.opt:
                self.tile_process()
            else:
                self.process()
            self.post_process()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if is_video_dataset:
                    video_name = val_data['video_name'][0]

                    save_dir = osp.join(self.opt['path']['visualization'], dataset_name)
                    os.makedirs(save_dir, exist_ok=True)

                    video_path = osp.join(save_dir, f"{video_name}.mp4")

                    h, w = sr_img.shape[:2]

                    if video_name not in video_writers:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        video_writers[video_name] = cv2.VideoWriter(
                            video_path, fourcc, 30, (w, h)
                        )

                    video_writers[video_name].write(sr_img)

                else:
                    # original image saving
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                     f'{img_name}_{self.opt["name"]}.png')

                    imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if is_video_dataset:
            for writer in video_writers.values():
                writer.release()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(dataset_name, metric,
                                                self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
