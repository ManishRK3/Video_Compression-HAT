import torch
from torch.nn import functional as F

from hat.utils.registry import MODEL_REGISTRY
from hat.models.sr_model import SRModel
from hat.metrics import calculate_metric
from hat.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp


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
        net = getattr(self, 'net_g_ema', self.net_g)
        net.eval()
        with torch.no_grad():
            self.output = net(self.img)

    def tile_process(self):
        batch, channel, height, width = self.img.shape
        output_shape = (batch, channel, height * self.scale, width * self.scale)
        self.output = self.img.new_zeros(output_shape)

        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        for y in range(tiles_y):
            for x in range(tiles_x):
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']

                x0 = ofs_x
                x1 = min(ofs_x + self.opt['tile']['tile_size'], width)
                y0 = ofs_y
                y1 = min(ofs_y + self.opt['tile']['tile_size'], height)

                x0p = max(x0 - self.opt['tile']['tile_pad'], 0)
                x1p = min(x1 + self.opt['tile']['tile_pad'], width)
                y0p = max(y0 - self.opt['tile']['tile_pad'], 0)
                y1p = min(y1 + self.opt['tile']['tile_pad'], height)

                input_tile = self.img[:, :, y0p:y1p, x0p:x1p]

                net = getattr(self, 'net_g_ema', self.net_g)
                net.eval()
                with torch.no_grad():
                    output_tile = net(input_tile)

                s = self.opt['scale']
                self.output[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = output_tile

    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[
            :, :,
            0:h - self.mod_pad_h * self.scale,
            0:w - self.mod_pad_w * self.scale,
        ]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):
                self.metric_results = {m: 0 for m in self.opt['val']['metrics']}
            self._initialize_best_metric_results(dataset_name)
            self.metric_results = {m: 0 for m in self.metric_results}

        metric_data = {}
        pbar = tqdm(total=len(dataloader), unit='image') if use_pbar else None

        for idx, val_data in enumerate(dataloader):
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
                metric_data['img2'] = tensor2img([visuals['gt']])
                del self.gt

            del self.lq, self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'],
                        img_name, f'{img_name}_{current_iter}.png'
                    )
                elif self.opt['val']['suffix']:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_{self.opt["val"]["suffix"]}.png'
                    )
                else:
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_{self.opt["name"]}.png'
                    )
                imwrite(sr_img, save_img_path)

            if with_metrics:
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)

            if pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results:
                self.metric_results[metric] /= (idx + 1)
                self._update_best_metric_result(
                    dataset_name, metric, self.metric_results[metric], current_iter
                )
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
