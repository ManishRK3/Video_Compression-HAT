import importlib
from copy import deepcopy
from os import path as osp

from hat.utils import get_root_logger, scandir
from hat.utils.registry import LOSS_REGISTRY
from .gan_loss import g_path_regularize, gradient_penalty_loss, r1_penalty

__all__ = ['build_loss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize']

loss_folder = osp.dirname(osp.abspath(__file__))
loss_filenames = [osp.splitext(osp.basename(v))[0] for v in scandir(loss_folder) if v.endswith('_loss.py')]
_loss_modules = [importlib.import_module(f'hat.losses.{file_name}') for file_name in loss_filenames]


def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
