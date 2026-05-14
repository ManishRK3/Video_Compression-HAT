import importlib
from copy import deepcopy
from os import path as osp

from hat.utils import get_root_logger, scandir
from hat.utils.registry import LOSS_REGISTRY
from .losses import g_path_regularize, gradient_penalty_loss, r1_penalty

__all__ = ['build_loss', 'gradient_penalty_loss', 'r1_penalty', 'g_path_regularize']

# import losses.py so all @LOSS_REGISTRY.register() decorators run
from . import losses  # noqa: F401


def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
