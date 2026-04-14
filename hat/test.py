# flake8: noqa
import os.path as osp

import hat.archs
import hat.data
import hat.models
from hat.test_pipeline_runner import test_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    test_pipeline(root_path)
