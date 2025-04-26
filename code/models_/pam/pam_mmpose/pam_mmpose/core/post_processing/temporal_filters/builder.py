# Copyright (c) OpenMMLab. All rights reserved.
from pam_mmcv.utils import Registry

FILTERS = Registry('filters')


def build_filter(cfg):
    """Build filters function."""
    return FILTERS.build(cfg)
