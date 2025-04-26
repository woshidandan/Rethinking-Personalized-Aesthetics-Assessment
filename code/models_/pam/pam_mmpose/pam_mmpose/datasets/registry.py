# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from .builder import DATASETS, PIPELINES

__all__ = ['DATASETS', 'PIPELINES']

warnings.simplefilter('once', DeprecationWarning)
warnings.warn(
    'Registries (DATASETS, PIPELINES) have been moved to '
    'pam_mmpose.datasets.builder. Importing from '
    'pam_mmpose.models.registry will be deprecated in the future.',
    DeprecationWarning)
