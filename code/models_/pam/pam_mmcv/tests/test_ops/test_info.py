# Copyright (c) OpenMMLab. All rights reserved.
import torch


class TestInfo:

    def test_info(self):
        if not torch.cuda.is_available():
            return
        from pam_mmcv.ops import get_compiler_version, get_compiling_cuda_version
        cv = get_compiler_version()
        ccv = get_compiling_cuda_version()
        assert cv is not None
        assert ccv is not None
