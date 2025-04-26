import os
from pathlib import Path
import numpy as np
import torch

from pam_mmpose.models import builder
from pam_mmpose.models.builder import POSENETS
from pam_mmpose.models.detectors.base import BasePose

from .backbone.swin_utils import load_pretrained
from .backbone.swin_transformer_v2 import SwinTransformerV2
from .keypoint_heads.head import PoseHead
import torch.nn.functional as F


class PoseAnythingModel(BasePose):
    def __init__(self):
        super().__init__()
        self.encoder_config = dict(
            type="SwinTransformerV2",
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=14,
            pretrained_window_sizes=[12, 12, 12, 6],
            drop_path_rate=0.1,
            img_size=224,
        )
        self.keypoint_head_input = dict(
            type="PoseHead",
            in_channels=1024,
            transformer=dict(
                type="EncoderDecoder",
                d_model=256,
                nhead=8,
                num_encoder_layers=3,
                num_decoder_layers=3,
                graph_decoder="pre",
                dim_feedforward=1024,
                dropout=0.1,
                similarity_proj_dim=256,
                dynamic_proj_dim=128,
                activation="relu",
                normalize_before=False,
                return_intermediate_dec=True,
            ),
            share_kpt_branch=False,
            num_decoder_layer=3,
            with_heatmap_loss=True,
            heatmap_loss_weight=2.0,
            support_order_dropout=-1,
            positional_encoding=dict(
                type="SinePositionalEncoding", num_feats=128, normalize=True
            ),
        )

        self.pretrained = os.path.join(
            Path(__file__).parent, "pretrained/swinv2_base_22k_500k.pth"
        )
        self.encoder_sample, self.backbone_type = self.init_backbone(
            self.pretrained, self.encoder_config
        )
        self.keypoint_head = builder.build_head(self.keypoint_head_input)
        self.keypoint_head.init_weights()
        self.train_cfg = (dict(),)
        self.test_cfg = dict(
            flip_test=False,
            post_process="default",
            shift_heatmap=True,
            modulate_kernel=11,
        )
        self.target_type = self.test_cfg.get(
            "target_type", "GaussianHeatMap"
        )  # GaussianHeatMap

    def init_backbone(self, pretrained, encoder_config):
        if "swin" in pretrained:
            encoder_sample = builder.build_backbone(encoder_config)
            pretext_model = torch.load(pretrained, map_location="cpu")["model"]
            model_dict = encoder_sample.state_dict()
            state_dict = {
                (
                    k.replace("encoder.", "")
                    if k.startswith("encoder.")
                    else k.replace("decoder.", "")
                ): v
                for k, v in pretext_model.items()
                if (
                    k.replace("encoder.", "")
                    if k.startswith("encoder.")
                    else k.replace("decoder.", "")
                )
                in model_dict.keys()
                and model_dict[
                    (
                        k.replace("encoder.", "")
                        if k.startswith("encoder.")
                        else k.replace("decoder.", "")
                    )
                ].shape
                == pretext_model[k].shape
            }
            model_dict.update(state_dict)
            encoder_sample.load_state_dict(model_dict)
            # load_pretrained(pretrained, encoder_sample, logger=None)
            backbone = "swin"
        return encoder_sample, backbone

    @property
    def with_keypoint(self):
        """Check if has keypoint_head."""
        return hasattr(self, "keypoint_head")

    def init_weights(self, pretrained=None):
        """Weight initialization for model."""
        self.encoder_sample.init_weights(pretrained)
        self.encoder_query.init_weights(pretrained)
        self.keypoint_head.init_weights()

    def forward(
        self,
        img_s,
        target_s,
        target_weight_s,
        target_q,
        target_weight_q,
        img_metas=None,
        **kwargs,
    ):

        batch_size, _, img_height, img_width = img_s.shape
        assert [i["sample_skeleton"][0] != i["query_skeleton"] for i in img_metas]
        skeleton = [i["sample_skeleton"][0] for i in img_metas]

        feature_q, feature_s = self.extract_features(img_s, img_s)

        mask_s = target_weight_s[0]
        for index in range(target_weight_s.shape[1]):
            mask_s = mask_s * target_weight_s[:, index]

        outs, _, _, _ = self.keypoint_head.forward_iaa(
            feature_q, feature_s, target_s, mask_s, skeleton
        )
        outs = outs.transpose(1, 0)
        outs = outs.transpose(1, -1)
        outs = F.adaptive_max_pool2d(outs, 1).squeeze(dim=-1).squeeze(dim=-1)

        return outs

    def extract_features(self, img_s, img_q):
        if self.backbone_type == "swin":
            feature_q = self.encoder_sample.forward_features(img_q)  # [bs, C, h, w]
            feature_s = torch.clone(feature_q)
        elif self.backbone_type == "dino":
            batch_size, _, img_height, img_width = img_q.shape
            feature_q = (
                self.encoder_sample.get_intermediate_layers(img_q, n=1)[0][:, 1:]
                .reshape(batch_size, img_height // 8, img_width // 8, -1)
                .permute(0, 3, 1, 2)
            )  # [bs, 3, h, w]
            feature_s = [
                self.encoder_sample.get_intermediate_layers(img, n=1)[0][:, 1:]
                .reshape(batch_size, img_height // 8, img_width // 8, -1)
                .permute(0, 3, 1, 2)
                for img in img_s
            ]
        elif self.backbone_type == "dinov2":
            batch_size, _, img_height, img_width = img_q.shape
            feature_q = self.encoder_sample.get_intermediate_layers(
                img_q, n=1, reshape=True
            )[
                0
            ]  # [bs, c, h, w]
            feature_s = [
                self.encoder_sample.get_intermediate_layers(img, n=1, reshape=True)[0]
                for img in img_s
            ]
        else:
            feature_s = [self.encoder_sample(img) for img in img_s]
            feature_q = self.encoder_query(img_q)

        return feature_q, feature_s

    def forward_test(self):
        print()
