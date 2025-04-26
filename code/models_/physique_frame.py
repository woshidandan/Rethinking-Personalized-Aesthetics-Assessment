import torch.nn as nn
import torch
from .pointmlp.pointmlp import PointMLPGenEncoder

import torch.nn.functional as F
from . import clip_vit, attention_fusion
from .pam.pam import PoseAnythingModel

from .model_config.config import (
    VIT,
    INPUT_DIMS,
    GROUPNORM,
    ADAPTER,
    FUSION,
    DROPOUT_RATE,
    HIDDEN_DIMS,
)


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class PhysiqueFrame(nn.Module):
    def __init__(self):
        super().__init__()

        self.vit = clip_vit.VisionTransformer(**VIT)
        self.pointmlp = PointMLPGenEncoder()
        self.pam = PoseAnythingModel()

        clip_input_num = INPUT_DIMS["clip_input_num"]
        mf_input_num = INPUT_DIMS["mf_input_num"]
        gt_input_num = INPUT_DIMS["gt_input_num"]
        people_answer_num = INPUT_DIMS["people_answer_num"]
        in_ch = clip_input_num + mf_input_num + gt_input_num + people_answer_num

        self.layernorm_x = nn.LayerNorm(clip_input_num)
        self.groupnorm_mf = nn.GroupNorm(**GROUPNORM["mf"])
        self.groupnorm_pam = nn.GroupNorm(**GROUPNORM["pam"])
        self.adapter = Adapter(clip_input_num, ADAPTER["reduction"])
        self.ratio = FUSION["ratio"]
        self.fusion_mode = FUSION["mode"]

        if (
            self.fusion_mode == "attention_image_q"
            or self.fusion_mode == "attention_preference_q"
        ):
            self.attention_fusion = attention_fusion.AttentionFusion()

        def make_classifier():
            return nn.Sequential(
                nn.Linear(in_ch, HIDDEN_DIMS["layer1"]),
                nn.BatchNorm1d(HIDDEN_DIMS["layer1"]),
                nn.ReLU(),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(HIDDEN_DIMS["layer1"], HIDDEN_DIMS["layer2"]),
                nn.BatchNorm1d(HIDDEN_DIMS["layer2"]),
                nn.ReLU(inplace=True),
                nn.Dropout(DROPOUT_RATE),
                nn.Linear(HIDDEN_DIMS["layer2"], 1),
                nn.Sigmoid(),
            )

        self.appearance_classifier = make_classifier()
        self.health_classifier = make_classifier()
        self.posture_classifier = make_classifier()

    def forward(
        self,
        x,
        mesh_point,
        target_s,
        target_weight_s,
        img_metas,
        preference_tensor,
    ):
        posture_feature = self.pam.forward(
            img_s=x,
            target_s=target_s,
            target_weight_s=target_weight_s,
            target_q=None,
            target_weight_q=None,
            img_metas=img_metas,
        )

        posture_feature = self.groupnorm_pam(posture_feature)

        mesh_feature = self.pointmlp.forward(mesh_point)
        mesh_feature = F.adaptive_max_pool1d(mesh_feature, 1).squeeze(dim=-1)
        mesh_feature = self.groupnorm_mf(mesh_feature)

        image_feature = self.vit.forward(x.type(torch.float32)).float()
        x_clip = self.adapter(image_feature)
        image_feature = self.ratio * x_clip + (1 - self.ratio) * image_feature
        image_feature = self.layernorm_x(image_feature)

        if self.fusion_mode == "cat":
            x = torch.cat(
                (image_feature, posture_feature, mesh_feature, preference_tensor), dim=1
            )
        elif self.fusion_mode == "attention_image_q":
            combined_feature = torch.cat(
                (image_feature, posture_feature, mesh_feature), dim=1
            )
            x = self.attention_fusion.forward(combined_feature, preference_tensor)
        elif self.fusion_mode == "attention_preference_q":
            combined_feature = torch.cat(
                (image_feature, posture_feature, mesh_feature), dim=1
            )
            x = self.attention_fusion.forward(preference_tensor, combined_feature)

        y_appearance_pred = self.appearance_classifier(x)
        y_health_pred = self.health_classifier(x)
        y_posture_pred = self.posture_classifier(x)

        return y_appearance_pred * 10, y_health_pred * 10, y_posture_pred * 10
