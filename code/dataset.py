import math
import os
from pathlib import Path
import trimesh

import argparse
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from SMPLer_X.common.utils_smpler_x.human_models import smpl_x

import sys
import os.path as osp

cfg_root_path = Path(__file__).parent
sys.path.insert(0, osp.join(cfg_root_path, "SMPLer_X", "main"))
sys.path.insert(0, osp.join(cfg_root_path, "SMPLer_X", "data"))
sys.path.insert(0, osp.join(cfg_root_path, "ultralytics_yolov5_master"))
sys.path.insert(0, osp.join(cfg_root_path, "SMPLer_X"))

from SMPLer_X.main.config import cfg
import cv2
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch
import option


opt = option.init()
map_location = "cuda:0"
device = torch.device(map_location)

cudnn.benchmark = True

from SMPLer_X.common.base import Demoer

demoer = Demoer()
demoer._make_model(device)

from SMPLer_X.common.utils_smpler_x.preprocessing import (
    load_img,
    process_bbox,
    generate_patch_image,
)

demoer.model.eval()
transform = transforms.ToTensor()

detector = torch.hub.load(
    os.path.join(Path(__file__).parent, "ultralytics_yolov5_master"),
    "yolov5s",
    os.path.join(
        Path(__file__).parent, "ultralytics_yolov5_master/checkpoints/yolov5s.pt"
    ),
    source="local",
)
detector.eval()


IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)


def cal_point_lenght(x, y):
    assert len(x) == len(y)
    index_list = range(0, len(x))
    length = 0.0
    for i in index_list:
        length += (x[i] - y[i]) ** 2
    return math.sqrt(length)


def no_mesh():
    return None, None, None, None, None, None, None


def get_mesh(original_img, resize_h):
    original_img_height, original_img_width = original_img.shape[:2]

    with torch.no_grad():
        results = detector(original_img)
        person_results = results.xyxy[0][results.xyxy[0][:, 5] == 0]
        class_ids, confidences, boxes = [], [], []

        # determine whether a person has been identified
        if len(person_results) == 0:
            return no_mesh()

        # choose the main person
        index = 0
        max_box = 0.0
        main_index = index
        for detection in person_results:
            x1, y1, x2, y2, confidence, class_id = detection.tolist()
            if (x2 - x1) * (y2 - y1) > max_box:
                max_box = (x2 - x1) * (y2 - y1)
                main_index = index
            index += 1

        x1, y1, x2, y2, confidence, class_id = person_results[main_index].tolist()
        class_ids.append(class_id)
        confidences.append(confidence)
        boxes.append([x1, y1, x2 - x1, y2 - y1])
        # indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        indice = 0
        bbox = boxes[indice]  # x,y,h,w
        bbox = process_bbox(bbox, original_img_width, original_img_height)

        # get main person image
        img, img2bb_trans, bb2img_trans = generate_patch_image(
            original_img, bbox, 1.0, 0.0, False, cfg.input_img_shape
        )
        inputs = transform(img.astype(np.float32)) / 255
        inputs = inputs.cuda()[None, :, :, :]
        inputs = {"img": inputs}
        targets = {}
        meta_info = {}

        # mesh recovery
        out = demoer.model(inputs, targets, meta_info, "test")

        # determine whether mesh has been generated
        if len(out["smplx_mesh_cam"]) == 0:
            return no_mesh()

        mesh_point = -out["smplx_mesh_cam_zero_pose"][0]
        mesh = trimesh.Trimesh(vertices=mesh_point.to("cpu"), faces=smpl_x.face)
        volume = -mesh.volume

        body_pose = torch.cat(
            [out["smplx_jaw_pose"], out["smplx_root_pose"], out["smplx_body_pose"]], 1
        )
        body_shape = out["smplx_shape"]

        joint_proj = out["smplx_joint_proj"].detach().cpu().numpy()[0]
        joint_proj[:, 0] = (
            joint_proj[:, 0] / cfg.output_hm_shape[2] * cfg.input_img_shape[1]
        )
        joint_proj[:, 1] = (
            joint_proj[:, 1] / cfg.output_hm_shape[1] * cfg.input_img_shape[0]
        )
        joint_proj = np.concatenate((joint_proj, np.ones_like(joint_proj[:, :1])), 1)
        joint_proj = np.dot(bb2img_trans, joint_proj.transpose(1, 0)).transpose(1, 0)

    return (
        mesh_point,
        body_pose,
        body_shape,
        out["joint_origin_root_cam"],
        [x1, y1, x2, y2],
        joint_proj[0:25],
        volume,
    )


def get_graph_point_position(joint_proj):
    point_position = joint_proj[0:14]
    append_position = [
        int((joint_proj[22][0] + joint_proj[23][0]) / 2),
        int((joint_proj[22][1] + joint_proj[23][1]) / 2),
    ]
    append_position = np.array(append_position)
    append_position = np.resize(append_position, [1, 2])
    point_position = np.append(point_position, append_position, axis=0).astype(np.int32)
    return point_position


from models_.pam.pipelines import TopDownGenerateTargetFewShot


class Dataset(Dataset):
    def __init__(
        self,
        path_to_csv,
        images_path,
        if_train,
        pre_mesh_point=False,
        pre_mesh_point_dir=None,
    ):

        self.genHeatMap = TopDownGenerateTargetFewShot()
        self.df = pd.read_csv(path_to_csv)

        self.images_path = images_path
        self.if_train = if_train
        self.h = 224
        self.mesh_point_dict = None
        if pre_mesh_point and pre_mesh_point_dir != None:
            self.mesh_point_dict = torch.load(pre_mesh_point_dir)

        if if_train:
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    # normalize
                ]
            )
            self.transform_clip = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize((self.h, self.h)),
                    transforms.ToTensor(),
                    # normalize
                ]
            )
        else:
            self.transform = transforms.Compose(
                [
                    # transforms.Resize((self.h, self.h)),
                    transforms.ToTensor(),
                    # normalize
                ]
            )
            self.transform_clip = transforms.Compose(
                [
                    transforms.Resize((self.h, self.h)),
                    transforms.ToTensor(),
                    # normalize
                ]
            )

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

        # scores
        score_appearance = np.array([row[f"score_appearance"]])
        score_health = np.array([row[f"score_health"]])
        score_posture = np.array([row[f"score_posture"]])

        # pic
        pic_name = row[f"pic_name"]
        pic_path = os.path.join(self.images_path, pic_name)
        pic = default_loader(pic_path)

        # preprocess
        x = self.transform(pic)
        h = x.shape[1]
        w = x.shape[2]
        if h > w:
            x = transforms.Resize((self.h, int((self.h * w) / h)))(x)
        else:
            x = transforms.Resize((int((self.h * h) / w), self.h))(x)
        x = transforms.CenterCrop(self.h)(x)
        mesh_x = np.transpose(x.numpy(), (1, 2, 0)) * 255
        x = normalize(x)

        # generate mesh
        if self.mesh_point_dict == None:
            (
                mesh_point,
                body_pose,
                body_shape,
                root_cam,
                boxes,
                joint_proj_resize,
                volume,
            ) = get_mesh(mesh_x, self.h)
        else:
            (
                mesh_point,
                body_pose,
                body_shape,
                root_cam,
                boxes,
                joint_proj_resize,
                volume,
            ) = self.mesh_point_dict[pic_name]

        if mesh_point != None:
            boxes = torch.tensor(boxes) / 1.0
            mesh_point = torch.tensor(mesh_point) / 1.0

            mesh_point = mesh_point.to("cpu")
            boxes = boxes.to("cpu")
            point_position = get_graph_point_position(joint_proj_resize)

        else:
            mesh_point = torch.zeros(10475, 3) / 1.0
            boxes = torch.zeros(1, 4) / 1.0
            point_position = torch.zeros(15, 2).numpy()

        edge_index = [
            # lower body
            [0, 1],
            [0, 2],
            [1, 3],
            [3, 5],
            [2, 4],
            [4, 6],
            [1, 0],
            [2, 0],
            [3, 1],
            [5, 3],
            [4, 2],
            [6, 4],
            # upper body excluding head
            [0, 8],
            [0, 9],
            [8, 10],
            [10, 12],
            [9, 11],
            [11, 13],
            [8, 0],
            [9, 0],
            [10, 8],
            [12, 10],
            [11, 9],
            [13, 11],
            # others
            [8, 9],
            [0, 7],
            [7, 14],
            [7, 8],
            [7, 9],
            [0, 8],
            [7, 0],
            [14, 7],
            [8, 7],
            [9, 7],
        ]
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_index = edge_index.to("cpu")

        channel_cfg = dict(
            num_output_channels=1,
            dataset_joints=1,
            dataset_channel=[
                [
                    0,
                ],
            ],
            inference_channel=[
                0,
            ],
            max_kpt_num=100,
        )

        data_cfg = dict(
            image_size=np.array([self.h, self.h]),
            heatmap_size=[64, 64],
            num_output_channels=channel_cfg["num_output_channels"],
            num_joints=channel_cfg["dataset_joints"],
            dataset_channel=channel_cfg["dataset_channel"],
            inference_channel=channel_cfg["inference_channel"],
        )
        data_cfg["joint_weights"] = None
        data_cfg["use_different_joint_weights"] = False

        kp_src = torch.tensor(point_position).float()
        kp_src_3d = torch.cat((kp_src, torch.zeros(kp_src.shape[0], 1)), dim=-1)
        kp_src_3d_weight = torch.cat(
            (torch.ones_like(kp_src), torch.zeros(kp_src.shape[0], 1)), dim=-1
        )
        target_s, target_weight_s = self.genHeatMap._msra_generate_target(
            data_cfg, kp_src_3d, kp_src_3d_weight, sigma=2
        )
        target_s = torch.tensor(target_s).float()[None]
        target_weight_s = torch.tensor(target_weight_s).float()[None]

        img_metas = [
            {
                "sample_skeleton": [edge_index],
                "query_skeleton": edge_index,
                "sample_joints_3d": [kp_src_3d],
                "query_joints_3d": kp_src_3d,
                "sample_center": [kp_src.mean(dim=0)],
                "query_center": kp_src.mean(dim=0),
                "sample_scale": [kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0]],
                "query_scale": kp_src.max(dim=0)[0] - kp_src.min(dim=0)[0],
                "sample_rotation": [0],
                "query_rotation": 0,
                "sample_bbox_score": [1],
                "query_bbox_score": 1,
                "query_image_file": "",
                "sample_image_file": [""],
            }
        ]

        target_s = target_s.to("cpu")
        target_weight_s = target_weight_s.to("cpu")

        preference_tensor = eval(row[f"preference_score_list"])
        preference_tensor = torch.tensor(preference_tensor).to("cpu")

        return (
            x,
            score_appearance.astype("float32"),
            score_health.astype("float32"),
            score_posture.astype("float32"),
            mesh_point,
            target_s,
            target_weight_s,
            img_metas,
            preference_tensor,
        )
