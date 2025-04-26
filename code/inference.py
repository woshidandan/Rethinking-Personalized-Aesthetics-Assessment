from torchvision import transforms
import os
import torch
from models_.physique_frame import PhysiqueFrame
from dataset import get_mesh, get_graph_point_position
import numpy as np
from torchvision.datasets.folder import default_loader
from models_.pam.pipelines import TopDownGenerateTargetFewShot

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

""" Input setting """
weights_path = "./PhysiqueFrame_ESFJ_vaccA0.746_srccA0.655_vlccA0.686_vaccH0.740_srccH0.544_vlccH0.573_vaccP0.766_srccP0.667_vlccP0.704.pth"

img_path = "./PAA-User/8_gB2UWrVPImE_1.jpg"

""" Input setting """

map_location = "cuda:0"

device = torch.device(map_location if torch.cuda.is_available() else "cpu")

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]

normalize = transforms.Normalize(mean=IMAGE_NET_MEAN, std=IMAGE_NET_STD)

data_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalize
    ]
)

transform_clip = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
)

# load model
model = PhysiqueFrame().to(device)
state_dict = torch.load(weights_path, map_location=map_location)
model.load_state_dict(state_dict)
model.eval()

# process img
img = default_loader(img_path)
x_clip = transform_clip(img)
img = data_transform(img)

h = img.shape[1]
w = img.shape[2]
if h > w:
    img = transforms.Resize((224, int((224 * w) / h)))(img)
else:
    img = transforms.Resize((int((224 * h) / w), 224))(img)
img = transforms.CenterCrop(224)(img)
mesh_x = np.transpose(img.numpy(), (1, 2, 0)) * 255

img = normalize(img)


mesh_point, body_pose, body_shape, root_cam, boxes, joint_proj_resize, volume = (
    get_mesh(mesh_x, 224)
)


if mesh_point != None:
    mesh_point = torch.tensor(mesh_point) / 1.0
    mesh_point = mesh_point.to("cpu")
    point_position = get_graph_point_position(joint_proj_resize)

else:
    mesh_point = torch.zeros(10475, 3) / 1.0
    point_position = torch.zeros(15, 2).numpy()

mesh_point = mesh_point.to(device)
mesh_point = torch.unsqueeze(mesh_point, dim=0)

img = torch.unsqueeze(img, dim=0)
img = img.to(device)

point_position_shape = point_position.shape
point_position = point_position.reshape(
    1, point_position_shape[0], point_position_shape[1]
)
point_position = torch.tensor(point_position).to(device)

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
edge_index = edge_index.to(device)

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
    image_size=np.array([224, 224]),
    heatmap_size=[64, 64],
    num_output_channels=channel_cfg["num_output_channels"],
    num_joints=channel_cfg["dataset_joints"],
    dataset_channel=channel_cfg["dataset_channel"],
    inference_channel=channel_cfg["inference_channel"],
)
data_cfg["joint_weights"] = None
data_cfg["use_different_joint_weights"] = False

kp_src = torch.tensor(point_position).float().to(device)
kp_src = torch.squeeze(kp_src, dim=0)
kp_src_3d = torch.cat((kp_src, torch.zeros(kp_src.shape[0], 1).to(device)), dim=-1)
kp_src_3d_weight = torch.cat(
    (torch.ones_like(kp_src).to(device), torch.zeros(kp_src.shape[0], 1).to(device)),
    dim=-1,
)

genHeatMap = TopDownGenerateTargetFewShot()
target_s, target_weight_s = genHeatMap._msra_generate_target(
    data_cfg, kp_src_3d.cpu(), kp_src_3d_weight.cpu(), sigma=2
)
target_s = torch.tensor(target_s).float()[None]
target_weight_s = torch.tensor(target_weight_s).float()[None]

kp_src_3d = torch.unsqueeze(kp_src_3d, dim=0).to(device)
kp_src = torch.unsqueeze(kp_src, dim=0).to(device)
target_s = torch.unsqueeze(target_s, dim=0).to(device)
target_weight_s = torch.unsqueeze(target_weight_s, dim=0).to(device)

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

preference_tensor = [0.5, 0.5, 0.5, 0.5, 0.5]
preference_tensor = torch.tensor(preference_tensor)
preference_tensor = torch.unsqueeze(preference_tensor, dim=0).to(device)

# predict
with torch.no_grad():

    output = model(
        img,
        mesh_point,
        target_s,
        target_weight_s,
        img_metas,
        preference_tensor,
    )

    print(
        "appearance: "
        + str("{:.2f}".format(float(output[0])))
        + "\n"
        + "health: "
        + str("{:.2f}".format(float(output[1])))
        + "\n"
        + "posture: "
        + str("{:.2f}".format(float(output[2])))
        + "\n"
    )
