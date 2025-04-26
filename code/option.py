import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument(
        "--path_to_images",
        type=str,
        default="/data/dataset/zhb/PAA-User",
        help="directory to images",
    )

    parser.add_argument(
        "--path_to_save_csv",
        type=str,
        default="/data/dataset/zhb/github/csv/user/ESFJ/",
        help="directory to csv_folder",
    )

    parser.add_argument(
        "--experiment_dir_name",
        type=str,
        default="./pth",
        help="directory to project",
    )
    parser.add_argument(
        "--path_to_model_weight",
        type=str,
        default="/data/dataset/zhb/github/final_pth/"
        + "PhysiqueFrame_ESFJ_vaccA0.746_srccA0.655_vlccA0.686_vaccH0.740_srccH0.544_vlccH0.573_vaccP0.766_srccP0.667_vlccP0.704.pth",
        help="directory to model",
    )

    parser.add_argument("--init_lr", type=int, default=0.0001, help="learning_rate")
    parser.add_argument("--num_epoch", type=int, default=20, help="epoch num for train")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="16how many pictures to process one time",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="num_workers",
    )

    parser.add_argument("--gpu_id", type=str, default="0", help="which gpu to use")
    parser.add_argument(
        "--pre_mesh_point_dir",
        type=str,
        default="/data/dataset/zhb/else/mesh_point_all_new.pt",
        help="directory to pre-cached mesh points",
    )
    parser.add_argument(
        "--pre_mesh_point",
        type=bool,
        default=False,
        help="choose to use the pre-cached mesh points",
    )

    args = parser.parse_args()
    return args
