import os

from tqdm import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader

from models_.physique_frame import PhysiqueFrame

from dataset import Dataset
from util import AverageMeter, calculate_metrics
import option


opt = option.init()

os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(opt.gpu_id)
map_location = "cuda:0"
device = torch.device(map_location)


def adjust_learning_rate(params, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = params.init_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def create_data_part(opt):
    train_csv_path = os.path.join(opt.path_to_save_csv, "train.csv")
    val_csv_path = os.path.join(opt.path_to_save_csv, "val.csv")

    train_ds = Dataset(
        train_csv_path,
        opt.path_to_images,
        if_train=True,
        pre_mesh_point=opt.pre_mesh_point,
        pre_mesh_point_dir=opt.pre_mesh_point_dir,
    )
    val_ds = Dataset(
        val_csv_path,
        opt.path_to_images,
        if_train=False,
        pre_mesh_point=opt.pre_mesh_point,
        pre_mesh_point_dir=opt.pre_mesh_point_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=opt.batch_size,
        num_workers=opt.num_workers,
        shuffle=False,
    )

    return train_loader, val_loader


def train(opt, model, loader, optimizer, criterion, global_step=None, name=None):
    model.train()
    train_losses = AverageMeter()
    for idx, (
        x,
        y_appearance,
        y_health,
        y_posture,
        mesh_point,
        target_s,
        target_weight_s,
        img_metas,
        preference_tensor,
    ) in enumerate(tqdm(loader)):
        x = x.to(device)
        y_appearance = y_appearance.to(device)
        y_health = y_health.to(device)
        y_posture = y_posture.to(device)

        mesh_point = mesh_point.to(device)
        target_s = target_s.to(device)
        target_weight_s = target_weight_s.to(device)
        preference_tensor = preference_tensor.to(device)

        y_appearance_pred, y_health_pred, y_posture_pred = model(
            x,
            mesh_point,
            target_s,
            target_weight_s,
            img_metas,
            preference_tensor,
        )

        loss = criterion(
            y_appearance,
            y_appearance_pred,
            y_health,
            y_health_pred,
            y_posture,
            y_posture_pred,
        ).requires_grad_(True)

        optimizer.zero_grad()

        loss.backward()

        # with amp.scale_loss(loss, optimizer) as scaled_loss:
        #     scaled_loss.backward()

        optimizer.step()
        train_losses.update(loss.item(), x.size(0))

    return train_losses.avg


def validate(
    opt,
    model,
    loader,
    criterion,
    global_step=None,
    name=None,
    test_or_valid_flag="test",
):
    model.eval()
    validate_losses = AverageMeter()
    true_score_appearance = []
    pred_score_appearance = []
    true_score_health = []
    pred_score_health = []
    true_score_posture = []
    pred_score_posture = []

    with torch.no_grad():
        for idx, (
            x,
            y_appearance,
            y_health,
            y_posture,
            mesh_point,
            target_s,
            target_weight_s,
            img_metas,
            preference_tensor,
        ) in enumerate(tqdm(loader)):

            x = x.to(device)
            y_appearance = y_appearance.to(device)
            y_health = y_health.to(device)
            y_posture = y_posture.to(device)

            mesh_point = mesh_point.to(device)
            target_s = target_s.to(device)
            target_weight_s = target_weight_s.to(device)
            preference_tensor = preference_tensor.to(device)

            y_appearance_pred, y_health_pred, y_posture_pred = model(
                x,
                mesh_point,
                target_s,
                target_weight_s,
                img_metas,
                preference_tensor,
            )

            for i in y_appearance.data.cpu().numpy().tolist():
                true_score_appearance.append(i)
            for j in y_appearance_pred.data.cpu().numpy().tolist():
                pred_score_appearance.append(j)
            for i in y_health.data.cpu().numpy().tolist():
                true_score_health.append(i)
            for j in y_health_pred.data.cpu().numpy().tolist():
                pred_score_health.append(j)
            for i in y_posture.data.cpu().numpy().tolist():
                true_score_posture.append(i)
            for j in y_posture_pred.data.cpu().numpy().tolist():
                pred_score_posture.append(j)

            loss = criterion(
                y_appearance,
                y_appearance_pred,
                y_health,
                y_health_pred,
                y_posture,
                y_posture_pred,
            ).requires_grad_(True)

            validate_losses.update(loss.item(), x.size(0))

    # calculate lcc srcc acc for 3 scores
    # For appearance
    lcc_mean_appearance, srcc_mean_appearance, acc_appearance = calculate_metrics(
        true_score_appearance, pred_score_appearance
    )

    # For health
    lcc_mean_health, srcc_mean_health, acc_health = calculate_metrics(
        true_score_health, pred_score_health
    )

    # For posture
    lcc_mean_posture, srcc_mean_posture, acc_posture = calculate_metrics(
        true_score_posture, pred_score_posture
    )

    print(
        "{}, acc_appearance: {}, lcc_mean_appearance: {}, srcc_mean_appearance: {}, validate_losses: {}".format(
            test_or_valid_flag,
            acc_appearance,
            lcc_mean_appearance[0],
            srcc_mean_appearance[0],
            validate_losses.avg,
        )
    )
    return (
        validate_losses.avg,
        acc_appearance,
        acc_health,
        acc_posture,
        lcc_mean_appearance,
        srcc_mean_appearance,
        lcc_mean_health,
        srcc_mean_health,
        lcc_mean_posture,
        srcc_mean_posture,
    )


def start_train(opt):
    train_loader, val_loader = create_data_part(opt)

    model = PhysiqueFrame()

    load_pth_path = opt.path_to_model_weight

    state_dict = torch.load(load_pth_path, map_location=map_location)
    model.load_state_dict(state_dict)

    # freeze clip encoder
    for name, param in model.named_parameters():
        if name.startswith("vit"):
            param.requires_grad = False

    import util

    criterion = util.CustomMultiLoss()
    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), eps=1e-05
    )

    srcc_best = 0.0
    vacc_best = 0.0
    for e in range(opt.num_epoch):
        adjust_learning_rate(opt, optimizer, e)
        train_loss = train(
            opt,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            global_step=len(train_loader) * e,
            name=f"{opt.experiment_dir_name}_by_batch",
        )

        (
            val_loss,
            acc_appearance,
            acc_health,
            acc_posture,
            lcc_mean_appearance,
            srcc_mean_appearance,
            lcc_mean_health,
            srcc_mean_health,
            lcc_mean_posture,
            srcc_mean_posture,
        ) = validate(
            opt,
            model=model,
            loader=val_loader,
            criterion=criterion,
            global_step=len(val_loader) * e,
            name=f"{opt.experiment_dir_name}_by_batch",
            test_or_valid_flag="valid",
        )

        if ((srcc_mean_appearance[0] > srcc_best or acc_appearance > vacc_best)) and (
            (acc_appearance > 0.60) and srcc_mean_appearance[0] > 0.5
        ):
            srcc_best = srcc_mean_appearance[0]
            vacc_best = acc_appearance
            model_save_name = "PhysiqueFrame"
            model_name = f"{model_save_name}_vaccA{acc_appearance}_srccA{srcc_mean_appearance[0]}_vlccA{lcc_mean_appearance[0]}_vaccH{acc_health}_srccH{srcc_mean_health[0]}_vlccH{lcc_mean_health[0]}_vaccP{acc_posture}_srccP{srcc_mean_posture[0]}_vlccP{lcc_mean_posture[0]}_{e}.pth"
            torch.save(
                model.state_dict(), os.path.join(opt.experiment_dir_name, model_name)
            )


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    start_train(opt)
