import os
import requests

import torch
import torch.nn as nn
from torch.autograd import Variable
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score
import numpy as np


Gl_z = torch.ones(64, 10)


def download_file(url, local_filename, chunk_size=1024):
    if os.path.exists(local_filename):
        return local_filename
    # 从网络指定处下载
    r = requests.get(url, stream=True)
    with open(local_filename, "wb") as f:
        # Response.iter_content边下载边存硬盘
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
    return local_filename


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EDMLoss(nn.Module):
    def __init__(self):
        super(EDMLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        cdf_target = torch.cumsum(p_target, dim=1)
        cdf_estimate = torch.cumsum(p_estimate, dim=1)

        cdf_diff = cdf_estimate - cdf_target
        samplewise_emd = torch.sqrt(
            torch.mean(torch.pow(torch.abs(cdf_diff), 2))
        )  # train

        return samplewise_emd.mean()

        """
        tensor([[ 1.0752e-01,  1.8369e-01,  2.7151e-01,  2.0681e-01, -7.2487e-02,
         -2.3670e-01, -2.3698e-01, -1.9482e-01, -1.0073e-01,  5.9605e-08],
        
        """


class MSEACCLoss(nn.Module):
    def __init__(self):
        super(MSEACCLoss, self).__init__()

    def forward(self, p_target, p_estimate):
        assert p_target.shape == p_estimate.shape
        B = p_target.shape[0]
        loss = torch.sum(torch.abs(p_estimate - p_target)) / B

        return loss.requires_grad_(True)


class CustomMultiLoss(nn.Module):
    def __init__(self, nb_outputs=3):
        super(CustomMultiLoss, self).__init__()
        self.nb_outputs = nb_outputs
        self.log_vars = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(1).requires_grad_())
                for _ in range(self.nb_outputs)
            ]
        )

    def forward(self, y, y_pred, y_shape, y_shape_pred, y_pose, y_pose_pred):
        ys_true = [y, y_shape, y_pose]
        ys_pred = [y_pred, y_shape_pred, y_pose_pred]
        assert len(ys_true) == self.nb_outputs and len(ys_pred) == self.nb_outputs
        loss = 0
        for y_true, y_pred, log_var in zip(ys_true, ys_pred, self.log_vars):
            precision = torch.exp(-log_var[0])
            loss += torch.sum(precision * (y_true - y_pred) ** 2.0 + log_var[0], -1)
        return torch.mean(loss)


def calculate_metrics(true_scores, pred_scores):
    # calculate LCC
    lcc = pearsonr(np.array(pred_scores).ravel(), np.array(true_scores).ravel())

    # calculate SRCC
    srcc = spearmanr(pred_scores, true_scores)

    # convert to binary labels (0 or 1)
    true_labels = np.where(np.array(true_scores) <= 5, 0, 1)
    pred_labels = np.where(np.array(pred_scores) <= 5, 0, 1)

    # calculate accuracy
    acc = accuracy_score(true_labels, pred_labels)

    return lcc, srcc, acc
