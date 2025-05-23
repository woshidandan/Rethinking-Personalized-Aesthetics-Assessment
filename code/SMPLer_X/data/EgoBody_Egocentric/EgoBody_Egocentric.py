import os
import os.path as osp
import numpy as np
import torch
import cv2
import json
import copy
from pycocotools.coco import COCO
from config import cfg
from utils_smpler_x.human_models import smpl_x
from utils_smpler_x.preprocessing import load_img, process_bbox, augmentation, process_db_coord, process_human_model_output, \
    get_fitting_error_3D
from utils_smpler_x.transforms import world2cam, cam2pixel, rigid_align
from humandata import HumanDataset


class EgoBody_Egocentric(HumanDataset):
    def __init__(self, transform, data_split):
        super(EgoBody_Egocentric, self).__init__(transform, data_split)
        if getattr(cfg, 'eval_on_train', False):
            self.data_split = 'eval_train'
            print("Evaluate on train set.")

        if 'train' in self.data_split:
            filename = getattr(cfg, 'filename', 'egobody_egocentric_train_230425_065_fix_betas.npz')
        else:
            filename = getattr(cfg, 'filename', 'egobody_egocentric_test_230425_043_fix_betas.npz')

        self.use_betas_neutral = getattr(cfg, 'egobody_fix_betas', False)

        self.img_dir = osp.join(cfg.data_dir, 'EgoBody')
        self.annot_path = osp.join(cfg.data_dir, 'preprocessed_datasets', filename)
        self.annot_path_cache = osp.join(cfg.data_dir, 'cache', filename)
        self.use_cache = getattr(cfg, 'use_cache', False)
        self.img_shape = (1080, 1920)  # (h, w)
        self.cam_param = {}

        # check image shape
        img_path = osp.join(self.img_dir, np.load(self.annot_path)['image_path'][0])
        img_shape = cv2.imread(img_path).shape[:2]
        assert self.img_shape == img_shape, 'image shape is incorrect: {} vs {}'.format(self.img_shape, img_shape)

        # load data or cache
        if self.use_cache and osp.isfile(self.annot_path_cache):
            print(f'[{self.__class__.__name__}] loading cache from {self.annot_path_cache}')
            self.datalist = self.load_cache(self.annot_path_cache)
        else:
            if self.use_cache:
                print(f'[{self.__class__.__name__}] Cache not found, generating cache...')
            self.datalist = self.load_data(
                train_sample_interval=getattr(cfg, f'{self.__class__.__name__}_train_sample_interval', 1))
            if self.use_cache:
                self.save_cache(self.annot_path_cache, self.datalist)