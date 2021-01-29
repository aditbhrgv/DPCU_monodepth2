# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import random
import numpy as np
import copy
from PIL import Image  # using pillow-simd for increased speed

import torch
import torch.utils.data as data
from torchvision import transforms
import yaml, csv
import PIL.Image as pil

euroc_cam0_data_file = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/mav0/cam0/data.csv"
euroc_cam0_sensor_file = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/mav0/cam0/sensor.yaml"
euroc_glob_state_data_file = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/mav0/state_groundtruth_estimate0/data.csv"
euroc_glob_state_sensor_file = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/mav0/state_groundtruth_estimate0/sensor.yaml"
euroc_leica0_sensor_file = "/media/adit/storage/Downloads/EUROC_dataset/MH_02_easy/mav0/leica0/sensor.yaml"

sensor_calib_file = "sensor_calib.yaml"
meta_data_file = "meta.csv"
glob_state_file = "globalState.csv"
camera_intrincs_file = "camera_intrinsics.csv"

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


class MonoPoseDataset(data.Dataset):
    """Superclass for monocular dataloaders

    Args:
        data_path
        seq_names
        height
        width
        frame_idxs
        num_scales
        is_train
        img_ext
    """
    def __init__(self,
                 data_path,
                 seq_names,
                 height,
                 width,
                 frame_idxs,
                 num_scales,
                 is_train=False,
                 img_ext='.png'):
        super(MonoPoseDataset, self).__init__()

        self.data_path = data_path
        self.seq_names = seq_names
        self.height = height
        self.width = width
        self.num_scales = num_scales
        self.interp = Image.ANTIALIAS

        self.frame_idxs = frame_idxs

        self.is_train = is_train
        self.img_ext = img_ext

        self.loader = pil_loader
        self.to_tensor = transforms.ToTensor()

        # We need to specify augmentations differently in newer versions of torchvision.
        # We first try the newer tuple version; if this fails we fall back to scalars
        try:
            self.brightness = (0.8, 1.2)
            self.contrast = (0.8, 1.2)
            self.saturation = (0.8, 1.2)
            self.hue = (-0.1, 0.1)
            transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        except TypeError:
            self.brightness = 0.2
            self.contrast = 0.2
            self.saturation = 0.2
            self.hue = 0.1

        self.resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.resize[i] = transforms.Resize((self.height // s, self.width // s),
                                               interpolation=self.interp)

        self.load_depth = self.check_depth()

        self.dataset_dict = {}
        for sequence in self.seq_names:
            self.dataset_dict[sequence] = {}
            self.dataset_dict[sequence]["filenames"] = np.array(sorted(os.listdir(os.path.join(self.data_path, sequence, 'camera'))))
            image = pil_loader(os.path.join(self.data_path, sequence, 'camera', self.dataset_dict[sequence]["filenames"][0]))
            self.full_res_shape = image.size

            # Read transformation matrices
            with open(os.path.join(self.data_path, sequence, sensor_calib_file), 'r') as fp:
                yaml_data_loaded = yaml.safe_load(fp)
                self.dataset_dict[sequence]["T_body2cam"]  = yaml_data_loaded['T_BODY_CAMERA']
                self.dataset_dict[sequence]["T_body2imu"]  = yaml_data_loaded['T_BODY_IMU']

            #Read the camera intrincs file
            with open(os.path.join(self.data_path, sequence, camera_intrincs_file)) as csvfile:
                cam_intrincs = csv.reader(csvfile, delimiter=' ')
                values = np.array(list(cam_intrincs))
                self.K  = np.identity(4).astype(np.float32)
                self.K[:3,:3] =  np.reshape(values, (3,3))
                self.K[0, 0] =  self.K[0, 0] / self.full_res_shape[1] #height
                self.K[0, 2] = self.K[0, 2] / self.full_res_shape[1]  # height
                self.K[1, 1] = self.K[1, 1] / self.full_res_shape[0]  # width
                self.K[1, 2] = self.K[1, 2] / self.full_res_shape[0]  # width


            # Extract glob state data from csv file
            globvalues = []
            with open(os.path.join(self.data_path, sequence, glob_state_file)) as csvfile:
                readglobstate = csv.reader(csvfile, delimiter=' ')
                gval = list(readglobstate)
                for i in range(len(gval)):
                    globvalues.append(gval[i][2:])

            self.dataset_dict[sequence]["globalstate"] = np.array(globvalues)
            self.dataset_dict[sequence]["filenames"] = self.dataset_dict[sequence]["filenames"][1:-1]
            self.dataset_dict[sequence]["globalstate"] = self.dataset_dict[sequence]["globalstate"][1:-1]
            self.dataset_dict[sequence]["num_samples"] = len(self.dataset_dict[sequence]["filenames"])

    def preprocess(self, inputs, color_aug):
        """Resize colour images to the required scales and augment if required

        We create the color_aug object in advance and apply the same augmentation to all
        images in this item. This ensures that all images input to the pose network receive the
        same augmentation.
        """
        for k in list(inputs):
            frame = inputs[k]
            if "color" in k:
                n, im, i = k
                for i in range(self.num_scales):
                    inputs[(n, im, i)] = self.resize[i](inputs[(n, im, i - 1)])

        for k in list(inputs):
            f = inputs[k]
            if "color" in k:
                n, im, i = k
                inputs[(n, im, i)] = self.to_tensor(f)
                inputs[(n + "_aug", im, i)] = self.to_tensor(color_aug(f))

    def __len__(self):
        num_samples = 0
        for seq in self.dataset_dict:
            num_samples += self.dataset_dict[seq]["num_samples"]
        return num_samples

    def __getitem__(self, index):
        """Returns a single training item from the dataset as a dictionary.

        Values correspond to torch tensors.
        Keys in the dictionary are either strings or tuples:

            ("color", <frame_id>, <scale>)          for raw colour images,
            ("color_aug", <frame_id>, <scale>)      for augmented colour images,
            ("K", scale) or ("inv_K", scale)        for camera intrinsics,
            "stereo_T"                              for camera extrinsics, and
            "depth_gt"                              for ground truth depth maps.

        <frame_id> is either:
            an integer (e.g. 0, -1, or 1) representing the temporal step relative to 'index',
        or
            "s" for the opposite image in the stereo pair.

        <scale> is an integer representing the scale of the image relative to the fullsize image:
            -1      images at native resolution as loaded from disk
            0       images resized to (self.width,      self.height     )
            1       images resized to (self.width // 2, self.height // 2)
            2       images resized to (self.width // 4, self.height // 4)
            3       images resized to (self.width // 8, self.height // 8)
        """
        inputs = {}

        do_color_aug = self.is_train and random.random() > 0.5
        do_flip = self.is_train and random.random() > 0.5

        #line = self.seq_names[index].split()
        #folder = line[0]

        ind = index #20, #70, 600
        # for key, seq in self.dataset_dict.items():
        #     ind -= seq["num_samples"]
        #     if(ind < 0):
        #         folder = key
        #         frame_index = ind + seq["num_samples"]
        #         break
        #     else:
        #         ind -= 1
        cnt = 0
        prev_cnt = 0
        for key, seq in self.dataset_dict.items():
            cnt += seq["num_samples"] #50, #100, #500  50+100 = 150, 150+500 = 650
            quotient = ind/cnt #70/50 = 1. , 70/150 = 0. , 600/50  , 600/650 <0
            if(quotient < 1):
                folder = key
                frame_index = ind - prev_cnt #70 - 50 = 20, 600-150 = 450
                break
            prev_cnt = cnt  #50, 150

        frame_name = self.dataset_dict[folder]["filenames"][frame_index]
        frameID, ext = frame_name.split('.')
        frameID = int(frameID)
        for i in self.frame_idxs:
            inputs[("color", i, -1)] = self.get_color(folder, frameID + i, do_flip)

        # adjusting intrinsics to match each scale in the pyramid
        for scale in range(self.num_scales):
            K = self.K.copy()

            K[0, :] *= self.width // (2 ** scale)
            K[1, :] *= self.height // (2 ** scale)

            inv_K = np.linalg.pinv(K)

            inputs[("K", scale)] = torch.from_numpy(K)
            inputs[("inv_K", scale)] = torch.from_numpy(inv_K)

        if do_color_aug:
            color_aug = transforms.ColorJitter.get_params(
                self.brightness, self.contrast, self.saturation, self.hue)
        else:
            color_aug = (lambda x: x)

        self.preprocess(inputs, color_aug)

        for i in self.frame_idxs:
            del inputs[("color", i, -1)]
            del inputs[("color_aug", i, -1)]

        if self.load_depth:
            depth_gt = self.get_depth(folder, frame_index, do_flip)
            inputs["depth_gt"] = np.expand_dims(depth_gt, 0)
            inputs["depth_gt"] = torch.from_numpy(inputs["depth_gt"].astype(np.float32))

        return inputs

    def get_image_path(self, folder, frame_index):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "camera", f_str)
        return image_path

    def get_color(self, folder, frame_index, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color

    def check_depth(self):
        return False

    def get_depth(self, folder, frame_index, side, do_flip):
        raise NotImplementedError
