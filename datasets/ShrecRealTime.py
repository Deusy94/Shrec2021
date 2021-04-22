import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import os
from pathlib import Path
import pickle

CLASS = ["ONE", "TWO", "THREE", "FOUR", "OK", "MENU",
         "POINTING", "LEFT", "RIGHT", "CIRCLE", "V",
         "CROSS", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND", "NO_GESTURE"]

class ShrecRealTime(Dataset):
    """NVGesture Dataset class"""
    def __init__(self, configer, path, split = "test", transforms = None, n_frames = 10):
        self.dataset_path = Path(path)
        self.split = split
        if split != "test":
            raise ValueError("Test in real time require test split")
        self.transforms = transforms
        self.window = n_frames
        self.configer = configer

        stride = configer.get("data", "stride")

        sequence = [x for x in self.dataset_path.glob('*') if x.is_file()]

        self.data = list()

        for seq in sequence:
            _seq = list()
            with open(seq, "rt") as infile:
                data = infile.readlines()
            prev_prev = self.extract_shrec_leapmotion(data[0])
            prev = self.extract_shrec_leapmotion(data[1])
            for i, line in enumerate(data[2:]):
                if len(line) <= 1:
                    continue
                extracted_data = self.extract_shrec_leapmotion(line)
                velocity = extracted_data[:, :3] - prev[:, :3]
                velocity_prev = prev[:, :3] - prev_prev[:, :3]
                acceleration = velocity - velocity_prev
                _seq.append(np.concatenate([extracted_data, velocity, acceleration], axis=1))
                prev_prev = prev.copy()
                prev = extracted_data.copy()
            self.data.append(np.array(_seq))

        _data = list()
        for i, seq in enumerate(self.data):
            sliding_start = 0
            _seq = list()
            _seq_gt = list()

            if self.configer.get("data", "distances"):
                tmp = list()
                for frame in seq:
                    mat = list()
                    for idx, joint in enumerate(frame):
                        mat.append(np.sqrt(
                            np.sum((frame[[i for i in range(frame.shape[0]) if i != idx], :3] - joint[:3]) ** 2,
                                   axis=-1)))
                    tmp.append(np.concatenate([frame, np.array(mat)], axis=1))
                seq = np.array(tmp)

            while True:
                if sliding_start + n_frames > len(seq):
                    break
                else:
                    _seq.append(seq[sliding_start:sliding_start + n_frames])
                sliding_start += stride
            _data.append(_seq)

        self.data = _data.copy()
        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = np.array(self.data[idx])

        mean_pos = np.abs(data[..., 4, :3] - data[..., 16, :3]).mean(axis=1)[..., np.newaxis, np.newaxis, :]

        # normalize pos
        if self.configer.get("data", "norm_hand"):
            data[..., :3] = data[..., :3] / mean_pos
        data[..., :3] = (data[..., :3] - data[..., :3].mean()) / data[..., :3].std()
        # normalize quat
        data[..., 3:7] = (data[..., 3:7] - data[..., 3:7].mean()) / data[..., 3:7].std()
        # normalize velocity
        data[..., 7:10] = (data[..., 7:10] - data[..., 7:10].mean()) / data[..., 7:10].std()
        #normalize acceleration
        data[..., 10:13] = (data[..., 10:13] - data[..., 10:13].mean()) / data[..., 10:13].std()

        return torch.FloatTensor(data)

    def extract_shrec_leapmotion(self, data):
        data.replace("\n", "")
        data = data.split(";")
        # pos(xyz) -> quat(xyzw), 21 data in total (doubled up)
        # palmpos(x;y;z); palmquat(x,y,z,w);
        # thumbApos(x;y;z); thumbAquat(x;y;z;w);
        # thumBposy;z); thumbBquat(x;y;z;w);
        # thumbEndpos(x;y;z); thumbEndquat(x;y;z;w);
        # indexApos(x;y;z); indexAquat(x;y;z;w);
        # indexBpos(x;y;z); indexBquat(x;y;z;w);
        # indexCpos(x;y;z); indexCquat(x;y;z;w);
        # indexEndpos(x;y;z); indexEndquat(x;y;z;w);
        # middleApos(x;y;z); middleAquat(x;y;z;w);
        data = np.array(data) # middleBpos(x;y;z); middleBquat(x;y;z;w);
        # middleCpos(x;y;z); middleCquat(x;y;z;w);
        # middleEndpos(x;y;z); middleEndquat(x;y;z;w); 11
        # ringApos(x;y;z); ringAquat(x;y;z;w);
        # ringBpos(x;y;z); ringBquat(x;y;z;w);
        # ringCpos(x;y;z); ringCquat(x;y;z;w);
        # ringEndpos(x;y;z); ringEndquat(x;y;z;w); 15
        # pinkyApos(x;y;z); pinkyAquat(x;y;z;w);
        # pinkyBpos(x;y;z); pinkyBquat(x;y;z;w);
        # pinkyCpos(x;y;z); pinkyCquat(x;y;z;w);
        # pinkyEndpos(x;y;z); pinkyEndquat(x;y;z;w)
        # _pos = list()
        # _quat = list()
        _data = list()
        for i in range(20):
            # _pos.append([data[0 + i*6], data[1 + i*6], data[2 + i*6]])                    # pos
            # _quat.append([data[3 + i*6], data[4 + i*6], data[5 + i*6], data[6 + i*6]])    # quat
            _data.append([data[0 + i*7], data[1 + i*7], data[2 + i*7], data[3 + i*7], data[4 + i*7], data[5 + i*7], data[6 + i*7]])
        return np.array(_data, dtype=np.float64)