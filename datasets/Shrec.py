import torch

from torch.utils.data.dataset import Dataset

import numpy as np
import os
from pathlib import Path
import pickle

CLASS = ["ONE", "TWO", "THREE", "FOUR", "OK", "MENU",
         "POINTING", "LEFT", "RIGHT", "CIRCLE", "V",
         "CROSS", "GRAB", "PINCH", "TAP", "DENY", "KNOB", "EXPAND", "NO_GESTURE"]

class Shrec(Dataset):
    """NVGesture Dataset class"""
    def __init__(self, configer, path, split = "train", transforms = None, n_frames = 40, idx_crossval=0):
        self.dataset_path = Path(path) / "training_set"
        self.split = split
        self.transforms = transforms
        self.window = n_frames
        self.configer = configer

        stride = configer.get("data", "stride")

        k_cross_validation = configer.get("data", "k_fold")

        sequence = [f"{i+1}.txt" for i in range(len(os.listdir(self.dataset_path / "sequences")))]
        annotations = list(self.dataset_path.glob('annotations/annotations.txt'))[0]

        if k_cross_validation is not None:
            with open("/home/andrea/TransformerBasedGestureRecognition/datasets/cross_val.pkl", "rb") as infile:
                cross_val_idx = pickle.load(infile)
            cross_val_idx = cross_val_idx.reshape(k_cross_validation, -1)

            if split in ["val", "test"]:
                cross_val_idx = cross_val_idx[idx_crossval].reshape(-1)
            else:
                cross_val_idx = np.array([cross_val_idx[i] for i in range(cross_val_idx.shape[0]) if i != idx_crossval]).reshape(-1)

            sequence = np.array(sequence)[cross_val_idx]
        else:
            sequence = np.array(sequence)

        self.data = list()
        self.gt = list()

        for seq in sequence:
            _seq = list()
            with open(self.dataset_path / "sequences" / seq, "rt") as infile:
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

        with open(annotations, "rt") as infile:
            data = infile.readlines()
        if k_cross_validation is not None:
            data = np.array(data)[cross_val_idx]
        else:
            data = np.array(data)

        self.start_end = list()
        for idx, line in enumerate(data):
            line = line.replace("; \n", "").split(";")
            _gt = np.ones((len(self.data[idx]),), dtype=np.int16) * CLASS.index("NO_GESTURE")
            same_video = list()
            for i in range(len(line[1:]) // 3):
                gt = CLASS.index(line[i*3 + 1])
                gt_start = int(line[i*3 + 1 + 1])
                gt_end = int(line[i * 3 + 2 + 1])
                _gt[gt_start:gt_end+1] = gt
                same_video.append([gt_start, gt_end])
            self.start_end.append(same_video)
            self.gt.append(_gt)

        if split == "train":
            _data = list()
            _gt = list()
            for i, seq in enumerate(self.data):
                sliding_start = 0

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
                        _data.append(seq[sliding_start:sliding_start + n_frames])
                        _gt.append(self.gt[i][sliding_start:sliding_start + n_frames])
                    sliding_start += stride
        elif split in ["val", "test"]:
            _data = list()
            _gt = list()
            for i, seq in enumerate(self.data):
                sliding_start = 0

                if self.configer.get("data", "distances"):
                    tmp = list()
                    for frame in seq:
                        mat = list()
                        for idx, joint in enumerate(frame):
                            mat.append(np.sqrt(np.sum((frame[[i for i in range(frame.shape[0]) if i != idx], :3] - joint[:3])**2, axis=-1)))
                        tmp.append(np.concatenate([frame, np.array(mat)], axis=1))
                    seq = np.array(tmp)

                _seq = list()
                _seq_gt = list()
                while True:
                    if sliding_start + n_frames > len(seq):
                        break
                    else:
                        _seq.append(seq[sliding_start:sliding_start + n_frames])
                        _seq_gt.append(self.gt[i][sliding_start:sliding_start + n_frames])
                    sliding_start += stride
                _data.append(_seq)
                _gt.append(_seq_gt)
        if split == "train":
            list_nogesture = list()
            list_gesture = list()
            _gt = np.array(_gt)
            for i, el in enumerate(_gt):
                if len(el[el != CLASS.index("NO_GESTURE")]) > 0:
                    list_gesture.append(i)
                else:
                    list_nogesture.append(i)
            indexes = np.array(list_nogesture)[np.random.permutation(len(list_nogesture))[:len(list_gesture)]]
            indexes = indexes.tolist() + list_gesture
            _data = np.array(_data)[np.array(indexes)].tolist()
            _gt = np.array(_gt)[np.array(indexes)].tolist()
        self.data = _data.copy()
        self.gt = _gt.copy()

        self.len = len(self.data)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        data = np.array(self.data[idx])
        gt = np.array(self.gt[idx])

        data = np.array(data)

        if self.split == "train":
            mean_pos = np.abs(data[..., 4, :3] - data[..., 16, :3]).mean(axis=1)[..., np.newaxis, np.newaxis]
        else:
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

        if self.split in ["val", "test"]:
            return torch.FloatTensor(data), torch.LongTensor(gt), torch.LongTensor(self.start_end[idx])
        else:
            return torch.FloatTensor(data), torch.LongTensor(gt), self.start_end[0]

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