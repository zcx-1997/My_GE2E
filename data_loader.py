# -*- coding: utf-8 -*-
"""
=================================================
@Project -> File    ：My_GE2E_Pytorch -> data_loader
@IDE                ：PyCharm
@Author             ：zcx
@Date               ：2021/8/17 上午11:17
@Description        ：
                    _ooOoo_    
                   o8888888o    
                   88" . "88    
                   (| -_- |)    
                    O\ = /O    
                ____/`---'\____    
                 .' \\| |// `.    
               / \\||| : |||// \    
             / _||||| -:- |||||- \    
               | | \\\ - /// | |    
             | \_| ''\---/'' | |    
              \ .-\__ `-` ___/-. /    
           ___`. .' /--.--\ `. . __    
        ."" '< `.___\_<|>_/___.' >'"".    
       | | : `- \`.;`\ _ /`;.`/ - ` : | |    
         \ \ `-. \_ __\ /__ _/ .-` / /    
 ======`-.____`-.___\_____/___.-`____.-'======    
                    `=---='    
 .............................................    
              佛祖保佑             永无BUG
==================================================
"""
import os
import glob
import random

import numpy as np
import torch

import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


class Vox1Dataset(Dataset):
    def __init__(self, train=True, num_frames=160):
        self.train = train
        self.num_frames = num_frames
        if self.train:
            self.path = r'data/train'
        else:
            self.path = r'data/test'
        self.data, self.labels = self._load_data(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = torch.tensor(data, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.int)
        return data, label

    def _load_data(self, data_path):
        data_all = []
        label_all = []
        if self.train:
            spk_list = os.listdir(data_path)
        else:
            spk_list = os.listdir(data_path)
        for spk_name in spk_list:
            spk_id = torch.tensor(int(spk_name[:-4]))
            spk_id = F.one_hot(spk_id)

            spk_uttr = np.load(os.path.join(data_path, spk_name),
                               allow_pickle=True)
            for i in range(len(spk_uttr)):
                data = spk_uttr[i][:self.num_frames]
                data_all.append(data)
                label_all.append(spk_id)
        return data_all, label_all


class Vox1Dataset_train_100(Dataset):
    def __init__(self, train=True, num_frames=160):
        self.train = train
        self.num_frames = num_frames
        if self.train:
            self.path = r'data/train_100'
        else:
            self.path = r'data/test'
        self.data, self.labels = self._load_data(self.path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx]
        data = torch.tensor(data, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.int)
        return data, label

    def _load_data(self, data_path):
        data_all = []
        label_all = []
        if self.train:
            spk_list = os.listdir(data_path)
        else:
            spk_list = os.listdir(data_path)
        for spk_name in spk_list:
            spk_id = spk_name[3:].lstrip('0')
            spk_id = torch.tensor(int(spk_id[:-4])) - 1
            # spk_id = F.one_hot(spk_id)

            spk_uttr = np.load(os.path.join(data_path, spk_name),
                               allow_pickle=True)
            for i in range(len(spk_uttr)):
                data = spk_uttr[i][:self.num_frames]
                data_all.append(data)
                label_all.append(spk_id)
        return data_all, label_all


class TIMITDataset(Dataset):

    def __init__(self, train=True, num_frames=160, shuffle=True):

        self.train = train
        self.num_frames = num_frames
        self.shuffle = shuffle
        if self.train:
            self.path = r'TIMIT/train'
        else:
            self.path = r'TIMIT/test'
        self.spks_list = os.listdir(self.path)

    def __getitem__(self, idx):
        self.spk_name = self.spks_list[idx]
        np_spk = np.load(os.path.join(self.path, self.spk_name))

        if self.shuffle:
            indices = random.sample(range(0, 10), 10)
            utters = np_spk[indices]
        else:
            utters = np_spk
        utters = utters[:, :self.num_frames]
        utters = torch.tensor(utters, dtype=torch.float32)
        return utters

    def __len__(self):
        return len(self.spks_list)


if __name__ == '__main__':
    # print("================== train dataset ======================")
    # train = Vox1Dataset(train=True)
    # print(len(train))
    # x, y = train[0]
    # print(x.shape)
    # print(x.dtype)
    # print(y)
    #
    # train_loader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
    # print(len(train_loader))
    # x, y = next(iter(train_loader))
    # print(x.shape)
    # print(y)
    #
    # print("================== test dataset ======================")
    # test = Vox1Dataset(train=False)
    # print(len(test))
    # x, y = test[0]
    # print(x.shape)
    # print(x.dtype)
    # print(y)
    # test_loader = DataLoader(test, batch_size=4, shuffle=True, drop_last=True)
    # print(len(test_loader))
    # x, y = next(iter(test_loader))
    # print(x.shape)
    # print(y)

    # print("================== train_100 dataset ======================")
    # train = Vox1Dataset_train_100()
    # print(len(train))
    # x, y = train[0]
    # print(x.shape)
    # print(x.dtype)
    # print(y)
    #
    # train_loader = DataLoader(train, batch_size=4, shuffle=True, drop_last=True)
    # print(len(train_loader))
    # x, y = next(iter(train_loader))
    # print(x.shape)
    # print(y)

    print("================== TIMIT dataset ======================")
    train_db = TIMITDataset()
    print(len(train_db))
    x = train_db[0]
    print(x.shape)
    print(x.dtype)

    test_db = TIMITDataset(train=False)
    print(len(test_db))
    x = test_db[0]
    print(x.shape)
    print(x.dtype)
