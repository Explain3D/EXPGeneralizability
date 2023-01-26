#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 13:54:09 2022

@author: tan
"""
from torch.utils.data import Dataset
import numpy as np
import os

class MNIST_EXP(Dataset):
    def __init__(self, split='train', data_dir='train_exp/', EM=None):
        self.ins = np.load(data_dir + split + '_ins.npy')
        self.exp = np.load(data_dir + split + '_' + EM + '.npy')
        self.label = np.load(data_dir + split + '_label.npy')
        print("EXP path: ", data_dir + split + '_' + EM + '.npy')
        
    def __len__(self):
        return self.ins.shape[0]

    def __getitem__(self, idx):
        ins = self.ins[idx]
        exp = self.exp[idx]
        label = self.label[idx]
        return ins, exp, label




if __name__ == '__main__':
    import torch

    data = MNIST_EXP(split='train', data_dir='train_exp/', EM='IG')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=False)
    for ins, exp, label in DataLoader:
        print("######")
        print(ins.shape)
        print(exp.shape)
        print(label.shape)
