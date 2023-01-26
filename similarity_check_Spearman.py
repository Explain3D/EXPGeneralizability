#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:39:33 2022

@author: tan
"""

import numpy as np
import time
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import spearman_corrcoef

from data_utils import MNIST_EXP
import os
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt
import random
from math import comb

num_cls = 10

def intra_class_sc(exp, label, n_sample=200):
    intra_score = np.zeros([num_cls, n_sample])
    for c in range(num_cls):
        print("######################")
        print("Class ", c)
        print("######################")
        cls_idx = np.argwhere(label == c)
        intra_ins = exp[cls_idx]
        num_ins = intra_ins.shape[0]
        intra_ins = torch.from_numpy(intra_ins.reshape(num_ins, -1))
        for s in range(n_sample):
            pairs = random.sample(range(num_ins),2)
            cur_sc = spearman_corrcoef(intra_ins[pairs[0]], intra_ins[pairs[1]])
            intra_score[c, s] = cur_sc
    
    return intra_score


def inter_class_sc(exp, label, n_sample=200):
    inter_score = np.zeros([comb(num_cls,2), n_sample])
    matrix_column_indice = 0
    for c1 in range(num_cls):
        for c2 in range(c1+1, num_cls):
            print("######################")
            print("Class ", c1, " and Class ", c2)
            print("######################")
            c1_idx = np.argwhere(label == c1)
            c2_idx = np.argwhere(label == c2)
            inter_ins_c1 = exp[c1_idx]
            inter_ins_c2 = exp[c2_idx]
            num_ins_c1 = inter_ins_c1.shape[0]
            num_ins_c2 = inter_ins_c2.shape[0]
            inter_ins_c1 = torch.from_numpy(inter_ins_c1.reshape(num_ins_c1, -1))
            inter_ins_c2 = torch.from_numpy(inter_ins_c2.reshape(num_ins_c2, -1))
            for s in range(n_sample):
                sample_c1 = random.sample(range(num_ins_c1),1)
                sample_c2 = random.sample(range(num_ins_c2),1)
                cur_sc = spearman_corrcoef(inter_ins_c1[sample_c1], inter_ins_c2[sample_c2])
                inter_score[matrix_column_indice, s] = cur_sc
            matrix_column_indice += 1
    return inter_score

EM = 'Lime30'

if EM == 'ins':
    gen_exp = np.load('gen_exps/' + EM + '/test_LRP.npy')
    gen_label = np.load('gen_exps/' + EM + '/test_label.npy')

else:
    gen_exp = np.load('gen_exps/' + EM + '/generated_exp.npy')
    gen_label = np.load('gen_exps/' + EM + '/generated_label.npy')

n_sample = 500

intra_sc = intra_class_sc(gen_exp, gen_label, n_sample=n_sample)
inter_sc = inter_class_sc(gen_exp, gen_label, n_sample=n_sample)

print(np.mean(inter_sc))
print(np.mean(intra_sc))

np.save('npys/spearman_LIME/inter_' + EM + '.npy' ,inter_sc)
np.save('npys/spearman_LIME/intra_' + EM + '.npy' ,intra_sc)