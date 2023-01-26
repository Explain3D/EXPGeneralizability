#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:37:22 2022

@author: tan
"""

import numpy as np
import time
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef, structural_similarity_index_measure

from captum.attr import IntegratedGradients,NoiseTunnel

from data_utils import MNIST_EXP
import os
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt
from models import Net

sys.path.append(os.getcwd())
EM = 'IG'

print("EXP method: ", EM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classifier = Net().to(device)

if torch.cuda.is_available() == True:
    checkpoint = torch.load("log/net.pt")
else:
    checkpoint = torch.load("log/net.pt",map_location=torch.device('cpu'))

classifier.load_state_dict(checkpoint)
classifier = classifier.eval()

# load dataset from numpy array and divide 90%-10% randomly for train and test sets
#train_loader, test_loader = GetDataLoaders(npArray=pc_array, batch_size=batch_size)
DATA_PATH = 'train_exp/'
TRAIN_DATASET = MNIST_EXP(split='train', data_dir='train_exp/', EM=EM)
TEST_DATASET = MNIST_EXP(split='test', data_dir='train_exp/', EM=EM)

ins = 0

data = TEST_DATASET[ins][0]
label = TEST_DATASET[ins][2]
data = torch.from_numpy(data).unsqueeze(0)
label = int(label)


ig = IntegratedGradients(classifier)
pure_ig_mask = ig.attribute(data, target=label, n_steps=50)

plt.imshow(pure_ig_mask.squeeze().detach().numpy())



ig = IntegratedGradients(classifier)
nt = NoiseTunnel(ig)
smooth_ig_mask = nt.attribute(data, nt_type='smoothgrad',nt_samples=10, target=label)

plt.figure()
plt.imshow(smooth_ig_mask.squeeze().detach().numpy())