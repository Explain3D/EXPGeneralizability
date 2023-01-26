#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 15:23:47 2022

@author: tan
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

#MNIST_exp = np.load('train_exp/train_exp.npy')

output_exp = np.load('train_exp/test_IG.npy')
output_data = np.load('train_exp/test_ins.npy')
output_gen = np.load('gen_exps/IG/generated_exp.npy')

selected_ins = 51

# =============================================================================
# tensor_exp = torch.from_numpy(MNIST_exp)
# 
# for e in range(tensor_exp.shape[0]):
#     cur_min = torch.min(tensor_exp[e])
#     tensor_exp[e] = tensor_exp[e] - cur_min
#     cur_max = torch.max(tensor_exp[e])
#     tensor_exp[e] = tensor_exp[e] / cur_max
# 
# MNIST_exp = tensor_exp.numpy()
# 
# print(MNIST_exp.shape)
# print(np.min(MNIST_exp))
# print(np.max(MNIST_exp))
# 
# cur_ins = MNIST_exp[7].squeeze()
# =============================================================================

cur_data = output_data[selected_ins].squeeze()
cur_output = output_exp[selected_ins].squeeze()
gen_exp = output_gen[selected_ins].squeeze()


fig = plt.figure(1)
plt.imshow(cur_data, cmap='gray')
plt.show()

fig = plt.figure(2)
plt.imshow(cur_output, cmap='gray')
plt.show()

fig = plt.figure(3)
plt.imshow(gen_exp, cmap='gray')
plt.show()