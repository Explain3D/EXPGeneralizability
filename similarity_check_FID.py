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
from scipy import linalg


class MNIST_EXP_AE(torch.nn.Module):
    def __init__(self):
        super(MNIST_EXP_AE, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 3, stride=1, padding=1),  # 
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1),
            torch.nn.Conv2d(64, 16, 3, stride=1, padding=1),  # b, 8, 3, 3
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(16, 64, 3, stride=1, padding=1),  # b, 16, 10, 10
            torch.nn.ReLU(True),
            torch.nn.Upsample(scale_factor=1, mode='nearest'),
            torch.nn.Conv2d(64, 1, 3, stride=1, padding=2),  # b, 8, 3, 3
            #torch.nn.Sigmoid()
        )

    def forward(self, x):
        coded = self.encoder(x)
        decoded = self.decoder(coded)
        return coded, decoded

num_cls = 10

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def intra_class_sc(model, exp, label, n_sample=200):
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
            
            input_1 = intra_ins[pairs[0]].reshape(28,-1).unsqueeze(0).unsqueeze(0)
            input_2 = intra_ins[pairs[1]].reshape(28,-1).unsqueeze(0).unsqueeze(0)
            latent_1,_ = model(input_1)
            latent_2,_ = model(input_2)
            latent_1 = latent_1.view(-1,1).detach().numpy()
            latent_2 = latent_2.view(-1,1).detach().numpy()
            mu1_g = np.mean(latent_1, axis=1)
            mu2_g = np.mean(latent_2, axis=1)
            sigma1_g = np.cov(latent_1, rowvar=False)
            sigma2_g = np.cov(latent_2, rowvar=False)
            cur_FID = calculate_frechet_distance(mu1_g,sigma1_g,mu2_g,sigma2_g)
            intra_score[c,s] = cur_FID
    return intra_score


def inter_class_sc(model, exp, label, n_sample=200):
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
                
                input_1 = inter_ins_c1[sample_c1].reshape(28,-1).unsqueeze(0).unsqueeze(0)
                input_2 = inter_ins_c2[sample_c2].reshape(28,-1).unsqueeze(0).unsqueeze(0)
                latent_1,_ = model(input_1)
                latent_2,_ = model(input_2)
                latent_1 = latent_1.view(-1,1).detach().numpy()
                latent_2 = latent_2.view(-1,1).detach().numpy()
                mu1_g = np.mean(latent_1, axis=1)
                mu2_g = np.mean(latent_2, axis=1)
                sigma1_g = np.cov(latent_1, rowvar=False)
                sigma2_g = np.cov(latent_2, rowvar=False)
                cur_FID = calculate_frechet_distance(mu1_g,sigma1_g,mu2_g,sigma2_g)
                inter_score[matrix_column_indice, s] = cur_FID
            matrix_column_indice += 1
    return inter_score

if torch.cuda.is_available() == True:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    
EM = 'Lime30'
model = torch.load('log/ins_AE.pth',map_location=device)

if EM == 'ins':
    gen_exp = np.load('gen_exps/' + EM + '/test_LRP.npy')
    gen_label = np.load('gen_exps/' + EM + '/test_label.npy')

else:
    gen_exp = np.load('gen_exps/' + EM + '/generated_exp.npy')
    gen_label = np.load('gen_exps/' + EM + '/generated_label.npy')


intra_sc = intra_class_sc(model, gen_exp, gen_label)
inter_sc = inter_class_sc(model, gen_exp, gen_label)

print(np.mean(intra_sc))
print(np.mean(inter_sc))
np.save('npys/FID_LIME/inter_' + EM + '.npy' ,inter_sc)
np.save('npys/FID_LIME/intra_' + EM + '.npy' ,intra_sc)