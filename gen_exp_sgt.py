#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:25:55 2021

@author: tan
"""

import torch
import os
import importlib
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from models import Net
from torchvision import datasets, transforms
from skimage.segmentation import slic


from captum.attr import Lime, KernelShap

def detach(tensor):
    if torch.cuda.is_available() == True:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()
    

# =============================================================================
# def single_output_forward(out_ind):
#     def forward(x):
#         yhat = classifier(x)
#         return yhat[out_ind]
#     return forward
# =============================================================================


# =============================================================================
# def IG_purify_top_k(data, model, gt_cls, ori_pred, n_csder=3, IG_step=25):
#     IG = IntegratedGradients(model)
#     gt_cls = int(gt_cls)
#     gt_pos_mask = IG.attribute(data, target=gt_cls, n_steps=IG_step)
#     gt_pos_mask = np.sum(detach(gt_pos_mask).squeeze(), axis=0)
#     
#     sorted_pred = torch.argsort(ori_pred[ori_pred!= gt_cls], descending=True)
#     sorted_pred_wo_gt = sorted_pred[sorted_pred!= gt_cls]
#     tar_cls = sorted_pred_wo_gt[:n_csder]
#     oth_pos_pool = []
#     for i in range(n_csder):
#         cur_label = int(tar_cls[i].detach().cpu().numpy())
#         cur_mask = IG.attribute(data, target=cur_label, n_steps=IG_step)
#         cur_mask = np.sum(detach(cur_mask).squeeze(), axis=0)
#         cur_residual_mask = gt_pos_mask.copy() - cur_mask
#         cur_pos_idx = np.argwhere(cur_residual_mask > 0).squeeze(-1)
#         set_cur_pos_idx = set(cur_pos_idx)
#         if cur_pos_idx.shape[0] > 0:
#             oth_pos_pool.append(set_cur_pos_idx)
#     
#     
#     idt_pos_idx = list(set.intersection(*oth_pos_pool))
#     new_ipt_tensor = data[:,:,idt_pos_idx]
#     return new_ipt_tensor
# =============================================================================

def IG_purify_top_k(data, model, gt_cls, ori_pred, n_csder=3, IG_step=25):
    explain = Lime(model)
    gt_cls = int(gt_cls)
    temp_seg = data.clone().squeeze()
    segments = slic(temp_seg, n_segments=50, sigma=5,start_label=1)-1
    segments = torch.from_numpy(segments)
    gt_pos_mask = explain.attribute(data, target=gt_cls, n_samples=30, feature_mask=segments)
# =============================================================================
#     #gt_pos_mask = np.sum(detach(gt_pos_mask).squeeze(), axis=0)
#     sorted_pred = torch.argsort(ori_pred[ori_pred!= gt_cls], descending=True)
#     sorted_pred_wo_gt = sorted_pred[sorted_pred!= gt_cls]
#     tar_cls = sorted_pred_wo_gt[:n_csder]
#     res_mask = np.zeros_like(gt_pos_mask)
#     for i in range(n_csder):
#         cur_label = int(tar_cls[i].detach().cpu().numpy())
#         cur_mask = IG.attribute(data, target=cur_label, n_steps=IG_step)
#         cur_mask = np.sum(detach(cur_mask).squeeze(), axis=0)
#         cur_residual_mask = gt_pos_mask.copy() - cur_mask
#         res_mask += cur_residual_mask
# # =============================================================================
# #         cur_pos_idx = np.argwhere(cur_residual_mask > 0).squeeze(-1)
# #         set_cur_pos_idx = set(cur_pos_idx)
# #         if cur_pos_idx.shape[0] > 0:
# #             oth_pos_pool.append(set_cur_pos_idx)
# #     
# #     
# #     idt_pos_idx = list(set.intersection(*oth_pos_pool))
# #     new_ipt_tensor = data[:,:,idt_pos_idx]
# # =============================================================================
# =============================================================================
    return gt_pos_mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_class = 10
IG_step = 50
cur_split = 'test'
EM = 'Lime30'


print("Explainability Method: ", EM)
print("Split: ", cur_split)

classifier = Net().to(device)

if torch.cuda.is_available() == True:
    checkpoint = torch.load("log/net.pt")           
else:
    checkpoint = torch.load("log/net.pt",map_location=torch.device('cpu'))

classifier.load_state_dict(checkpoint)
classifier = classifier.eval()

batch_size = 256
test_batch_size = 256
train_kwargs = {'batch_size': batch_size}
test_kwargs = {'batch_size': test_batch_size}
transform=transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
       ])

dataset_train = datasets.MNIST('./data', train=True, download=True,
                   transform=transform)
dataset_test = datasets.MNIST('./data', train=False,
                   transform=transform)
train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)
test_loader = torch.utils.data.DataLoader(dataset_test, **test_kwargs)

print(dataset_train[0][0].mean())

acc_wto_pur = 0
acc_wt_pur = 0

classifier = classifier.to(device)

if cur_split == 'train':
    DATASET = dataset_train
elif cur_split == 'test':
    DATASET = dataset_test

print(DATASET.__len__())

for ins in range(DATASET.__len__()):
    
# =============================================================================
#     if ins >1:
#         break
# =============================================================================
    
    print("Processing number ", ins, "instances...")
    input_tensor = DATASET[ins][0]
    ori_cls = DATASET[ins][1]
    
    input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)
    
    
    pred = classifier(input_tensor)
    
# =============================================================================
#     model = single_output_forward(0)
# =============================================================================

    gt_mask = IG_purify_top_k(input_tensor, classifier, ori_cls, pred, n_csder=2)
    #gt_mask = np.expand_dims(gt_mask, 0)

    pred_feat = detach(pred)
    ori_img = detach(input_tensor)
    gt_label = np.expand_dims(ori_cls,0)
    gt_mask = detach(gt_mask)
    if ins == 0:
        final_exp = gt_mask
        final_pred = pred_feat
        final_ins = ori_img
        final_label = gt_label
    else:
        final_exp = np.concatenate((final_exp, gt_mask), 0)
        final_pred = np.concatenate((final_pred, pred_feat), 0)
        final_ins = np.concatenate((final_ins, ori_img), 0)
        final_label = np.concatenate((final_label, gt_label), 0)
        

print("Exp size: ", final_exp.shape)
print("Pred size: ", final_pred.shape)
print("Image size: ", final_ins.shape)
print("Label size: ", final_label.shape)

np.save('train_exp/' + cur_split + '_' + EM + '.npy', final_exp)
# =============================================================================
# np.save('train_exp/' + cur_split + '_pred.npy', final_pred)
# np.save('train_exp/' + cur_split + '_ins.npy', final_ins)
# np.save('train_exp/' + cur_split + '_label.npy', final_label)
# =============================================================================

    
        
    
