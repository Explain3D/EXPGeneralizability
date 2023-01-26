#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 17:20:44 2022

@author: tan
"""

#!/usr/bin/env python
# coding: utf-8

import numpy as np
import time
#get_ipython().run_line_magic('matplotlib', 'inline')
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef, structural_similarity_index_measure

from data_utils import MNIST_EXP
import os
from torch.utils.data import Dataset
import sys
import matplotlib.pyplot as plt

def normalize_01(input_tensor):
    batch_size = input_tensor.shape[0]
    w,h = input_tensor.shape[-2:]
    input_tensor = input_tensor.reshape(batch_size,-1)
    input_tensor -= input_tensor.min(1, keepdim=True)[0]
    input_tensor /= input_tensor.max(1, keepdim=True)[0]
    input_tensor = input_tensor.reshape(batch_size,1,w,h)
    return input_tensor

def topk_idx(exp, prop):
    num_ins = exp.shape[0]
    exp_len = exp.shape[1]
    topk_value, topk_idx = torch.topk(exp, int(exp_len * prop), dim=1)
    binary_idx = torch.zeros_like(exp) - 1
    for n in range(num_ins):
        binary_idx[n][topk_idx[n]] = 1
    return binary_idx

# =============================================================================
# class MNIST_EXP_AE(nn.Module):
# 
#     def __init__(self):
#         super(MNIST_EXP_AE,self).__init__()
#         
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5),
#             nn.ReLU(True),
#             nn.Conv2d(16,32,kernel_size=5),
#             nn.ReLU(True))
# 
#         self.decoder = nn.Sequential(             
#             nn.ConvTranspose2d(32,16,kernel_size=5),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16,1,kernel_size=5)
# # =============================================================================
# #             nn.ReLU(True),
# #             nn.Sigmoid()
# # =============================================================================
#             )
# 
#     def forward(self,x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x
# =============================================================================


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
        return decoded


def detach(tensor):
    if torch.cuda.is_available() == True:
        return tensor.detach().cpu().numpy()
    else:
        return tensor.detach().numpy()


def imprt_acc(pred, gt_exp, percent, batch_size):
    pred = pred.squeeze(-1).view(batch_size, -1)
    gt_exp = gt_exp.squeeze(-1).view(batch_size, -1)
    ida = torch.argsort(pred, descending=True, dim=1)
    idb = torch.argsort(gt_exp, descending=True, dim=1)
    num_ins = ida.shape[0]
    total_len = ida.shape[1]
    num_compare = int(np.ceil(total_len * percent))
    
    
    cand_a = ida[:,:num_compare]
    cand_b = idb[:,:num_compare]
    
    total_acc = 0
    for ins in range(num_ins):
        intersect = np.intersect1d(detach(cand_a[ins].clone()), detach(cand_b[ins].clone()))
        num_impt = len(intersect)
        cur_impt_acc = num_impt / num_compare
        total_acc += cur_impt_acc
    
    total_acc = total_acc / num_ins
    return total_acc


def train_epoch():
    epoch_loss = []
    epoch_topk_acc = []
    epoch_spearman_coef = []
    epoch_pearson_coef = []
    epoch_SSIM = []
    epoch_MSE = []
    epoch_L1D = []
    for batch_id, data in enumerate(trainDataLoader, 0):
        
# =============================================================================
#         if batch_id > 5:
#             break
# =============================================================================
        
        optimizer.zero_grad()
        ins = data[0].to(device)
        exp = data[1].to(device)
        #exp = F.normalize(exp,dim=1)
        output = net(ins)
        #binary_mask = binary_mask.unsqueeze(-1)
        loss = criterion(exp.float(), output.float())
        #loss = criterion(idx_exp[:, :focus_points], idx_output[:, :focus_points])
        #loss = torch.mean(earth_mover_distance(idx_exp[:,:focus_points].unsqueeze(-1).repeat(1,1,3), idx_output[:,:focus_points].unsqueeze(-1).repeat(1,1,3), transpose=False))
        cur_batch_acc = imprt_acc(output, exp, focus_top, batch_size)
        
        exp = exp.float()
        avg_spear_coef = []
        avg_pearson_coef = []
        for b in range(exp.shape[0]):
            cur_output = output[b].view(1,-1)
            cur_exp = exp[b].view(1,-1)
            
            cur_spear_coef = spearman_corrcoef(cur_output, cur_exp)
            avg_spear_coef.append(detach(cur_spear_coef))
            
            cur_pearson_coef = pearson_corrcoef(cur_output, cur_exp)
            avg_pearson_coef.append(detach(cur_pearson_coef))
        
        avg_L1D = detach(L1(exp.float(), output.float()))
        avg_MSE = detach(L2(exp.float(), output.float()))
        avg_spear_coef = np.mean(avg_spear_coef)
        avg_pearson_coef = np.mean(avg_pearson_coef)
        output_for_ssim = output.clone().detach()
        exp_for_ssim = exp.clone().detach()
        avg_SSIM = detach(structural_similarity_index_measure(normalize_01(output_for_ssim), normalize_01(exp_for_ssim)))


        print("Train number ", batch_id, "L1 Dis: ", avg_L1D," MSE Dis: ", avg_MSE, "Topk acc: ", cur_batch_acc, "Spear Coef :", avg_spear_coef, "Pearson Coef :", avg_pearson_coef, "SSIM: ", avg_SSIM)
        loss.backward()
        optimizer.step()
        
        epoch_L1D.append(avg_L1D)
        epoch_MSE.append(avg_MSE)
        epoch_loss.append(loss.item())
        epoch_topk_acc.append(cur_batch_acc)
        epoch_spearman_coef.append(avg_spear_coef)
        epoch_pearson_coef.append(avg_pearson_coef)
        epoch_SSIM.append(avg_SSIM)
        
    return np.mean(epoch_loss), np.mean(epoch_L1D), np.mean(epoch_MSE), np.mean(epoch_topk_acc), np.mean(epoch_spearman_coef), np.mean(epoch_pearson_coef), np.mean(epoch_SSIM)


def test_epoch(): # test with all test set
    with torch.no_grad():
        epoch_loss = []
        epoch_topk_acc = []
        epoch_spearman_coef = []
        epoch_pearson_coef = []
        epoch_SSIM = []
        epoch_MSE = []
        epoch_L1D = []
        for j, data in enumerate(testDataLoader, 0):
            
# =============================================================================
#             if j > 5:
#                 break
# =============================================================================

            ins = data[0].float()
            exp = data[1].float()
            label = data[2]
            with torch.no_grad():
                ins = ins.to(device)
                exp = exp.to(device)
                #exp = F.normalize(exp,dim=1)
                output = net(ins)
                loss = criterion(exp.float(), output.float())
                #loss,_ = chamfer_distance(IG_exp, output)
                #loss = criterion(idx_exp[:, :focus_points], idx_output[:, :focus_points])
                #loss = torch.mean(earth_mover_distance(idx_exp[:,:focus_points].unsqueeze(-1).repeat(1,1,3), idx_output[:,:focus_points].unsqueeze(-1).repeat(1,1,3), transpose=False))
                cur_batch_acc = imprt_acc(output, exp, focus_top, batch_size)
                
                avg_spear_coef = []
                avg_pearson_coef = []
                
    
# =============================================================================
#                 np.save('visu/visu_data.npy', ins.clone().detach().cpu().numpy())
#                 np.save('visu/visu_out.npy', output.clone().detach().cpu().numpy())
# =============================================================================
                for b in range(exp.shape[0]):
                    cur_output = output[b].view(1,-1)
                    cur_exp = exp[b].view(1,-1)
                    
                    cur_spear_coef = spearman_corrcoef(cur_output, cur_exp)
                    avg_spear_coef.append(detach(cur_spear_coef))
                    
                    cur_pearson_coef = pearson_corrcoef(cur_output, cur_exp)
                    avg_pearson_coef.append(detach(cur_pearson_coef))
                
                avg_L1D = detach(L1(exp.float(), output.float()))
                avg_MSE = detach(L2(exp.float(), output.float()))
                avg_spear_coef = np.mean(avg_spear_coef)
                avg_pearson_coef = np.mean(avg_pearson_coef)
                output_for_ssim = output.clone().detach()
                exp_for_ssim = exp.clone().detach()
                avg_SSIM = detach(structural_similarity_index_measure(normalize_01(output_for_ssim), normalize_01(exp_for_ssim)))
                print("Test topk acc: ", cur_batch_acc, "L1 Dis: ", avg_L1D," MSE Dis: ", avg_MSE, "Spear Coef :", avg_spear_coef, "Pearson Coef :", avg_pearson_coef, "SSIM: ", avg_SSIM)
                
                if j == 0:
                    output_pool = output.clone()
                    label_pool = label.clone()
                else:
                    output_pool = torch.cat([output_pool, output], dim=0)
                    label_pool = torch.cat([label_pool, label], dim=0)
            
            
            epoch_L1D.append(avg_L1D)
            epoch_loss.append(loss.item())
            epoch_topk_acc.append(cur_batch_acc)
            epoch_spearman_coef.append(avg_spear_coef)
            epoch_pearson_coef.append(avg_pearson_coef)
            epoch_SSIM.append(avg_SSIM)
            epoch_MSE.append(avg_MSE)
        
        
    return np.mean(epoch_loss), np.mean(epoch_L1D), np.mean(epoch_MSE), np.mean(epoch_topk_acc), np.mean(epoch_spearman_coef), np.mean(epoch_pearson_coef), np.mean(epoch_SSIM), output_pool, label_pool


def weighted_MSE_loss(input, target):
        target_tmp = target.clone().squeeze().view(batch_size,-1)
        weights = torch.argsort(target_tmp)
        weights = weights.reshape(batch_size,28,28).unsqueeze(1)
        return (weights * (input - target) ** 2).mean()


if __name__ == '__main__':
    
    sys.path.append(os.getcwd())
    batch_size = 100
    epoch = 100
    EM = 'Lime30'

    print("EXP method: ", EM)
    print("Num of epoch: ", epoch)
    print("Batch Size: ", batch_size)
    
    # load dataset from numpy array and divide 90%-10% randomly for train and test sets
    #train_loader, test_loader = GetDataLoaders(npArray=pc_array, batch_size=batch_size)
    DATA_PATH = 'train_exp/'
    TRAIN_DATASET = MNIST_EXP(split='train', data_dir='train_exp/', EM=EM)
    TEST_DATASET = MNIST_EXP(split='test', data_dir='train_exp/', EM=EM)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=batch_size, shuffle=True, num_workers=1)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=batch_size, shuffle=False, num_workers=1)
    
    
    #Normalizing:
    img_size = TRAIN_DATASET[0][0].shape
    print(img_size)
    #train_loader = torch.from_numpy(train_loader)
    #test_loader = torch.from_numpy(test_loader)
    
    focus_top = 0.25
    latent_size = 128
    focus_points = int(img_size[1] * img_size[2] * focus_top)
    
    net = MNIST_EXP_AE()
    
    
    if torch.cuda.is_available() == True:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    net = net.to(device)
    
    optimizer = torch.optim.Adam(net.parameters(),lr=1e-5)
    
    #criterion = nn.KLDivLoss(log_target=True)
    
    
    criterion = nn.L1Loss(reduction='mean')
    
    
    L1 = nn.L1Loss(reduction='mean')
    L2 = nn.MSELoss(reduction='mean')
    train_L1_list = []
    train_MSE_list = []  
    train_topk_acc_list = []
    train_spearman_coef_list = []
    train_pearson_coef_list = []
    train_SSIM_list = []
    
    test_L1_list = []
    test_MSE_list = []
    test_topk_acc_list = []
    test_spearman_coef_list = []
    test_pearson_coef_list = []
    test_SSIM_list = []
    
    best_loss = float('inf')
    
    net = net.eval()
    test_loss, test_L1, test_MSE, test_topk_acc, test_spearman_coef, test_pearson_coef, test_SSIM, output_set, label_set = test_epoch() # test with test set
    test_L1_list.append(test_L1)
    test_MSE_list.append(test_MSE)
    test_topk_acc_list.append(test_topk_acc)
    test_spearman_coef_list.append(test_spearman_coef)
    test_pearson_coef_list.append(test_pearson_coef)
    test_SSIM_list.append(test_SSIM)
    
    
    for i in range(epoch) :
    
        startTime = time.time()
        
        net = net.train()
        train_loss, train_L1, train_MSE, train_topk_acc, train_spearman_coef, train_pearson_coef, train_SSIM = train_epoch() #train one epoch, get the average loss
        train_L1_list.append(train_L1)
        train_MSE_list.append(train_MSE)
        train_topk_acc_list.append(train_topk_acc)
        train_spearman_coef_list.append(train_spearman_coef)
        train_pearson_coef_list.append(train_pearson_coef)
        train_SSIM_list.append(train_SSIM)
        
        print("################ Begin Test ##################")
        
        net = net.eval()
        test_loss, test_L1, test_MSE, test_topk_acc, test_spearman_coef, test_pearson_coef, test_SSIM, output_set, label_set = test_epoch() # test with test set
        test_L1_list.append(test_L1)
        test_MSE_list.append(test_MSE)
        test_topk_acc_list.append(test_topk_acc)
        test_spearman_coef_list.append(test_spearman_coef)
        test_pearson_coef_list.append(test_pearson_coef)
        test_SSIM_list.append(test_SSIM)
        
        if test_loss < best_loss:
            best_loss = test_loss
            print("\nBetter model found, saving parameters...\n")
            torch.save(net, 'log/' + EM + '_AE.pth')
        
        epoch_time = time.time() - startTime
        print("\n\nepoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n")
        #writeString = "epoch " + str(i) + " train loss : " + str(train_loss) + " test loss : " + str(test_loss) + " epoch time : " + str(epoch_time) + "\n"
    
    train_L1_array = np.array(train_L1_list)
    train_MSE_array = np.array(train_MSE_list)
    train_topk_acc_array = np.array(train_topk_acc_list)
    train_spearman_coef_array = np.array(train_spearman_coef_list)
    train_pearson_coef_array = np.array(train_pearson_coef_list)
    train_SSIM_array = np.array(train_SSIM_list)
    
    test_L1_array = np.array(test_L1_list)
    test_MSE_array = np.array(test_MSE_list)
    test_topk_acc_array = np.array(test_topk_acc_list)
    test_spearman_coef_array = np.array(test_spearman_coef_list)
    test_pearson_coef_array = np.array(test_pearson_coef_list)
    test_SSIM_array = np.array(test_SSIM_list)
    
    np.save('res_array/' + EM + '/test_L1.npy', test_L1_array)
    np.save('res_array/' + EM + '/test_MSE.npy', test_MSE_array)
    np.save('res_array/' + EM + '/test_topk_acc.npy', test_topk_acc_array)
    np.save('res_array/' + EM + '/test_spearman_coef.npy', test_spearman_coef_array)
    np.save('res_array/' + EM + '/test_pearson_coef.npy', test_pearson_coef_array)
    np.save('res_array/' + EM + '/test_SSIM.npy', test_SSIM_array)
    
    np.save('res_array/' + EM + '/train_L1.npy', train_L1_array)
    np.save('res_array/' + EM + '/train_MSE.npy', train_MSE_array)
    np.save('res_array/' + EM + '/train_topk_acc.npy', train_topk_acc_array)
    np.save('res_array/' + EM + '/train_spearman_coef.npy', train_spearman_coef_array)
    np.save('res_array/' + EM + '/train_pearson_coef.npy', train_pearson_coef_array)
    np.save('res_array/' + EM + '/train_SSIM.npy', train_SSIM_array)
    
net = net.eval()
_, _, _, _, _, _, _, output_set, label_set = test_epoch() # test with test set
output_set = output_set.detach().cpu().numpy()
label_set = label_set.detach().cpu().numpy()

np.save('gen_exps/' + EM + '/generated_exp.npy', output_set)
np.save('gen_exps/' + EM + '/generated_label.npy', label_set)

        

