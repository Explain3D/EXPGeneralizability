#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:24:01 2022

@author: tan
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def get_mean_and_fluct(array):
    mean = np.mean(array)
    fluct = np.max((np.max(array)-mean, mean-np.min(array)))
    return mean, fluct


path = '../res_array_s/'

EMs = os.listdir(path)
print(EMs)

fig = plt.figure()

ax0 = fig.add_subplot(2, 3, 1)
ax0.title.set_text('L1')
ax0.set_facecolor('lightgrey')
ax0.grid()

ax1 = fig.add_subplot(2, 3, 2)
ax1.title.set_text('MSE')
ax1.set_facecolor('lightgrey')
ax1.grid()

ax2 = fig.add_subplot(2, 3, 3)
ax2.title.set_text('SSIM')
ax2.set_facecolor('lightgrey')
ax2.grid()

ax3 = fig.add_subplot(2, 3, 4)
ax3.title.set_text('Top-k Acc.')
ax3.set_facecolor('lightgrey')
ax3.grid()

ax4 = fig.add_subplot(2, 3, 5)
ax4.title.set_text('Spearman Coef.')
ax4.set_facecolor('lightgrey')
ax4.grid()

ax5 = fig.add_subplot(2, 3, 6)
ax5.title.set_text('Pearson Coef.')
ax5.set_facecolor('lightgrey')
ax5.grid()



num_EMs = len(EMs)

topk_list = []
spearman_list = []
pearson_list = []

for idx in range(num_EMs):
    EM = EMs[idx]
    if not EM.startswith('.DS'):

        test_L1 = np.load(path + EM + '/test_L1.npy')
        test_MSE = np.load(path + EM + '/test_MSE.npy')
        test_topk_acc = np.load(path + EM + '/test_topk_acc.npy')
        test_spearman_coef = np.load(path + EM + '/test_spearman_coef.npy')
        test_pearson_coef = np.load(path + EM + '/test_pearson_coef.npy')
        test_SSIM = np.load(path + EM + '/test_SSIM.npy')
        
        train_L1 = np.load(path + EM + '/train_L1.npy')
        train_MSE = np.load(path + EM + '/train_MSE.npy')
        train_topk_acc = np.load(path + EM + '/train_topk_acc.npy')
        train_spearman_coef = np.load(path + EM + '/train_spearman_coef.npy')
        train_pearson_coef = np.load(path + EM + '/train_pearson_coef.npy')
        train_SSIM = np.load(path + EM + '/train_SSIM.npy')
        
        
        if EM.startswith('V'):
            plot_color = 'r'
        elif EM.startswith('IxG'):
            plot_color = 'g'
        elif EM.startswith('IG'):
            plot_color = 'b'

        if EM.endswith('_s'):
            marker = 'o'
        else:
            marker = 's'
            
# =============================================================================
#         if idx % 4 == 0:
#             marker = 'o'
#         elif idx % 4 == 1:
#             marker = 'v'
#         elif idx % 4 == 2:
#             marker = '^'
#         elif idx % 4 == 3:
#             marker = 's'
# =============================================================================
        
        marker_dis = 20
        
        last_ten_L1 = test_L1[-10:]
        last_ten_MSE = test_MSE[-10:]
        last_ten_SSIM = test_SSIM[-10:]
        
        last_ten_topk = test_topk_acc[-10:]
        last_ten_spearman = test_spearman_coef[-10:]
        last_ten_pearson = test_pearson_coef[-10:]
        
        u_L1, f_L1 = get_mean_and_fluct(last_ten_L1)
        u_MSE, f_MSE = get_mean_and_fluct(last_ten_MSE)
        u_SSIM, f_SSIM = get_mean_and_fluct(last_ten_SSIM)
        u_topk, f_topk = get_mean_and_fluct(last_ten_topk)
        u_spearman, f_spearman = get_mean_and_fluct(last_ten_spearman)
        u_pearson, f_pearson = get_mean_and_fluct(last_ten_pearson)
        
        print("\n")
        print(EM)
        print("L1: ", u_L1, f_L1)
        print("MSE: ", u_MSE, f_MSE)
        print("SSIM: ", u_SSIM, f_SSIM)
        print("topk: ", u_topk, f_topk)
        print("spearman: ", u_spearman, f_spearman)
        print("pearson: ", u_pearson, f_pearson)
        print('Avg. LAB: ', np.mean([u_topk, u_spearman, u_pearson]))
        
        topk_list.append(u_topk)
        spearman_list.append(u_spearman)
        pearson_list.append(u_pearson)
        
# =============================================================================
#         ax0.plot(train_L1,label='Train L1')
# =============================================================================
        ax0.plot(test_L1,label= EM, markevery=marker_dis, marker=marker, color = plot_color)
# =============================================================================
#         ax1.plot(train_MSE,label='Train MSE')
# =============================================================================
        ax1.plot(test_MSE,label=EM, markevery=marker_dis, marker=marker, color = plot_color)
# =============================================================================
#         ax5.plot(train_SSIM,label='Train SSIM')
# =============================================================================
        ax2.plot(test_SSIM,label=EM, markevery=marker_dis, marker=marker, color = plot_color)
# =============================================================================
#         ax2.plot(train_topk_acc,label='Train topk_acc')
# =============================================================================
        ax3.plot(test_topk_acc,label=EM, markevery=marker_dis, marker=marker, color = plot_color) 
# =============================================================================
#         ax3.plot(train_spearman_coef,label='Train Spear_coef')
# =============================================================================
        ax4.plot(test_spearman_coef,label=EM, markevery=marker_dis, marker=marker, color = plot_color)
# =============================================================================
#         ax4.plot(train_pearson_coef,label='Train Pearson_coef')
# =============================================================================
        ax5.plot(test_pearson_coef,label=EM, markevery=marker_dis, marker=marker, color = plot_color)

# =============================================================================
# ax0.legend()
# ax1.legend()
# ax2.legend()
# ax3.legend()
# ax4.legend()
# ax5.legend()
# =============================================================================


topk_array = np.expand_dims(np.array(topk_list), -1)
spearman_array = np.expand_dims(np.array(spearman_list), -1)
pearson_array = np.expand_dims(np.array(pearson_list), -1)
TSP = np.concatenate([topk_array, spearman_array, pearson_array], axis=1)


np.save('npys/concluded/Train_all.npy', TSP)


handles, labels = ax0.get_legend_handles_labels()

labels = ['IG_s', 'IG', 'IxG', 'V_s', 'IxG_s', 'V']
# =============================================================================
# labels = ['n_sample=50', 'n_sample=10', 'n_sample=30', 'n_sample=100', 'n_sample=500']
# =============================================================================
order = [5,3,2,4,1,0]
fig.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc=(0.332, 0.04),ncol=10)



plt.show()

plt.savefig('visu/' + EM + '.png')