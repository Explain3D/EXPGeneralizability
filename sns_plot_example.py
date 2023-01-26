#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 14:16:42 2022

@author: tan
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 
plt.rcParams["axes.grid"] = False

ins_path = 'train_exp/test_ins.npy'
ins_array = np.load(ins_path)

path = 'gen_exps/'
exps = os.listdir(path)
lime_exps = [exp for exp in exps if exp.startswith('Lime')]
lime_exps = sorted(lime_exps, key = lambda x: int(x[4:]))

exp_list = []

for exp in lime_exps:
        n_sample = exp[4:]
        cur_exp = np.load(path + exp + '/generated_exp.npy')
        print(exp)
        exp_list.append(cur_exp)
        
#2,13, 54
show_index = 2

titles = ['n_samples=10', 'n_sample=30', 'n_samples=50','n_samples=100','n_samples=500']

fig, axn = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(13.5,3))
cbar_ax = fig.add_axes([.91, .15, .03, .7])

for i, ax in enumerate(axn.flat):
    sns.heatmap(exp_list[i][show_index][0], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                cbar_ax=None if i else cbar_ax)
    ax.tick_params(left=False, bottom=False)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.title.set_text(titles[i])

plt.show()
#fig.tight_layout(rect=[0, 0, .9, 1])


