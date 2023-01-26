#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 16:24:01 2022

@author: tan
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns


path = '../gen_exps_s/'

EMs = os.listdir(path)
outliers = [x for x in EMs if x.startswith('.')]
EMs = [x for x in EMs if x not in outliers]

# =============================================================================
# EMs = sorted(EMs)
# =============================================================================

EMs = ['V', 'IxG', 'IG', 'V_s', 'IxG_s', 'IG_s']

num_EM = len(EMs)

exps = []
for em in EMs:
    exp_path = path + em + '/'
    exp_file = np.load(exp_path + 'generated_exp.npy')
    exps.append(exp_file)
    

selected_ins = 22#79  #1899 

fig, axn = plt.subplots(2, 3, sharex=True, sharey=True)
cbar_ax = fig.add_axes([.91, .3, .03, .4])

for i, ax in enumerate(axn.flat):
    sns.heatmap(exps[i][selected_ins][0], ax=ax,
                cbar=i == 0,
                vmin=0, vmax=1,
                cbar_ax=None if i else cbar_ax)
    ax.set(xticklabels=[])
    ax.set(yticklabels=[])
    ax.tick_params(bottom=False)
    ax.tick_params(left=False)
    ax.set(xlabel=EMs[i])

fig.tight_layout(rect=[0, 0, .9, 1])



plt.show()