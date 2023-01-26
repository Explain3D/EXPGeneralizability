#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 17:31:38 2023

@author: tan
"""

import numpy as np
import matplotlib.pyplot as plt

rc = np.load('../generated_exp.npy')
ins_path = 'train_exp/test_ins.npy'
ins_array = np.load(ins_path)

select_idx = 1568

fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)
plt.imshow(ins_array[select_idx][0], cmap='gray')
ax1.set_xticks([])
ax1.set_yticks([])

ax2 = fig.add_subplot(2,1,2)
plt.imshow(rc[select_idx][0], cmap='gray')
ax2.set_xticks([])
ax2.set_yticks([])
    
plt.show()
