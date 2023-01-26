#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 11:37:24 2023

@author: tan
"""

import numpy as np
import os
import matplotlib.pyplot as plt

ins_path = 'train_exp/test_ins.npy'
ins_array = np.load(ins_path)


#5  15  26
show_index = 1899


fig, ax = plt.subplots(figsize=(13.5,3))
ax.imshow(ins_array[show_index][0], cmap='gray')    
ax.tick_params(left=False, bottom=False)
ax.set(xticklabels=[])
ax.set(yticklabels=[])
ax.title.set_text('Image')

plt.show()