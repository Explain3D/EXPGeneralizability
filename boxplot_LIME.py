#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 17:21:50 2022

@author: tan
"""

import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

path = 'npys/FID/'
npys = os.listdir(path)
npys = [x for x in npys if not x.startswith('.')]

inter = []
intra = []
mean = []
num_iter = 0

df = []
for file in npys:
    cur_data = np.load(path + '/' + file, allow_pickle=True)
    cur_data = np.ravel(cur_data)
    split = file[:5]
    name = file[6:-4]
    n_data = cur_data.shape[0]
    
    cur_split = [split + ' class'] * n_data

    cur_name = [name] * n_data

    cur_tmp = {'Spearman Coef': cur_data, 'Type': cur_split, 'Explanation': cur_name}

    cur_df = pd.DataFrame(cur_tmp)
    df.append(cur_df)
    

    if file[10:-4] == '500' and file[:5] == 'inter':
        print(cur_df['Spearman Coef'].mean())
        print(max(cur_df['Spearman Coef'].max() - cur_df['Spearman Coef'].mean(),
                     cur_df['Spearman Coef'].mean() - cur_df['Spearman Coef'].min()))
        
    
df = pd.concat(df)



fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
sns.set(rc={'axes.facecolor':'lightgray', 'figure.facecolor':'white'})
bplot = sns.boxplot(data=df, x="Explanation", y="Spearman Coef", hue="Type", order=['n_sample=10', 'n_sample=50', 'n_sample=100', 'n_sample=500'])
bplot.legend_.set_title(None)
bplot.tick_params(labelsize=15)
plt.setp(ax1.get_legend().get_texts(), fontsize='15')
ax1.set_xlabel('Explanation', fontsize=15)
ax1.set_ylabel('Fréchet Inception Distance', fontsize=15)