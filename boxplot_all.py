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

def del_smooth(list_words, ending):
    return [word for word in list_words if word[len(word)-len(ending):] != ending]

plt.rcParams.update(plt.rcParamsDefault)

path = 'npys/FID/'
npys = os.listdir(path)
npys = [x for x in npys if not x.startswith('.')]

inter = []
intra = []
mean = []
num_iter = 0

df = []

exp_name_list = []
for file in npys:
    cur_data = np.load(path + '/' + file, allow_pickle=True)
    cur_data = np.ravel(cur_data)
    split = file[:5]
    name = file[6:-4]

    exp_name_list.append(name)
    n_data = cur_data.shape[0]
    
    cur_split = [split + ' class'] * n_data
    cur_name = [name] * n_data
    
    cur_tmp = {'Spearman Coef': cur_data, 'Type': cur_split, 'Explanation': cur_name}
    cur_df = pd.DataFrame(cur_tmp)
    df.append(cur_df)



# =============================================================================
# exp_name_list = list(dict.fromkeys(exp_name_list))
# exp_name_list = sorted(exp_name_list, key=lambda x: int(x[4:]))
# print(exp_name_list)
# 
# =============================================================================
# =============================================================================
# exp_name_list.remove('Lime')
# exp_name_list.remove('KSHAP')
# =============================================================================

# =============================================================================
# exp_name_list =['Lime', 'KSHAP']
# =============================================================================

# =============================================================================
#     if file[10:-4] == '500' and file[:5] == 'inter':
#         print(cur_df['Spearman Coef'].mean())
#         print(max(cur_df['Spearman Coef'].max() - cur_df['Spearman Coef'].mean(),
#                      cur_df['Spearman Coef'].mean() - cur_df['Spearman Coef'].min()))
# =============================================================================


# =============================================================================
# exp_name_list = del_smooth(exp_name_list, '_s')
# exp_name_list = list(dict.fromkeys(exp_name_list))
# =============================================================================

exp_name_list = ['V', 'V_s', 'IxG', 'IxG_s', 'IG', 'IG_s']

df = pd.concat(df)



df_mean = df.groupby(["Type", "Explanation"], as_index = False)["Spearman Coef"].mean()
for exp_name in exp_name_list:
    statistics = df_mean.loc[df_mean['Explanation'] == exp_name,'Spearman Coef']
    inter = statistics.iloc[0]
    intra = statistics.iloc[1]
# =============================================================================
#     print('Delta '+exp_name +': ',  (intra-inter))
# =============================================================================
    print('Delta '+exp_name +': ',  (inter-intra)/intra)



fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(7, 6))
# =============================================================================
# sns.set(rc={'axes.facecolor':'lightgray', 'figure.facecolor':'white'})
# =============================================================================
bplot = sns.boxplot(data=df, x="Explanation", y="Spearman Coef", hue="Type", order=exp_name_list)

labels = ax1.get_xticklabels()
# =============================================================================
# labels = ['n_sample=10', 'n_sample=30', 'n_sample=50', 'n_sample=100', 'n_sample=500']
# =============================================================================
ax1.set_xticklabels(labels)

bplot.legend_.set_title(None)
bplot.tick_params(labelsize=10)
plt.setp(ax1.get_legend().get_texts(), fontsize='12')
ax1.set_xlabel('Explanation', fontsize=12)
ax1.set_ylabel('Fréchet Inception Distance', fontsize=15)
# =============================================================================
# ax1.set_ylabel('Spearman Coefficient', fontsize=15)
# =============================================================================
plt.grid(axis='y', color='w')
ax1.set_axisbelow(True) 
ax1.set_facecolor('lightgrey')
# =============================================================================
# ax1.yaxis.tick_right()
# =============================================================================
plt.show()
