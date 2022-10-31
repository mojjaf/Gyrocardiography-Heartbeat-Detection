#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 10:20:46 2019

@author: Mojtaba Jafaritadi
"""
import time
start = time.time()

from HeartBeatDetector import *
from HRV_parameters import *
from h5_helperfunctions import *

import pandas as pd

MODE_AF_signals=load_dict_from_hdf5('MODE_AF_rawsignals.h5')
UniHill_signals=load_dict_from_hdf5('UniHill_rawsignals.h5')


list_of_key_MODEAFnames=list(MODE_AF_signals.keys())     
list_of_key_UniHillnames=list(UniHill_signals.keys())     


############################################### training data (Mode-AF)
signals_fused=dict()
heart_beat_locs=dict()
HRV_features=dict()

fs=200 #Hz

for i in list_of_key_MODEAFnames[:]:

    
    plt.close('all')
    
####################################### Processing Heartrates from MODE AF dataset
    data=preprocess_resample(MODE_AF_signals[i]['acc'], MODE_AF_signals[i]['gyro'])
    
    fused_signal,Peak_locs= heartbeatdetector(data) ## detect heartbeats
    signals_fused[i]=fused_signal
    heart_beat_locs[i]=Peak_locs
    
    time_domain_features=get_hrv_features(Peak_locs,fs)## calculate HRV features
    HRV_features[i]=time_domain_features
    
############################################ Converts HRV  features to dataframe 
HRV_feat_MODEAF=pd.DataFrame.from_dict(HRV_features, orient='index') ## transform HRV features to list

####### Statistical Analysis for HR Validation and Testing

############################################
import statsmodels.api as sm
import matplotlib as mpl
import pingouin as pg
########################################## Loading csv files to read heart rataes from ECG and Manual annotations
HR_ekg_tyks = pd.read_csv('MODE_AF_sub.csv', sep = ';', header = 0) ### ECG
HR_mcg_manual = pd.read_csv('HeartRates_MODEAF_07082019.csv', sep = ',', header = 0)## manually calculated HRs

######################################################## BLAND-ALTMAN PLOT


ax1 = pg.plot_blandaltman(np.array(HR_mcg_manual['MeanHR']), np.array(HRV_feat_MODEAF['mean_hr']))
ax1.set_title('Bland-Altman Plot True (Manual) vs. Estimated  HR')

ax2 = pg.plot_blandaltman(np.array(HR_ekg_tyks['pulse']), np.array(HRV_feat_MODEAF['mean_hr']))
ax2.set_title('Bland-Altman Plot True (ECG) HR vs. Estimated HR')


######################################################## CORRELATION analysis
correlation_test_manual_vs_pred=pg.corr(np.array(HRV_feat_MODEAF['mean_hr']), np.array(HR_mcg_manual['MeanHR']), method="pearson")
correlation_test_ECG_vs_pred=pg.corr(np.array(HRV_feat_MODEAF['mean_hr']), np.array(HR_ekg_tyks['pulse']), method="pearson")



####################################################### Boxplot and Violin plots for visual comparison
HR_all=np.stack((np.array(HRV_feat_MODEAF['mean_hr']),np.array(HR_ekg_tyks['pulse']), np.array(HR_mcg_manual['MeanHR'])))

HR_list=HR_all.tolist()

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4))

# plot violin plot
axes[0].violinplot(HR_list,
                   showmeans=False,
                   showmedians=True)
axes[0].set_title('Violin plot')

# plot box plot
axes[1].boxplot(HR_list)
axes[1].set_title('Box plot')

# adding horizontal grid lines
for ax in axes:
    ax.yaxis.grid(True)
    ax.set_xticks([y + 1 for y in range(len(HR_list))])
    ax.set_xlabel('Three separate samples')
    ax.set_ylabel('Observed values')

# add x-tick labels
plt.setp(axes, xticks=[y + 1 for y in range(len(HR_list))],
         xticklabels=['Estimated', 'ECG HR', 'Manual HR'])
plt.show()


end = time.time()
print(end - start)

