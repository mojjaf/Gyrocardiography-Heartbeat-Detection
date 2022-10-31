#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 14:19:23 2019

@author: Mojtaba Jafaritadi
"""
##############
import matplotlib.pyplot as plt
from h5_helperfunctions import load_dict_from_hdf5
from HeartBeatDetector import butterworth_filter,heartbeatdetector,preprocess_resample
from HRV_parameters import *
import pandas as pd
import scipy as sp
import time
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,NullFormatter)
from matplotlib.backends.backend_pdf import PdfPages
def plot_signal_to_user(input_sig,envelope):
    
    fs=200
    lfc=4
    hfc=30
    sig_filt = butterworth_filter(input_sig[5*fs:fs*65], lowcut = lfc, highcut = hfc, fs = fs)
      
    signal_norm=np.array((sig_filt-np.mean(sig_filt))/np.std(sig_filt))
    
    t= np.arange(0, len(signal_norm)/fs, 0.005)
    
    win_len=60*fs
    num_of_subfig=int(np.floor(len(signal_norm)/win_len) )    
    
    sig_waveforms=signal_norm.reshape(num_of_subfig,win_len)
    #    yhat =sp.signal.savgol_filter(envelope, 51, 3)
    envelope_sig=(envelope-np.mean(envelope))/np.std(envelope)
    env_waveforms=envelope_sig.reshape(num_of_subfig,win_len)
    t_sig_sliced=t.reshape(num_of_subfig,win_len)
    t_env=t-0.2
    t_env_sliced=t_env.reshape(num_of_subfig,win_len)
    
    y_low=np.min(signal_norm)
    y_up=np.max(signal_norm)
    sig_range=y_up-y_low
    y_low_env=np.min(envelope)
    y_up_env=np.max(envelope)
    env_range=y_up_env-y_low_env
    
    figures=list()
    for i in range(num_of_subfig):
        figures.append(sig_plotter(t_sig_sliced[i,:],t_env_sliced[i,:],sig_waveforms[i,:],env_waveforms[i,:],env_range,y_low))
    
#              
#    fig, ax = plt.subplots(figsize=(50, 4))
#     
#    plt.minorticks_on()
#    # Set axis ranges; by default this will put major ticks every 25.
#    y_low=np.min(signal_norm)
#    y_up=np.max(signal_norm)
#    sig_range=y_up-y_low
#    y_low_env=np.min(envelope)
#    y_up_env=np.max(envelope)
#    env_range=y_up_env-y_low_env
#    ax.set_xlim(0, 10)
#    ax.set_ylim(y_low-env_range, y_up+3)
#    
#    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
#    # Customize the minor grid
#    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
#    # Turn off the display of all ticks.
#    plt.tick_params(which='both', # Options for both major and minor ticks
#                    top='off', # turn off top ticks
#                    left='off', # turn off left ticks
#                    right='off',  # turn off right ticks
#                    bottom='off') # turn off bottom ticks
#    
##    plotlabel_sig="Average Heart Rate = %.2f"%(HR)  
#    ax.yaxis.set_major_formatter(NullFormatter())
#    ax.set_frame_on(False)
#    plt.plot(t,signal_norm,'b', label='GCG Y-axis')
#    plt.plot(t_env,envelope_sig-(env_range+1),'purple',label='GCG Pulsewave Velocity')
##    plt.plot([], [], ' ', label=plotlabel_sig)
#    plt.xlabel('Time (second)')
#    plt.ylabel('Angular Velocity (a.u)')
#    ax.axes.get_yaxis().set_visible(False)
    #plt.legend(loc='upper right')
   # plt.title('Gyrocardiogram')
    
    plt.tight_layout()

    return


def write_pdf(fname, figures):
    doc = PdfPages(fname)
    for fig in figures:
        fig.savefig("{}.png".format(i), format='pdf', dpi=600)
    doc.close()

def sig_plotter(t_sig_,t_env_,sig_,env_,env_range,y_low):

    
    
    fig, ax = plt.subplots(figsize=(800, 4))
    fig.set_size_inches(30, fig.get_figheight(), forward=True)
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # Turn off the display of all ticks.
    ax.tick_params(which='both', # Options for both major and minor ticks
                top='off', # turn off top ticks
                left='off', # turn off left ticks
                right='off',  # turn off right ticks
                bottom='off') # turn off bottom ticks
    
#   ax.set_xlim(0, 5)
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.set_frame_on(False)
    ax.set_ylim(y_low-env_range, y_up+3)
    plt.plot(t_sig_,sig_,'b')
    plt.plot(t_env_,env_-(env_range+1),'purple')
    plt.tight_layout()  
    

    plt.show()
    plt.savefig('test3.jpg', dpi = my_dpi)
    #    plt.plot([], [], ' ', label=plotlabel_sig)
    #    plt.xlabel('Time (second)')
    #    plt.ylabel('Angular Velocity (a.u)')
    #    ax.axes.get_yaxis().set_visible(False)
        #plt.legend(loc='upper right')
        # plt.title('Gyrocardiogram')
        
    return fig 
   
MODE_AF_signals=load_dict_from_hdf5('MODE_AF_rawsignals.h5')
list_of_key_MODEAFnames=list(MODE_AF_signals.keys())     
fs=200 #Hz

for i in list_of_key_MODEAFnames[147:148]:

    
    plt.close('all')
    
####################################### Detecting Heartrates and extracting features 
    data=preprocess_resample(MODE_AF_signals[i]['acc'], MODE_AF_signals[i]['gyro'])
    
    fused_signal,Peak_locs= heartbeatdetector(data)
    time_domain_features=get_hrv_features(Peak_locs,fs)
    
    
#    
#def main():
#    a = fig_barh(['a','b','c'], [1, 2, 3], 'Test #1')
#    b = fig_barh(['x','y','z'], [5, 3, 1], 'Test #2')
#    write_pdf('test.pdf', [a, b])
#
#if __name__=="__main__":
#    main()    
plot_signal_user(data['gy'],fused_signal,time_domain_features['mean_hr'])

    
