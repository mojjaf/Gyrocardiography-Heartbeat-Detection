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
import numpy as np
import time
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator, NullFormatter)

def plot_signal_to_user(input_sig,envelope):
    plt.close('all')
    fs=200
    lfc=4
    hfc=30
    sig_filt = butterworth_filter(input_sig[5*fs:fs*65], lowcut = lfc, highcut = hfc, fs = fs)
      
    signal_norm=np.array((sig_filt-np.mean(sig_filt))/np.std(sig_filt))
    
    t= np.arange(0, len(signal_norm)/fs, 0.005)
    
    #    yhat =sp.signal.savgol_filter(envelope, 51, 3)
    envelope_sig=(envelope-np.mean(envelope))/np.std(envelope)
    t_env=t-0.2
    
    y_low=np.min(signal_norm)
    y_up=np.max(signal_norm)
    sig_range=y_up-y_low
    y_low_env=np.min(envelope)
    y_up_env=np.max(envelope)
    env_range=y_up_env-y_low_env
    
    
    fig=sig_plotter(t,t_env,signal_norm,envelope_sig,env_range,y_low,y_up)
    
    plt.savefig('meas_fig_af3.jpg', dpi = 600)
    return


def sig_plotter(t_sig_,t_env_,sig_,env_,env_range,y_low,y_up):
    
    fig, ax = plt.subplots(figsize=(860, 4))
    fig.set_size_inches(30, fig.get_figheight(), forward=True)
    ax.minorticks_on()
    
    major_ticks = np.arange(0, 60, 1)
    minor_ticks = np.arange(0, 60, 0.25)
    
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks, minor=True)
#    ax.set_yticks(minor_ticks)
    
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    # Customize the minor grid
    ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    # Turn off the display of all ticks.
    ax.tick_params(which='both' # Options for both major and minor ticks
            ) # turn off bottom ticks
#    ax.grid(True)
#    ax.xaxis.set_major_locator(MultipleLocator(1))
#    ax.yaxis.set_major_locator(MultipleLocator(1))
#    
#    # Change minor ticks to show every 5. (20/4 = 5)
#    ax.xaxis.set_minor_locator(AutoMinorLocator(0.25))
#    ax.yaxis.set_minor_locator(AutoMinorLocator(0.25))
#    
#    # Turn grid on for both major and minor ticks and style minor slightly
#    # differently.
#    ax.grid(which='major', color='red', linestyle='-')
#    ax.grid(which='minor', color='black', linestyle=':')

##    
    ax.set_xlim(0, 60)
    ax.yaxis.set_major_formatter(NullFormatter())
#    ax.set_frame_on(False)
    ax.set_ylim(y_low-env_range, y_up+1)
    ax.plot(t_sig_,sig_,'b')
    ax.plot(t_env_,env_-(env_range+1),'purple')
    plt.tight_layout()  
    plt.xlabel('Time (second)')
    plt.ylabel('Angular Velocity (a.u)')
    plt.show()
 
    #    plt.plot([], [], ' ', label=plotlabel_sig)

    #    ax.axes.get_yaxis().set_visible(False)
        #plt.legend(loc='upper right')
        
    
    plt.title('Gyrocardiogram')
        
    return fig 
   
    
def main():
    MODE_AF_signals=load_dict_from_hdf5('MODE_AF_rawsignals.h5')
    list_of_key_MODEAFnames=list(MODE_AF_signals.keys())     
    fs=200 #Hz

    for i in list_of_key_MODEAFnames[153:154]:

    
        plt.close('all')
        
    ####################################### Detecting Heartrates and extracting features 
        data=preprocess_resample(MODE_AF_signals[i]['acc'], MODE_AF_signals[i]['gyro'])
        
        fused_signal,Peak_locs= heartbeatdetector(data)
    plot_signal_to_user(data['gy'],fused_signal)
        
if __name__=="__main__":
    main()   