# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 09:12:13 2019

@author: mkaist
"""


import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math


#the code allows constants to be called outside from this file
#if left here, they can be removed from function arguments
#if outside, easier to do parameter sensitivit analysis
# in that case these would be variables
 
#Constants
MINRRI     = 110    #autocorr window size for detecting period CHANGE TO SEC
STEP_LEN   = 1      #step length used to segment signal (sec)
AC_LEN     = [5,7]  #autocorr window length (sec)
RRILIM     = 300    #lim for finding period, first two autocorr peaks CHANGE TO SEC
DRRILIM    = 10     #lim for RRI differences CHANGE TO SEC
AMPLIM     = 0.4    #lim for finding period, first two autocorr peaks



def segment_one_axis(signal_in,STEP_LEN,w,fs):
    """ Segments a single 1D vector into segments and returns segments in
    a list
        
    Keyword arguments:
    signal_in -- input vector to segment
    STEP_LEN -- distance between segments starting time (sec)
    w -- length of segments (sec) 
    fs -- sample frequency
    """   
            
    n_segments = round((signal_in.size/fs)/STEP_LEN)-2 #reasoning for -2 is unclear 
    segments = [] 
    
    #segments signal with STEP_LEN interval with w length window
    for i in range(0, n_segments + 1):
        startpos    = i*STEP_LEN*fs 
        endpos      = startpos + fs*w - 1
        endpos      = min(endpos,signal_in.size)
        segments.append(signal_in[startpos:endpos + 1]) 
    return segments 


def segment_all_axis(sensor_data, STEP_LEN, w, fs):
    """Segments 6 axis data and return 6 separate lists each containing
    a segmented signal
        
    Keyword arguments:
    sensor_data -- dictionary containing 6 axis data
    STEP_LEN -- distance between segments starting time (sec)
    w -- length of segments (sec) 
    fs -- sample frequenc
    """
    
    
    #ACHTUNG!!!basic filter move to pre-processing"!!!!!!
    LOWER_BAND = 4
    HIGHER_BAND = 40
    
    wbut = [2*LOWER_BAND/fs, 2*HIGHER_BAND/fs] #should be fs/2
    bbut,abut = signal.butter(2,wbut,btype='bandpass') #changed to 3
    axisDataAccX = signal.lfilter(bbut,abut,sensor_data['ax'])
    axisDataAccY = signal.lfilter(bbut,abut,sensor_data['ay'])
    axisDataAccZ = signal.lfilter(bbut,abut,sensor_data['az'])
    axisDataGyrX = signal.lfilter(bbut,abut,sensor_data['gx'])
    axisDataGyrY = signal.lfilter(bbut,abut,sensor_data['gy'])
    axisDataGyrZ = signal.lfilter(bbut,abut,sensor_data['gz'])
    
        
    fs = int(fs) #Conversion to integer
    #Cut fs samples from the beginning to give time filter to settle
    ax = axisDataAccX[fs-1:]
    ay = axisDataAccY[fs-1:]
    az = axisDataAccZ[fs-1:]
    gx = axisDataGyrX[fs-1:]
    gy = axisDataGyrY[fs-1:]
    gz = axisDataGyrZ[fs-1:]
    
    """
    ax = sensor_data['ax']
    ay = sensor_data['ay']
    az = sensor_data['az']
    gx = sensor_data['gx']
    gy = sensor_data['gy']
    gz = sensor_data['gz']
    """
      
    ax_segmented = segment_one_axis(ax,STEP_LEN,w,fs)
    ay_segmented = segment_one_axis(ay,STEP_LEN,w,fs)
    az_segmented = segment_one_axis(az,STEP_LEN,w,fs)
    gx_segmented = segment_one_axis(gx,STEP_LEN,w,fs)
    gy_segmented = segment_one_axis(gy,STEP_LEN,w,fs)
    gz_segmented = segment_one_axis(gz,STEP_LEN,w,fs)
    return (ax_segmented, ay_segmented, az_segmented,\
            gx_segmented, gy_segmented, gz_segmented)
   


def correlate_segments(signals_in):
    """Autocorrelate each segment and return
    a list with autocorrelated segments
    
    Keyword arguments: 
    signals_in -- a list of signal segments
    """
    
    n_segments  = len(signals_in) 
    ac = []
    
    #go through all segments and autocorrelate each
    for i in range (0, n_segments): 
        tmp = signal.correlate(signals_in[i], signals_in[i])  
        tmp = tmp[signals_in[i].size-9:] #ACHTUNG WE CHANGED THIS TO -2, GOOD DECISION?
        tmp = tmp/max(tmp)
        #plt.plot(tmp)
        ac.append(tmp)
        
    return ac


    

def find_sidepeaks_one_segment(signal_in, MINRRI):
    """Find side peaks from one 1D vector and 
    returns peak locations and amplitudes
        
    Keyword arguments: 
    signal_in -- input signal from which peaks are detected
    MINRRI -- backward min distance from where a peak is searched        
    """
    all_locs, _ = signal.find_peaks(signal_in)
    all_pks     = signal_in[all_locs]
    
    #peak elinimation rules
    tmp  = []
    locs = [all_locs[0]]
    pks  = [all_pks[0]]
    for i in range(1, all_locs.size): 
        startpos = max(1,all_locs[i] - MINRRI)
        tmp      = max(signal_in[startpos:all_locs[i]])
        if all_pks[i] >= max(all_pks[i:]) and\
        all_pks[i] >= tmp and\
        all_locs[i] > locs[-1] + MINRRI: 
            
        #if all conditions are true, append the peaks and location
           locs.append(all_locs[i])
           pks.append(all_pks[i])
           
    return locs, pks

    
    

def find_period_segments(signals_in, MINRRI, AMPLIM, RRILIM, DRRILIM):
    """Find periodicity of a segmented signal
    
    Keyword arguments: 
    signals_in -- a segmented signal
    MINRRI -- passed onward 
    AMPLIM -- required ratio between first two found peaks
    RRILIM -- max distance between found peaks
    DRRILIM -- limit of interval differences       
    """
    n_segments  = len(signals_in)  #segments is a list
    periodfound = np.zeros(n_segments)
    rri         = np.zeros(n_segments)
    
    for i in range (0, n_segments): 
        #find all local maximas
        locs_tmp, pks_tmp = find_sidepeaks_one_segment(signals_in[i], MINRRI)

        #period finding rules
        max_locs_diff = math.inf
        max_rri       = math.inf        
        if len(locs_tmp) > 3 and (pks_tmp[1] / pks_tmp[0]) > AMPLIM:
            locs_diff     = np.diff(locs_tmp[0:3])
            max_rri       = locs_tmp[1] - locs_tmp[0]
            max_locs_diff = max(abs(np.diff(locs_diff)))
            
        if max_locs_diff < DRRILIM and max_rri < RRILIM:
            periodfound[i] = 1
            rri[i] = max_rri
        else:
            periodfound[i] = 0   
            rri[i] = 0
            
      
    return periodfound, rri

    

def find_period_all_axis(ax, ay, az, gx, gy, gz, MINRRI, AMPLIM, RRILIM, fs):
    """Find period from 6 axis data. Returns a numpy vector
    containing 1 (periodfound) or 0 (period not found) for each
    segment in each 6 axis. Returns a numpy vector containing the average 
    over az, gy, gz lag between 1st and 2nd peaks
    
    Keyword arguments:
    ax, ay, az, gx ,gy, gz -- segmented and autocorrelated inputs
    MINRRI -- passed onward 
    AMPLIM -- passed onward 
    RRILIM -- passed onward 
    fs -- passed onward     
    """
    periodfound_ax, rri_ax = find_period_segments(ax, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_ay, rri_ay = find_period_segments(ay, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_az, rri_az = find_period_segments(az, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_gx, rri_gx = find_period_segments(gx, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_gy, rri_gy = find_period_segments(gy, MINRRI, AMPLIM, RRILIM, DRRILIM)
    periodfound_gz, rri_gz = find_period_segments(gz, MINRRI, AMPLIM, RRILIM, DRRILIM)
    
    P = np.vstack((periodfound_ax, periodfound_ay, periodfound_az,
                   periodfound_gx, periodfound_gy, periodfound_gz))
    
    #periodfound vector size 1 x n where n is number of segments for a 
    #spesific autocorrelation window length
    periodfound_all_axis = P.sum(axis=0)
    periodfound_all_axis[periodfound_all_axis >= 1] = 1
    
    # mean rri averaged over all segments and all axis where period was found
    # with one correlation window  
    R   = np.vstack((rri_az, rri_ay, rri_az, rri_gx, rri_gy, rri_gz)) #take mean lag from az, gy, gz autocorrelations
    R   = R[np.where(R > 0) ]
    
   
    if len(R) > 0:
        rri_all_axis = R.mean(axis=0)    
    else: 
        rri_all_axis = math.nan
    
    
    
    return periodfound_all_axis, rri_all_axis

#==============================MAIN============================================

def detect_period(sensor_data, fs):
         
    for i in range (0, len(AC_LEN)):
    
        ax_segmented,\
        ay_segmented,\
        az_segmented,\
        gx_segmented,\
        gy_segmented,\
        gz_segmented = segment_all_axis(sensor_data, STEP_LEN, AC_LEN[i], fs)
     
        ax_ac = correlate_segments(ax_segmented)
        ay_ac = correlate_segments(ay_segmented)
        az_ac = correlate_segments(az_segmented)
        gx_ac = correlate_segments(gx_segmented)
        gy_ac = correlate_segments(gy_segmented)
        gz_ac = correlate_segments(gz_segmented)
        
        
        #initialize empty matrix if vars do not exist
        if 'periodfound_all_axis' in locals():
            None
        else:
            periodfound_all_axis = np.empty([len(ax_ac), len(AC_LEN)])
        
        if 'rri_all_axis' in locals():
            None
        else:
            rri_all_axis = np.empty([len(AC_LEN)])
    
        periodfound_all_axis[:,i], rri_all_axis[i] = find_period_all_axis(ax_ac, ay_ac, az_ac, gx_ac, gy_ac, gz_ac, MINRRI, AMPLIM, RRILIM, fs)
     
#   
    tmp = periodfound_all_axis.sum(axis=1)
    tmp[tmp >= 1] = 1
    periodval = np.mean(tmp)
    
    
    #fraction (%) of segments that period was found from final 
    #periodfound vector 1 x n where n is number of segments
    regularity_index = tmp.sum()/len(tmp) *100 
    
    
    

    if periodval < 0.1:
        afibval = 1
        rrival = np.nan
    else:
        afibval = 0
        rrival  = np.mean(rri_all_axis)
        
    
    
    rv = {
        'afibval': afibval,
        'rrival': rrival,
        'regularity_index': regularity_index
    }
    
    return rv
    #return afibval, rrival, regularity_index

