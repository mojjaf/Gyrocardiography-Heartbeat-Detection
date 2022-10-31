#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 16:55:19 2019

@author: differet developers
"""

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd 
from numpy import *
from scipy.signal import hilbert
import scipy.io as sio
import scipy.signal
from scipy import signal
import glob
import csv, json, time, base64, os, sys, time
import os.path
from random import randint
from scipy import interpolate
from pprint import pprint
from datetime import datetime
from array import array
import csv, json, time, base64, os, sys, time
import matplotlib.pyplot as plt
import openpyxl
from pathlib import Path
import numba
from numba import *

import neurokit as nk
from numpy import diff
import os
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.decomposition import FastICA

from typing import List, Tuple
from collections import namedtuple
#import nolds
from astropy.stats import LombScargle





def butter_bandpass_filter(data,  fs, lowcut, highcut = None,  order=4):
    
    from scipy.signal import butter, filtfilt

    nyq = 0.5 * fs
    
    if (not (highcut is None)):
    
        high = highcut / nyq
        low = lowcut / nyq
        b, a = butter(order, [low, high], btype='band')
        y = filtfilt(b, a, data)
        
    else: 
    
        low = lowcut / nyq
        
        b, a = butter(order, low, btype='high')
        y = filtfilt(b, a, data)
        
    return y

def plotsig(signal):
    plt.figure()
    plt.plot(signal)
    
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y
    
    
def nextfastpower (n, base=2.0):
    """Return the next integral power of two greater than the given number.
    Specifically, return m such that
        m >= n
        m == 2**x
    where x is an integer. Use base argument to specify a base other than 2.
    This is useful for ensuring fast FFT sizes.
    """
    x = base**ceil (log (n) / log (base))
    if type(n) == np.ndarray:
        return np.asarray (x, dtype=int)
    else:
        return int (x)
    
    
def signal_envelope1D(data, *, sigma=None, fs=None):
    """Docstring goes here

    TODO: this is not yet epoch-aware!

    sigma = 0 means no smoothing (default 4 ms)
    """

    if sigma is None:
        sigma = 0.004   # 4 ms standard deviation
    if fs is None:
        if isinstance(data, (np.ndarray, list)):
            raise ValueError("sampling frequency must be specified!")
        elif isinstance(data, core.AnalogSignalArray):
            fs = data.fs

    if isinstance(data, (np.ndarray, list)):
        # Compute number of samples to compute fast FFTs
        padlen = nextfastpower(len(data)) - len(data)
        # Pad data
        paddeddata = np.pad(data, (0, padlen), 'constant')
        # Use hilbert transform to get an envelope
        envelope = np.absolute(hilbert(paddeddata))
        # Truncate results back to original length
        envelope = envelope[:len(data)]
        if sigma:
            # Smooth envelope with a gaussian (sigma = 4 ms default)
            EnvelopeSmoothingSD = sigma*fs
            smoothed_envelope = scipy.ndimage.filters.gaussian_filter1d(envelope, EnvelopeSmoothingSD, mode='constant')
            envelope = smoothed_envelope
    elif isinstance(data, core.AnalogSignalArray):
        newasa = copy.deepcopy(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cum_lengths = np.insert(np.cumsum(data.lengths), 0, 0)

        # for segment in data:
        for idx in range(data.n_epochs):
            # print('hilberting epoch {}/{}'.format(idx+1, data.n_epochs))
            segment_data = data._ydata[:,cum_lengths[idx]:cum_lengths[idx+1]]
            n_signals, n_samples = segment_data.shape
            assert n_signals == 1, 'only 1D signals supported!'
            # Compute number of samples to compute fast FFTs:
            padlen = nextfastpower(n_samples) - n_samples
            # Pad data
            paddeddata = np.pad(segment_data.squeeze(), (0, padlen), 'constant')
            # Use hilbert transform to get an envelope
            envelope = np.absolute(hilbert(paddeddata))
            # free up memory
            del paddeddata
            # Truncate results back to original length
            envelope = envelope[:n_samples]
            if sigma:
                # Smooth envelope with a gaussian (sigma = 4 ms default)
                EnvelopeSmoothingSD = sigma*fs
                smoothed_envelope = scipy.ndimage.filters.gaussian_filter1d(envelope, EnvelopeSmoothingSD, mode='constant')
                envelope = smoothed_envelope
            newasa._ydata[:,cum_lengths[idx]:cum_lengths[idx+1]] = np.atleast_2d(envelope)
        return newasa
    return envelope 



def homomorphic_envelope(x, fs, f_LPF=4, order=3):
    from scipy.signal import butter, filtfilt
    """
    
    Computes the homomorphic envelope of x

    Args:
        x : array
        fs : float
            Sampling frequency. Defaults to 1000 Hz
        f_LPF : float
            Lowpass frequency, has to be f_LPF < fs/2. Defaults to 8 Hz
    Returns:
        time : numpy array
    """
    b, a = butter(order, 2 * f_LPF / fs, 'low')
    he = np.exp(filtfilt(b, a, np.log(np.abs(hilbert(x)))))
    return he 


def signal_envelope_triang(series):
    from scipy import signal
    dsrate=1
    nfir=6
    nfir2=8
    bhp=-(np.ones((1,nfir))/nfir)
    bhp[0]=bhp[0]+1
    blp=signal.triang(nfir2)
    final_filterlength=np.round(51/dsrate)
    finalmask=signal.triang(final_filterlength)
    series_filt_hp=signal.lfilter(bhp[0,:],1, series)
    series_filt_lp=signal.lfilter(blp,1, series_filt_hp)
    series_env=signal.lfilter(finalmask,1,np.abs(series_filt_lp))

    
    return series_env



def ampd(sigInput, LSMlimit = 1):
	"""Find the peaks in the signal with the AMPD algorithm.
	
		Original implementation by Felix Scholkmann et al. in
		"An Efficient Algorithm for Automatic Peak Detection in 
		Noisy Periodic and Quasi-Periodic Signals", Algorithms 2012,
		 5, 588-603
		Parameters
		----------
		sigInput: ndarray
			The 1D signal given as input to the algorithm
		lsmLimit: float
			Wavelet transform limit as a ratio of full signal length.
			Valid values: 0-1, the LSM array will no longer be calculated after this point
			  which results in the inability to find peaks at a scale larger than this factor.
			  For example a value of .5 will be unable to find peaks that are of period 
			  1/2 * signal length, a default value of 1 will search all LSM sizes.
		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput
	"""
		
	# Create preprocessing linear fit	
	sigTime = np.arange(0, len(sigInput))
	
	# Detrend
	dtrSignal = (sigInput - np.polyval(np.polyfit(sigTime, sigInput, 1), sigTime)).astype(float)
	
	N = len(dtrSignal)
	L = int(np.ceil(N*LSMlimit / 2.0)) - 1
	
	# Generate random matrix
	LSM = np.ones([L,N], dtype='uint8')
	
	# Local minima extraction
	for k in range(1, L):
		LSM[k - 1, np.where((dtrSignal[k:N - k - 1] > dtrSignal[0: N - 2 * k - 1]) & (dtrSignal[k:N - k - 1] > dtrSignal[2 * k: N - 1]))[0]+k] = 0
	
	pks = np.where(np.sum(LSM[0:np.argmin(np.sum(LSM, 1)), :], 0)==0)[0]
	return pks


# Fast AMPD		
def ampdFast(sigInput, order, LSMlimit = 1):
	"""A slightly faster version of AMPD which divides the signal in 'order' windows
		Parameters
		----------
		sigInput: ndarray
			The 1D signal given as input to the algorithm
		order: int
			The number of windows in which sigInput is divided
		Returns
		-------
		pks: ndarray
			The ordered array of peaks found in sigInput 
	"""

	# Check if order is valid (perfectly separable)
	if(len(sigInput)%order != 0):
		print("AMPD: Invalid order, decreasing order")
		while(len(sigInput)%order != 0):
			order -= 1
		print("AMPD: Using order " + str(order))

	N = int(len(sigInput) / order / 2)

	# Loop function calls
	for i in range(0, len(sigInput)-N, N):
		print("\t sector: " + str(i) + "|" + str((i+2*N-1)))
		pksTemp = ampd(sigInput[i:(i+2*N-1)], LSMlimit)
		if(i == 0):
			pks = pksTemp
		else:
			pks = np.concatenate((pks, pksTemp+i))
		
	# Keep only unique values
	pks = np.unique(pks)
	
	return pks


def matlab_struct_to_dict(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    import scipy
    import numpy as np
    import scipy.io as spio
    from numba.decorators import autojit

    ######################################
    
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    ######################################
    
    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    ######################################
    
    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


    #########################################

    def smqt(y,level,res):

       if level>res:
           y_t=np.zeros(np.size(y))
           return y_t
       mean_y=np.nanmean(y)

       D0=y
       D1=y

       if ~(np.isnan(mean_y)):
           D0=np.where(D0>mean_y, D0, nan)
           D1=np.where(D1<=mean_y, D1, nan)

       y_t=~(np.isnan(D1))*(2**(res-level))

       if (level==res):
           return y_t

       y0=smqt(D0,level+1,res)
       y1=smqt(D1,level+1,res)
       y_t=y_t+y0+y1

       return y_t

        
     ##############################################    LOAD FUNCTION FOR WINDOWS USERS 
def load_data_win(fullpath):
   #  
    fname=os.listdir(fullpath) #list of files
    if fname[0].endswith(".json"):
        #dict_of_data = { i : fname[i] for i in range(0, len(fname) ) }
        dict_of_data = {}
        for item in fname:
            with open(os.path.join(fullpath,item), 'r') as data_file:
                dict_of_data[item] = json.load(data_file) 
                del item
    #Read file 
        dict_of_MEMS_data = {}
        for key in dict_of_data.keys():
        
                MEMS_dict = dict()
        #first check order of sensor data
                if dict_of_data[key][0]['sensordata_set'][0]["sensor_type"] == "GYROSCOPE":
                    accDataIndex = 1
                    gyrDataIndex = 0  
                else:
                    accDataIndex = 0
                    gyrDataIndex = 1
            
                #Get timestamp information
                accTS = array('Q') #unsigned long long, 8 bytes
                accTS.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['timestamp']))
                print('Accelerator event count:'+str(len(accTS)))
                gyrTS = array('Q') #unsigned long long, 8 bytes
                gyrTS.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['timestamp']))
                print('Gyroscope event count:'+str(len(gyrTS)))
            
                #Get accelerometer data
                pprint(dict_of_data[key][0]['sensordata_set'][accDataIndex]['sensor_type'])
                accX = array('f') #unsigned long long, 8 bytes
                accX.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['x']))
                print('Length Accelerator x-axis:'+str(len(accX)))
                accY = array('f') #unsigned long long, 8 bytes
                accY.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['y']))
                print('Length Accelerator y-axis:'+str(len(accY)))
                accZ = array('f') #unsigned long long, 8 bytes
                accZ.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['z']))
                print('Length Accelerator z-axis:'+str(len(accZ)))
                accelerometer_data_raw = (np.array(accX), np.array(accY), np.array(accZ), np.array(accTS))
        
                #Get gyroscope data
                pprint(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['sensor_type'])
                gyrX = array('f') #unsigned long long, 8 bytes
                gyrX.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['x']))
                print('Length Gyroscope x-axis:'+str(len(gyrX)))
                gyrY = array('f') #unsigned long long, 8 bytes
                gyrY.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['y']))
                print('Length Gyroscope y-axis:'+str(len(gyrY)))
                gyrZ = array('f') #unsigned long long, 8 bytes
                gyrZ.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['z']))
                print('Length Gyroscope z-axis:'+str(len(gyrZ)))
                gyroscope_data_raw = (np.array(gyrX), np.array(gyrY), np.array(gyrZ), np.array(gyrTS))
            
           
        
                MEMS_dict['acc'] = np.transpose(accelerometer_data_raw)
                MEMS_dict['gyro'] = np.transpose(gyroscope_data_raw)
        
                dict_of_MEMS_data[key] = MEMS_dict
                list_of_key_names = list(dict_of_data.keys())     
        
    if fname[0].endswith(".mat"):
        dict_of_data = {}
        for item in range(0,len(fname)):
            #with open(os.path.join(fullpath,fname[item]), 'r') as data_file:
                dict_of_data[fname[item]] = matlab_struct_to_dict(os.path.join(fullpath,fname[item])) 
                
                
    
        dict_of_MEMS_data = {}
        for key in dict_of_data.keys():
            MEMS_dict = dict()

            MEMS_dict['acc'] = dict_of_data[key]['accdata']
            MEMS_dict['gyro'] = dict_of_data[key]['gyrodata']
    
            dict_of_MEMS_data[key] = MEMS_dict
               
            list_of_key_names = list(dict_of_data.keys()) 

    return dict_of_MEMS_data,list_of_key_names


###################################### LOAD FUNCTION FOR MAC USERS

def load_data_mac(fullpath):
   #  
    fname=os.listdir(fullpath)
    if fname[0].endswith(".json"):
        extension="*.json"
        meas_files = glob.glob(fullpath+extension)
        
        dict_of_data = dict()
    
        for item in meas_files:
            name =  item.split('/')[-1]
            with open(item, 'r') as data_file:
                dict_of_data[name] = json.load(data_file) 
                del item
                del name
    #Read file 
        dict_of_MEMS_data = {}
        for key in dict_of_data.keys():
        
                MEMS_dict = dict()
        #first check order of sensor data
                if dict_of_data[key][0]['sensordata_set'][0]["sensor_type"] == "GYROSCOPE":
                    accDataIndex = 1
                    gyrDataIndex = 0  
                else:
                    accDataIndex = 0
                    gyrDataIndex = 1
            
                #Get timestamp information
                accTS = array('Q') #unsigned long long, 8 bytes
                accTS.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['timestamp']))
                print('Accelerator event count:'+str(len(accTS)))
                gyrTS = array('Q') #unsigned long long, 8 bytes
                gyrTS.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['timestamp']))
                print('Gyroscope event count:'+str(len(gyrTS)))
            
                #Get accelerometer data
                pprint(dict_of_data[key][0]['sensordata_set'][accDataIndex]['sensor_type'])
                accX = array('f') #unsigned long long, 8 bytes
                accX.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['x']))
                print('Length Accelerator x-axis:'+str(len(accX)))
                accY = array('f') #unsigned long long, 8 bytes
                accY.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['y']))
                print('Length Accelerator y-axis:'+str(len(accY)))
                accZ = array('f') #unsigned long long, 8 bytes
                accZ.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][accDataIndex]['z']))
                print('Length Accelerator z-axis:'+str(len(accZ)))
                accelerometer_data_raw = (np.array(accX), np.array(accY), np.array(accZ), np.array(accTS))
        
                #Get gyroscope data
                pprint(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['sensor_type'])
                gyrX = array('f') #unsigned long long, 8 bytes
                gyrX.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['x']))
                print('Length Gyroscope x-axis:'+str(len(gyrX)))
                gyrY = array('f') #unsigned long long, 8 bytes
                gyrY.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['y']))
                print('Length Gyroscope y-axis:'+str(len(gyrY)))
                gyrZ = array('f') #unsigned long long, 8 bytes
                gyrZ.frombytes(base64.b64decode(dict_of_data[key][0]['sensordata_set'][gyrDataIndex]['z']))
                print('Length Gyroscope z-axis:'+str(len(gyrZ)))
                gyroscope_data_raw = (np.array(gyrX), np.array(gyrY), np.array(gyrZ), np.array(gyrTS))
            
           
        
                MEMS_dict['acc'] = np.transpose(accelerometer_data_raw)
                MEMS_dict['gyro'] = np.transpose(gyroscope_data_raw)
        
                dict_of_MEMS_data[key] = MEMS_dict
                list_of_key_names = list(dict_of_data.keys())     
        
    if fname[0].endswith(".mat"):
        extension="*.mat"
        meas_files = glob.glob(fullpath+extension)
        dict_of_data = dict()
    
        for item in meas_files:
            name =  item.split('/')[-1]
            dict_of_data[name] = matlab_struct_to_dict(item) # ['accdata']
            del item
            del name
    
        dict_of_MEMS_data = {}
        for key in dict_of_data.keys():
            MEMS_dict = dict()

            MEMS_dict['acc'] = dict_of_data[key]['accdata']
            MEMS_dict['gyro'] = dict_of_data[key]['gyrodata']
    
            dict_of_MEMS_data[key] = MEMS_dict
               
            list_of_key_names = list(dict_of_data.keys()) 

    return dict_of_MEMS_data,list_of_key_names

######################################################
    
def preprocess_resample(acc_data_raw, gyr_data_raw):
    '''
    Calculate the duration of timestamp vectors
    uses index 10 instead of 0, because there can be significant delays
    in the first samples. First 10 samples are therefore ignored.
    
    '''
    
    sampleinterval = 5*1e6 # data sampled to 200Hz
    
    duration_acc_ns = acc_data_raw[-1,3] - acc_data_raw[10,3]
    duration_gyr_ns = gyr_data_raw[-1,3] - gyr_data_raw[10,3]
    #subtract value at index from each element to cancel possible time delay
    acc_data_time = np.array(acc_data_raw[:,3])
    acc_data_time = acc_data_time - acc_data_time[10]
    gyr_data_time = np.array(gyr_data_raw[:,3])
    gyr_data_time = gyr_data_time - gyr_data_time[10]
    sampling_freq_new = 200 #resampled freq is 200
    
    '''
    Compute interpolation functions for each data axis.
    Note original samples are unevenly spaced. 
    They are evenly spaced after resampling.
    '''
    spl_accx=interpolate.interp1d(acc_data_time[10:], 
                                  np.array(acc_data_raw[10:,0]), kind='nearest')
    spl_accy=interpolate.interp1d(acc_data_time[10:], 
                                  np.array(acc_data_raw[10:,1]), kind='nearest')
    spl_accz=interpolate.interp1d(acc_data_time[10:], 
                                  np.array(acc_data_raw[10:,2]), kind='nearest')
    spl_gyrx=interpolate.interp1d(gyr_data_time[10:], 
                                  np.array(gyr_data_raw[10:,0]), kind='nearest')
    spl_gyry=interpolate.interp1d(gyr_data_time[10:], 
                                  np.array(gyr_data_raw[10:,1]), kind='nearest')
    spl_gyrz=interpolate.interp1d(gyr_data_time[10:], 
                                  np.array(gyr_data_raw[10:,2]), kind='nearest')
    '''
    Create new timebase from 0 to the end of data with time interval 5ms
    '''
    resampled_index_acc = np.arange(0, duration_acc_ns, sampleinterval)
    resampled_index_gyr = np.arange(0, duration_gyr_ns, sampleinterval)
    #Check which one has shorter data vector after resampling
    cutval = min(len(resampled_index_acc), len(resampled_index_gyr))
    #apply interpolation function to new xvalues
    ax = spl_accx(resampled_index_acc)
    ay = spl_accy(resampled_index_acc)
    az = spl_accz(resampled_index_acc)
    gx = spl_gyrx(resampled_index_gyr)
    gy = spl_gyry(resampled_index_gyr)
    gz = spl_gyrz(resampled_index_gyr)
    '''
    Finally make both data equal in size.
    There is no need to create time vectors because 
    the step size is always 5ms (200Hz)
    '''
    ax = ax[0:cutval]
    ay = ay[0:cutval]
    az = az[0:cutval]
    gx = gx[0:cutval]
    gy = gy[0:cutval]
    gz = gz[0:cutval]
    #uncomment plt-lines to compare the resulting data to original
    #plt.figure(kk+20)
    #plt.plot(acc_data_time[1:], np.array(acc_data_raw[2][1:]), '-g*', 
    #         resampled_index_acc[0:cutval], az[0:cutval], '-bo' )
    rv = {
        'ax': ax,
        'ay': ay,
        'az': az,
        'gx': gx,
        'gy': gy,
        'gz': gz,
        'samplingFreq': float(sampling_freq_new)
    }

    return rv

def get_time_domain_features(nn_intervals: List[float]) -> dict:
    
    """
    Returns a dictionary containing time domain features for HRV analysis.
    Mostly used on long term recordings (24h) but some studies use some of those features on
    short term recordings, from 1 to 5 minutes window.

    Parameters
    ----------
    nn_intervals : list
        list of Normal to Normal Interval

    Returns
    -------
    time_domain_features : dict
        dictionary containing time domain features for HRV analyses. There are details
        about each features below.

    Notes
    -----
    Here are some details about feature engineering...

    - **mean_nni**: The mean of RR-intervals.

    - **sdnn** : The standard deviation of the time interval between successive normal heart beats \
    (i.e. the RR-intervals).

    - **sdsd**: The standard deviation of differences between adjacent RR-intervals

    - **rmssd**: The square root of the mean of the sum of the squares of differences between \
    adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV \
    (*i.e.*, those influencing larger changes from one beat to the next).

    - **median_nni**: Median Absolute values of the successive differences between the RR-intervals.

    - **nni_50**: Number of interval differences of successive RR-intervals greater than 50 ms.

    - **pnni_50**: The proportion derived by dividing nni_50 (The number of interval differences \
    of successive RR-intervals greater than 50 ms) by the total number of RR-intervals.

    - **nni_20**: Number of interval differences of successive RR-intervals greater than 20 ms.

    - **pnni_20**: The proportion derived by dividing nni_20 (The number of interval differences \
    of successive RR-intervals greater than 20 ms) by the total number of RR-intervals.

    - **range_nni**: difference between the maximum and minimum nn_interval.

    - **cvsd**: Coefficient of variation of successive differences equal to the rmssd divided by \
    mean_nni.

    - **cvnni**: Coefficient of variation equal to the ratio of sdnn divided by mean_nni.

    - **mean_hr**: The mean Heart Rate.

    - **max_hr**: Max heart rate.

    - **min_hr**: Min heart rate.

    - **std_hr**: Standard deviation of heart rate.

    References
    ----------
    .. [1] Heart rate variability - Standards of measurement, physiological interpretation, and \
    clinical use, Task Force of The European Society of Cardiology and The North American Society \
    of Pacing and Electrophysiology, 1996
    """

    diff_nni = np.diff(nn_intervals)
    length_int = len(nn_intervals)

    # Basic statistics
    mean_nni = np.mean(nn_intervals)
    median_hrv =abs(np.median(diff_nni))
    range_nni = max(nn_intervals) - min(nn_intervals)

    sdsd = np.std(diff_nni)
    rmssd = np.sqrt(np.mean(diff_nni ** 2))

    nni_50 = sum(np.abs(diff_nni) > 50)
    pnni_50 = 100 * nni_50 / length_int
    nni_20 = sum(np.abs(diff_nni) > 20)
    pnni_20 = 100 * nni_20 / length_int

    # Feature found on github and not in documentation
    cvsd = rmssd / mean_nni

    # Features only for long term recordings
    sdnn = np.std(nn_intervals, ddof=1)  # ddof = 1 : unbiased estimator => divide std by n-1
    cvnni = sdnn / mean_nni

    # Heart Rate equivalent features
    heart_rate_list = np.divide(60000, nn_intervals)
    mean_hr = np.mean(heart_rate_list)
    min_hr = min(heart_rate_list)
    max_hr = max(heart_rate_list)
    std_hr = np.std(heart_rate_list)

    time_domain_features = {
        'mean_nni': mean_nni,
        'sdnn': sdnn,
        'sdsd': sdsd,
        'nni_50': nni_50,
        'pnni_50': pnni_50,
        'nni_20': nni_20,
        'pnni_20': pnni_20,
        'rmssd': rmssd,
        'median_hrv': median_hrv,
        'range_nni': range_nni,
        'cvsd': cvsd,
        'cvnni': cvnni,
        'mean_hr': mean_hr,
        "max_hr": max_hr,
        "min_hr": min_hr,
        "std_hr": std_hr,
    }

    return time_domain_features

def heartbeatdetect(data):
    """
     Returns a dictionary containing time domain features for HRV analysis.
    
    Parameters
    ----------
    data : dictionary of 3 axis accelerometer and gyroscope signals

    Returns
    -------
    time_domain_features : dict
        dictionary containing time domain features for HRV analyses. There are details
        about each features in the get_time_domain_features function.

    
    """
    ##################################BANDPASS FILTERING
    fs=200
    lfc=10
    hfc=70
    sig_gyroX_filt = butter_bandpass_filter(data['gx'][5*fs:fs*55], lowcut = lfc, highcut = hfc, fs = fs)
    sig_gyroY_filt = butter_bandpass_filter(data['gy'][5*fs:fs*55], lowcut = lfc, highcut = hfc, fs = fs)
    sig_gyroZ_filt = butter_bandpass_filter(data['gz'][5*fs:fs*55], lowcut = lfc, highcut = hfc,fs = fs)
    sig_accX_filt = butter_bandpass_filter(data['ax'][5*fs:fs*55], lowcut = lfc,  highcut = hfc,fs = fs)
    sig_accY_filt = butter_bandpass_filter(data['ay'][5*fs:fs*55], lowcut = lfc, highcut = hfc,fs = fs)
    sig_accZ_filt = butter_bandpass_filter(data['az'][5*fs:fs*55], lowcut = lfc, highcut = hfc,fs = fs)
    
    ##################################### Envelope Extraction 
    env_sig_accX=signal_envelope_triang((sig_accX_filt))
    env_sig_accY=signal_envelope_triang((sig_accY_filt))
    env_sig_accZ=signal_envelope_triang((sig_accZ_filt))
    env_sig_gyroX=signal_envelope_triang((sig_gyroX_filt))
    env_sig_gyroY=signal_envelope_triang((sig_gyroY_filt))
    env_sig_gyroZ=signal_envelope_triang((sig_gyroZ_filt))
    
    ###########################Principal component analysis on the original ACC and GYRO signals  
    IMU_signals=np.stack((env_sig_accX,env_sig_accY,env_sig_accZ,env_sig_gyroX,sig_gyroY_filt)).T
    IMU_signals_norm= StandardScaler().fit_transform(IMU_signals)
    
    pca = PCA(n_components=5)
    principalComponents_IMUs = pca.fit_transform(IMU_signals_norm)
    
    fused_envelope=principalComponents_IMUs[:,0]

    fused_envelope=butter_highpass_filter(fused_envelope, 0.5, fs, order=3)


    ############################ Beat to beat estimation of heart rate
    Peak_indx=ampdFast(fused_envelope, 1, LSMlimit = 1)
    RR_int=diff(Peak_indx)/fs    

    ############################# HRV and Complexity analysis 
    
    time_domain_features = get_time_domain_features(RR_int*1000)
    complexity=get_complexity_features(fused_envelope, fs)
    cplx_feat=pd.DataFrame.from_dict(complexity, orient='index')
    cplx_median=list(np.median(cplx_feat,axis=0))
    tmp=complexity[0]
    key_names=list(tmp.keys())
    
    complexity_features=dict(zip(key_names, cplx_median))
    
    return time_domain_features,complexity_features


##################################################### PLOT CONFUSION MATRIX
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def get_complexity_features(signal, fs):
    
    K = 1500                     #length of each segment
    N =int( np.floor(len(signal) / K) )#number of segments
    M=N*K
    rdm_seg = np.reshape(np.array(signal[0:M,]), (N,K)) # reshape, each column contains K consecutive samples
    
    complexity_features=dict()
    
    for i in range(rdm_seg.shape[0]):
        complexity_features[i]=nk.complexity(rdm_seg[i], sampling_rate=fs, shannon=True, sampen=False,
        multiscale=False, spectral=True, svd=True, correlation=True, higushi=True, petrosian=True, 
        fisher=True, hurst=True, dfa=True, lyap_r=False, lyap_e=False, emb_dim=2, 
        tolerance='default', k_max=8, bands=None, tau=1)
    
    return complexity_features

###################################### SAVING Python dictionary files
import h5py
def save_dict_to_hdf5(dic, filename):
    """
    ....
    """
    with h5py.File(filename, 'w') as h5file:
        recursively_save_dict_contents_to_group(h5file, '/', dic)

def recursively_save_dict_contents_to_group(h5file, path, dic):
    """
    ....
    """
    for key, item in dic.items():
        if isinstance(item, (np.ndarray, np.int64, np.float64, str, bytes)):
            h5file[path + key] = item
        elif isinstance(item, dict):
            recursively_save_dict_contents_to_group(h5file, path + key + '/', item)
        else:
            raise ValueError('Cannot save %s type'%type(item))

def load_dict_from_hdf5(filename):
    """
    ....
    """
    with h5py.File(filename, 'r') as h5file:
        return recursively_load_dict_contents_from_group(h5file, '/')

def recursively_load_dict_contents_from_group(h5file, path):
    """
    ....
    """
    ans = {}
    for key, item in h5file[path].items():
        if isinstance(item, h5py._hl.dataset.Dataset):
            ans[key] = item.value
        elif isinstance(item, h5py._hl.group.Group):
            ans[key] = recursively_load_dict_contents_from_group(h5file, path + key + '/')
    return ans

