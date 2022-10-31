#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:25:12 2019

@author: Mojtaba Jafaritadi
"""
import numpy as np

def butterworth_filter(data,  fs, lowcut, highcut = None,  order=4):
    
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


def preprocess_resample(acc_data_raw, gyr_data_raw):
    from scipy import interpolate

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
                                  np.array(acc_data_raw[10:,0]), kind='cubic')
    spl_accy=interpolate.interp1d(acc_data_time[10:], 
                                  np.array(acc_data_raw[10:,1]), kind='cubic')
    spl_accz=interpolate.interp1d(acc_data_time[10:], 
                                  np.array(acc_data_raw[10:,2]), kind='cubic')
    spl_gyrx=interpolate.interp1d(gyr_data_time[10:], 
                                  np.array(gyr_data_raw[10:,0]), kind='cubic')
    spl_gyry=interpolate.interp1d(gyr_data_time[10:], 
                                  np.array(gyr_data_raw[10:,1]), kind='cubic')
    spl_gyrz=interpolate.interp1d(gyr_data_time[10:], 
                                  np.array(gyr_data_raw[10:,2]), kind='cubic')
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
    
def heartbeatdetector(data):
    """
     This function find the position of heartbeats using PCA sensor fusion.
    
    Parameters
    ----------
    data : dictionary of 3 axis accelerometer and gyroscope signals

    Returns
    -------
    A new pulsatile shaped signal is generated along with the detected peaks 

    
    """
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    ##################################BANDPASS FILTERING
    fs=200
    lfc=10
    hfc=70
    sig_gyroX_filt = butterworth_filter(data['gx'][5*fs:fs*65], lowcut = lfc, highcut = hfc, fs = fs)
    sig_gyroY_filt = butterworth_filter(data['gy'][5*fs:fs*65], lowcut = lfc, highcut = hfc, fs = fs)
    sig_gyroZ_filt = butterworth_filter(data['gz'][5*fs:fs*65], lowcut = lfc, highcut = hfc,fs = fs)
    sig_accX_filt = butterworth_filter(data['ax'][5*fs:fs*65], lowcut = lfc,  highcut = hfc,fs = fs)
    sig_accY_filt = butterworth_filter(data['ay'][5*fs:fs*65], lowcut = lfc, highcut = hfc,fs = fs)
    sig_accZ_filt = butterworth_filter(data['az'][5*fs:fs*65], lowcut = lfc, highcut = hfc,fs = fs)
    
    ##################################### Envelope Extraction from all axes
    env_sig_accX=signal_envelope_triang((sig_accX_filt))
    env_sig_accY=signal_envelope_triang((sig_accY_filt))
    env_sig_accZ=signal_envelope_triang((sig_accZ_filt))
    env_sig_gyroX=signal_envelope_triang((sig_gyroX_filt))
    env_sig_gyroY=signal_envelope_triang((sig_gyroY_filt))
    env_sig_gyroZ=signal_envelope_triang((sig_gyroZ_filt))
    
    ########################### Principal component analysis on the original ACC and GYRO signals  
    IMU_signals=np.stack((env_sig_accX,env_sig_accY,env_sig_accZ,env_sig_gyroX,env_sig_gyroY)).T
    IMU_signals_norm= StandardScaler().fit_transform(IMU_signals)
    
    pca = PCA(n_components=5)
    principalComponents_IMUs = pca.fit_transform(IMU_signals_norm)
    
    fused_envelope=principalComponents_IMUs[:,0]

    fused_envelope=butterworth_filter(fused_envelope, lowcut=0.5, highcut= None, fs=fs)


    ############################ Beat to beat estimation of heart rate
    Peak_indx=ampdFast(fused_envelope, 1, LSMlimit = 1)
#    RR_int=diff(Peak_indx)/fs    

    ############################# HRV and Complexity analysis 
    
#    time_domain_features = get_time_domain_features(RR_int*1000)
#    complexity=get_complexity_features(fused_envelope, fs)
#    cplx_feat=pd.DataFrame.from_dict(complexity, orient='index')
#    cplx_median=list(np.median(cplx_feat,axis=0))
#    tmp=complexity[0]
#    key_names=list(tmp.keys())
    
#    complexity_features=dict(zip(key_names, cplx_median))
    
    return fused_envelope,Peak_indx

def calculate_heartrate(Peak_indx):
    fs=200
    RR_int=1./np.diff(Peak_indx)*fs
    HR=np.mean(60/RR_int)
    
    return HR