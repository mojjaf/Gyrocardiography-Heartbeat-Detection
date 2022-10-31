#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:26:51 2019

@author: Mojtaba Jafaritadi
"""
import numpy as np

def get_autoregressive_feature(Peak_indx,fs):
    
    r"""Estimate the complex autoregressive parameters by the Burg algorithm.

    .. math:: x(n) = \sqrt{(v}) e(n) + \sum_{k=1}^{P+1} a(k) x(n-k)

    :param x:  Array of complex data samples (length N)
    :param order: Order of autoregressive process (0<order<N)
    :param criteria: select a criteria to automatically select the order

    :return:
        * A Array of complex autoregressive parameters A(1) to A(order). First
          value (unity) is not included !!
        * P Real variable representing driving noise variance (mean square
          of residual noise) from the whitening operation of the Burg
          filter.
        * reflection coefficients defining the filter of the model.

    .. plot::
        :width: 80%
        :include-source:

        from pylab import plot, log10, linspace, axis
        from spectrum import *

        AR, P, k = arburg(marple_data, 15)
        PSD = arma2psd(AR, sides='centerdc')
        plot(linspace(-0.5, 0.5, len(PSD)), 10*log10(PSD/max(PSD)))
        axis([-0.5,0.5,-60,0])

    .. note::
        1. no detrend. Should remove the mean trend to get PSD. Be careful if
           presence of large mean.
        2. If you don't know what the order value should be, choose the
           criterion='AKICc', which has the least bias and best
           resolution of model-selection criteria.

    .. note:: real and complex results double-checked versus octave using
        complex 64 samples stored in marple_data. It does not agree with Marple
        fortran routine but this is due to the simplex precision of complex
        data in fortran.

    :reference: [Marple]_ [octave]_
    
    requires spectrum package : pip install spectrum 
    """
    
    
    from spectrum import arburg, arma2psd
    
    RR_int=np.diff(Peak_indx)/fs 
    
    AR, P, k = arburg(RR_int, 5)
    
    estimated_variance = {
        'autoreg_error': P,
    }
    return estimated_variance


    
    