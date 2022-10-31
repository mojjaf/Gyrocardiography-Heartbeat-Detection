#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:06:57 2019

@author: Mojtaba Jafaritadi
"""

def get_complexity_features(signal, fs):
    import neurokit as nk
    import numpy as np
    import pandas as pd
    
    """
    Computes several chaos/complexity indices of a signal (including entropy, fractal dimensions, Hurst and Lyapunov exponent etc.).

    Parameters
    ----------
    signal : list or array
        List or array of values.
    sampling_rate : int
        Sampling rate (samples/second).
    shannon : bool
        Computes Shannon entropy.
    sampen : bool
        Computes approximate sample entropy (sampen) using Chebychev and Euclidean distances.
    multiscale : bool
        Computes multiscale entropy (MSE). Note that it uses the 'euclidean' distance.
    spectral : bool
        Computes Spectral Entropy.
    svd : bool
        Computes the Singular Value Decomposition (SVD) entropy.
    correlation : bool
        Computes the fractal (correlation) dimension.
    higushi : bool
        Computes the Higushi fractal dimension.
    petrosian : bool
        Computes the Petrosian fractal dimension.
    fisher : bool
        Computes the Fisher Information.
    hurst : bool
        Computes the Hurst exponent.
    dfa : bool
        Computes DFA.
    lyap_r : bool
        Computes Positive Lyapunov exponents (Rosenstein et al. (1993) method).
    lyap_e : bool
        Computes Positive Lyapunov exponents (Eckmann et al. (1986) method).
    emb_dim : int
        The embedding dimension (*m*, the length of vectors to compare). Used in sampen, fisher, svd and fractal_dim.
    tolerance : float
        Distance *r* threshold for two template vectors to be considered equal. Default is 0.2*std(signal). Used in sampen and fractal_dim.
    k_max : int
        The maximal value of k used for Higushi fractal dimension. The point at which the FD plateaus is considered a saturation point and that kmax value should be selected (Gómez, 2009). Some studies use a value of 8 or 16 for ECG signal and other 48 for MEG.
    bands : int
        Used for spectral density. A list of numbers delimiting the bins of the frequency bands. If None the entropy is computed over the whole range of the DFT (from 0 to `f_s/2`).
    tau : int
        The delay. Used for fisher, svd, lyap_e and lyap_r.

    Returns
    ----------
    complexity : dict
        Dict containing values for each indices.


    Example
    ----------
    >>> import neurokit as nk
    >>> import numpy as np
    >>>
    >>> signal = np.sin(np.log(np.random.sample(666)))
    >>> complexity = nk.complexity(signal)

    Notes
    ----------
    *Details*

    - **Entropy**: Entropy is a measure of unpredictability of the state, or equivalently, of its average information content.

      - *Shannon entropy*: Shannon entropy was introduced by Claude E. Shannon in his 1948 paper "A Mathematical Theory of Communication". Shannon entropy provides an absolute limit on the best possible average length of lossless encoding or compression of an information source.
      - *Sample entropy (sampen)*: Measures the complexity of a time-series, based on approximate entropy. The sample entropy of a time series is defined as the negative natural logarithm of the conditional probability that two sequences similar for emb_dim points remain similar at the next point, excluding self-matches. A lower value for the sample entropy therefore corresponds to a higher probability indicating more self-similarity.
      - *Multiscale entropy*: Multiscale entropy (MSE) analysis is a new method of measuring the complexity of finite length time series.
      - *SVD Entropy*: Indicator of how many vectors are needed for an adequate explanation of the data set. Measures feature-richness in the sense that the higher the entropy of the set of SVD weights, the more orthogonal vectors are required to adequately explain it.

    - **fractal dimension**: The term *fractal* was first introduced by Mandelbrot in 1983. A fractal is a set of points that when looked at smaller scales, resembles the whole set. The concept of fractak dimension (FD) originates from fractal geometry. In traditional geometry, the topological or Euclidean dimension of an object is known as the number of directions each differential of the object occupies in space. This definition of dimension works well for geometrical objects whose level of detail, complexity or *space-filling* is the same. However, when considering two fractals of the same topological dimension, their level of *space-filling* is different, and that information is not given by the topological dimension. The FD emerges to provide a measure of how much space an object occupies between Euclidean dimensions. The FD of a waveform represents a powerful tool for transient detection. This feature has been used in the analysis of ECG and EEG to identify and distinguish specific states of physiologic function. Many algorithms are available to determine the FD of the waveform (Acharya, 2005).

      - *Correlation*: A measure of the fractal (or correlation) dimension of a time series which is also related to complexity. The correlation dimension is a characteristic measure that can be used to describe the geometry of chaotic attractors. It is defined using the correlation sum C(r) which is the fraction of pairs of points X_i in the phase space whose distance is smaller than r.
      - *Higushi*: Higuchi proposed in 1988 an efficient algorithm for measuring the FD of discrete time sequences. As the reconstruction of the attractor phase space is not necessary, this algorithm is simpler and faster than D2 and other classical measures derived from chaos theory. FD can be used to quantify the complexity and self-similarity of a signal. HFD has already been used to analyse the complexity of brain recordings and other biological signals.
      - *Petrosian Fractal Dimension*: Provide a fast computation of the FD of a signal by translating the series into a binary sequence.

    - **Other**:

      - *Fisher Information*:  A way of measuring the amount of information that an observable random variable X carries about an unknown parameter θ of a distribution that models X. Formally, it is the variance of the score, or the expected value of the observed information.
      - *Hurst*: The Hurst exponent is a measure of the "long-term memory" of a time series. It can be used to determine whether the time series is more, less, or equally likely to increase if it has increased in previous steps. This property makes the Hurst exponent especially interesting for the analysis of stock data.
      - *DFA*: DFA measures the Hurst parameter H, which is very similar to the Hurst exponent. The main difference is that DFA can be used for non-stationary processes (whose mean and/or variance change over time).
      - *Lyap*: Positive Lyapunov exponents indicate chaos and unpredictability. Provides the algorithm of Rosenstein et al. (1993) to estimate the largest Lyapunov exponent and the algorithm of Eckmann et al. (1986) to estimate the whole spectrum of Lyapunov exponents.

    *Authors*

    - Dominique Makowski (https://github.com/DominiqueMakowski)
    - Christopher Schölzel (https://github.com/CSchoel)
    - tjugo (https://github.com/nikdon)
    - Quentin Geissmann (https://github.com/qgeissmann)

    *Dependencies*

    - nolds
    - numpy

    *See Also*

    - nolds package: https://github.com/CSchoel/nolds
    - pyEntropy package: https://github.com/nikdon/pyEntropy
    - pyrem package: https://github.com/gilestrolab/pyrem

    References
    -----------
    - Accardo, A., Affinito, M., Carrozzi, M., & Bouquet, F. (1997). Use of the fractal dimension for the analysis of electroencephalographic time series. Biological cybernetics, 77(5), 339-350.
    - Pierzchalski, M. Application of Higuchi Fractal Dimension in Analysis of Heart Rate Variability with Artificial and Natural Noise. Recent Advances in Systems Science.
    - Acharya, R., Bhat, P. S., Kannathal, N., Rao, A., & Lim, C. M. (2005). Analysis of cardiac health using fractal dimension and wavelet transformation. ITBM-RBM, 26(2), 133-139.
    - Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. American Journal of Physiology-Heart and Circulatory Physiology, 278(6), H2039-H2049.
    - Costa, M., Goldberger, A. L., & Peng, C. K. (2005). Multiscale entropy analysis of biological signals. Physical review E, 71(2), 021906.
    """

    
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
        
    cplx_feat=pd.DataFrame.from_dict(complexity_features, orient='index')   
    cplx_median=list(np.median(cplx_feat,axis=0))
    tmp=complexity_features[0]
    key_names=list(tmp.keys())
    
    complexity_features=dict(zip(key_names, cplx_median))
    
    return complexity_features

