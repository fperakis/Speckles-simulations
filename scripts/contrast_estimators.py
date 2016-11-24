#!/usr/bin/env python

import numpy as np
from scipy import optimize
from scipy.signal import fftconvolve
from scipy.special import gamma, gammaln
from scipy.special import psi as digamma


def analytical_estimator(k0,p1,p2):
    """
    Estimate the contrast based on the toal photon density k0 and the number
    of 1 and 2 photon events (p1,p2) in the low photon count limit.
    """
    contrast =  (k0-2.*(p2/p1)) / (2.*k0*(p2/p1-1./2.))

    return contrast


"""
Negative binomial scripts adapted from TJLANE/SCPECKLES
https://github.com/tjlane/speckle
"""

def fit_negative_binomial(samples, method='ml', limit=1e-4):
    """
    Estimate the parameters of a negative binomial distribution, using either
    a maximum-likelihood fit or an analytic estimate appropriate in the low-
    photon count limit.
    
    Parameters
    ----------
    samples : np.ndarray, int
        An array of the photon counts for each pixel.
    
    method : str, {"ml", "expansion", "lsq"}
        Which method to use to estimate the contrast.

    limit : float
        If method == 'ml', then this sets the lower limit for which contrast
        can be evaluated. Note that the solution can be sensitive to this
        value.
        
    Returns
    -------
    contrast : float
        The contrast of the samples.
        
    sigma_contrast : float
        The first moment of the parameter estimation function.
        
    References
    ----------
    [1] PRL 109, 185502 (2012)
    """
    h = np.bincount(samples)
    return fit_negative_binomial_from_hist(h, method=method, limit=limit)



def fit_negative_binomial_from_hist(empirical_pmf, method='ml', limit=1e-2):
    """
    Just like fit_negative_binomial, but uses the argument `empirical_pmf`,
    simply the empirical distribution of counts. I.e. empirical_pmf[n] should yield
    an integer counting how many samples there were with `n` counts.

    Parameters
    ----------
    
    Returns
    -------
 Also
    --------
    fit_negative_binomial : function
    """
    
    #k = samples.flatten()
    N = float( empirical_pmf.sum() )
    n = np.arange( len(empirical_pmf) )
    k_bar = np.sum(n * empirical_pmf) / N #np.mean(samples)
    pmf = empirical_pmf # just a shorthand
    #print "CORE:  k_bar=", k_bar

    if method == 'ml': # use maximum likelihood estimation

        def logL_prime(contrast):
            M = 1.0 / contrast
            t1 = -N * (np.log(k_bar/M + 1.0) + digamma(M))
            t2 = np.sum( pmf * ((k_bar - n) / (k_bar + M) + digamma(n + M)) )
            return t1 + t2

        try:
            contrast = optimize.brentq(logL_prime, limit, 1.0)
        except ValueError as e:
            print e
            #raise ValueError('log-likelihood function has no maximum given'
            #                 ' the empirical example provided. Please samp'
            #                 'le additional points and try again.')
            contrast = limit
            sigma_contrast = np.inf

            return contrast, sigma_contrast

        def logL_dbl_prime(contrast):
            M = 1.0 / (contrast + 1e-100)
            n = np.arange( len(empirical_pmf) )
            t1 = np.sum( pmf * ((np.square(k_bar) - n*M) / (M * np.square(k_bar + M))) )
            t2 = - N * digamma(M)
            t3 = np.sum( pmf * digamma(n + M) )
            return t1 + t2 + t3

        dbl_prime = logL_dbl_prime(contrast)
        if dbl_prime < 0.0:
            raise RuntimeError('Maximum likelihood optimization found a local '
                               'minimum instead of maximum! sigma = %s' % sigma_contrast)

        # f''(x)_(x=mu) = - ( sqrt{2*pi} * sigma^3 )^{-1} for normal dist
        sigma_contrast = np.power(np.sqrt(2.0*np.pi) * dbl_prime, -1./3.)


    elif method == 'lsq': # use least-squares fit

        k_range = np.arange(len(empirical_pmf))

        err = lambda contrast : negative_binomial_pmf(k_range, k_bar, contrast) - empirical_pmf

        c0 = 0.5
        contrast, success = optimize.leastsq(err, c0)
        if success:
            contrast = contrast[0]
            sigma_contrast = success
        else:
            raise RuntimeError('least-squares fit did not converge.')

    elif method == 'expansion': # use low-order expansion
        # directly from the SI of the paper in the doc string
        p1 = empirical_pmf[1]
        p2 = empirical_pmf[2]
        contrast = (2.0 * p2 * (1.0 - p1) / np.square(p1)) - 1.0

        # this is not quite what they recommend, but it's close...
        # what they recommend is a bit confusing to me atm --TJL
        sigma_contrast = np.power(2.0 * (1.0 + contrast) / N, 0.5) / k_bar

    else:
        raise ValueError('`method` must be one of {"ml", "ls", "expansion"}')

    return contrast, sigma_contrast

def negative_binomial_pmf(k_range, k_bar, contrast):
    """
    Evaluate the negative binomial probablility mass function.
    
    Parameters
    ----------
    k_range : ndarray, int
        The integer values in the domain of the PMF to evaluate.
        
    k_bar : float
        The mean count density. Greater than zero.
        
    contrast : float
        The contrast parameter, in [0.0, 1.0).
        
    Returns
    -------
    pmf : np.ndarray, float
        The normalized pmf for each point in `k_range`.
    """

    if type(k_range) == int:
        k_range = np.arange(k_range)
    elif type(k_range) == np.ndarray:
        pass
    else:
        raise ValueError('invalid type for k_range: %s' % type(k_range))

    M = 1.0 / contrast
    norm = np.exp(gammaln(k_range + M) - gammaln(M) - gammaln(k_range+1))
    f1 = np.power(1.0 + M/k_bar, -k_range)
    f2 = np.power(1.0 + k_bar/M, -M)
    
    return norm * f1 * f2


