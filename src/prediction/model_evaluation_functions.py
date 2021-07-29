#!/usr/bin/python

import numpy as np
import pdb
import sys, os, re, time

###############################
### Prediction metrics
###############################
def my_mean_squared_error(y_hat, y_true, multioutput='', relative=False):
    if relative:
        this_loss = np.power((y_hat-y_true)/y_true, 2)
    else:
        this_loss = np.power((y_hat-y_true), 2)
    if multioutput=='raw_values':
        this_loss_mean=np.nanmean(this_loss,axis=0)
    else:
        this_loss_mean=np.nanmean(this_loss)
    return this_loss_mean
    
def my_median_squared_error(y_hat, y_true, multioutput='', relative=False):
    if relative:
        this_loss = np.power((y_hat-y_true)/y_true, 2)
    else:
        this_loss = np.power((y_hat-y_true), 2)
    if multioutput=='raw_values':
        this_loss_median=np.nanmedian(this_loss,axis=0)
    else:
        this_loss_median=np.nanmedian(this_loss)
    return this_loss_median
    
def my_mean_absolute_error(y_hat, y_true, multioutput='', relative=False):
    if relative:
        this_loss = np.abs((y_hat-y_true)/y_true)
    else:
        this_loss = np.abs(y_hat-y_true)
    if multioutput=='raw_values':
        this_loss_mean=np.nanmean(this_loss,axis=0)
    else:
        this_loss_mean=np.nanmean(this_loss)
    return this_loss_mean
    
def my_median_absolute_error(y_hat, y_true, multioutput='', relative=False):
    if relative:
        this_loss = np.abs((y_hat-y_true)/y_true)
    else:
        this_loss = np.abs(y_hat-y_true)
    if multioutput=='raw_values':
        this_loss_median=np.nanmedian(this_loss,axis=0)
    else:
        this_loss_median=np.nanmedian(this_loss)
    return this_loss_median

###############################
### Calibration metrics
###############################
### Data predictive likelihood
def my_likelihood(y_pmf, y_true, average=False, cumulative=False):
    # y_pmf should have y coordinate in last dimension, so:
    assert y_pmf.shape[0]==y_true.size
    # y_range
    y_range=np.arange(y_pmf.shape[-1])
    y_after_day_range=(y_true>y_range.max())
    y_true[y_after_day_range]=y_range.max()
    # Likelihood given pmf
    lik=y_pmf[np.arange(y_true.size),...,y_true]
    # Zero probability to those outside range
    lik[y_after_day_range,:]=0
    
    # Average over first axes
    if average:
        lik=np.nanmean(lik,axis=0)
    # Or cumulative over first axes
    elif cumulative:
        lik=np.nanprod(lik,axis=0)
    # Return
    return lik
    
def my_loglikelihood(y_pmf, y_true, average=False, cumulative=False):
    # loglikelihood
    loglik=np.log(my_likelihood(y_pmf, y_true))
    
    # Average over first axes
    if average:
        loglik=np.nanmean(loglik,axis=0)
    # Or cumulative over first axes
    elif cumulative:
        loglik=np.nansum(loglik,axis=0)
    # Return
    return loglik

### R^2
def my_r2(y_true, y_predicted, axis=0):
    numerator = np.nansum(np.power(y_true - y_predicted, 2), axis=axis, dtype=np.float64)
    denominator = np.nansum(np.power(y_true - np.nanmean(y_true,axis=axis), 2), axis=axis, dtype=np.float64)
    
    return 1 - (numerator/denominator)

def my_mean_r2(y_pmf, y_true, average=True):
    # y_pmf should have y coordinate in last dimension, so:
    assert y_pmf.shape[0]==y_true.size
    # Y and day range
    y_range=np.arange(y_pmf.shape[-1])[None,None,:]
    day_range=np.arange(y_pmf.shape[1])
    
    # Predict
    y_mean=(y_range*y_pmf).sum(axis=-1)

    # Remove 'unreasonable' predictions
    y_true_per_day=(y_true[:,None] * np.ones((1,day_range.size), dtype=float))
    y_true_per_day[day_range[None,:]>y_true[:,None]]=np.nan
    y_mean[day_range[None,:]>y_true[:,None]]=np.nan
    
    # Compute r2: average over axis
    r2=my_r2(y_true_per_day, y_mean, axis=0)
    
    # Return
    return r2
    
def my_mode_r2(y_pmf, y_true, average=True):
    # y_pmf should have y coordinate in last dimension, so:
    assert y_pmf.shape[0]==y_true.size
    # Y and day range
    y_range=np.arange(y_pmf.shape[-1])[None,None,:]
    day_range=np.arange(y_pmf.shape[1])
    
    # Predict
    y_mode=np.argmax(y_pmf, axis=2).astype(dtype=float)

    # Remove 'unreasonable' predictions
    y_true_per_day=(y_true[:,None] * np.ones((1,day_range.size), dtype=float))
    y_true_per_day[day_range[None,:]>y_true[:,None]]=np.nan
    y_mode[day_range[None,:]>y_true[:,None]]=np.nan

    # Compute r2: average over axis
    r2=my_r2(y_true_per_day, y_mode, axis=0)
    
    # Return
    return r2

### Calibration visualizations
# As per https://sites.stat.washington.edu/raftery/Research/PDF/Gneiting2007jrssb.pdf

# Probability integral transform (PIT) 
def my_pit_plot(y_pmf, y_true, plot_filename=None):
    # p_t=y_CDF(y_true)
    y_range=np.arange(y_pmf.shape[-1])
    y_after_day_range=(y_true>y_range.max())
    # If y_true is longer than y_range, just truncate...
    y_true[y_after_day_range]=y_range.max()
    return y_pmf.cumsum(axis=-1)[np.arange(y_true.size),...,y_true]    
    
# Marginal calibration plot (MCP)
def my_mcp_plot(y_pmf, y_true):
    # p_t=y_CDF(y_true)
    # q_t=empirical_CDF(y_true)
    y_range=np.arange(y_pmf.shape[-1])
    # MCP
    return y_pmf.cumsum(axis=-1).mean(axis=0) - (y_true[:,None,None]<=y_range).mean(axis=0)
    
# Reliability diagram: UNCLEAR, TODO
# As per https://arxiv.org/pdf/1706.04599.pdf
# Maybe check this: https://www.jstor.org/stable/2987588
def my_reliability_plot(y_pmf, y_true, n_bins=10):
    # Accuracies
    # Average confidence
    pass


### Proper scoring rules: categorical also ordinal variables
# As per https://viterbi-web.usc.edu/~shaddin/cs699fa17/docs/GR07.pdf
# Note that the following do not account for ordering, i.e., are invariant to permutations!
# Thee are not sensitive to distance, 
#   meaning that no credit is given for assigning high probabilities 
#   to values near but not identical to the one materializing

# Logarithmic score: just my_loglikelihood above

# Brier score
def my_brier_score(y_pmf, y_true, average=False):
    # Brier
    brier=2*my_likelihood(y_pmf, y_true) \
            - np.power(y_pmf,2).sum(axis=-1) - 1
            
    # Average over first axes
    if average:
        brier=np.nanmean(brier,axis=0)
    # Return
    return brier
    
# Spherical score
def my_spherical_score(y_pmf, y_true, average=False, alpha=2):
    # Spherical
    spherical=(
                np.power(my_likelihood(y_pmf, y_true), alpha-1)
                ) / (
                np.power( 
                    np.power(y_pmf,alpha).sum(axis=-1),
                    (alpha-1)/alpha
                    )
                )
            
    # Average over first axes
    if average:
        spherical=np.nanmean(spherical,axis=0)
    # Return
    return spherical
    
# Zero-one (misclassification) score 
def my_zeroone_score(y_pmf, y_true, average=False):
    # Zero-one based on mode
    zeroone=np.argmax(y_pmf, axis=2)==y_true[:,None]
            
    # Average over first axes
    if average:
        zeroone=np.nanmean(zeroone,axis=0)
    # Return
    return zeroone

# Zero-one (misclassification) range score 
def my_zeroone_inrange_score(y_pmf, y_true, day_range=1, average=False):
    # Mode
    y_mode=np.argmax(y_pmf, axis=2)
    # Zero-one based on true within mode +/- day_range
    zeroone_inrange = (y_true[:,None]>=(y_mode-day_range)) & (y_true[:,None]<=(y_mode+day_range))
            
    # Average over first axes
    if average:
        zeroone_inrange=np.nanmean(zeroone_inrange,axis=0)
    # Return
    return zeroone_inrange
    
### Proper scoring rules: continuous variables
# As per https://viterbi-web.usc.edu/~shaddin/cs699fa17/docs/GR07.pdf

# Continuous ranked probability score (CRPS)
def my_crps(y_pmf, y_true, average=False):
    # CDF over y_range
    y_range=np.arange(y_pmf.shape[-1])
    y_cdf=y_pmf.cumsum(axis=-1)
    y_ecdf=y_range[None,None,:]>=y_true[:,None,None]
    # crps
    crps=-np.power(y_cdf-y_ecdf,2).sum(axis=-1)
            
    # Average over first axes
    if average:
        crps=np.nanmean(crps,axis=0)
    # Return
    return crps

# Interval width
# As per https://viterbi-web.usc.edu/~shaddin/cs699fa17/docs/GR07.pdf   
def my_interval_width(y_pmf, y_true, alpha=0.5, average=False):
    # CDF over y_range
    y_range=np.arange(y_pmf.shape[-1])
    y_cdf=y_pmf.cumsum(axis=-1)
    
    # Quantiles
    # l is the alpha/2 quantile
    l_idx=(y_cdf<=alpha/2).sum(axis=-1)-1
    l=y_range[l_idx]
    # u is the (1-alpha/2) quantile
    u_idx=(y_cdf<=(1-alpha/2)).sum(axis=-1)-1
    u=y_range[u_idx]
    
    # Interval width
    int_width=(u-l)
               
    # Average over first axes
    if average:
        int_width=np.nanmean(int_width,axis=0)
    # Return
    return int_width
    
# Interval coverage
# As per https://viterbi-web.usc.edu/~shaddin/cs699fa17/docs/GR07.pdf   
def my_interval_coverage(y_pmf, y_true, alpha=0.5, average=False):
    # CDF over y_range
    y_range=np.arange(y_pmf.shape[-1])
    y_cdf=y_pmf.cumsum(axis=-1)
    
    # Quantiles
    # l is the alpha/2 quantile
    l_idx=(y_cdf<=alpha/2).sum(axis=-1)-1
    l=y_range[l_idx]
    # u is the (1-alpha/2) quantile
    u_idx=(y_cdf<=(1-alpha/2)).sum(axis=-1)-1
    u=y_range[u_idx]
    
    # Coverage
    int_coverage=(y_true[:,None]>=l) & (y_true[:,None]<=u)
               
    # Average over first axes
    if average:
        int_coverage=np.nanmean(int_coverage,axis=0)
    # Return
    return int_coverage
    
# Interval score
# As per https://viterbi-web.usc.edu/~shaddin/cs699fa17/docs/GR07.pdf   
def my_interval_score(y_pmf, y_true, alpha=0.5, average=False):
    # CDF over y_range
    y_range=np.arange(y_pmf.shape[-1])
    y_cdf=y_pmf.cumsum(axis=-1)
    
    # Quantiles
    # l is the alpha/2 quantile
    l_idx=(y_cdf<=alpha/2).sum(axis=-1)-1
    l=y_range[l_idx]
    # u is the (1-alpha/2) quantile
    u_idx=(y_cdf<=(1-alpha/2)).sum(axis=-1)-1
    u=y_range[u_idx]
    
    # Interval score
    int_score=(u-l) \
                + 2/alpha*(l-y_true[:,None]) * (y_true[:,None]<l) \
                + 2/alpha*(y_true[:,None]-u) * (y_true[:,None]>u)
               
    # Average over first axes
    if average:
        int_score=np.nanmean(int_score,axis=0)
    # Return (negative)
    return -int_score
