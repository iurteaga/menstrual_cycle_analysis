## Imports/
import sys, os, re, time
import timeit
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import gammaln
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors
# Pytorch
import torch
import torch.nn as nn
import torch.distributions as torch_dist
#from torch.utils.data import * (not working with torch >v1.7)
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

# Add path to our source directory
sys.path.append('../src/prediction')
# Helpful functions
from aux_functions import *
# Import generalized poisson model  
from my_generalized_poisson import *

## Global variables
# Memory allocation limit
max_memory_in_bytes=32*np.power(1024,3) # 32 GiB

# Epsilon for minimum numerical probability
prob_eps=1e-32
# Minus infinity, so that exp(minus_inf)==0
minus_inf=-1e32
# Maximum x_max to use for computation of log_Z (because xis \approx 0 result in x_max \rightarrow \infty)
log_Z_max_x_max=1e3

## Class definition
# Hierarchical Generalized poisson model with latent skipped report variable for observed cycle lengths
class hierarchical_generalized_poisson_with_skipped_cycles_model(nn.Module):
    '''
        Class inheriting from nn.Module
    '''
    # Init module
    def __init__(self, kappa=1, gamma=1, alpha_xi=1, beta_xi=1, alpha_pi=1, beta_pi=1, xi_max=1, x_max=float('inf'), s_max=100, config_file=None):
        assert kappa>0 and gamma>0
        assert alpha_xi>0 and beta_xi>0
        assert alpha_pi>0 and beta_pi>0
        assert xi_max>=-1 and xi_max<=1
        assert x_max>0
        super(hierarchical_generalized_poisson_with_skipped_cycles_model, self).__init__()
        # Keep config file name
        self.config_file=config_file
        
        # Model parameters
        if x_max < float('inf'):
            # Truncation is only possible in underdispersed Generalized Poisson distributions
            self.truncated_underdispersed=True
        else:
            # Keep track of flags
            self.truncated_underdispersed=False
        # xi_max
        self.xi_max = xi_max
        # x_max
        self.x_max=x_max*torch.ones(1)
        
        # Skipped cycle distribution
        # s_max = float(inf) means no truncation
        assert s_max>0
        self.s_max = s_max
        
        # Log parameters for gamma prior
        self.log_kappa=nn.Parameter(data=torch.log(kappa*torch.ones(1)), requires_grad=True)
        self.log_gamma=nn.Parameter(data=torch.log(gamma*torch.ones(1)), requires_grad=True)
        # Log parameters for xi beta prior
        if not self.truncated_underdispersed:
            # Xi is a free parameter if we are not truncating the underdispersed Generalized Poisson
            self.log_alpha_xi=nn.Parameter(data=torch.log(alpha_xi*torch.ones(1)), requires_grad=True)
            self.log_beta_xi=nn.Parameter(data=torch.log(beta_xi*torch.ones(1)), requires_grad=True)
        
        # Log parameters for pi beta prior
        self.log_alpha_pi=nn.Parameter(data=torch.log(alpha_pi*torch.ones(1)), requires_grad=True)
        self.log_beta_pi=nn.Parameter(data=torch.log(beta_pi*torch.ones(1)), requires_grad=True)
        
        # Torch generative model
        self.gamma_dist=torch_dist.gamma.Gamma
        self.beta_dist=torch_dist.beta.Beta
        self.geometric_dist=torch_dist.geometric.Geometric
        self.generalized_poisson_dist=generalized_poisson
        
    # Exponentiate Generalized Poisson prior log-parameters
    def exponentiate_prior_log_params(self):
        # Exponentiate all parameters
        self.kappa=torch.exp(self.log_kappa)
        self.gamma=torch.exp(self.log_gamma)
        if not self.truncated_underdispersed:
            self.alpha_xi=torch.exp(self.log_alpha_xi)
            self.beta_xi=torch.exp(self.log_beta_xi)
        self.alpha_pi=torch.exp(self.log_alpha_pi)
        self.beta_pi=torch.exp(self.log_beta_pi)

    def get_hyperparameters(self, return_limits=False):
        # Return object's hyperparameter attribute values as array
        
        # Make sure they have been exponentiated
        self.exponentiate_prior_log_params()
        if not self.truncated_underdispersed:
            if return_limits:
                u=np.array([
                        self.kappa.detach().numpy()[0],
                        self.gamma.detach().numpy()[0],
                        self.alpha_xi.detach().numpy()[0],
                        self.beta_xi.detach().numpy()[0],
                        self.xi_max,
                        self.x_max.detach().numpy()[0],
                        self.alpha_pi.detach().numpy()[0],
                        self.beta_pi.detach().numpy()[0]
                        ])
            else:
                u=np.array([
                        self.kappa.detach().numpy()[0],
                        self.gamma.detach().numpy()[0],
                        self.alpha_xi.detach().numpy()[0],
                        self.beta_xi.detach().numpy()[0],
                        self.alpha_pi.detach().numpy()[0],
                        self.beta_pi.detach().numpy()[0]
                        ])
                assert np.all(u>0), 'Hyperparameters must be positive!'
        else:
            u=np.array([
                    self.kappa.detach().numpy()[0],
                    self.gamma.detach().numpy()[0],
                    self.alpha_pi.detach().numpy()[0],
                    self.beta_pi.detach().numpy()[0]
                    ])
            assert np.all(u>0), 'Hyperparameters must be positive!'
        
        return u

    # Draw Generalized Poisson parameters
    def draw_params(self, sample_size):
        # Gradients do not propagate via samples: https://pytorch.org/docs/stable/distributions.html
        # NOTE: Draw via reparameterization trick
        
        # Lambda from its own gamma prior distribution
        self.lambdas=self.gamma_dist(self.kappa, self.gamma).rsample([*sample_size]).double()[...,0]
    
        # Xi from beta prior distribution, or set by the underdispersed truncation limit
        if self.truncated_underdispersed:
            # Xi is set to the biggest value that matches x_max
            self.xis=-self.lambdas/(self.x_max+1)
        else:
            # Figure out xi_min limits
            # In general
            self.xi_min=-1*torch.ones(*sample_size)
            # Xi from shifted/scaled beta prior
            self.xis=self.xi_min+(self.xi_max-self.xi_min)*self.beta_dist(self.alpha_xi, self.beta_xi).rsample([*sample_size]).double()[...,0]
        
        # Pi from its own beta prior distribution
        self.pis=self.beta_dist(self.alpha_pi, self.beta_pi).rsample([*sample_size]).double()[...,0]
            
    # Generalized poisson's normalizing log constant, given parameters, and range of skipped cycles to consider
    def compute_log_Z_generalized_poisson(self,s_max=None):
        # Dimensions
        assert self.lambdas.shape == self.xis.shape
        assert self.lambdas.shape == self.pis.shape
        
        # log of Generalized poisson partition, of dimension: I (might be 1) times M times S
        # Truncated range of skipped cycles
        if s_max is None:
            s_max_value=self.s_max # Use initial s_max
        elif torch.is_tensor(s_max):
            s_max_value=s_max.item()
        else:
            s_max_value=s_max
        
        # s+1: (I by M) by S by x_max
        s_plus_1=(torch.arange(s_max_value, dtype=torch.double)+1 if s_max_value>0 else torch.ones(1))[None,:,None]
        # Z=1, log_Z=0
        log_Z=torch.zeros(self.lambdas.shape[0],self.lambdas.shape[1], s_plus_1.shape[1], dtype=torch.double) 
        
        # For those individual/samples that are underdispersed
        if (self.xis<0).any():
            # Need to compute log_Z for underdispersed Generalized Poisson
            log_Z_i, log_Z_m=torch.nonzero(self.xis<0, as_tuple=True) # Indexes of interest
            if s_max_value < float('inf'):                
                # x_max for those i,m of interest
                # Note that this also depends on (s+1), so x_max might become quite big!
                x_max=torch.ceil(-(self.lambdas[log_Z_i, log_Z_m]/self.xis[log_Z_i, log_Z_m])[:,None]*s_plus_1[:,:,0] - 1).long()
                # We nevertheless limit this so that memory request is not unbounded
                truncated_x_max=np.minimum(torch.max(x_max), s_max_value*log_Z_max_x_max)

                # We will sum over a range of x from 0 to x_max
                #   which we allocate here due to pytorch allocation shennanigans:
                #   https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/10
                print('\t\t\tComputing logZ with max(x_max)={} truncated to truncated_x_max={}'.format(torch.max(x_max), truncated_x_max))
                x_range=torch.arange(truncated_x_max+1, dtype=torch.double)[(None,)*2]
                            
                # Memory requirement of log_Z computation in bytes: (selected I times M) times S time x_max
                mem_bytes_needed=np.prod([log_Z_i.size(0),s_plus_1.shape[1],truncated_x_max+1]) * log_Z.element_size() 

                # Number of "splits" needed
                n_splits=np.ceil(mem_bytes_needed/max_memory_in_bytes).astype(np.uint)
                # Resulting split sizes
                assert log_Z_i.size(0) == log_Z_m.size(0)
                assert log_Z_i.size(0) == x_max.size(0)
                split_size=int(log_Z_i.size(0)/n_splits)
                if split_size>0:
                    # Split arguments, across sample axis
                    this_Z_i=torch.split(log_Z_i, split_size)
                    this_Z_m=torch.split(log_Z_m, split_size)
                    this_x_max=torch.split(x_max, split_size)
                    
                    # Splits should match
                    final_n_splits=len(this_Z_i)
                    assert final_n_splits==len(this_Z_m)
                    assert final_n_splits==len(this_x_max)
                    # Iterate over splits
                    for n_split in torch.arange(final_n_splits, dtype=torch.int):
                        print('\t\t\t\t... Log Z computation n_split={}/{} with size={}'.format(n_split, final_n_splits, this_Z_i[n_split].size(0)))                        
                        # Unnecessary terms
                        clamp_mask=(x_range>this_x_max[n_split][:,:,None])
                        
                        # We clamp key term that can not be less than zero (i.e., where x_range>x_max)
                        must_be_positive_term=torch.clamp(
                                                s_plus_1*self.lambdas[this_Z_i[n_split],this_Z_m[n_split]][:,None,None]
                                                + self.xis[this_Z_i[n_split],this_Z_m[n_split]][:,None,None]*x_range
                                                , min=prob_eps) # Clamp at "almost" zero (if not log(0)=-inf creates backward pass issues)
                        # Compute for full x_range, clamp x_range>x_max
                        inner_log_terms=(
                                            (x_range-1) * torch.log(must_be_positive_term)
                                            - must_be_positive_term
                                            - gammaln(x_range+1) # Gamma(x+1)=x!
                                        )
                        del must_be_positive_term
                        # Clamp terms where x_range>x_max
                        inner_log_terms[clamp_mask]=minus_inf
                        assert ~ torch.isnan(inner_log_terms).any() and ~ torch.isinf(inner_log_terms).any(), 'Computations of inner terms did not work as expected'
                        del clamp_mask
                        # Compute partition, via summing extra dimension x with logsumexp
                        log_Z[this_Z_i[n_split],this_Z_m[n_split],:]=(
                                                                        torch.log(s_plus_1)[:,:,0]
                                                                        + torch.log(self.lambdas[this_Z_i[n_split],this_Z_m[n_split]])[:,None]
                                                                        + torch.logsumexp(inner_log_terms, dim=-1) # Sum over x_range
                                                                    )
                        del inner_log_terms
                else:
                    raise ValueError('We can not split {} instances of log_Z in {} splits needed for {} maximum bytes per split'.format(log_Z_i.size(0), n_splits, max_memory_in_bytes))
            else:
                raise ValueError('Need to use polylog, which is not vectorized, not implemented')
        
        # Return computed log partition function, of size I (might be 1) times M times S
        return log_Z
        
    # Generalized poisson's log sum pdf, given parameters, and range of skipped cycles to consider
    def compute_log_unnormalized_sum_pdf_generalized_poisson(self,s_max=None,day_range=None):
        # Dimensions
        assert self.lambdas.shape == self.xis.shape
        assert self.lambdas.shape == self.pis.shape
        
        # Truncated range of skipped cycles
        if s_max is None:
            s_max_value=self.s_max # Use initial s_max
        elif torch.is_tensor(s_max):
            s_max_value=s_max.item()
        else:
            s_max_value=s_max
        
        # s+1: (I by M) by day_range.size by S by x_max
        s_plus_1=(torch.arange(s_max_value, dtype=torch.double)+1 if s_max_value>0 else torch.ones(1))[None,None,:,None]
        # Log of the sum of the pdf over x_range: (I by M) by day_range.size by S    
        log_sum_pdf=torch.zeros(self.lambdas.shape[0],self.lambdas.shape[1], day_range.size()[0], s_plus_1.shape[2], dtype=torch.double) 
        
        # Individual/samples that are underdispersed
        underdispersed=self.xis<0
        
        # For those individual/samples that are underdispersed
        if underdispersed.any():
            # Indexes of interest
            ind_i, ind_m=torch.nonzero(underdispersed, as_tuple=True)
            if s_max_value < float('inf'):                
                # x_max for those i,m of interest
                # Note that this also depends on (s+1), so x_max might become quite big!
                x_max=torch.ceil(-(self.lambdas[ind_i, ind_m]/self.xis[ind_i, ind_m])[:,None]*s_plus_1[0,0,:,0] - 1).long()
                # We nevertheless limit this so that memory request is not unbounded
                truncated_x_max=np.minimum(torch.max(x_max), s_max_value*log_Z_max_x_max)

                # We will sum over a range of x from day_range.min to x_max
                #   which we allocate here due to pytorch allocation shennanigans:
                #   https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/10
                print('\t\t\tComputing GP log_unnormalized_sum_pdf with max(x_max)={} truncated to truncated_x_max={}'.format(torch.max(x_max), truncated_x_max))
                x_range=torch.arange(day_range.min(),truncated_x_max+1, dtype=torch.double)[(None,)*3]*torch.ones((day_range.size()[0]))[None,:,None,None]
                    
                # Memory requirement of CDF computation in bytes: (selected I times M) times day_range.size times S times x_max
                mem_bytes_needed=np.prod([ind_i.size(0),day_range.size()[0], s_plus_1.shape[2], truncated_x_max+1]) * log_sum_pdf.element_size() 

                # Number of "splits" needed
                n_splits=np.ceil(mem_bytes_needed/max_memory_in_bytes).astype(np.uint)
                # Resulting split sizes
                assert ind_i.size(0) == ind_m.size(0)
                assert ind_i.size(0) == x_max.size(0)
                split_size=int(ind_i.size(0)/n_splits)
                if split_size>0:
                    # Split arguments, across sample axis
                    this_ind_i=torch.split(ind_i, split_size)
                    this_ind_m=torch.split(ind_m, split_size)
                    this_x_max=torch.split(x_max, split_size)
                    
                    # Splits should match
                    final_n_splits=len(this_ind_i)
                    assert final_n_splits==len(this_ind_m)
                    assert final_n_splits==len(this_x_max)
                    # Iterate over splits
                    for n_split in torch.arange(final_n_splits, dtype=torch.int):
                        print('\t\t\t\t... GP log_unnormalized_sum_pdf computation n_split={}/{}'.format(n_split, final_n_splits))
                        print('\t\t\t\t\t... with size={} and memory={} bytes (max={})'.format(
                                                        this_ind_i[n_split].size(0),
                                                        np.prod([this_ind_i[n_split].size(0),day_range.size()[0], s_plus_1.shape[2], truncated_x_max+1]) * log_sum_pdf.element_size(),
                                                        max_memory_in_bytes)
                                                        )
                        # Unnecessary terms
                        clamp_mask=(x_range<day_range[None,:,None,None]) | (x_range>this_x_max[n_split][:,None,:,None])
                        # We clamp key term that can not be less than zero (i.e., where x_range>x_max)
                        must_be_positive_term=torch.clamp(
                                                s_plus_1*self.lambdas[this_ind_i[n_split],this_ind_m[n_split]][:,None,None,None]
                                                + self.xis[this_ind_i[n_split],this_ind_m[n_split]][:,None,None,None]*x_range
                                                , min=prob_eps) # Clamp at "almost" zero (if not log(0)=-inf creates backward pass issues)
                        # Compute for full x_range, clamp x_range>x_max
                        inner_log_terms=(
                                            (x_range-1) * torch.log(must_be_positive_term)
                                            - must_be_positive_term
                                            - gammaln(x_range+1) # Gamma(x+1)=x!
                                        )
                        del must_be_positive_term
                        # Clamp terms
                        inner_log_terms[clamp_mask]=minus_inf
                        assert ~ torch.isnan(inner_log_terms).any() and ~ torch.isinf(inner_log_terms).any(), 'Computations of inner terms did not work as expected'
                        del clamp_mask
                        # Compute cdf, via summing extra dimension x with logsumexp
                        log_sum_pdf[this_ind_i[n_split],this_ind_m[n_split],:,:]=(
                                                                        torch.log(s_plus_1)[:,:,:,0]
                                                                        + torch.log(self.lambdas[this_ind_i[n_split],this_ind_m[n_split],None,None])
                                                                        + torch.logsumexp(inner_log_terms, dim=-1) # Sum over x_range
                                                                    )
                        assert ~ torch.isnan(log_sum_pdf[this_ind_i[n_split],this_ind_m[n_split],:,:]).any(), 'Computations of log_sum_pdf did not work as expected'
                        del inner_log_terms
                else:
                    raise ValueError('We can not split {} instances of log_sum_pdf in {} splits needed for {} maximum bytes per split'.format(ind_i.size(0), n_splits, max_memory_in_bytes))
            else:
                raise ValueError('Need to use polylog, which is not vectorized, not implemented')
        
        # For those individual/samples that are not underdispersed
        if (~ underdispersed).any():
            # Indexes of interest
            ind_i, ind_m=torch.nonzero(~ underdispersed, as_tuple=True)
            if s_max_value < float('inf'):                
                # For those not underdispersed, we compute via 1-sum_{x=0}^{day_range}
                #   which we allocate here due to pytorch allocation shennanigans:
                #   https://discuss.pytorch.org/t/how-to-debug-causes-of-gpu-memory-leaks/6741/10
                x_range=torch.arange(day_range.max()+1, dtype=torch.double)[(None,)*3]*torch.ones((day_range.size()[0]))[None,:,None,None]
                # Memory requirement of CDF computation in bytes: (selected I times M) times day_range.size times S times max day_range
                mem_bytes_needed=np.prod([ind_i.size(0),day_range.size()[0],s_plus_1.shape[2],day_range.max()+1]) * log_sum_pdf.element_size() 

                # Number of "splits" needed
                n_splits=np.ceil(mem_bytes_needed/max_memory_in_bytes).astype(np.uint)
                # Resulting split sizes
                assert ind_i.size(0) == ind_m.size(0)
                split_size=int(ind_i.size(0)/n_splits)
                if split_size>0:
                    # Split arguments, across sample axis
                    this_ind_i=torch.split(ind_i, split_size)
                    this_ind_m=torch.split(ind_m, split_size)
                    
                    # Splits should match
                    final_n_splits=len(this_ind_i)
                    assert final_n_splits==len(this_ind_m)
                    # Iterate over splits
                    for n_split in torch.arange(final_n_splits, dtype=torch.int):
                        print('\t\t\t\t... GP log_unnormalized_sum_pdf computation n_split={}/{}'.format(n_split, final_n_splits))
                        print('\t\t\t\t\t... with size={} and memory={} bytes (max={})'.format(
                                                        this_ind_i[n_split].size(0),
                                                        np.prod([this_ind_i[n_split].size(0),day_range.size()[0],s_plus_1.shape[2],day_range.max()+1]) * log_sum_pdf.element_size(),
                                                        max_memory_in_bytes)
                                                        )
                        # Unnecessary terms
                        clamp_mask=(x_range<=day_range[None,:,None,None]) * torch.ones((this_ind_i[n_split].size()[0], 1, s_plus_1.shape[2], 1), dtype=torch.bool)
                        
                        # We clamp key term that can not be less than zero (i.e., where x_range>x_max)
                        must_be_positive_term=torch.clamp(
                                                s_plus_1*self.lambdas[this_ind_i[n_split],this_ind_m[n_split]][:,None,None,None]
                                                + self.xis[this_ind_i[n_split],this_ind_m[n_split]][:,None,None,None]*x_range
                                                , min=prob_eps) # Clamp at "almost" zero (if not log(0)=-inf creates backward pass issues)
                        # Compute for full x_range
                        inner_log_terms=(
                                            (x_range-1) * torch.log(must_be_positive_term)
                                            - must_be_positive_term
                                            - gammaln(x_range+1) # Gamma(x+1)=x!
                                        )
                        del must_be_positive_term
                        # Clamp terms
                        inner_log_terms[clamp_mask]=minus_inf
                        assert ~ torch.isnan(inner_log_terms).any() and ~ torch.isinf(inner_log_terms).any(), 'Computations of inner terms did not work as expected'
                        del clamp_mask
                        # Compute sum, via summing extra dimension x with logsumexp
                        key_log_sum=(
                                    torch.log(s_plus_1)[:,:,:,0]
                                    + torch.log(self.lambdas[this_ind_i[n_split],this_ind_m[n_split],None,None])
                                    + torch.logsumexp(inner_log_terms, dim=-1) # Sum over x_range
                                )
                        assert ~ torch.isnan(key_log_sum).any() and ~ torch.isinf(key_log_sum).any(), 'Computations of key_log_sum did not work as expected'
                        del inner_log_terms 
                        # To avoid numerical errors, note that for day_range=0, key_log_sum==0 (CDF=1)
                        almost_zero_idx, almost_zero_s_idx=torch.where(torch.isclose(key_log_sum[:,0,:], torch.zeros(key_log_sum[:,0,:].shape, dtype=torch.double)))
                        key_log_sum[almost_zero_idx,0,almost_zero_s_idx]=0
                        # CDF as (1-sum)
                        log_sum_pdf[this_ind_i[n_split],this_ind_m[n_split],:,:]=torch.log(1-torch.exp(key_log_sum))
                        assert ~ torch.isnan(log_sum_pdf[this_ind_i[n_split],this_ind_m[n_split],:,:]).any(), 'Computations of log_sum_pdf did not work as expected'
                else:
                    raise ValueError('We can not split {} instances of log_sum_pdf in {} splits needed for {} maximum bytes per split'.format(ind_i.size(0), n_splits, max_memory_in_bytes))
            else:
                raise ValueError('Need to use polylog, which is not vectorized, not implemented')
            
        # Return computed log sum of pdf, of size I (might be 1) times M times day_range.size times S
        return log_sum_pdf  
        
    # Per observation data loglikelihood, given parameters
    def data_loglik_per_observation(self,x,s_max=None,log_normalized=False):
        # Should be I times M
        assert self.lambdas.shape == self.xis.shape
        assert self.lambdas.shape == self.pis.shape
        
        # Whether to consider truncated geometric distribution
        if s_max is None: s_max=self.s_max # Use initial s_max
        
        if s_max < float('inf'):
            # Truncated range of s+1, as an extra 4th dimension
            s_plus_1=(torch.arange(s_max, dtype=torch.double)+1 if s_max>0 else torch.ones(1))[(None,)*3]
            
            # log of Generalized poisson partition, of dimension: I times M times S (x is summed over)
            log_Z=self.compute_log_Z_generalized_poisson(s_max)[:,None,:,:] # Add C_i dimension
            assert not torch.isnan(log_Z).any()
            
            # We clamp key term that can not be less than zero (i.e., if x>x_max)
            must_be_positive_term=torch.clamp(
                                    s_plus_1*self.lambdas[:,None,:,None]
                                    + self.xis[:,None,:,None]*x[(...,)+(None,)*2]
                                    , min=prob_eps) # Clamp at "almost" zero (if not log(0)=-inf creates backward pass issues)
            # Per-observation likelihood: I times C_i times M (4th dimesion S is summed over)
            loglik=(
                    (torch.log(self.lambdas) 
                        + torch.log(1-self.pis)-torch.log(self.pis) 
                        - torch.log(1-torch.pow(self.pis,s_max+1))
                        )[:,None,:] 
                    - self.xis[:,None,:]*x[...,None]
                    + torch.logsumexp(
                            torch.log(s_plus_1) 
                            +(x[(...,)+(None,)*2]-1) * torch.log(must_be_positive_term)
                            + s_plus_1*(
                                torch.log(self.pis)[:,None,:,None]
                                - self.lambdas[:,None,:,None]
                                )
                            - log_Z
                            , dim=-1) # Truncated polylog, by summing over last dimension
                )
            # If normalized loglik is desired
            if log_normalized:
                loglik-=gammaln(x[...,None]+1) # Gamma(x+1)=x! (unnecessary for optimization, but necessary for other uses of this function)
            
        else:
            raise ValueError('Need to use polylog, which is not vectorized, not implemented')
        
        # Return should be I times C_i times M
        assert loglik.shape[0] == self.lambdas.shape[0]
        assert loglik.shape[1] == x.shape[1]
        assert loglik.shape[2] == self.lambdas.shape[1]
        assert ~torch.isnan(loglik).any()
        return loglik
    
    # Per individual data loglikelihood, given parameters
    def data_loglik_per_individual(self,x,s_max=None,log_normalized=False):
        # Should be I times M
        assert self.lambdas.shape == self.xis.shape
        assert self.lambdas.shape == self.pis.shape
        
        # Whether to consider truncated geometric distribution
        if s_max is None: s_max=self.s_max # Use initial s_max
        
        if s_max < float('inf'):
            # Truncated range of s+1, as an extra 4th dimension
            s_plus_1=(torch.arange(s_max, dtype=torch.double)+1 if s_max>0 else torch.ones(1))[(None,)*3]
            
            # Per-individual likelihood: I times times M
            # 2th dimension C_i is summed over
            # TODO: C_i might not be equal for all!
            C_i=x.shape[1]
            # 4th dimesion S is summed over in logsumexp
            
            # log of Generalized poisson partition, of dimension: I times M times S (x is summed over)
            log_Z=self.compute_log_Z_generalized_poisson(s_max)[:,None,:,:] # Add C_i dimension
            assert not torch.isnan(log_Z).any()
            
            # We clamp key term that can not be less than zero (i.e., if x>x_max)
            must_be_positive_term=torch.clamp(
                                    s_plus_1*self.lambdas[:,None,:,None]
                                    + self.xis[:,None,:,None]*x[(...,)+(None,)*2]
                                    , min=prob_eps) # Clamp at "almost" zero (if not log(0)=-inf creates backward pass issues)
            # Compute vectorized log-likelihood 
            loglik=( C_i*(torch.log(self.lambdas)
                        + torch.log(1-self.pis)-torch.log(self.pis) 
                        -torch.log(1-torch.pow(self.pis,s_max+1))
                        ) 
                    + torch.sum(
                        - self.xis[:,None,:]*x[...,None]
                        +torch.logsumexp(
                            torch.log(s_plus_1) 
                            +(x[(...,)+(None,)*2]-1) * torch.log(must_be_positive_term)
                            + s_plus_1*(
                                torch.log(self.pis)[:,None,:,None]
                                - self.lambdas[:,None,:,None]
                                )
                            - log_Z
                            , dim=-1) # Truncated polylog, by summing over last dimension
                        , dim=1) # Sum over cycles
                    )
            # If normalized loglik is desired
            if log_normalized:
                loglik-=torch.sum(
                        gammaln(x[...,None]+1) # Gamma(x+1)=x! (unnecessary for optimization, but necessary for other uses of this function)
                        , dim=1) # Sum over cycles
            
        else:
            raise ValueError('Need to use polylog, which is not vectorized, not implemented')
        
        # Return should be I times M
        assert loglik.shape[0] == x.shape[0]
        assert loglik.shape[1] == self.lambdas.shape[1]
        assert not torch.isnan(loglik).any()
        return loglik
    
    # Per individual parameter posterior, given observed data
    def param_posterior_weights_per_individual(self,x,M):
        # Exponentiate parameters
        self.exponentiate_prior_log_params()
        # Draw individual parameters (from prior)
        self.draw_params(M)

        # Loglikelihood per individual    
        loglik_i=self.data_loglik_per_individual(x,log_normalized=True) # loglik for all cycles of individual
        # Log-normalized weights per individual
        log_weights_i=loglik_i-torch.logsumexp(loglik_i, dim=1, keepdim=True) # Sum over MC samples            
        # Weights per individual
        weights_i=torch.exp(log_weights_i)
        # Renormalize again (logsumexp does not do enough)
        weights_i=weights_i/torch.sum(weights_i,dim=-1, keepdims=True)
        
        # Should be of size I times M
        assert weights_i.shape[0]==x.shape[0] and weights_i.shape[1]==M[1]
        # And sum to (almost) one
        assert torch.allclose(torch.sum(weights_i, dim=-1, dtype=torch.double), torch.ones(weights_i.shape[0], dtype=torch.double))
        return weights_i
    
    # Per individual parameter posterior estimates, given observed data
    def estimate_param_posterior_per_individual(self,x,M,posterior_type):
        with torch.no_grad():
            # Dimensionalities
            I = x.shape[0]
            C = x.shape[1]
            # Posterior items to compute
            posterior_items=[]
            if 'mean' in posterior_type or 'sufficient_statistics' in posterior_type or 'full' in posterior_type:
                posterior_items+=['mean']
            if 'sufficient_statistics' in posterior_type or 'full' in posterior_type:
                posterior_items+=['var']
            if 'full' in posterior_type:
                 posterior_items+=['samples']
            # Pre-allocate posterior
            parameter_posterior={}
            for param in ['lambda', 'xi', 'pi']:
                parameter_posterior[param]={}
                for item in posterior_items:
                    if item=='mean':
                        parameter_posterior[param][item]=torch.zeros((I, 1) , dtype=torch.double)
                    elif item=='var':
                        parameter_posterior[param][item]=torch.zeros((I, I, 1) , dtype=torch.double)
                    elif item=='samples':
                        parameter_posterior[param][item]=torch.zeros((I, M[1]), dtype=torch.double)
            if 'full' in posterior_type:
                parameter_posterior['weights']=torch.zeros((I, M[1]), dtype=torch.double)
            
            ### Serialization
            # get max of tmp bytes needed, I x C x M x S
            mem_bytes_needed=np.prod([I, C, M[1], self.s_max+1]) * 8 # Needed for computing loglik

            # If no memory constraints are given or apply
            if max_memory_in_bytes is None or mem_bytes_needed<max_memory_in_bytes:
                # Compute per-individual weights
                weights_i=self.param_posterior_weights_per_individual(x,M)
                
                # Parameter posterior
                if 'samples' in posterior_items:
                    parameter_posterior['lambda']['samples']=self.lambdas
                    parameter_posterior['xi']['samples']=self.xis
                    parameter_posterior['pi']['samples']=self.pis
                    parameter_posterior['weights']=weights_i
                
                # Sufficient statistics of parameter posterior
                if 'mean' in posterior_items:
                    # Lambda
                    parameter_posterior['lambda']['mean']=torch.sum(self.lambdas * weights_i, dim=-1, keepdim=True)
                    # Xi
                    parameter_posterior['xi']['mean']=torch.sum(self.xis * weights_i, dim=-1, keepdim=True)
                    # Pi
                    parameter_posterior['pi']['mean']=torch.sum(self.pis * weights_i, dim=-1, keepdim=True)
                    
                    if 'var' in posterior_items:
                        # Lambda
                        parameter_posterior['lambda']['var']=mc_variance(self.lambdas, parameter_posterior['lambda']['mean'], weights_i, max_memory_in_bytes)
                        # Xi
                        parameter_posterior['xi']['var']=mc_variance(self.xis, parameter_posterior['xi']['mean'], weights_i, max_memory_in_bytes)
                        # Pi
                        parameter_posterior['pi']['var']=mc_variance(self.pis, parameter_posterior['pi']['mean'], weights_i, max_memory_in_bytes)
                    
            else:
                # Number of "splits" needed
                n_splits=np.ceil(mem_bytes_needed/max_memory_in_bytes).astype(np.uint)
    
                # Resulting split sizes
                split_size=int(I/n_splits)
                if split_size>0:
                    print('\t... splitting I={} in {} splits in estimate_param_posterior_per_individual')
                    # Split arguments, across sample axis
                    x_splitted=torch.split(x, split_size, 0)
                    # Iterate over splits (as determined by torch.split), sum across splits
                    for n_split in torch.arange(len(x_splitted), dtype=torch.int):
                        # This split indexes
                        split_idx=torch.arange(n_split*split_size,np.minimum((n_split+1)*split_size, I))
                        
                        # Compute per-individual weights: I by M
                        weights_i=self.param_posterior_weights_per_individual(x_splitted[n_split],M)

                        # Parameter posterior
                        if 'samples' in posterior_items:
                            parameter_posterior['lambda']['samples'][split_idx]=self.lambdas
                            parameter_posterior['xi']['samples'][split_idx]=self.xis
                            parameter_posterior['pi']['samples'][split_idx]=self.pis
                            parameter_posterior['weights'][split_idx]=weights_i
                        
                        # Sufficient statistics of parameter posterior
                        if 'mean' in posterior_items:
                            # Lambda
                            parameter_posterior['lambda']['mean'][split_idx]=torch.sum(self.lambdas * weights_i, dim=-1, keepdim=True)
                            # Xi
                            parameter_posterior['xi']['mean'][split_idx]=torch.sum(self.xis * weights_i, dim=-1, keepdim=True)
                            # Pi
                            parameter_posterior['pi']['mean'][split_idx]=torch.sum(self.pis * weights_i, dim=-1, keepdim=True)
                            
                            if 'var' in posterior_items:
                                # Lambda
                                parameter_posterior['lambda']['var'][split_idx,split_idx]=mc_variance(self.lambdas, parameter_posterior['lambda']['mean'][split_idx], weights_i, max_memory_in_bytes)
                                # Xi
                                parameter_posterior['xi']['var'][split_idx,split_idx]=mc_variance(self.xis, parameter_posterior['xi']['mean'][split_idx], weights_i, max_memory_in_bytes)
                                # Pi
                                parameter_posterior['pi']['var'][split_idx,split_idx]=mc_variance(self.pis, parameter_posterior['pi']['mean'][split_idx], weights_i, max_memory_in_bytes)
                else:
                    raise ValueError('We can not split I={} instances in {} splits needed for {} maximum bytes per split'.format(I, n_splits, max_memory_in_bytes))
        
        # Return posterior
        return parameter_posterior
        
    # per-individual posterior probability of skipping
    def estimate_posterior_skipping_prob_per_day_per_individual(self,x,s_predict,M,day_range=None,posterior_type='full',posterior_self_normalized=True):
        with torch.no_grad():
            # Just making sure data is tensor
            if not torch.is_tensor(x):
                x=torch.from_numpy(x).double()
            if not torch.is_tensor(day_range):
                day_range=torch.from_numpy(day_range).int()
            
            # Dimensionalities
            I = x.shape[0]
            
            # Posterior items to compute
            posterior_items=[]
            if 'mean' in posterior_type or 'sufficient_statistics' in posterior_type or 'full' in posterior_type:
                posterior_items+=['mean']
            if 'sufficient_statistics' in posterior_type or 'full' in posterior_type:
                posterior_items+=['var']
            if 'pmf' in posterior_type or 'full' in posterior_type:
                 posterior_items+=['pmf']
            # Pre-allocate posterior
            posterior_skipping={}
            for item in posterior_items:
                if item=='mean':
                    posterior_skipping[item]=torch.zeros((I, 1) , dtype=torch.double)
                elif item=='var':
                    posterior_skipping[item]=torch.zeros((I, I, 1) , dtype=torch.double)
                elif item=='pmf':
                    posterior_skipping[item]=torch.zeros((I, day_range.size()[0], int(s_predict)), dtype=torch.double)
            
            ### Serialization
            # get max of memory bytes needed (I x day_range x M x S)
            mem_bytes_needed=np.prod([I, M[1], day_range.size()[0], s_predict]) * 8 
            # If no memory constraints are given or apply
            if max_memory_in_bytes is None or mem_bytes_needed<max_memory_in_bytes:
                # Compute per-individual weights, I x M
                weights_i=self.param_posterior_weights_per_individual(x,M)
                
                # sum of GP pdf is of shape I x M x day_range x S
                log_unnormalized_sum_pdf=self.compute_log_unnormalized_sum_pdf_generalized_poisson(s_predict, day_range)
                # log p(s|pi), of shape I x M x day_range x S
                s_range=(torch.arange(s_predict, dtype=torch.double) if s_predict>0 else torch.zeros(1))[(None,)*3]
                log_p_s = s_range*torch.log(self.pis[:,:,None,None]) + torch.log(1-self.pis[:,:,None,None])
                
                # Compute PMF
                if posterior_self_normalized:
                    # No need to keep normalizing constant, as they cancel out when normalizing below
                    posterior_skipping['pmf']=torch.sum(
                                                torch.exp(log_unnormalized_sum_pdf + log_p_s)
                                                * weights_i[:,:,None,None],
                                                dim=1, # Sum over M
                                                keepdim=False
                                                )

                    # Normalized over s range
                    posterior_skipping['pmf']=posterior_skipping['pmf']/torch.sum(posterior_skipping['pmf'],dim=-1, keepdims=True, dtype=torch.double)
                    assert torch.allclose(torch.sum(posterior_skipping['pmf'], dim=-1, dtype=torch.double), torch.ones(posterior_skipping['pmf'].shape[1], dtype=torch.double))
                    
                else:                   
                    # log of Generalized poisson partition, of dimension: I times M times S (x is summed over)
                    log_Z=self.compute_log_Z_generalized_poisson(s_predict)[:,:,None,:] # Add day_range dimension
                    # we need to normalize terms in numerator
                    posterior_skipping['pmf']=torch.sum(
                                                torch.exp(
                                                    log_unnormalized_sum_pdf - log_Z
                                                    + log_p_s - torch.log(1-torch.pow(self.pis[:,:,None,None],s_predict+1))
                                                    )
                                                * weights_i[:,:,None,None],
                                                dim=1, # Sum over M
                                                keepdim=False
                                                )
            # if memory constraints
            else:
                # Number of "splits" needed
                n_splits=np.ceil(mem_bytes_needed/max_memory_in_bytes).astype(np.uint)
                
                # Resulting split sizes
                split_size=int(I/n_splits)
                if split_size>0:
                    print('\t... splitting I={} with split size {} in estimate_predictive_posterior_per_individual'.format(str(I), str(split_size)))
                    # Split arguments, across sample axis
                    x_splitted=torch.split(x, split_size, 0)
                    # Iterate over splits (as determined by torch.split), sum across splits
                    for n_split in torch.arange(len(x_splitted), dtype=torch.int):
                        # This split indexes
                        split_idx=torch.arange(n_split*split_size,np.minimum((n_split+1)*split_size, I))
                        
                        # Compute per-individual weights
                        weights_i=self.param_posterior_weights_per_individual(x_splitted[n_split],M)
                        
                        # sum of GP pdf is of shape I x M x day_range x S
                        log_unnormalized_sum_pdf=self.compute_log_unnormalized_sum_pdf_generalized_poisson(s_predict, day_range)
                        # log p(s|pi), of shape I x M x day_range x S
                        s_range=(torch.arange(s_predict, dtype=torch.double) if s_predict>0 else torch.zeros(1))[(None,)*3]
                        log_p_s = s_range*torch.log(self.pis[:,:,None,None]) + torch.log(1-self.pis[:,:,None,None])
                        
                        # Compute PMF
                        if posterior_self_normalized:
                            # No need to keep normalizing constant, as they cancel out when normalizing below
                            posterior_skipping['pmf'][split_idx]=torch.sum(
                                                        torch.exp(log_unnormalized_sum_pdf + log_p_s)
                                                        * weights_i[:,:,None,None],
                                                        dim=1, # Sum over M
                                                        keepdim=False
                                                        )

                            # Normalized over s range
                            posterior_skipping['pmf'][split_idx]=posterior_skipping['pmf'][split_idx]/torch.sum(posterior_skipping['pmf'][split_idx],dim=-1, keepdims=True, dtype=torch.double)
                            assert torch.allclose(torch.sum(posterior_skipping['pmf'][split_idx], dim=-1, dtype=torch.double), torch.ones(posterior_skipping['pmf'][split_idx].shape[1], dtype=torch.double))
                            
                        else:                   
                            # log of Generalized poisson partition, of dimension: I times M times S (x is summed over)
                            log_Z=self.compute_log_Z_generalized_poisson(s_predict)[:,:,None,:] # Add day_range dimension
                            # we need to normalize terms in numerator
                            posterior_skipping['pmf'][split_idx]=torch.sum(
                                                        torch.exp(
                                                            log_unnormalized_sum_pdf - log_Z
                                                            + log_p_s - torch.log(1-torch.pow(self.pis[:,:,None,None],s_predict+1))
                                                            )
                                                        * weights_i[:,:,None,None],
                                                        dim=1, # Sum over M
                                                        keepdim=False
                                                        )

        # Return posterior
        return posterior_skipping
    
    # Per individual posterior predictive distribution, given observed data
    def estimate_predictive_posterior_per_individual(self,x,s_predict,M,x_predict_max,posterior_type='full',posterior_self_normalized=True):
        with torch.no_grad():
            # Dimensionalities
            I = x.shape[0]
            C = x.shape[1]
            # Posterior items to compute
            posterior_items=[]
            if 'mean' in posterior_type or 'sufficient_statistics' in posterior_type or 'full' in posterior_type:
                posterior_items+=['mean']
            if 'sufficient_statistics' in posterior_type or 'full' in posterior_type:
                posterior_items+=['var']
            if 'pmf' in posterior_type or 'full' in posterior_type:
                 posterior_items+=['pmf']
            # Pre-allocate posterior
            predictive_posterior={}
            for item in posterior_items:
                if item=='mean':
                    predictive_posterior[item]=torch.zeros((I, 1) , dtype=torch.double)
                elif item=='var':
                    predictive_posterior[item]=torch.zeros((I, I, 1) , dtype=torch.double)
                elif item=='pmf':
                    predictive_posterior[item]=torch.zeros((I, x_predict_max+1), dtype=torch.double)
            
            ### Serialization
            # get max of memory bytes needed (I x C x M x S)
            mem_bytes_needed=np.prod([I, C, M[1], self.s_max+1]) * 8 # Needed for computing loglik

            # If no memory constraints are given or apply
            if max_memory_in_bytes is None or mem_bytes_needed<max_memory_in_bytes:
                # Compute per-individual weights
                weights_i=self.param_posterior_weights_per_individual(x,M)
                
                # Predictive posterior distribution
                if 'pmf' in posterior_items:
                    # We will work with an I by x_predict_max+1 matrix
                    x_predict_range=torch.arange(x_predict_max+1)[None,:]
                    x_predict_log_prob=self.data_loglik_per_observation(x_predict_range,s_predict,log_normalized=True)
                    if posterior_self_normalized:
                        # Normalized over x_predict_max range: note that if x_predict_max is small, then bias is introduced
                        predictive_posterior['pmf']=torch.sum(
                                                torch.exp(
                                                    x_predict_log_prob - torch.logsumexp(x_predict_log_prob, dim=1, keepdims=True)
                                                    )
                                                * weights_i[:,None,:],
                                                dim=-1,
                                                keepdim=False)
                        # And sum to (almost) one
                        # We renormalize again, to avoid numerical errors
                        predictive_posterior['pmf']=predictive_posterior['pmf']/torch.sum(predictive_posterior['pmf'],dim=-1, keepdims=True, dtype=torch.double)
                        assert torch.allclose(torch.sum(predictive_posterior['pmf'], dim=-1, dtype=torch.double), torch.ones(predictive_posterior['pmf'].shape[0], dtype=torch.double))
                    else:
                        # Unnormalized posterior
                        predictive_posterior['pmf']=torch.sum(
                                                torch.exp(
                                                    x_predict_log_prob
                                                    )
                                                * weights_i[:,None,:],
                                                dim=-1,
                                                keepdim=False)
                        # This will not sum to (almost) one over x_predict_max

                    # Should be of size I times x_predict_max+1
                    assert predictive_posterior['pmf'].shape[0]==x.shape[0] and predictive_posterior['pmf'].shape[1]==x_predict_max+1
                
                # Sufficient statistics of predictive posterior distribution
                if 'mean' in posterior_items:
                    # Samples, given s_predict
                    if s_predict < float('inf'):
                        # Marginalize over provided s_predict
                        s_range=(torch.arange(s_predict, dtype=torch.double) if s_predict>0 else torch.zeros(1))[(None,)*2]
                        predictive_samples=self.lambdas*(1-self.pis)/(1-self.xis)*torch.sum(
                                                                        torch.pow(self.pis[:,:,None], s_range) * (s_range+1),
                                                                         dim=-1,
                                                                         keepdim=False)
                    else:
                        # Marginalize over s=inf
                        predictive_samples=self.lambdas/((1-self.xis)*(1-self.pis))
                    
                    # Expected value
                    predictive_posterior['mean']=torch.sum(predictive_samples * weights_i, dim=-1, keepdim=True)
                            
                    if 'var' in posterior_items:                    
                        # Variance
                        predictive_posterior['var']=mc_variance(predictive_samples, predictive_posterior['mean'], weights_i, max_memory_in_bytes)
            
            # if memory constraints
            else:
                # Number of "splits" needed
                n_splits=np.ceil(mem_bytes_needed/max_memory_in_bytes).astype(np.uint)
                
                # Resulting split sizes
                split_size=int(I/n_splits)
                if split_size>0:
                    print('\t... splitting I={} in {} splits in estimate_predictive_posterior_per_individual')
                    # Split arguments, across sample axis
                    x_splitted=torch.split(x, split_size, 0)
                    # Iterate over splits (as determined by torch.split), sum across splits
                    for n_split in torch.arange(len(x_splitted), dtype=torch.int):
                        # This split indexes
                        split_idx=torch.arange(n_split*split_size,np.minimum((n_split+1)*split_size, I))
                        
                        # Compute per-individual weights
                        weights_i=self.param_posterior_weights_per_individual(x_splitted[n_split],M)
                        
                        # Predictive posterior distribution
                        if 'pmf' in posterior_items:
                            # We will work with an I by x_predict_max+1 matrix
                            x_predict_range=torch.arange(x_predict_max+1)[None,:]
                            x_predict_log_prob=self.data_loglik_per_observation(x_predict_range,s_predict,log_normalized=True)
                            if posterior_self_normalized:
                                # Normalized over x_predict_max range: note that if x_predict_max is small, then bias is introduced
                                predictive_posterior['pmf'][split_idx]=torch.sum(
                                                        torch.exp(
                                                            x_predict_log_prob - torch.logsumexp(x_predict_log_prob, dim=1, keepdims=True)
                                                            )
                                                        * weights_i[:,None,:],
                                                        dim=-1,
                                                        keepdim=False)
                                # And sum to (almost) one
                                # We renormalize again, to avoid numerical errors
                                predictive_posterior['pmf'][split_idx]=predictive_posterior['pmf'][split_idx]/torch.sum(predictive_posterior['pmf'][split_idx],dim=-1, keepdims=True, dtype=torch.double)
                                assert torch.allclose(torch.sum(predictive_posterior['pmf'][split_idx], dim=-1, dtype=torch.double), torch.ones(predictive_posterior['pmf'][split_idx].shape[0], dtype=torch.double))
                            else:
                                # Unnormalized posterior
                                predictive_posterior['pmf'][split_idx]=torch.sum(
                                                        torch.exp(
                                                            x_predict_log_prob
                                                            )
                                                        * weights_i[:,None,:],
                                                        dim=-1,
                                                        keepdim=False)
                                # This will not sum to (almost) one over x_predict_max

                            # Should be of size I times x_predict_max+1
                            assert predictive_posterior['pmf'][split_idx].shape[0]==x_splitted[n_split].shape[0] and predictive_posterior['pmf'][split_idx].shape[1]==x_predict_max+1
                        
                        # Sufficient statistics of predictive posterior distribution
                        if 'mean' in posterior_items:
                            # Samples, given s_predict
                            if s_predict < float('inf'):
                                # Marginalize over provided s_predict
                                s_range=(torch.arange(s_predict, dtype=torch.double) if s_predict>0 else torch.zeros(1))[(None,)*2]
                                predictive_samples=self.lambdas*(1-self.pis)/(1-self.xis)*torch.sum(
                                                                                torch.pow(self.pis[:,:,None], s_range) * (s_range+1),
                                                                                 dim=-1,
                                                                                 keepdim=False)
                            else:
                                # Marginalize over s=inf
                                predictive_samples=self.lambdas/((1-self.xis)*(1-self.pis))
                            
                            # Expected value
                            predictive_posterior['mean'][split_idx]=torch.sum(predictive_samples * weights_i, dim=-1, keepdim=True)
                                    
                            if 'var' in posterior_items:                    
                                # Variance
                                predictive_posterior['var'][split_idx,split_idx]=mc_variance(predictive_samples, predictive_posterior['mean'][split_idx], weights_i, max_memory_in_bytes)
        
                else:
                    raise ValueError('We can not split I={} instances in {} splits needed for {} maximum bytes per split'.format(I, n_splits, max_memory_in_bytes))
                
        # Return distribution
        return predictive_posterior
    
    # Per individual posterior per-day predictive distribution, given observed data
    def estimate_predictive_posterior_per_day_per_individual(self,x,s_predict,M,x_predict_max,posterior_type='full',day_range=None,posterior_self_normalized=True):
        with torch.no_grad():                                     
            # The predictive posterior needs to be computed via MC
            # These (due posterior_self_normalized=False) to  are just likelihoods over x_predict_max, they do not sum up to 1
            predictive_pmf_per_day=self.estimate_predictive_posterior_per_individual(x,s_predict,M,x_predict_max,'pmf',posterior_self_normalized=False)
            # Since we work with an I by day_range.size by x_predict_max+1 matrix, add extra day_range dimension to returned posterior pmf values
            predictive_pmf_per_day=(
                                            predictive_pmf_per_day['pmf'][:,None,:]
                                            *torch.ones(
                                                (x.shape[0], day_range.size()[0], x_predict_max+1),
                                                dtype=torch.double
                                                ) # We will work with an I by day_range.size by x_predict_max+1 matrix
                                            )
            # Indicator function
            predictive_pmf_per_day[:,torch.arange(x_predict_max+1)[None,:] <= day_range[:,None]]=0
            
            if posterior_self_normalized:
                # Normalize across x_predict_max
                # This posterior convergest to true as x_predict_max goes to infnty
                predictive_pmf_per_day=predictive_pmf_per_day/torch.sum(predictive_pmf_per_day, dim=2, keepdim=True)
                
            # x prediction support
            x_predict_range=torch.arange(x_predict_max+1)[(None,)*2]

            # Predictive posterior and suff statistics
            predictive_posterior_per_day={}
            if 'pmf' in posterior_type or 'full' in posterior_type:
                # Empirical PMF
                predictive_posterior_per_day['pmf']=predictive_pmf_per_day
            
            if 'mean' in posterior_type or 'full' in posterior_type:
                # Empirical mean
                # Note that this mean is computed with respect to x_predict_range:
                #   This computation get more accurate as x_predict_max goes to infinity
                predictive_posterior_per_day['mean']=torch.sum(
                                                    x_predict_range*predictive_pmf_per_day
                                                    ,dim=2)
                
                if 'var' in posterior_type or 'full' in posterior_type:
                    # Empirical variance
                    # Note that this mean is computed with respect to x_predict_range:
                    #   This computation get more accurate as x_predict_max goes to infinity
                    predictive_posterior_per_day['var']=torch.sum(
                                                        torch.pow(x_predict_range-predictive_posterior_per_day['mean'][:,:,None], 2)
                                                        * predictive_pmf_per_day
                                                        ,dim=2)                            
        # Return predictive posterior per day
        return predictive_posterior_per_day
    
    # Model's marginalized negative loglikelihood
    def nll(self,x):
        '''
            Input:
                x: data to compute negative loglikelihood for
            Output:
                nll: computed negative loglikelihood
        '''
        # Directly: Analytically not possible?
        raise ValueError('Fully marginalized nll for this model is not implemented')
    
    # Model's marginalized negative data log-likelihood, MC marginalizing parameters per observation
    def nll_mc_per_observation(self,x,M):
        '''
            Input:
                x: data to compute negative loglikelihood for
                M: number of samples to use for MC, tuple (1,M) or (I,M)
            Output:
                nll: computed negative loglikelihood
        '''
        # Exponentiate parameters
        self.exponentiate_prior_log_params()
        
        # Draw individual parameters
        self.draw_params(M)
        
        # Compute and return
        return -torch.sum(
                    torch.sum(
                        torch.logsumexp(
                            self.data_loglik_per_observation(x, log_normalized=True), 
                            dim=2) # MC
                    ,dim=1) # Cycles
                , dim=0) # Individuals
                    
    # Model's marginalized negative data log-likelihood, MC marginalizing parameters per individual joint observations
    def nll_mc_per_individual(self,x,M):
        '''
            Input:
                x: data to compute negative loglikelihood for
                M: number of samples to use for MC, tuple (1,M) or (I,M)
            Output:
                nll: computed negative loglikelihood
        '''
        # Exponentiate parameters
        self.exponentiate_prior_log_params()
        
        # Draw individual parameters
        self.draw_params(M)
        
        # Compute and return
        return -torch.sum(
                    torch.logsumexp(
                        self.data_loglik_per_individual(x,log_normalized=True) # loglik for all cycles of individual
                      ,  dim=1) # MC
                    , dim=0) # Individuals
    
    # Forward pass: draw samples from model
    def forward(self, I, C_i):
        '''
            Input:
                I: number of individuals
                C_i: number of cycles per individual
            Output:
                x: sampled cycle length data, of size I by C_i
                skipped: skipped cycle indicator data, of size I by C_i
        '''
        with torch.no_grad():
            # Exponentiate parameters
            self.exponentiate_prior_log_params()
                     
            # Draw individual parameters
            self.draw_params((I,))
            
            # Draw skipped cycles per-individual
            # NOTE: our pi is probability of skipping, not of success
            self.skipped=self.geometric_dist(1-self.pis).sample([C_i]).T.long()
            # Draw data from per-individual
            x = self.generalized_poisson_dist((self.skipped+1)*self.lambdas[:,None],self.xis[:,None]*np.ones(C_i)).rvs(1)[...,0]
            self.x = torch.from_numpy(x).long()
            
            # Return output
            return self.x, self.skipped

    # Fit the generative process
    def fit(self, x, optimizer, criterion='nll', M=(1,1000), n_epochs=100, batch_size=None, loss_epsilon=0.000001, grad_norm_max=0):
        '''
            Input:
                x: data to fit to
                optimizer: to be used
                criterion: criterion to fit, usually negative log-likelihood
                M: number of samples to use for MC, tuple (1,M) or (I,M)
                n_epochs: to train for
                batch_size: what batch size to use for fitting, if None, full dataset is used
                loss_epsilon: minimum relative loss diference to consider as converged training
                grad_norm_max: whether we are clipping gradient norms, very useful for RNN type models
            Output:
                None: the model will be trained after executing this function
        '''
        # Input type:
        if not torch.is_tensor(x):
            x=torch.from_numpy(x).double()
        
        # And make it a Torch dataloader type dataset
        if batch_size is None:
            # Use full dataset
            batch_size = int(x.shape[0])
        dataset=DataLoader(TensorDataset(x), batch_size=int(batch_size), shuffle=True)
        
        # Make sure MC sample size makes sense
        assert len(M)==2 and (M[0]==1 or M[0]==x.shape[0]) and M[1]>0, 'Unreasonable sample size {}'.format(M)
        
        # Training Run
        debug_epoch=np.floor(n_epochs/100).astype(int)
        # used for debugging and plotting, otherwise unnecessary
        self.training_loss_values = []    # loss function values
        self.training_u_values = []   # optimized hyperparameter values
        
        # Epoch variables
        epoch=0
        prev_loss=0
        this_loss=np.inf
        
        # Initiate fit-time counter
        start_time = timeit.default_timer()
                
        # Iterate
        while (epoch < n_epochs) and (abs(this_loss - prev_loss) >= loss_epsilon*abs(prev_loss)):
            # Option to catch errors sooner
            #with torch.autograd.detect_anomaly():
            # Keep track of losses over batches
            batch_count=0
            batches_loss=0
            # Mini-batching
            for data_batch in dataset:
                # When per-individual, adjust to batch size
                if batch_size is not None and M[0]>1:
                    M=(int(data_batch[0].shape[0]),M[1])
                # Fit criterion
                if 'mc' in criterion:
                    loss=getattr(self,criterion)(*data_batch,M)
                else:
                    loss=getattr(self,criterion)(*data_batch)

                # Keep track of this batch and its loss
                batch_count+=1
                batches_loss+=loss.item()
                    
                # Backpropagation and calculate gradients
                optimizer.zero_grad() # clear existing gradients from previous epoch
                loss.backward() # perform a backward pass
                if grad_norm_max>0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.parameters(), grad_norm_max)
                optimizer.step() # update the weights
        
            # Debugging
            if epoch%debug_epoch == 0:
                print('\t\tEpoch {}/{} with per-batch average loss={}'.format(epoch, n_epochs, batches_loss/batch_count))
            # Keep track of iterations
            epoch+=1
            prev_loss=this_loss
            this_loss=batches_loss
            
            # For debugging purposes
            # Keep hyperparams and losses
            self.training_u_values.append(self.get_hyperparameters())
            self.training_loss_values.append(this_loss)
        
        # Number of epochs and fit-time
        self.n_epochs_conv = epoch
        self.time_elapsed = timeit.default_timer() - start_time
                
        print('\tModel trained after {} epochs with per-batch average loss={}'.format(epoch, this_loss))
        
    # Inference of parameter posterior using the (learned) model
    def parameter_inference(self, x, M=(1,1000),posterior_type='mean'):
        '''
            Input:
                x: data to use for inference
                M: number of samples to use for MC, tuple (1,M) or (I,M)
                posterior_type: string indicating what posterior information we are interested
                    'mean': Just the mean of the posterior
                    'sufficient_statistics': The posterior's sufficient statistics (mean and var)
                    'full': samples and weights of the MC posterior
            Output:
                param_posterior: the model's infered posterior, as dictionary
                    param_posterior['lambda']
                        'mean': expected value 
                        'var': variance
                        'samples': drawn parameter samples (if posterior_type='full')
                    param_posterior['xi']
                        'mean': expected value 
                        'var': variance
                        'samples': drawn parameter samples (if posterior_type='full')
                    param_posterior['pi']
                        'mean': expected value 
                        'var': variance
                        'samples': drawn parameter samples (if posterior_type='full')
                    param_posterior['weights'] (if posterior_type='full')
                        weights of drawn parameters
                        
        '''
        # Make sure x is tensor
        x_is_numpy=False
        if not torch.is_tensor(x):
            x_is_numpy=True
            x=torch.from_numpy(x).float()
        
        with torch.no_grad():
            # Estimate parameter posterior
            estimated_param_posterior = self.estimate_param_posterior_per_individual(x,M,posterior_type)
        
        # Return parameter posterior dictionary as numpy or torch
        param_posterior={}
        for item in [*estimated_param_posterior]:
            if item =='weights':
                param_posterior[item]=estimated_param_posterior[item].numpy() if x_is_numpy else estimated_param_posterior[item]
            else:
                param_posterior[item]={k: v.numpy() for k, v in estimated_param_posterior[item].items()} if x_is_numpy else estimated_param_posterior[item]
        return param_posterior
    
    # Predict using the (learned) model
    def predict(self, x, s_predict=float('inf'), M=(1,1000), x_predict_max=100, posterior_type='full', day_range=None):
        '''
            Input:
                x: data to use for prediction
                s_predict: type of predictive assumption:
                    either we integrate out probability of skipping (s_predict=inf)
                    or we assume no probability of skipping (s_predict=0)
                M: number of samples to use for MC, tuple (1,M) or (I,M)
                x_predict_max: maximum prediction day support to consider for numerical computation of posterior
                posterior_type: string indicating what posterior information we are interested
                    'pmf': Just the posterior
                    'sufficient_statistics': Just the posterior's sufficient statistics
                    'full': all of the above
                day_range: day range (array) to consider for conditional per-day prediction: current day = 0, 1, 2, ...
                    if day_range is None, prediction is done for current day = 0
            Output:
                predictive: the model's predicted posterior
        '''
        # Make sure x is tensor
        x_is_numpy=False
        if not torch.is_tensor(x):
            x_is_numpy=True
            x=torch.from_numpy(x).double()
            
        # Make sure MC sample size makes sense
        assert len(M)==2 and (M[0]==1 or M[0]==x.shape[0]) and M[1]>0, 'Unreasonable sample size {}'.format(M)
        
        # Predict
        with torch.no_grad():
            if day_range is None:
                # Estimate predictive posterior
                predictive = self.estimate_predictive_posterior_per_individual(x,s_predict,M,x_predict_max,posterior_type)
            else:
                if not torch.is_tensor(day_range):
                    day_range=torch.from_numpy(day_range).float()
                # Estimate predictive posterior per _day
                predictive = self.estimate_predictive_posterior_per_day_per_individual(x,s_predict,M,x_predict_max,posterior_type,day_range)
        
        # Return predictive distribution dictionary
        return {k: v.numpy() for k, v in predictive.items()} if x_is_numpy else predictive
    
