#!/usr/bin/python

import numpy as np
import torch
import pdb
import sys, os, re, time
import pickle
    
# Compute variance of MC samples
def mc_variance(samples, mean, weights, max_memory_in_bytes=None):
    '''
        Input:
            samples: I by M matrix of MC samples
            mean: I by 1 matrix of samples' empirical mean
            weights: I by M matrix of MC weights
            max_memory_in_bytes: maximum memory size to use in bytes
        Output:
            mc_variance: I by I by M matrix of MC sample variance
    '''
    # Figure out key dimensionalities
    # I can be equal to 1
    I=np.maximum(samples.shape[0], weights.shape[0])
    # M has to be given and equal in both inputs
    assert samples.shape[1] == weights.shape[1]
    M=samples.shape[1]
    
    # For Numpy objects
    if not torch.is_tensor(weights):
        # Memory requirement of mc-variance computation in bytes
        tmp_bytes_needed=np.prod([I, I, M]) * weights.itemsize
        
        # If no memory constraints are given or apply
        if max_memory_in_bytes is None or tmp_bytes_needed<max_memory_in_bytes:
            # Vectorized operation via einsum
            tmp=(samples-mean)
            mc_variance = np.sum(np.einsum('im,jm->ijm', tmp, tmp) * weights, axis=2, keepdims=True)
        
        # If computation can not be done once        
        else:
            print('MC variance computation requires {} bytes, while maximum {} bytes are given'.format(tmp_bytes_needed,max_memory_in_bytes))
            # Pre-allocate mc_variance
            mc_variance=np.zeros((I,I,1))
            # Number of "splits" needed
            n_splits=np.ceil(tmp_bytes_needed/max_memory_in_bytes).astype(np.uint)
            # Resulting split sizes
            split_size=M/n_splits
            if split_size>=1:
                # Split arguments, across sample axis
                samples_splitted=np.array_split(samples, n_splits, 1)
                weights_splitted=np.array_split(weights, n_splits, 1)
                # Iterate over splits, sum across splits
                for n_split in np.arange(n_splits, dtype=np.uint):
                    #print('Split {}/{} with {}/{}'.format(n_split,n_splits, samples_splitted[n_split].size, samples.size))
                    tmp=(samples_splitted[n_split]-mean)
                    mc_variance += np.sum(np.einsum('im,jm->ijm', tmp, tmp) * weights_splitted[n_split], axis=2, keepdims=True)
            else:
                raise ValueError('We can not split {} samples in {} splits needed for {} maximum bytes per split'.format(M, n_splits, max_memory_in_bytes))
    
    # For torch objects
    else:
        # Memory requirement of mc-variance computation in bytes
        tmp_bytes_needed=np.prod([I, I, M]) * weights.element_size()
        
        # If no memory constraints are given or apply
        if max_memory_in_bytes is None or tmp_bytes_needed<max_memory_in_bytes:
            # Vectorized operation via einsum
            tmp=(samples-mean)
            mc_variance = torch.sum(torch.einsum('im,jm->ijm', tmp, tmp) * weights, dim=2, keepdim=True)
        
        # If computation can not be done once        
        else:
            print('MC variance computation requires {} bytes, while maximum {} bytes are given'.format(tmp_bytes_needed,max_memory_in_bytes))
            # Pre-allocate mc_variance
            mc_variance=torch.zeros((I,I,1), dtype=torch.double)
            # Number of "splits" needed
            n_splits=np.ceil(tmp_bytes_needed/max_memory_in_bytes).astype(np.uint)
            # Resulting split sizes
            split_size=int(M/n_splits)
            if split_size>0:
                # Split arguments, across sample axis
                samples_splitted=torch.split(samples, split_size, 1)
                weights_splitted=torch.split(weights, split_size, 1)
                # Iterate over splits (as determined by torch.split), sum across splits
                for n_split in torch.arange(len(samples_splitted), dtype=torch.int):
                    #print('Split {}/{} with {}/{} elements'.format(n_split,n_splits,samples_splitted[n_split].nelement(), samples.nelement()))
                    tmp=(samples_splitted[n_split]-mean)
                    mc_variance += torch.sum(
                                torch.einsum('im,jm->ijm', tmp, tmp) * weights_splitted[n_split],
                                dim=2,
                                keepdim=True,
                                dtype=torch.double)
            else:
                raise ValueError('We can not split {} samples in {} splits needed for {} maximum bytes per split'.format(M, n_splits, max_memory_in_bytes))
            
    return mc_variance

