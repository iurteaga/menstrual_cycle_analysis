#!/usr/bin/python

# Imports
import sys, os, re, time
import pdb
import pickle
# Science
import numpy as np
import scipy.stats as stats
import pandas as pd
import sklearn
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors

# Import generalized poisson model
from my_generalized_poisson import *
from plotting_functions import *

# Data generating processes/model
def generate_sim_model(model, hyperparameters, I, C, save_data_dir, debug_plotting=False, plot_dir=False):
    '''
        Input:
            model: one of the following strings
                poisson
                generalized_poisson
            hyperparameters of the model:
                for poisson model:
                    kappa, gamma, alpha, beta
                for generalized_poisson model:
                    kappa, gamma, alpha_xi, beta_xi, xi_max, x_max, alpha, beta
            Number of individuals I
            Number of cycles per individual C
            save_data_dir (string): if not None, directory where to save this data
            debug_plotting: whether to plot hyperparameter distributions
            plot_dir: where to plot hyperparameter distributions
        Output:
            N: Cycle lengths (I by C matrix)
            S: Skipped cycles (I by C matrix)
            true_params dictionary
                lambda: lambda per individual (I)
                xi: xi (if generalized poisson) per individual (I) 
                pi: pi per individual (I)
    '''
    
    # Expand hyperparameters
    if model == 'poisson':
        kappa,gamma,alpha,beta=hyperparameters
    elif model == 'generalized_poisson':
        kappa,gamma,alpha_xi,beta_xi,xi_max,x_max,alpha,beta=hyperparameters
    
    # Draw true parameters 
    pis=stats.beta.rvs(alpha, beta, size=I)
    
    lambdas=stats.gamma.rvs(kappa, loc=0, scale=1/gamma, size=I)
    if model == 'generalized_poisson':
        # Draw xis
        # Set by the underdispersed truncation limit
        if x_max < float('inf'):
            # Xi is set to the biggest value that matches x_max
            xis=-lambdas/(x_max+1)
        else:
            # Xi drawn from shifted beta prior distribution
            # Figure out xi_min limits [np.maximum(-1*np.ones(I), -lambdas/x_max), 1]
            xi_min=-1*np.ones((I))
            # Xi from shifted/scaled beta prior
            xis=xi_min+(xi_max-xi_min)*stats.beta.rvs(alpha_xi, beta_xi, size=I)
    
    # True parameters
    if model == 'poisson':
        true_params={
                    'lambda':lambdas,
                    'pi':pis
                    }
    elif model == 'generalized_poisson':
        true_params={
                    'lambda':lambdas,
                    'xi':xis,
                    'pi':pis
                    }
    
    # Generate data
    # Skip indicator
    S = stats.geom.rvs(p=(1-pis)[:,None], loc=-1, size=(I,C))
    # Cycle-lengths
    if model == 'poisson':
        # Draw cycle-length from poisson distribution
        N = stats.poisson.rvs((S+1)*lambdas[:,None], size=(I,C))
    elif model == 'generalized_poisson':
        # Draw cycle-length from my Generalized poisson distribution
        N = generalized_poisson((S+1)*lambdas[:,None], xis[:,None]*np.ones(C)).rvs(1)[...,0]
    else:
        raise ValueError('Unknown model {}'.format(model))
    
    # If needed
    if save_data_dir is not None:
        # Save data for later use
        try:
            with open('{}/cycle_lengths.npz'.format(save_data_dir), 'wb') as f:
                np.savez_compressed(f,
                                    data_model=model,
                                    I=I,
                                    C=C,
                                    hyperparameters=hyperparameters,
                                    cycle_lengths=N,
                                    cycle_skipped=S,
                                    true_params=true_params
                                    )
        except Exception as error:
            raise ValueError('Could not save sim data in {} with error {}'.format(save_data_dir, error))
    ##### Debugging/plotting #####
    if debug_plotting:
        plot_generative_model_hyperparams(model, hyperparameters, true_params, plot_dir=plot_dir)
    #############################
    
    # Return data and parameters
    return N,S,true_params

def get_data(data_model, save_data_dir, I, I_init, C, hyperparameters, train_test_splits, data_stamp, shuffle_C):
    '''Loads real or simulated data, or generates simulated data; creates main_dir based on train / test split'''
    if data_model == 'load':
        # Load data
        try:
            # Open file
            f=open('{}/cycle_lengths.npz'.format(save_data_dir), 'rb')
            # Load
            loaded_data=np.load(f, allow_pickle=True)
            # Simulation parameters
            I=loaded_data['I']
            C=loaded_data['C']
            true_N=loaded_data['cycle_lengths']
            true_S=loaded_data['cycle_skipped']
            # Was more info saved?
            try:
                # Only possible if simulated data
                true_params=dict(loaded_data['true_params'].tolist())
                # This can only occur with simulated data, so it is ok to rename:
                hyperparameters=loaded_data['hyperparameters']
                hyperparameter_string=str(hyperparameters).replace(' ','').replace('[','').replace(']','').replace(',','_')
                data_model=str(loaded_data['data_model'])
            except:
                true_params=None # Don't know real data's parameters
                # This can only occur with real data, so it is ok to rename:
                data_model=str(loaded_data['data_model'])
                hyperparameters=dict(loaded_data['hyperparameters'].tolist())
                hyperparameter_string='_'.join(['{}_{}'.format(i,str(hyperparameters[i])) for i in hyperparameters])
            
            # Main result dir
            main_dir = '../results/evaluate_predictive_models/loaded_{}_data/I_{}/C_{}/{}/{}_fold'.format(
                    data_model,
                    I,
                    C,
                    hyperparameter_string,
                    train_test_splits
                    )
            # Make sure directory is ready
            os.makedirs(main_dir, exist_ok = True)
            
        except:
            raise ValueError('Provided data file {} can not be loaded'.format('{}/cycle_lengths.npz'.format(save_data_dir)))

    # Or simulate
    else:
        # Main result dir
        main_dir = '../results/evaluate_predictive_models/{}_data/I_{}/C_{}/{}/{}_fold'.format(
                    data_model,
                    I,
                    C,
                    str(hyperparameters).replace(' ','').replace('[','').replace(']','').replace(',','_'),
                    train_test_splits
                    )
        # Make sure directory is ready
        os.makedirs(main_dir, exist_ok = True)
        # If we want to save data, but save_data_dir does not exist,
        if (save_data_dir is not None) and (not os.path.isdir(save_data_dir)):
            # Just use main_dir
            save_data_dir=main_dir
        
        # Draw from simulated data
        true_N,true_S,true_params=generate_sim_model(data_model, hyperparameters, I, C, save_data_dir, debug_plotting=True, plot_dir=main_dir)

    return true_N, true_S, true_params, hyperparameters, main_dir

