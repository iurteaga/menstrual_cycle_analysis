#!/usr/bin/python

########################################################################
### Script to evaluate predictive models in simulated data
########################################################################
## Imports
import sys, os, re, time
import timeit
import argparse
from configparser import ConfigParser
import pdb, traceback
import pickle
import gzip
from itertools import *
# Science
import numpy as np
import pandas as pd
import scipy.stats as stats
import sklearn
from sklearn.model_selection import train_test_split
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors

# Import Hierarchical latent var model
# Add path
sys.path.append('../src/prediction')

# Import Generative model src
from poisson_with_skipped_cycles_models import *
# And other helpful functions
from evaluation_utils import *

# Main code
def main(data_dir, model_name):
    
    #### Data handling
    # Open file
    with open('{}/cycle_lengths.npz'.format(data_dir), 'rb') as f:
        # Load
        loaded_data=np.load(f, allow_pickle=True)
        # Simulation parameters
        I=loaded_data['I']
        C=loaded_data['C']
        X=loaded_data['cycle_lengths']
    
    ###### Model config from file
    # Type and parameters of the generative model
    model_config = ConfigParser()
    model_config.read('./generative_model_config/{}'.format(model_name))
    
    try:
        ### Create generative model object, with given parameters
        my_model=getattr(
                        sys.modules[__name__],
                        model_config.get('generative_model','model_name')
                    )(
                        **cast_dict_values(model_config._sections['model_params'], float),
                        config_file=model_name,
                    )
        
        #######################
        ### Model config
        # Model fit criterion
        model_fit_criterion = model_config.get(
                                'model_fitting_criterion', 'criterion'
                            ) #sampling_criterion
        # Model fit MC type
        model_fit_MC_samples = model_config.get(
                                'model_fitting_criterion', 'MC_samples',
                                fallback='per_individual'
                            )
        if model_fit_MC_samples == 'per_individual':
            # M samples per individual, sample_size=(I,M)
            model_fit_M = (
                            X.shape[0],
                            model_config.getint(
                                'model_fitting_criterion', 'M',
                                fallback=1000
                            )
                        )
        elif model_fit_MC_samples == 'per_cohort':
            # M samples for all, sample_size=(1,M)
            model_fit_M = (
                            1,
                            model_config.getint(
                                'model_fitting_criterion',
                                'M',
                                fallback=1000
                            )
                        )
        else:
            raise ValueError('Fitting MC sampling type {} not implemented yet'.format(model_fit_MC_samples))
        
        # Model optimizer
        model_optimizer = getattr(
                                optim,
                                model_config.get('model_optimizer', 'optimizer')
                            )(
                                my_model.parameters(),
                                lr=model_config.getfloat(
                                    'model_optimizer', 'learning_rate'
                                    )
                            )
        other_fitting_args=cast_dict_values(
                        model_config._sections['model_fitting_params'],
                        float
                    )
        
        # Model prediction MC type
        model_predict_MC_samples = model_config.get(
                                        'model_prediction_criterion', 'MC_samples',
                                        fallback='per_individual'
                                    )
        # Default is M samples for all, sample_size=(1,M)
        model_predict_M = (
                            1,
                            model_config.getint(
                                'model_prediction_criterion', 'M',
                                fallback=1000
                            )
                        )
        # Number of skipped cycles used for prediction:
        s_predict = model_config.getfloat(
                        'model_prediction_criterion', 's_predict',
                        fallback=100
                    )
        
        # max x (cycle length) - for predictive posterior
        x_predict_max = model_config.getint(
                            'model_prediction_criterion', 'x_predict_max',
                            fallback=x_predict_max_default
                        )

        if model_predict_MC_samples == 'per_individual':
            # M samples per individual, sample_size=(I,M)
            model_predict_M = (
                                X.shape[0],
                                model_config.getint(
                                        'model_prediction_criterion', 'M',
                                        fallback=1000
                                    )
                                )
        #######################

        ###### Model fitting, given train data 
        my_model.fit(X,
                        optimizer=model_optimizer,
                        criterion=model_fit_criterion,
                        M=model_fit_M,
                        **other_fitting_args
                        )   
        ###### Predict with learned model,
        # based on provided data X
        # For the first 30 days
        my_model_predictions=my_model.predict(
                X,
                s_predict=s_predict,
                M=model_predict_M,
                x_predict_max=x_predict_max,
                posterior_type='mean', # Only mean predictions
                day_range=np.arange(0,30) #First 30 days
        )
        
    except Exception as error:
        print('Could not create {} model with error {}'.format(model_name, error))
    
    print('FINISHED!')
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Fit and predict with generative model')
    parser.add_argument('-data_dir', type=str, help='Directory to load the data from')
    parser.add_argument('-model_name', type=str, help='What generative model config to load')
    
    # Get arguments
    args = parser.parse_args()
    
    # Call main function
    main(
        args.data_dir,
        args.model_name
    )
