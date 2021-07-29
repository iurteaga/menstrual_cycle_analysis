#!/usr/bin/python

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

# Torch optim utilities
import torch.optim as optim

# Import Hierarchical latent var model
# Add path
sys.path.append('../src/prediction')
# Import data functions
from data_functions import *
# Import plotting functions
from plotting_functions import *
# Import evaluation functions
from model_evaluation_functions import *

# Import baseline models
from baseline import *
# Import neural network models
from neural_network_models import *
# Import Generative models
from poisson_with_skipped_cycles_models import *
from generalized_poisson_with_skipped_cycles_models import *  

# Global defaults
# This suffices for real data
x_predict_max_default = 90

###############################
### Script execution utils
###############################
# Cast dictionary values
def cast_dict_values(dict, dtype):
    '''
        Input:
            dict: dictionary for which to cast values
            dtype: what data-type to cast values to
        Output:
            dict: dictionary with casted values
    '''
    return {key:dtype(value) for (key,value) in dict.items()}

### Load script configuration file into dictionary
def load_exec_mode_config(exec_mode_file):
    exec_config = ConfigParser()
    exec_config.read('./execution_config/{}'.format(exec_mode_file))
    exec_mode = {}
    for section in exec_config.sections():
        exec_mode[section] = {}
        for option in exec_config.options(section):
            option_string=exec_config.get(section, option,fallback=False)
            # Make string boolean if 'True'
            if option_string == 'True':
                exec_mode[section][option] = True 
            elif option_string == 'False': 
                exec_mode[section][option] = False
            else:
                exec_mode[section][option] = option_string
    return exec_mode
    
### Prepare ranges of I and C to evaluate for
def prep_I_C_ranges(prediction_type, I, C, C_init_online, C_step_online, I_init_online, I_step_online):
    '''Prepare I and C ranges depending on whether prediction type is online and if initial / step values are provided'''
    # TODO: this does not fully work if I_init_online and I_step_online are both not specified
    n_I_online=5
    if prediction_type == 'online':
        if C_init_online is not None:
            C_init=C_init_online
        else:
            C_init=1
        if C_step_online is not None:
            C_step=C_step_online
        else:
            C_step=1
        if I_init_online is not None:
            I_init=I_init_online
        else:
            I_init=np.linspace(1,I,n_I_online+1).astype(int)[1]
        if I_step_online is not None:
            I_step=I_step_online
        else:
            I_step=np.floor(I/n_I_online).astype(int)
    else:
        C_init=C-1
        C_step=1
        I_init=I
        I_step=1

    # Get I,C ranges ready
    I_range=np.arange(I_init,I+1,I_step)
    C_range=np.arange(C_init,C,C_step)
    assert I_range.size>0 and C_range.size>0, 'Provided I_range={} and C_range={} are not enough'.format(I_range, C_range)

    return I_range, C_range, I_init

# Figure out the number of individuals I in data to use (only makes sense with real or loaded datasets)
def get_full_I_size(data_model, save_data_dir, I, C, hyperparameters):
    '''gets full I size (all candidates)'''
    if data_model == 'load':
        # Load data
        try:
            # Open file
            #f=open('{}/cycle_lengths.npz'.format(save_data_dir), 'rb')
            # To accomodate Kathy Li
            f=open('{}'.format(save_data_dir), 'rb')
            # Load
            loaded_data=np.load(f, allow_pickle=True)
            # Simulation parameters
            I=loaded_data['I']
        except:
            raise ValueError('Provided data file {} can not be loaded'.format('{}/cycle_lengths.npz'.format(save_data_dir)))
    # Or generate data
    elif data_model == 'real':
        ### Clue dataset
        # Directory
        preprocessed_data_dir='../preprocessed_data'
        
        # Cycles, with excluded flags
        with open('{}/cohort_cycles_flagged.pickle'.format(preprocessed_data_dir), 'rb') as f:
            cohort_cycles = pickle.load(f)

        print('Cycles-data loaded')
        
        # Whether to remove excluded cycles
        if hyperparameters['clean']:
            # Remove cycles flagged as badly tracked 
            cohort_cycles = cohort_cycles[cohort_cycles['badly_tracked_cycle'] == 'f']
        
            # Cycle stats computed after cleaning cycles
            cycle_stats_file='cohort_clean_cycle_stats.pickle'
        else:
            # Cycle stats computed for all cycles
            cycle_stats_file='cohort_cycle_stats.pickle'
        
        # Load cycle stats, which have pre-computed statistic of interest
        # It provides quick access to stats such as median_inter_cycle_length or num_cycles_tracked 
        with open('{}/{}'.format(preprocessed_data_dir,cycle_stats_file), 'rb') as f:
            cohort_cycle_stats = pickle.load(f)
        
        print('Cycle-stats loaded')
        
        # Identify users with greater than C cycles
        user_ids_enough_data=cohort_cycle_stats[cohort_cycle_stats['num_cycles_tracked']>C]['user_id']
        # Filter-out users with less than C cycles
        cohort_cycles=cohort_cycles[cohort_cycles['user_id'].isin(user_ids_enough_data)]
        cohort_cycle_stats=cohort_cycle_stats[cohort_cycle_stats['user_id'].isin(user_ids_enough_data)]
        all_candidate_ids=cohort_cycles.user_id.unique()

        # Get C cycles per individual
        cycle_lengths_df=cohort_cycles.groupby('user_id')['cycle_length'].apply(lambda x: np.array(x)[:C])
        # Cutoff
        cld_cutoff=9
        # as numpy 2D array
        cycle_lengths=np.stack(cycle_lengths_df.values)
        median_CLD = np.median(np.abs(cycle_lengths[:,:-1] - cycle_lengths[:,1:]), axis=1)
        regular_ids = all_candidate_ids[np.where(median_CLD <= cld_cutoff)]
        irregular_ids = all_candidate_ids[np.where(median_CLD > cld_cutoff)]

        if hyperparameters['regular_proportion'] == 0:
            I=len(irregular_ids)
        elif hyperparameters['regular_proportion'] == 1:
            I=len(regular_ids)
        else:
            I=all_candidate_ids.size

    return I

### Pre-allocate necessary resources based on the script execution defined in the config file
def preallocate_resources_based_on_exec_mode(main_dir, model_names, exec_mode, train_test_splits, train_test_ratio, I_range, C_range): 
    # Fit evaluation
    fit_results=fit_model_dir=fit_plot_dir=None
    if 'fit' in exec_mode:
        # Fitted models
        if exec_mode['fit']['save_fitted_model']:
            fit_model_dir='{}/fitted_models'.format(main_dir)
            os.makedirs(fit_model_dir, exist_ok=True)
        
        if 'fit_eval_metrics' in exec_mode['fit']:
            # Results
            fit_results=preallocate_fit_results(exec_mode['fit']['fit_eval_metrics'], model_names, train_test_splits, I_range, C_range)
        
        # Fitted model evaluation
        if exec_mode['fit']['plot_fit_eval']:
            fit_plot_dir='{}/plots_fit'.format(main_dir)
            os.makedirs(fit_plot_dir, exist_ok=True)
    
    # Inference Allocate dict for inference results for every model, every split, every I, every C
    inference_results=inference_plot_dir=None
    if 'inference' in exec_mode and 'inference_eval_metrics' in exec_mode['inference']:
        # Results
        inference_results=preallocate_inference_results(exec_mode['inference']['inference_eval_metrics'], model_names, train_test_splits, I_range, C_range)
        # Inference evaluation
        if 'inference_eval_metrics' in exec_mode['inference']:
            inference_plot_dir='{}/plots_inference'.format(main_dir)
            os.makedirs(inference_plot_dir, exist_ok=True)
            
    # Prediction
    predict_mode=prediction_plot_dir=prediction_posterior_dir=None
    day_range=None
    pred_loss_train=pred_loss_test=None
    pred_mode_loss_train=pred_mode_loss_test=None
    pred_calib_train=pred_calib_test=calibration_plot_dir=None    
    
    if 'prediction' in exec_mode:
        # Defaults
        predict_mode='prediction'
        
        # If per-day prediction requested
        if 'day_range' in exec_mode['prediction']:
            predict_mode = 'prediction_by_day'
            # Determine day_range 
            if len(exec_mode['prediction']['day_range'].split(','))==1:
                # Compute as range
                day_range=np.arange(int(exec_mode['prediction']['day_range'])+1)
            else:
                # Range is given
                day_range=np.array(exec_mode['prediction']['day_range'].split(','), dtype=int)

        # Point-estimate evaluation metrics
        if 'prediction_eval_metrics' in exec_mode['prediction']:
            # Allocate dict for prediction results for every model, every split, every I, every C
            pred_loss_train, pred_loss_test = preallocate_prediction_results(exec_mode['prediction']['prediction_eval_metrics'].split(','), model_names, train_test_splits, train_test_ratio, I_range, C_range, day_range, exec_mode['prediction']['predict_s_0'])
            
            # Mode prediction for generative models
            if 'generative' in str(model_names):
                # Evaluate prediction accuracy
                pred_mode_loss_train, pred_mode_loss_test=preallocate_prediction_results(exec_mode['prediction']['prediction_eval_metrics'].split(','), model_names, train_test_splits, train_test_ratio, I_range, C_range, day_range, exec_mode['prediction']['predict_s_0'])
        
        if 'plot_predictions' in exec_mode['prediction'] and exec_mode['prediction']['plot_predictions']:
            # Prediction plots
            prediction_plot_dir = '{}/plots_prediction'.format(main_dir)
            os.makedirs(prediction_plot_dir, exist_ok=True)
    
        if 'save_predictions' in exec_mode['prediction'] and exec_mode['prediction']['save_predictions']:
            # We want to save predictive posteriors
            prediction_posterior_dir = '{}/predictive_posterior'.format(main_dir)
            os.makedirs(prediction_posterior_dir, exist_ok=True)
                                    
            
        # Calibration evaluation metrics    
        if 'prediction_eval_calibration' in exec_mode['prediction']:
            # Allocate dict for prediction calibration results for every model, every split, every I, every C
            pred_calib_train, pred_calib_test = preallocate_prediction_calibration_results(exec_mode['prediction']['prediction_eval_calibration'].split(','), model_names, train_test_splits, train_test_ratio, I_range, C_range, day_range, exec_mode['prediction']['predict_s_0'])
    
            # Calibration plots
            calibration_plot_dir = '{}/plots_calibration'.format(main_dir)
            os.makedirs(calibration_plot_dir, exist_ok=True)

    # Return
    return (
            fit_results, fit_model_dir, fit_plot_dir,
            inference_results, inference_plot_dir,
            predict_mode, prediction_plot_dir, prediction_posterior_dir, day_range,
            pred_loss_train, pred_loss_test, pred_mode_loss_train, pred_mode_loss_test,
            pred_calib_train, pred_calib_test, calibration_plot_dir
            )
    
###############################
### Model fitting utils
###############################
def preallocate_fit_results(fit_eval_metrics, prediction_models, train_test_splits, I_range, C_range):    
    # Create fit evaluation result dictionary
    fit_results={} # Empty for now
    
    # Per model lists of hyperparameter and parameters
    for prediction_model in prediction_models:
        fit_metrics=[]            
        # The inference metrics of interest
        if (fit_eval_metrics == 'all') or ('loss' in fit_eval_metrics):
            fit_metrics += ['training_loss']
        if (fit_eval_metrics == 'all') or ('convergence' in fit_eval_metrics):
            fit_metrics += ['n_epochs_conv']
        if (fit_eval_metrics == 'all') or ('exec_time' in fit_eval_metrics):
            fit_metrics += ['time_elapsed']

        # Fill-out inference result dictionary for this model
        fit_results[prediction_model]={
                            metric:np.zeros((train_test_splits, I_range.size, C_range.size)) for metric in fit_metrics
                            }
    
    # Return dictionary
    return fit_results

### Model fit function
def fit_model(model_name, data_model, exec_mode_fit, I_train, C_train, true_params, X, Y, fit_model_dir, stamp):
    
    this_model=None
    fitted=False
    n_try=0
    n_fit_tries=int(exec_mode_fit['n_fit_tries'])
    while (n_try < n_fit_tries) and (not fitted):
        print('\tFitting of {} try {}/{}'.format(model_name, n_try, n_fit_tries))
        #################
        # Baseline models
        #################
        if 'baseline' in model_name:
            # Baseline model statistic
            pred_statistic=getattr(np, model_name.split('_')[1])
            # Create baseline object
            try:
                # Average weights if needed
                weights=None
                if 'average' in model_name:
                    if 'linear' in model_name:
                        weights=np.arange(X.shape[1])+1
                    elif 'exponential' in model_name:
                        weights=np.exp(np.arange(X.shape[1]))
                
                # Create baseline object
                this_model=baseline(pred_statistic, weights)
                fitted=True
            except Exception as error:
                print('Could not create {} model with error {}'.format(model_name, error))

        #################    
        # Neural network models
        #################
        elif 'nnet' in model_name:
            # Parameters of the NN
            nn_config = ConfigParser()
            nn_config.read('./nnet_config/{}'.format(model_name))
            # Create NN object
            try:
                if 'conv_nnet' in nn_config.get('nn_model','nn_model_name'):
                    # get kernel size
                    if ('kernel_size_type' in exec_mode_fit) and (exec_mode['fit']['kernel_size_type']=='num_cycles'):
                        kernel_size = C_train
                    else:
                        kernel_size=nn_config.getint('nn_model_params', 'kernel_size', fallback=2)

                    #Input size is C_train
                    this_model=getattr(sys.modules[__name__], nn_config.get('nn_model','nn_model_name'))(
                                                                                    input_size=C_train,
                                                                                    output_size=1,
                                                                                    n_layers=nn_config.getint('nn_model_params', 'n_layers', fallback=1),
                                                                                    kernel_size=kernel_size,
                                                                                    stride=nn_config.getint('nn_model_params', 'stride', fallback=1),
                                                                                    padding=nn_config.getint('nn_model_params', 'padding', fallback=0),
                                                                                    dilation=nn_config.getint('nn_model_params', 'dilation', fallback=1),
                                                                                    nonlinearity=nn_config.get('nn_model_params', 'nonlinearity', fallback='tanh'),
                                                                                    dropout=nn_config.getfloat('nn_model_params', 'dropout', fallback=0.0),
                                                                                    config_file=model_name,
                                                                                    )
                elif 'rnn_nnet' in nn_config.get('nn_model','nn_model_name'):
                    # get hidden size
                    if ('kernel_size_type' in exec_mode_fit) and (exec_mode['fit']['kernel_size_type']=='num_cycles'):
                        hidden_size = C_train
                    else:
                        hidden_size=nn_config.getint('nn_model_params', 'hidden_size', fallback=1)
                    
                    #Input size is 1, C_train is sequence length
                    this_model=getattr(sys.modules[__name__], nn_config.get('nn_model','nn_model_name'))(
                                                                                    rnn_type=nn_config.get('nn_model_params', 'rnn_type', fallback='RNN'),
                                                                                    input_size=1,
                                                                                    output_size=1,
                                                                                    hidden_size=hidden_size,
                                                                                    n_layers=nn_config.getint('nn_model_params', 'n_layers', fallback=1),
                                                                                    nonlinearity=nn_config.get('nn_model_params', 'nonlinearity', fallback='tanh'),
                                                                                    dropout=nn_config.getfloat('nn_model_params', 'dropout', fallback=0.0),
                                                                                    bidirectional=nn_config.getboolean('nn_model_params', 'bidirectional', fallback=False),
                                                                                    config_file=model_name,
                                                                                    )
                
                # NN Optimizer and criterion
                nn_optimizer = getattr(optim, nn_config.get('nn_optimizer', 'optimizer'))(this_model.parameters(), lr=nn_config.getfloat('nn_optimizer', 'learning_rate'))
                nn_train_criterion = getattr(nn, nn_config.get('nn_training_criterion', 'criterion'))()
                
                # Train
                this_model.train(X, Y,
                                optimizer=nn_optimizer,
                                criterion=nn_train_criterion,
                                **cast_dict_values(nn_config._sections['nn_training_params'], float)
                                )
               
                # Check convergence                
                max_n_epochs = nn_config.getint('nn_fitting_params', 'n_epochs', fallback=100)
                loss_epsilon = nn_config.getfloat('nn_fitting_params', 'loss_epsilon', fallback=0.0001)
                if len(this_model.training_loss_values) == max_n_epochs and (abs(this_model.training_loss_values[-1] - this_model.training_loss_values[-2]) >= loss_epsilon*abs(this_model.training_loss_values[-2])):
                    print('Unsuccessful fitting of {} due to reaching max epochs {} with loss difference {}'.format(model_name, max_n_epochs, abs(this_model.training_loss_values[-1] - this_model.training_loss_values[-2])))
                    n_try+=1
                else:
                    fitted=True
            
            except Exception as error:
                print('Could not create {} model with error {}'.format(model_name, error))
                           
        #################
        # Generative models
        #################
        elif 'generative' in model_name:
            # Type and parameters of the generative model
            model_config = ConfigParser()
            model_config.read('./generative_model_config/{}'.format(model_name))
            
            try:
                # Create generative model object, with given parameters
                this_model=getattr(sys.modules[__name__], model_config.get('generative_model','model_name'))(
                                                                                **cast_dict_values(model_config._sections['model_params'], float),
                                                                                config_file=model_name,
                                                                                )
                # Model fit criterion
                model_fit_criterion = model_config.get('model_fitting_criterion', 'criterion') #sampling_criterion
                # Model fit MC type
                model_fit_MC_samples = model_config.get('model_fitting_criterion', 'MC_samples', fallback='per_individual')
                if model_fit_MC_samples == 'per_individual':
                    # M samples per individual, sample_size=(I,M)
                    model_fit_M = (X.shape[0],model_config.getint('model_fitting_criterion', 'M', fallback=1000))
                elif model_fit_MC_samples == 'per_cohort':
                    # M samples for all, sample_size=(1,M)
                    model_fit_M = (1,model_config.getint('model_fitting_criterion', 'M', fallback=1000))
                else:
                    raise ValueError('Fitting MC sampling type {} not implemented yet'.format(model_fit_MC_samples))
                
                # Model optimizer
                model_optimizer = getattr(optim, model_config.get('model_optimizer', 'optimizer'))(this_model.parameters(), lr=model_config.getfloat('model_optimizer', 'learning_rate'))
                other_fitting_args=cast_dict_values(model_config._sections['model_fitting_params'], float)
                
                # Fit model, given train data
                this_model.fit(X,
                                optimizer=model_optimizer,
                                criterion=model_fit_criterion,
                                M=model_fit_M,
                                **other_fitting_args
                                )

                # Check convergence
                max_n_epochs = model_config.getint('model_fitting_params', 'n_epochs', fallback=100)
                loss_epsilon = model_config.getfloat('model_fitting_params', 'loss_epsilon', fallback=0.0001)
                if len(this_model.training_loss_values) == max_n_epochs and (abs(this_model.training_loss_values[-1] - this_model.training_loss_values[-2]) >= loss_epsilon*abs(this_model.training_loss_values[-2])):
                    print('Unsuccessful fitting of {} due to reaching max epochs {} with loss difference {}'.format(model_name, max_n_epochs, abs(this_model.training_loss_values[-1] - this_model.training_loss_values[-2])))
                    n_try+=1
                else:
                    fitted=True
                
            except Exception as error:
                print('Could not create {} model with error {}'.format(model_name, error))
    
    if fitted and exec_mode_fit['save_fitted_model']:
        # save fitted model
        with gzip.open('{}/{}_{}.picklegz'.format(fit_model_dir, model_name, stamp), 'wb') as f:
            pickle.dump(this_model, f)
    elif fitted:
        print('Fitted {}, not saving'.format(data_model))
    else:
        print('Unsuccessful fitting of {}!'.format(data_model))

    # Return fitted model
    return this_model

### Load already fitted model
def load_fitted_model(model_name, main_dir, stamp):
    this_model=None
    # load fitted model
    save_dir='{}/fitted_models'.format(main_dir)
    with gzip.open('{}/{}_{}.picklegz'.format(save_dir, model_name, stamp), 'rb') as f:
        this_model=pickle.load(f)

    # Return fitted model
    return this_model

### Evaluate fit
def eval_fit(fit_results, exec_mode_fit, fitted_model, model_name, n_split, I_idx, C_idx, fit_plot_dir, split_stamp):
    
    # Collect fit results from fitted model
    fit_results[model_name]['training_loss'][n_split,I_idx, C_idx] = fitted_model.training_loss_values[-1]
    print('\t... with final training loss {}'.format(fit_results[model_name]['training_loss'][n_split,I_idx, C_idx]))
    fit_results[model_name]['n_epochs_conv'][n_split,I_idx, C_idx] = fitted_model.n_epochs_conv
    print('\t... n epochs {}'.format(fit_results[model_name]['n_epochs_conv'][n_split,I_idx, C_idx]))
    fit_results[model_name]['time_elapsed'][n_split,I_idx, C_idx] = fitted_model.time_elapsed
    print('\t... time elapsed {}'.format(fit_results[model_name]['time_elapsed'][n_split,I_idx, C_idx]))
                        
    # Plotting
    if exec_mode_fit['plot_fit_eval']:
        # Type and parameters of the generative model
        model_config = ConfigParser()
        model_config.read('./generative_model_config/{}'.format(fitted_model.config_file))
        model_predict_M = (1,model_config.getint('model_prediction_criterion', 'M', fallback=1000))
        
        # Plot cost
        plot_fitting_cost(
                            fitted_model.training_loss_values,
                            fit_plot_dir,
                            stamp='_M{}_{}_{}'.format(model_predict_M, model_name, split_stamp)
                            )
                            
    # Return results
    return fit_results
    
###############################
### Inference utils
###############################
def preallocate_inference_results(inference_eval_metrics, prediction_models, train_test_splits, I_range, C_range):   
    # Create inference result dictionary
    inference_results={} # Empty for now
    
    # Per model lists of hyperparameter and parameters
    for prediction_model in prediction_models:
        inference_metrics=[]
        if 'generative' in prediction_model:          
            # Generalized poisson
            if 'generalized_poisson' in prediction_model:
                # Hyperparameters
                if 'truncated' in prediction_model:
                    # With truncation, no prior on xi
                    hyperparameter_list=[
                                            'kappa', 'gamma',
                                            'alpha', 'beta'
                                        ]
                else:
                    # Full list
                    hyperparameter_list=[
                                        'kappa', 'gamma',
                                        'alpha_xi', 'beta_xi',
                                        'xi_max', 'x_max',
                                        'alpha', 'beta'
                                    ]
                # Parameters
                parameter_list=['lambda', 'xi', 'pi']
            
            # Poisson hyperparameters
            elif 'poisson' in prediction_model:
                # Hyperparameters
                hyperparameter_list=['kappa', 'gamma', 'alpha', 'beta']
                # Parameters
                parameter_list=['lambda', 'pi']
        
            # RMSE
            # Hyperparameters
            rmse_hyperparameters=['RMSE_hyperparams', 'rRMSE_hyperparams']
            for hyperparam in hyperparameter_list:
                rmse_hyperparameters+=['RMSE_{}'.format(hyperparam)]
                rmse_hyperparameters+=['rRMSE_{}'.format(hyperparam)]
            # Parameters
            rmse_parameters=[]
            for param in parameter_list:
                rmse_parameters+=['RMSE_params_{}'.format(param)]
                rmse_parameters+=['rRMSE_params_{}'.format(param)]
                
            # The inference metrics of interest
            if (inference_eval_metrics == 'all') or ('hyperparameters' in inference_eval_metrics):
                # Add hyperparam RMSE metrics
                inference_metrics+=rmse_hyperparameters
            if (inference_eval_metrics == 'all') or ('parameters' in inference_eval_metrics):
                # Add param RMSE metrics
                inference_metrics+=rmse_parameters
            if (inference_eval_metrics == 'all') or ('data' in inference_eval_metrics):
                # Data related RMSE
                inference_metrics+=[
                        'RMSE_data_mean_est_actual',
                        'RMSE_data_mean_est_generated',
                        'RMSE_data_mean_actual_generated'
                        ]
        # Fill-out inference result dictionary for this model
        inference_results[prediction_model]={
                            metric:np.zeros((train_test_splits, I_range.size, C_range.size)) for metric in inference_metrics
                            }
    
    # Return dictionary
    return inference_results

### Inference evaluation for given model 
def eval_inference(inference_results, inference_eval_metrics, model, model_name, hyperparameters, plot_hyperparameters, true_params, parameter_posterior, plot_parameters, X, plot_data_statistics, n_split, I_idx, C_idx, inference_plot_dir, stamp):
    #################
    # Baseline models
    #################
    if 'baseline' in model_name:        
        print('Inference not implemented for {} model'.format(model.__class__.__name__))

    #################    
    # Neural network models
    #################
    elif 'nnet' in model_name:
        print('Inference not implemented for {} model'.format(model.__class__.__name__))
                       
    #################
    # Generative models
    #################
    elif 'generative' in model_name:
        # Type and parameters of the generative model
        model_config = ConfigParser()
        model_config.read('./generative_model_config/{}'.format(model.config_file))
        model_fit_criterion = model_config.get('model_fitting_criterion', 'criterion')
        
        #####################################
        ### Hyperparameters
        # Get prior hyperparameters
        prior_hyperparameters = cast_dict_values(model_config._sections['model_params'], float).values()
                
        # Get fitted hyperparameters
        if 'generalized' in model_name:
            estimated_hyperparameters = model.get_hyperparameters(return_limits=True)
        else:
            estimated_hyperparameters = model.get_hyperparameters()
        
        # Hyperparameter names
        # Full generalized poisson
        if estimated_hyperparameters.size>4:
            hyperparameter_list=[
                                    'kappa', 'gamma',
                                    'alpha_xi', 'beta_xi',
                                    'xi_max', 'x_max',
                                    'alpha', 'beta'
                                ]

        # Poisson
        elif estimated_hyperparameters.size==4:
            hyperparameter_list=[
                                    'kappa', 'gamma',
                                    'alpha', 'beta'
                                ]
        else:
            raise ValueError('Unexpected number of estimated hyperparameters')
        
        # Hyperparameter evaluation
        if (inference_eval_metrics == 'all') or ('hyperparameters' in inference_eval_metrics):
            # If hyperparameters are of same length
            if len(hyperparameters) == estimated_hyperparameters.size:
                # Note that it's possible to have Generalized and Poisson have same number of hyperparameters with different meaning.
                for hyperparam_idx, hyperparam in enumerate(hyperparameter_list):
                    inference_results[model_name]['RMSE_{}'.format(hyperparam)][n_split,I_idx, C_idx] = np.sqrt(my_mean_squared_error(hyperparameters[hyperparam_idx], estimated_hyperparameters[hyperparam_idx])) # Using mine because sklearn does not like singletons
                
        #####################################
        ### Parameters
        # Get estimated parameters
        # MC needed for inference of skipped cycle-based models
        if 'skipped_cycles' in model.__class__.__name__:
            ## Inferred posteriors
            # Model fit MC type
            model_fit_MC_samples = model_config.get('model_fitting_criterion', 'MC_samples', fallback='per_individual')
            # Default is M samples for all, sample_size=(1,M)
            model_infer_M = (1,model_config.getint('model_fitting_criterion', 'M', fallback=1000))
            if model_fit_MC_samples == 'per_individual':
                # M samples per individual, sample_size=(I,M)
                model_infer_M = (X.shape[0],model_config.getint('model_fitting_criterion', 'M', fallback=1000))
            
            # get s_inference 
            s_max = model_config.get('model_params', 's_max', fallback=100)
            # Estimated posterior
            estimated_params = model.parameter_inference(X, model_infer_M, parameter_posterior)
            
        else:
            # No MC needed, closed form available
            ## Inferred posteriors
            estimated_params = model.parameter_inference(X)
        
        # Parameter evaluation
        if (inference_eval_metrics == 'all') or ('parameters' in inference_eval_metrics):
            if (true_params is not None) and (len(true_params) == len(estimated_params)):
                # For all parameters (but weights)
                for parameter in filter(lambda param: 'weights' not in param, [*estimated_params]):
                    # Compute parameter RMSE
                    inference_results[model_name]['RMSE_params_{}'.format(parameter)][n_split,I_idx, C_idx] = np.sqrt(my_mean_squared_error(true_params[parameter], estimated_params[parameter]['mean']))
                    inference_results[model_name]['rRMSE_params_{}'.format(parameter)][n_split,I_idx, C_idx] = np.sqrt(my_mean_squared_error(true_params[parameter], estimated_params[parameter]['mean'], relative=True))
        
        #####################################
        ### Data statistics
        # Generated data statistics
        data_mean=np.mean(X, axis=1)
        data_std=np.std(X, axis=1)
        
        # Estimated (given parameters) expectation of data
        s_inference=float('inf') # Ideal expectation, integrating out skipping probabilities
        if 'skipped' in model_name:
            if 'generalized' in model_name:
                estimation_data_model='generative_generalized_poisson_with_skipped_cycles'
                if s_inference < float('inf'):
                    # Use provided s
                    estimated_data_mean=estimated_params['lambda']['mean']*(s_inference+1)/(1-estimated_params['xi']['mean'])
                else:
                    # Marginalize over s
                    estimated_data_mean=estimated_params['lambda']['mean']/((1-estimated_params['xi']['mean'])*(1-estimated_params['pi']['mean']))
            else:
                estimation_data_model='generative_poisson_with_skipped_cycles'
                if s_inference < float('inf'):
                    # Use provided s
                    estimated_data_mean=estimated_params['lambda']['mean']*(s_inference+1)
                else:
                    # Marginalize over s
                    estimated_data_mean=estimated_params['lambda']['mean']/(1-estimated_params['pi']['mean'])
        else:
            # Estimated expectation of data
            estimation_data_model='generative_poisson'
            estimated_data_mean=estimated_params['lambda']['mean']

        # Data sufficient statistics evaluation
        if (inference_eval_metrics == 'all') or ('data' in inference_eval_metrics):
            # RMSE between estimated and actual data mean
            inference_results[model_name]['RMSE_data_mean_est_actual'][n_split,I_idx, C_idx] = np.sqrt(sklearn.metrics.mean_squared_error(np.array(estimated_data_mean), np.array(data_mean)))
            
            # True model's expectation of data
            true_data_mean=None
            true_data_model=None
            if true_params is not None:
                # Based on true_params, we figure out true model
                if [*true_params] == ['lambda']:
                    # Poisson
                    true_data_model='generative_poisson'
                    true_data_mean=true_params['lambda']
                elif [*true_params] == ['lambda', 'pi']:
                    # Poisson with skipped cycles
                    true_data_model='generative_poisson_with_skipped_cycles'
                    if s_inference < float('inf'):
                        # Use provided s
                        true_data_mean=true_params['lambda']*(s_inference+1)
                    else:
                        # Marginalize over s
                        true_data_mean=true_params['lambda']/(1-true_params['pi'])
                elif [*true_params] == ['lambda', 'xi', 'pi']:
                    # Generalized poisson with skipped cycles
                    true_data_model='generative_generalized_poisson_with_skipped_cycles'
                    if s_inference < float('inf'):
                        # Use provided s
                        true_data_mean=true_params['lambda']*(s_inference+1)/(1-true_params['xi'])
                    else:
                        # Marginalize over s
                        true_data_mean=true_params['lambda']/((1-true_params['xi'])*(1-true_params['pi']))
                else:
                    raise ValueError('Unexpected true parameters={}'.format(true_params))
                
                # RMSE between estimated and generative model's true expected data mean    
                inference_results[model_name]['RMSE_data_mean_est_generated'][n_split,I_idx, C_idx] = np.sqrt(sklearn.metrics.mean_squared_error(np.array(estimated_data_mean), np.array(true_data_mean[:,None])))
                # RMSE between actual and generative model's true expected data mean
                inference_results[model_name]['RMSE_data_mean_actual_generated'][n_split,I_idx, C_idx] = np.sqrt(sklearn.metrics.mean_squared_error(np.array(data_mean), np.array(true_data_mean[:,None])))
            
        #####################################
        ### Inference plotting
        ### Hyper-parameters
        if plot_hyperparameters and (estimation_data_model==true_data_model):
            # Dir setup
            hyperparams_plot_dir = '{}/hyperparams'.format(inference_plot_dir)
            os.makedirs(hyperparams_plot_dir, exist_ok=True)
            # Plot
            plot_hyperparameter_estimation(
                                            model_name,
                                            prior_hyperparameters, hyperparameters, estimated_hyperparameters,
                                            estimated_params,
                                            plot_dir=hyperparams_plot_dir, stamp='{}_{}'.format(model_fit_criterion,stamp)
                                            )
        ### Parameters
        if plot_parameters and (estimation_data_model==true_data_model):
            # Dir setup
            params_plot_dir = '{}/params'.format(inference_plot_dir)
            os.makedirs(params_plot_dir, exist_ok=True)
            # Plot
            plot_parameter_estimation(
                                        model_name, 
                                        estimated_params, true_params,
                                        plot_dir=params_plot_dir, stamp='{}_{}'.format(model_fit_criterion,stamp)
                                        )
        ### Data statistics
        if plot_data_statistics:
            # Dir setup
            data_statistics_plot_dir = '{}/data_statistics'.format(inference_plot_dir)
            os.makedirs(data_statistics_plot_dir, exist_ok=True)
            # Plot
            plot_data_statistics_estimation(
                                            model_name,
                                            data_mean, data_std,
                                            estimated_data_mean, true_data_mean,
                                            estimated_params,
                                            plot_dir=data_statistics_plot_dir, stamp='{}_{}'.format(model_fit_criterion,stamp)
                                            )
        
    # Return inference results
    return inference_results

###############################
### Prediction utils
###############################
def preallocate_prediction_results(prediction_eval_metrics, prediction_models, train_test_splits, train_test_ratio, I_range, C_range, day_range, predict_s_0=False):
    if day_range is None:
        day_range = np.array([0])

    '''preallocates evaluation data for pred_loss_train (and pred_loss_test if train_test_ratio < 1)'''
    pred_loss_test=None
    # Preallocate evaluation data
    pred_loss_train={
        prediction_model:{
                            metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in prediction_eval_metrics
                            } for prediction_model in prediction_models
                }
    # In case we are evaluating generative models with skipped cycles, add prediction assuming predict_s_0
    if predict_s_0:
        for prediction_model in filter(lambda model: 'skipped_cycles' in model, prediction_models):
            pred_loss_train['{}_predict_s_0'.format(prediction_model)]={
                                    metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in prediction_eval_metrics
                                }

    # If test data as well        
    if train_test_ratio<1:
        pred_loss_test={
            prediction_model:{
                                metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in prediction_eval_metrics
                                } for prediction_model in prediction_models
                    }
        # In case we are evaluating generative models with skipped cycles, add prediction assuming predict_s_0
        if predict_s_0:
            for prediction_model in filter(lambda model: 'skipped_cycles' in model, prediction_models):
                pred_loss_test['{}_predict_s_0'.format(prediction_model)]={
                                        metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in prediction_eval_metrics
                                        }
                    
    # In case we are interested in MSE, also add RMSE
    if 'mean_squared_error' in prediction_eval_metrics:
        for prediction_model in prediction_models:
            pred_loss_train[prediction_model]['root_mean_squared_error']=np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size))
            if train_test_ratio<1:
                pred_loss_test[prediction_model]['root_mean_squared_error']=np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size))
        
        # In case we are evaluating generative models with skipped cycles, add prediction assuming predict_s_0
        if predict_s_0:
            for prediction_model in filter(lambda model: 'skipped_cycles' in model, prediction_models):
                pred_loss_train['{}_predict_s_0'.format(prediction_model)]['root_mean_squared_error']=np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size))
                if train_test_ratio<1:
                    pred_loss_test['{}_predict_s_0'.format(prediction_model)]['root_mean_squared_error']=np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size))
    
    # Return dictionary
    return pred_loss_train, pred_loss_test

### Prediction based on provided execution mode
def predict(exec_mode, model_name, fitted_model,
                X_train, X_test, day_range,
                prediction_plot_dir, prediction_posterior_dir, stamp):

    # Define all prediction possibilities
    # Default predictions
    predictions_train=Y_hat_train=predictions_test=Y_hat_test=None
    # For skipping models, predictions with no skipping assumption
    predictions_train_s_0=Y_hat_train_predict_s_0=predictions_test_s_0=Y_hat_test_predict_s_0=None
    # For generative models, predictions based on mode of posterior
    Y_hat_mode_train=Y_hat_mode_test=Y_hat_mode_train_predict_s_0=Y_hat_mode_test_predict_s_0=None
    
    # Populate as needed                                                   
    if 'generative' in model_name:
        # Whether to compute predictions for skipping probability as well
        predictive_posterior_s=exec_mode['prediction']['predictive_posterior_s'] if 'predictive_posterior_s' in exec_mode['prediction'] else False
        
        # Default prediction
        # Training predictions
        predictions_train = predict_with_model(
                                fitted_model, model_name,
                                X_train,
                                exec_mode['prediction']['plot_predictions'], 
                                day_range=day_range,
                                posterior_type=exec_mode['prediction']['predictive_posterior'], predictive_posterior_s=predictive_posterior_s,
                                prediction_plot_dir=prediction_plot_dir, stamp=stamp
                                )
        # Expected values as point estimates
        Y_hat_train=predictions_train['mean']
        # If PMF, available
        if 'pmf' in predictions_train:
            # Mode as prediction
            Y_hat_mode_train=predictions_train['pmf'].argmax(axis=2)
        
        # Save predictions if required
        if predictions_train and exec_mode['prediction']['save_predictions'] if 'save_predictions' in exec_mode['prediction'] else False:
            # Save compressed
            with gzip.open('{}/{}_predictive_train_{}_{}.picklegz'.format(prediction_posterior_dir,
                                                                        model_name,
                                                                        exec_mode['prediction']['predictive_posterior'].replace(',', '_'),
                                                                        stamp
                                                                        ), 'wb') as f:
                pickle.dump(predictions_train, f)
        
        # Test set predictions
        if X_test is not None:
            predictions_test = predict_with_model(
                                fitted_model, model_name,
                                X_test,
                                exec_mode['prediction']['plot_predictions'],
                                day_range=day_range,
                                posterior_type=exec_mode['prediction']['predictive_posterior'], predictive_posterior_s=predictive_posterior_s,
                                prediction_plot_dir=prediction_plot_dir, stamp=stamp
                                )
            # Expected values as point estimates
            Y_hat_test=predictions_test['mean']
            # If PMF, available
            if 'pmf' in predictions_test:
                # Mode as prediction
                Y_hat_mode_test=predictions_test['pmf'].argmax(axis=2)
            
            # Save predictions if required
            if predictions_test and exec_mode['prediction']['save_predictions'] if 'save_predictions' in exec_mode['prediction'] else False:
                # Save compressed
                with gzip.open('{}/{}_predictive_test_{}_{}.picklegz'.format(prediction_posterior_dir,
                                                                            model_name,
                                                                            exec_mode['prediction']['predictive_posterior'].replace(',', '_'),
                                                                            stamp
                                                                            ), 'wb') as f:
                    pickle.dump(predictions_test, f)
    
        # For generative models with skipped cycles, we can compare prediction with s_predict=0
        if exec_mode['prediction']['predict_s_0'] and 'skipped_cycles' in fitted_model.__class__.__name__:
            # Training predictions
            predictions_train_s_0 = predict_with_model(
                                    fitted_model, model_name,
                                    X_train,
                                    exec_mode['prediction']['plot_predictions'],
                                    day_range=day_range,
                                    posterior_type=exec_mode['prediction']['predictive_posterior'], prediction_with_s=0, 
                                    prediction_plot_dir=prediction_plot_dir, stamp=stamp
                                    )
            # Expected values as point estimates
            Y_hat_train_predict_s_0=predictions_train_s_0['mean']
            # If PMF, available
            if 'pmf' in predictions_train_s_0:
                # Mode as prediction
                Y_hat_mode_train_predict_s_0=predictions_train_s_0['pmf'].argmax(axis=2)
            
            # Save predictions if required
            if predictions_train_s_0 and exec_mode['prediction']['save_predictions'] if 'save_predictions' in exec_mode['prediction'] else False:
                # Save compressed
                with gzip.open('{}/{}_predictive_train_{}_s0_{}.picklegz'.format(prediction_posterior_dir,
                                                                            model_name,
                                                                            exec_mode['prediction']['predictive_posterior'].replace(',', '_'),
                                                                            stamp
                                                                            ), 'wb') as f:
                    pickle.dump(predictions_train_s_0, f)
            
            # Test set predictions
            if X_test is not None:
                predictions_test_s_0 = predict_with_model(
                                    fitted_model, model_name,
                                    X_test,
                                    exec_mode['prediction']['plot_predictions'],
                                    day_range=day_range,
                                    posterior_type=exec_mode['prediction']['predictive_posterior'], prediction_with_s=0,
                                    prediction_plot_dir=prediction_plot_dir, stamp=stamp
                                    )
                # Expected values as point estimates
                Y_hat_test_predict_s_0=predictions_test_s_0['mean']
                # If PMF, available
                if 'pmf' in predictions_test_s_0:
                    # Mode as prediction
                    Y_hat_mode_test_predict_s_0=predictions_test_s_0['pmf'].argmax(axis=2)
                
                # Save predictions if required
                if predictions_test_s_0 and exec_mode['prediction']['save_predictions'] if 'save_predictions' in exec_mode['prediction'] else False:
                    # Save compressed
                    with gzip.open('{}/{}_predictive_test_{}_s0_{}.picklegz'.format(prediction_posterior_dir,
                                                                                model_name,
                                                                                exec_mode['prediction']['predictive_posterior'].replace(',', '_'),
                                                                                stamp
                                                                                ), 'wb') as f:
                        pickle.dump(predictions_test_s_0, f)

    # Non-generative models
    else:
        # Training predictions
        predictions_train = predict_with_model(
                                fitted_model, model_name,
                                X_train,
                                exec_mode['prediction']['plot_predictions'], 
                                day_range=day_range,
                                prediction_plot_dir=prediction_plot_dir, stamp=stamp
                                )
        Y_hat_train=np.repeat(predictions_train, day_range.size, axis=1).astype(float) # To accommodate prediction per-day
        
        # Test set predictions
        if X_test is not None:
            predictions_test = predict_with_model(
                                fitted_model, model_name,
                                X_test,
                                exec_mode['prediction']['plot_predictions'],
                                day_range=day_range,
                                prediction_plot_dir=prediction_plot_dir, stamp=stamp
                                )
            Y_hat_test=np.repeat(predictions_test, day_range.size, axis=1).astype(float)  # To accommodate prediction per-day

    # Return all prediction variables
    return (
        predictions_train, Y_hat_train, predictions_test, Y_hat_test, 
        predictions_train_s_0, Y_hat_train_predict_s_0, predictions_test_s_0, Y_hat_test_predict_s_0,
        Y_hat_mode_train, Y_hat_mode_test, Y_hat_mode_train_predict_s_0, Y_hat_mode_test_predict_s_0
    )
    
### Predict with given model
def predict_with_model(model, model_name, X, plot_predictions=True, day_range=None, posterior_type=None, prediction_with_s=None, predictive_posterior_s=False, prediction_plot_dir=None, stamp=None):
    #################
    # Baseline models
    #################
    if 'baseline' in model_name:        
        # Predictions of the model
        y_hat=model.predict(X)

    #################    
    # Neural network models
    #################
    elif 'nnet' in model_name:
        # Predict
        y_hat=model.predict(X)
                       
    #################
    # Generative models
    #################
    elif 'generative' in model_name:
        # Type and parameters of the generative model
        model_config = ConfigParser()
        model_config.read('./generative_model_config/{}'.format(model.config_file))
        model_fit_criterion = model_config.get('model_fitting_criterion', 'criterion')

        # MC needed for prediction of skipped cycle-based models
        if 'skipped_cycles' in model.__class__.__name__: 
            ## Prediction
            # Model prediction MC type
            model_predict_MC_samples = model_config.get('model_prediction_criterion', 'MC_samples', fallback='per_individual')
            # Default is M samples for all, sample_size=(1,M)
            model_predict_M = (1,model_config.getint('model_prediction_criterion', 'M', fallback=1000))
            # Number of skipped cycles used for prediction:
            if prediction_with_s is None:
                # Read from config file
                s_predict = model_config.getfloat('model_prediction_criterion', 's_predict', fallback=100)
            else:
                # Provided value: 0 or integrate out 
                s_predict = prediction_with_s
            
            # max x (cycle length) - for predictive posterior
            # This suffices for real data
            x_predict_max = model_config.getint('model_prediction_criterion', 'x_predict_max', fallback=x_predict_max_default)

            if model_predict_MC_samples == 'per_individual':
                # M samples per individual, sample_size=(I,M)
                model_predict_M = (X.shape[0],model_config.getint('model_prediction_criterion', 'M', fallback=1000))
            
            y_hat=model.predict(X, s_predict=s_predict, M=model_predict_M, x_predict_max=x_predict_max, posterior_type=posterior_type, day_range=day_range)
            
        else:
            # No MC needed, closed form available
            ## Prediction                                
            y_hat=model.predict(X)
        
        if plot_predictions:
            # Day-range to use
            day_range=0 if day_range is None else day_range
            # Cycle-length predictive posterior
            plot_predictive_posterior_by_day(
                                                y_hat['pmf'],
                                                day_range,
                                                x_predict_max=x_predict_max,
                                                y_true=None,
                                                save_dir='{}/plot_posterior'.format(prediction_plot_dir),
                                                stamp='{}_s_predict_{}_{}_{}'.format(model.config_file, str(s_predict), model_fit_criterion, stamp)
                                            )
            
            # Skipping probability predictive posterior
            if s_predict>0 and predictive_posterior_s:
                # Compute posterior
                posterior_s = model.estimate_posterior_skipping_prob_per_day_per_individual(
                                                X,
                                                s_predict,
                                                model_predict_M,
                                                day_range=day_range,
                                                posterior_type=posterior_type,
                                                posterior_self_normalized=True
                                                )
                # Plot
                plot_skipping_posterior_by_day(
                                                posterior_s['pmf'].detach().numpy(),
                                                day_range=day_range,
                                                s_predict=s_predict,
                                                y_true=None,
                                                save_dir='{}/plot_posterior_s'.format(prediction_plot_dir),
                                                stamp='{}_s_{}_posterior_{}_{}'.format(model.config_file, str(s_predict), model_fit_criterion, stamp)
                                                )
                
                
    return y_hat

# Per-model evaluation of all prediction metrics
def eval_model_prediction(exec_mode, model_name, fitted_model,
                            pred_loss_train, pred_loss_test, pred_mode_loss_train, pred_mode_loss_test,
                            X_train, Y_train_per_day,
                            Y_hat_train, Y_hat_mode_train, Y_hat_train_predict_s_0, Y_hat_mode_train_predict_s_0,
                            Y_test_per_day,
                            Y_hat_test, Y_hat_mode_test, Y_hat_test_predict_s_0, Y_hat_mode_test_predict_s_0,
                            n_split, I_idx, C_idx,
                            prediction_plot_dir, stamp
                            ):
    
    # Repeat evaluation for each provided metric
    for evaluation_metric in exec_mode['prediction']['prediction_eval_metrics'].split(','):
        # Evaluate prediction accuracy
        pred_loss_train, pred_loss_test=eval_model_prediction_metric(
                                            pred_loss_train, pred_loss_test,
                                            model_name, evaluation_metric,
                                            Y_hat_train, Y_train_per_day,
                                            Y_hat_test, Y_test_per_day,
                                            n_split,I_idx,C_idx
                                            )
        
        # If we want to accommodate s_0 
        if exec_mode['prediction']['predict_s_0'] and 'generative' in model_name and 'skipped_cycles' in fitted_model.__class__.__name__:
            pred_loss_train, pred_loss_test=eval_model_prediction_metric(
                                                pred_loss_train, pred_loss_test,
                                                '{}_predict_s_0'.format(model_name), evaluation_metric,
                                                Y_hat_train_predict_s_0, Y_train_per_day,
                                                Y_hat_test_predict_s_0, Y_test_per_day,
                                                n_split,I_idx,C_idx
                                                )
        # Also try with mode predictions
        if 'generative' in model_name:
            # Evaluate prediction accuracy
            pred_mode_loss_train, pred_mode_loss_test=eval_model_prediction_metric(
                                            pred_mode_loss_train, pred_mode_loss_test,
                                            model_name, evaluation_metric,
                                            Y_hat_mode_train, Y_train_per_day,
                                            Y_hat_mode_test, Y_test_per_day,
                                            n_split,I_idx,C_idx
                                            )
            
            # If we want to accommodate s_0 
            if exec_mode['prediction']['predict_s_0'] and 'generative' in model_name and 'skipped_cycles' in fitted_model.__class__.__name__:
                pred_mode_loss_train, pred_mode_loss_test=eval_model_prediction_metric(
                                                    pred_mode_loss_train, pred_mode_loss_test,
                                                    '{}_predict_s_0'.format(model_name), evaluation_metric,
                                                    Y_hat_mode_train_predict_s_0, Y_train_per_day,
                                                    Y_hat_mode_test_predict_s_0, Y_test_per_day,
                                                    n_split,I_idx,C_idx
                                                    )
                                                    
    
    # If necessary to plot prediction with respect to user regularity
    # TODO: This plotting should be replicated for all metrics
    if 'plot_reg_vs_pred' in exec_mode['prediction']:
        plot_reg_vs_pred(X_train, np.array(Y_hat_train), np.array(Y_train_per_day), model_name, prediction_plot_dir, stamp)
        if Y_hat_train_predict_s_0 is not None:
            plot_reg_vs_pred(X_train, np.array(Y_hat_train_predict_s_0), np.array(Y_train_per_day), model_name+'_predict_s_0', prediction_plot_dir, stamp)

    
    # Return dictionary of evaluated losses
    return pred_loss_train, pred_loss_test, pred_mode_loss_train, pred_mode_loss_test

# Per-model and evaluation metric prediction evaluation
def eval_model_prediction_metric(pred_loss_train, pred_loss_test, model_name, evaluation_metric, Y_hat_train, Y_true_train, Y_hat_test, Y_true_test, n_split, I_idx, C_idx):
    # Define metric evaluation function
    # Evaluation function not via sklearn, to accommodate NaNs
    evaluation_function=eval('my_{}'.format(evaluation_metric))
    
    # Training
    pred_loss_train[model_name][evaluation_metric][n_split,I_idx,C_idx,:] = evaluation_function(
                                                                                    Y_hat_train,
                                                                                    Y_true_train,
                                                                                    multioutput='raw_values'
                                                                                    )
    
    if pred_loss_test is not None:
        pred_loss_test[model_name][evaluation_metric][n_split,I_idx,C_idx,:] = evaluation_function(
                                                                            Y_hat_test,
                                                                            Y_true_test,
                                                                            multioutput='raw_values'
                                                                            )
                                                                            
    # For RMSE
    if evaluation_metric == 'mean_squared_error':
        pred_loss_train[model_name]['root_mean_squared_error'][n_split,I_idx,C_idx,:] = np.sqrt(
            pred_loss_train[model_name][evaluation_metric][n_split,I_idx,C_idx,:]
            )
        if pred_loss_test is not None:
            pred_loss_test[model_name]['root_mean_squared_error'][n_split,I_idx,C_idx,:] = np.sqrt(
                pred_loss_test[model_name][evaluation_metric][n_split,I_idx,C_idx]
                )
                                            
    # Return dictionary
    return pred_loss_train, pred_loss_test                                                           


###############################
### Calibration utils
###############################
def preallocate_prediction_calibration_results(prediction_eval_calibration, prediction_models, train_test_splits, train_test_ratio, I_range, C_range, day_range, predict_s_0=False):
    if day_range is None:
        day_range = np.array([0])

    '''preallocates evaluation data for pred_calib_train (and pred_calib_test if train_test_ratio < 1)'''
    pred_calib_test=None
    # Preallocate evaluation data
    pred_calib_train={
        prediction_model:{
                            metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in filter(lambda m: 'plot' not in m, prediction_eval_calibration)
                            } for prediction_model in prediction_models
                }
    # In case we are evaluating generative models with skipped cycles, add prediction assuming predict_s_0
    if predict_s_0:
        for prediction_model in filter(lambda model: 'skipped_cycles' in model, prediction_models):
            pred_calib_train['{}_predict_s_0'.format(prediction_model)]={
                                    metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in filter(lambda m: 'plot' not in m, prediction_eval_calibration)
                                }

    # If test data as well        
    if train_test_ratio<1:
        pred_calib_test={
            prediction_model:{
                                metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in filter(lambda m: 'plot' not in m, prediction_eval_calibration)
                                } for prediction_model in prediction_models
                    }
        # In case we are evaluating generative models with skipped cycles, add prediction assuming predict_s_0
        if predict_s_0:
            for prediction_model in filter(lambda model: 'skipped_cycles' in model, prediction_models):
                pred_calib_test['{}_predict_s_0'.format(prediction_model)]={
                                        metric:np.zeros((train_test_splits, I_range.size, C_range.size, day_range.size)) for metric in filter(lambda m: 'plot' not in m, prediction_eval_calibration)
                                        }
    
    # Return dictionary
    return pred_calib_train, pred_calib_test



# Per-model evaluation of all calibration metrics
def eval_model_calibration(exec_mode, model_name, fitted_model,
                            pred_calib_train, pred_calib_test,
                            Y_train, Y_test, day_range,
                            predictions_train, predictions_train_s_0,
                            predictions_test, predictions_test_s_0,
                            n_split,I_idx,C_idx,
                            calibration_plot_dir, stamp
                        ):
    # Truth
    y_true_train=Y_train[:,0]
    y_true_test=None

    # Just make sure predictive distributions exist
    if 'generative' in model_name:
        # Predictions are probabilistic
        predictions_train_pmf=predictions_train['pmf']
        predictions_train_s_0_pmf=predictions_train_s_0['pmf']
        if Y_test is not None:
            y_true_test=Y_test[:,0]
            predictions_test_pmf=predictions_test['pmf']
            predictions_test_s_0_pmf=predictions_test_s_0['pmf']
        else:
            predictions_test_pmf=None
            predictions_test_s_0_pmf=None
            
    else:
        # 0/1 predictions as pmf, using default x_predict_max
        predictions_train_pmf=np.zeros((Y_train.shape[0],day_range.max()+1,x_predict_max_default))
        predicted_day=np.round(predictions_train[:,0]).astype(int)
        predicted_day[predicted_day>x_predict_max_default]=x_predict_max_default
        predictions_train_pmf[np.arange(Y_train.shape[0]),:,predicted_day]=1
        if Y_test is not None:
            y_true_test=Y_test[:,0]
            predictions_test_pmf=np.zeros((Y_test.shape[0],day_range.max()+1,x_predict_max_default))
            predicted_day=np.round(predictions_test[:,0]).astype(int)
            predicted_day[predicted_day>x_predict_max_default]=x_predict_max_default
            predictions_test_pmf[np.arange(Y_test.shape[0]),:,predicted_day]=1
        else:
            predictions_test_pmf=None

    # For each calibration metric
    for calibration_metric in exec_mode['prediction']['prediction_eval_calibration'].split(','):
        # Compute
        pred_calib_train, pred_calib_test=eval_model_calibration_metric(
                                            pred_calib_train, pred_calib_test,
                                            model_name, calibration_metric,
                                            predictions_train_pmf, y_true_train,
                                            predictions_test_pmf, y_true_test,
                                            n_split,I_idx,C_idx,
                                            calibration_plot_dir=calibration_plot_dir, stamp=stamp
                                            )
        
        # If we want to accommodate s_0 
        if exec_mode['prediction']['predict_s_0'] and 'skipped_cycles' in fitted_model.__class__.__name__:
            pred_calib_train, pred_calib_test=eval_model_calibration_metric(
                                                pred_calib_train, pred_calib_test,
                                                '{}_predict_s_0'.format(model_name), calibration_metric,
                                                predictions_train_s_0_pmf, y_true_train,
                                                predictions_test_s_0_pmf, y_true_test,
                                                n_split,I_idx,C_idx,
                                                calibration_plot_dir=calibration_plot_dir, stamp=stamp
                                                )

    # Return model calibration dictionary results
    return pred_calib_train, pred_calib_test

# Per-model and calibration metric evaluation
def eval_model_calibration_metric(pred_calib_train, pred_calib_test, model_name, calibration_metric, predictive_train_pmf, Y_true_train, predictive_test_pmf, Y_true_test, n_split, I_idx, C_idx, calibration_plot_dir=None, stamp=None):
    # Define calibration function
    calibration_function=eval('my_{}'.format(calibration_metric.replace('_cumulative', '').split('_alpha')[0]))
    
    # Training
    if 'plot' in calibration_metric:
        save_dir = '{}/pit_mcp'.format(calibration_plot_dir)
        os.makedirs(save_dir, exist_ok=True)
        filename='{}/{}_{}_train_{}'.format(save_dir, model_name, calibration_metric, stamp)
        
        # Compute calibration
        calibration_result=calibration_function(
                            predictive_train_pmf,
                            Y_true_train,
                            )                 
        # Plot calibration
        eval('plot_{}'.format(calibration_metric))(
                                calibration_result,
                                day_range=np.arange(predictive_train_pmf.shape[1]),
                                y_range=np.arange(predictive_train_pmf.shape[2]),
                                plot_filename=filename
                                )
        # Save calibration plot data
        with gzip.open('{}.picklegz'.format(filename), 'wb') as f:
            pickle.dump(calibration_result, f)
        
    else:
        if 'cumulative' in calibration_metric:
            pred_calib_train[model_name][calibration_metric][n_split,I_idx,C_idx,:] = calibration_function(
                                                                                        predictive_train_pmf,
                                                                                        Y_true_train,
                                                                                        average=False,
                                                                                        cumulative=True
                                                                                        )
        elif 'alpha' in calibration_metric:
            pred_calib_train[model_name][calibration_metric][n_split,I_idx,C_idx,:] = calibration_function(
                                                                                        predictive_train_pmf,
                                                                                        Y_true_train,
                                                                                        alpha=float(calibration_metric.split('_alpha')[-1])/100,
                                                                                        average=True,
                                                                                        )
        else:
            pred_calib_train[model_name][calibration_metric][n_split,I_idx,C_idx,:] = calibration_function(
                                                                                        predictive_train_pmf,
                                                                                        Y_true_train,
                                                                                        average=True
                                                                                        )
    if pred_calib_test is not None:
        if 'plot' in calibration_metric:
            filename='{}/{}_{}_test_{}'.format(save_dir, model_name, calibration_metric, stamp)
            
            # Compute calibration
            calibration_function(
                                predictive_train_pmf,
                                Y_true_train,
                                )                
            # Plot calibration
            eval('plot_{}'.format(calibration_metric))(
                                    calibration_result,
                                    day_range=np.arange(predictive_train_pmf.shape[1]),
                                    y_range=np.arange(predictive_train_pmf.shape[2]),
                                    plot_filename=filename
                                    )
            # Save calibration plto data
            with gzip.open('{}.picklegz'.format(filename), 'wb') as f:
                pickle.dump(calibration_result, f)
        else:
            if 'cumulative' in calibration_metric:
                pred_calib_test[model_name][calibration_metric][n_split,I_idx,C_idx,:] = calibration_function(
                                                                                        predictive_test_pmf,
                                                                                        Y_true_test,
                                                                                        average=False,
                                                                                        cumulative=True
                                                                                        )
            elif 'alpha' in calibration_metric:
                pred_calib_test[model_name][calibration_metric][n_split,I_idx,C_idx,:] = calibration_function(
                                                                                            predictive_test_pmf,
                                                                                            Y_true_test,
                                                                                            alpha=float(calibration_metric.split('_alpha')[-1])/100,
                                                                                            average=True,
                                                                                            )
            else:
                pred_calib_test[model_name][calibration_metric][n_split,I_idx,C_idx,:] = calibration_function(
                                                                                        predictive_test_pmf,
                                                                                        Y_true_test,
                                                                                        average=True
                                                                                        )
    # Return dictionary
    return pred_calib_train, pred_calib_test

