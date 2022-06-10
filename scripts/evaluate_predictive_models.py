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
# Import data functions
from data_functions import *
# Import plotting functions
from plotting_functions import *
# Import evaluation functions
from model_evaluation_functions import *
# And other helpful functions
from evaluation_utils import *

# Import baseline models
from baseline import *
# Import neural network models
from neural_network_models import *

# Import Generative models
from poisson_with_skipped_cycles_models import *
from generalized_poisson_with_skipped_cycles_models import *  

# Main code
def main(data_model, save_data_dir, hyperparameters, I, C, train_test_ratio, train_test_splits, prediction_type, C_init_online, C_step_online, I_init_online, I_step_online, model_names, exec_mode, exec_stamp, data_stamp, shuffle_C):   
    
    ######################## SCRIPT EXECUTION SET-UP ###########################
    # Online or batch processing
    # Default number of Is if not specified
    if I == -1:
        I = get_full_I_size(data_model, save_data_dir, I, C, hyperparameters)
        print('Using full I, I = '+str(I))

    I_range, C_range, I_init = prep_I_C_ranges(prediction_type, I, C, C_init_online, C_step_online, I_init_online, I_step_online)
    ######################## DATA GENERATION ###########################
    print('Getting data...')
    true_N, true_S, true_params, hyperparameters, main_dir = get_data(data_model, save_data_dir, I, I_init, C, hyperparameters, train_test_splits, data_stamp, shuffle_C)
    print('... data ready!')
    
    ######################## SCRIPT RESULTS ALLOCATION ###########################
    (
        fit_results, fit_model_dir, fit_plot_dir,
        inference_results, inference_plot_dir,
        predict_mode, prediction_plot_dir, prediction_posterior_dir, day_range, 
        pred_loss_train, pred_loss_test, pred_mode_loss_train, pred_mode_loss_test,
        pred_calib_train, pred_calib_test, calibration_plot_dir
    ) = preallocate_resources_based_on_exec_mode(main_dir, model_names, exec_mode, train_test_splits, train_test_ratio, I_range, C_range)
    
    ######################## SCRIPT EXECUTION ###########################
    # For each train_test_split
    for n_split in np.arange(train_test_splits):
        print('Splitting the data... {}/{}'.format(n_split, train_test_splits))

        # For each number of Individuals, we will pick the first this_I
        for this_I_idx, this_I in enumerate(I_range):
            # For each training length
            for this_C_idx, C_train in enumerate(C_range):
                # split_stamp
                split_stamp='split{}_I_{}_C_{}_{}'.format(n_split, this_I, C_train,exec_stamp)
                
                # Train/test
                if train_test_ratio==1:
                    X_train=true_N[:this_I,:C_train] # X is first C_train cycles
                    X_test=None
                    Y_train=true_N[:this_I,C_train:C_train+1] # Y is next cycle
                    Y_test=None
                    idx_train=np.arange(this_I) # to keep track of train/test indexes
                    idx_test=None
                    
                    # If we have access to true parameters
                    if true_params is not None:
                        true_params_train={}
                        true_params_test={}
                        for param in true_params:
                            true_params_train[param]=true_params[param][idx_train]
                            true_params_test[param]=None
                    else:
                        true_params_train = None
                        true_params_test = None
                else:
                    # Train-test split of cycle data
                    X_train, X_test, Y_train, Y_test, idx_train, idx_test = train_test_split(
                                                            true_N[:this_I,:C_train], # X is first C_train cycles
                                                            true_N[:this_I,C_train:C_train+1], # Y is next cycle
                                                            np.arange(this_I), # to keep track of train/test indexes
                                                            train_size=train_test_ratio
                                                            )
                    
                    # If we have access to true parameters
                    if true_params is not None:
                        # Collect true parameters of model
                        true_params_train={}
                        true_params_test={}
                        for param in true_params:
                            true_params_train[param]=true_params[param][idx_train]
                            true_params_test[param]=true_params[param][idx_test]
                            
                    else:
                        true_params_train = None
                        true_params_test = None
                        
                ###############################
                ### For each model of interest
                ###############################
                for model_name in model_names:
                    print('Starting {} for split {}/{} and {} users with {} training cycles'.format(model_name, n_split, train_test_splits, this_I, C_train))
                    
                    ###############################
                    # Fit model
                    fitted_model=None
                    if 'fit' in exec_mode:
                        fitted_model = fit_model(
                                            model_name, data_model,
                                            exec_mode['fit'],
                                            this_I, C_train,
                                            true_params_train, X_train, Y_train,
                                            fit_model_dir, split_stamp
                                            )
                        print('\t... finished fitting {}'.format(model_name))
                    # Load model
                    elif 'load_fitted' in exec_mode:
                        fitted_model = load_fitted_model(
                                            model_name, main_dir, split_stamp
                                            )
                        print('\t... finished loading fitted model {}'.format(model_name))

                    # Evaluate fitting
                    if 'fit_eval_metrics' in exec_mode['fit']:
                        if fitted_model is not None: 
                            # Fitting evaluation
                            fit_results=eval_fit(
                                                fit_results, exec_mode['fit'],
                                                fitted_model, model_name, 
                                                n_split, this_I_idx, this_C_idx,
                                                fit_plot_dir, split_stamp
                                            )                        
                        else:
                            raise ValueError('Can not evaluate models if we do not fit or load fitted!')
                    
                    ###############################
                    # Now that we have a fitted model
                    if 'inference' in exec_mode:
                        # Inference evaluation
                        inference_results=eval_inference(
                                                    inference_results, exec_mode['inference']['inference_eval_metrics'],
                                                    fitted_model, model_name,
                                                    hyperparameters, exec_mode['inference']['plot_hyperparameters'],
                                                    true_params_train, exec_mode['inference']['parameter_posterior'], exec_mode['inference']['plot_parameters'], 
                                                    X_train, exec_mode['inference']['plot_data_statistics'], 
                                                    n_split, this_I_idx, this_C_idx,
                                                    inference_plot_dir, split_stamp
                                                    )
                            
                        print('\t... finished inference for {}'.format(model_name))
                    
                    ###############################
                    # Prediction
                    if 'prediction' in exec_mode:
                        print('\t... starting predict mode {} for {}'.format(predict_mode, model_name))
                        
                        ### True cycle-lengths to predict
                        # In case we are doing prediction by day
                        if day_range is not None:
                            # Repeat truth by day range
                            Y_train_per_day=np.repeat(Y_train, day_range.size, axis=1).astype(float)
                            Y_test_per_day=None
                            if train_test_ratio<1:
                                Y_test_per_day=np.repeat(Y_test, day_range.size, axis=1).astype(float)
                            # Then remove 'unreasonable' truth
                            Y_train_per_day[day_range>Y_train]=np.nan
                            if train_test_ratio<1:
                                Y_test_per_day[day_range>Y_test]=np.nan
                        else:
                            # There is only one day to evaluate
                            Y_train_per_day=Y_train
                            Y_test_per_day=None
                            if train_test_ratio<1:
                                Y_test_per_day=Y_test
                        
                        ### Model predictions
                        (predictions_train, Y_hat_train, predictions_test, Y_hat_test, 
                            predictions_train_s_0, Y_hat_train_predict_s_0, predictions_test_s_0, Y_hat_test_predict_s_0,
                            Y_hat_mode_train, Y_hat_mode_test, Y_hat_mode_train_predict_s_0, Y_hat_mode_test_predict_s_0
                            )=predict(
                                exec_mode, model_name, fitted_model, X_train, X_test, day_range, prediction_plot_dir, prediction_posterior_dir, split_stamp
                                )
                        print('\t... finished prediction for {}'.format(model_name))
                        
                        ### Per-model prediction evaluation
                        print('\t ...Evaluating {} prediction for split {}/{} and {} users with {} training cycles'.format(model_name, n_split, train_test_splits, this_I, C_train))
                        # Point estimates
                        if 'prediction_eval_metrics' in exec_mode['prediction']:
                            # Evaluate all predictions as needed
                            (pred_loss_train, pred_loss_test, pred_mode_loss_train, pred_mode_loss_test
                                )=eval_model_prediction(
                                    exec_mode, model_name, fitted_model,
                                    pred_loss_train, pred_loss_test, pred_mode_loss_train, pred_mode_loss_test,
                                    X_train, Y_train_per_day,
                                    Y_hat_train, Y_hat_mode_train, Y_hat_train_predict_s_0, Y_hat_mode_train_predict_s_0,
                                    Y_test_per_day,
                                    Y_hat_test, Y_hat_mode_test, Y_hat_test_predict_s_0, Y_hat_mode_test_predict_s_0,
                                    n_split, this_I_idx, this_C_idx,
                                    prediction_plot_dir, split_stamp
                            )
                            
                        # Calibration
                        if 'prediction_eval_calibration' in exec_mode['prediction']:
                            # Evaluate calibration as needed
                            (pred_calib_train, pred_calib_test)=eval_model_calibration(
                                exec_mode, model_name, fitted_model,
                                pred_calib_train, pred_calib_test,
                                Y_train, Y_test, day_range,
                                predictions_train, predictions_train_s_0,
                                predictions_test, predictions_test_s_0,
                                n_split,this_I_idx, this_C_idx,
                                calibration_plot_dir, split_stamp
                            )
                    
                    print('Finished evaluating {} for split {}/{} and {} users with {} training cycles'.format(model_name, n_split, train_test_splits, this_I, C_train))
                    
                    ###############################
                    # Reduce memory consumption
                    del fitted_model
                    ###############################
                    
                print('Done evaluating split {}/{}'.format(n_split, train_test_splits))
    
    ####################### EVALUATION ACROSS MODELS ###############################
    ### Fitting
    if 'fit' in exec_mode and 'fit_eval_metrics' in exec_mode['fit']:
        # save fit eval results
        fit_stamp='{}_I{}_C{}_nsplits_{}_{}'.format(data_model, I, C, str(train_test_splits), exec_stamp)
        with gzip.open('{}/fit_results_{}.picklegz'.format(main_dir, fit_stamp), 'wb') as f:
            pickle.dump(fit_results, f)
        
    ### Inference
    if 'inference' in exec_mode:
        # save inference results
        inf_stamp='{}_I{}_C{}_nsplits_{}_{}'.format(data_model, I,C, str(train_test_splits), exec_stamp)
        with gzip.open('{}/inference_results_{}.picklegz'.format(main_dir, inf_stamp), 'wb') as f:
            pickle.dump(inference_results, f)

        # plot inf results
        if true_params is not None:
            if (exec_mode['inference']['inference_eval_metrics'] == 'all') or ('hyperparameters' in exec_mode['inference']['inference_eval_metrics']):
                plot_hyperparam_inf_results_sim_data(inference_plot_dir, inference_results, fit_results, I_range, C_range)
            if (exec_mode['inference']['inference_eval_metrics'] == 'all') or ('parameters' in exec_mode['inference']['inference_eval_metrics']):
                plot_param_inf_results_sim_data(inference_plot_dir, inference_results, I_range, C_range)
        else:
            if (exec_mode['inference']['inference_eval_metrics'] == 'all') or ('parameters' in exec_mode['inference']['inference_eval_metrics']):
                plot_param_inf_results_real_data(inference_plot_dir, inference_results, I_range, C_range)
        
    ### Prediction
    if 'prediction' in exec_mode:
        # Save prediction results
        loss_stamp='{}_I{}_C{}_nsplits_{}_{}'.format(predict_mode, I,C, str(train_test_splits), exec_stamp)
        with gzip.open('{}/pred_loss_train_{}.picklegz'.format(main_dir, loss_stamp), 'wb') as f:
            pickle.dump(pred_loss_train, f)

        # Save Mode prediction for generative models
        if 'generative' in str(model_names):
            with gzip.open('{}/pred_mode_loss_train_{}.picklegz'.format(main_dir, loss_stamp), 'wb') as f:
                pickle.dump(pred_mode_loss_train, f)
            
        if pred_loss_test is not None:
            with gzip.open('{}/pred_loss_test_{}.picklegz'.format(main_dir, loss_stamp), 'wb') as f:
                pickle.dump(pred_loss_test, f)
                
            if 'generative' in str(model_names):
                with gzip.open('{}/pred_mode_loss_test_{}.picklegz'.format(main_dir, loss_stamp), 'wb') as f:
                    pickle.dump(pred_mode_loss_test, f)
        
        print('Saved all evaluation losses')
        
        # Plot losses
        predict_stamp='I{}_C{}_nsplits_{}_{}'.format(I,C,str(train_test_splits), exec_stamp)
        
        # Figure out metrics to plot
        # In principle, all evaluated
        eval_metric_to_plot=exec_mode['prediction']['prediction_eval_metrics'].split(',')
        # Replace MSE with RMSE
        if 'mean_squared_error' in eval_metric_to_plot:
            eval_metric_to_plot.remove('mean_squared_error')
            eval_metric_to_plot.append('root_mean_squared_error')
        
        # Plotting
        if C_range.size>1:
            # Train loss over training cycles
            plot_prediction_loss_over_training_cycles(
                                                        pred_loss_train,
                                                        I_range,
                                                        C_range,
                                                        evaluation_metrics=eval_metric_to_plot,
                                                        save_dir='{}/train'.format(prediction_plot_dir),
                                                        stamp='{}'.format(predict_stamp),
                                                        colors=default_colors,
                                                        day_range=day_range
                                                    )
            if pred_loss_test is not None:
                # Test loss over training cycles
                plot_prediction_loss_over_training_cycles(
                                                            pred_loss_test,
                                                            I_range,
                                                            C_range,
                                                            evaluation_metrics=eval_metric_to_plot,
                                                            save_dir='{}/test'.format(prediction_plot_dir),
                                                            stamp='{}'.format(predict_stamp),
                                                            colors=default_colors,
                                                            day_range=day_range
                                                        )
        
        
        # Prediction loss by day
        if predict_mode == 'prediction_by_day':            
            # Mean loss by day
            plot_predictive_posterior_mean_by_day_loss_over_models(
                                                                    pred_loss_train,
                                                                    day_range,
                                                                    I_range,
                                                                    C_range,
                                                                    evaluation_metrics=eval_metric_to_plot,
                                                                    save_dir='{}/loss_by_day'.format(prediction_plot_dir),
                                                                    stamp='{}'.format(predict_stamp)
                                                                    )
                                                                    
            if 'generative' in str(model_names):
                # Mode loss by day
                plot_predictive_posterior_mean_by_day_loss_over_models(
                                                                    pred_mode_loss_train,
                                                                    day_range,
                                                                    I_range,
                                                                    C_range,
                                                                    evaluation_metrics=eval_metric_to_plot,
                                                                    save_dir='{}/loss_by_day'.format(prediction_plot_dir),
                                                                    stamp='mode_{}'.format(predict_stamp)
                                                                    )
        
    # Prediction calibration
    if 'prediction' in exec_mode and 'prediction_eval_calibration' in exec_mode['prediction']:
        # Save prediction calibration results
        calib_stamp='I{}_C{}_nsplits_{}_{}'.format(I,C,str(train_test_splits), exec_stamp)
        with gzip.open('{}/pred_calib_train_{}.picklegz'.format(main_dir, calib_stamp), 'wb') as f:
            pickle.dump(pred_calib_train, f)
                        
        if pred_calib_test is not None:
            with gzip.open('{}/pred_calib_test_{}.picklegz'.format(main_dir, calib_stamp), 'wb') as f:
                pickle.dump(pred_calib_test, f)
        
        print('Saved all evaluation calibration')
        
        # Plot all evaluation calibration results
        # Training
        plot_predictive_posterior_calibration_by_day_over_models(
                                                                    pred_calib_train,
                                                                    day_range,
                                                                    I_range,
                                                                    C_range,
                                                                    calibration_metrics=exec_mode['prediction']['prediction_eval_calibration'].split(','),
                                                                    save_dir=calibration_plot_dir,
                                                                    stamp='{}'.format(calib_stamp)
                                                                    )
        
        if pred_calib_test is not None:
            # Testing
            plot_predictive_posterior_calibration_by_day_over_models(
                                                                        pred_calib_test,
                                                                        day_range,
                                                                        I_range,
                                                                        C_range,
                                                                        calibration_metrics=exec_mode['prediction']['prediction_eval_calibration'].split(','),
                                                                        save_dir=calibration_plot_dir,
                                                                        stamp='{}'.format(calib_stamp)
                                                                        )
    
    print('FINISHED!')
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Input parser
    parser = argparse.ArgumentParser(description='Evaluate predictive models')
    parser.add_argument('-data_model', type=str, default='real', help='Data model: real, poisson, generalized_poisson, load')
    parser.add_argument('-save_data_dir', type=str, default=None, help='Whether to save the data generated by the model, or if data_model==load where to read the data from')
    parser.add_argument('-hyperparameters', nargs='+', type=float, default=None, help='Hyperparameters for the data model')
    parser.add_argument('-I', type=int, default=10, help='Number of individuals')
    parser.add_argument('-C', type=int, default=10, help='Number of cycles per individual')
    parser.add_argument('-train_test_ratio', type=float, default='0.75', help='Train/test ratio')
    parser.add_argument('-train_test_splits', type=int, default='5', help='Train/test splits')
    parser.add_argument('-prediction_type', type=str, default='batch', help='Whether to predict based on batch or online data')
    parser.add_argument('-C_init_online', type=int, default=None, help='Number of minimum cycles to start for online evaluation')
    parser.add_argument('-C_step_online', type=int, default=None, help='Number of cycles to step in cycle range for online evaluation')
    parser.add_argument('-I_init_online', type=int, default=None, help='Number of minimum individuals to start with for online evaluation')
    parser.add_argument('-I_step_online', type=int, default=None, help='Number of individuals to to step in individual range for online evaluation')
    parser.add_argument('-model_names', nargs='+', type=str, default='baseline_mean', help='What models to evaluate')
    parser.add_argument('-exec_mode_file', type=str, default='exec_all', help='String to execution config file')
    parser.add_argument('-exec_stamp', type=str, default='', help='Stamp to identify results from this execution')
    parser.add_argument('-data_stamp', type=str, default='', help='Stamp to identify cycle length data file; use this for when running multiple random runs')
    parser.add_argument('-shuffle_C', type=str, default='no', help='Whether to shuffle cycles')
    # Get arguments
    args = parser.parse_args()
    
    # If loading data
    if args.data_model == 'load':
        # save_data_dir must exist and have cycle_lengths.npz file
        assert os.path.isdir(args.save_data_dir) and os.path.isfile('{}/cycle_lengths.npz'.format(args.save_data_dir)) , '{} must exist and have cycle_lengths.npz file within'.format(args.save_data_dir)
        
    # Given hyperparameters
    if args.hyperparameters != None:
        if args.data_model == 'real':
            assert len(args.hyperparameters)==3, 'Not enough true hyperparameters={} provided for {} data model'.format(args.hyperparameters, args.data_model)
            hyperparameters={
                'clean':bool(args.hyperparameters[0]),
                'regular_proportion':None if args.hyperparameters == 'None' else float(args.hyperparameters[1]),
                'random':bool(args.hyperparameters[2])
                }
        elif args.data_model == 'toy':
            assert len(args.hyperparameters)==3, 'Not enough true hyperparameters={} provided for {} data model'.format(args.hyperparameters, args.data_model)
            hyperparameters={
                'user_type':str(args.hyperparameters[0]),
                'mean':float(args.hyperparameters[1]),
                'variance':float(args.hyperparameters[2])
                }                  
        elif args.data_model == 'poisson':
            assert len(args.hyperparameters)==4, 'Not enough true hyperparameters={} provided for {} data model'.format(args.hyperparameters, args.data_model)
            hyperparameters=np.array(args.hyperparameters)
        elif args.data_model == 'generalized_poisson':
            if len(args.hyperparameters)==6:
                kappa,gamma,alpha_xi,beta_xi,alpha,beta=args.hyperparameters
                # default
                xi_max=1
                x_max=float('inf')
                hyperparameters=np.array([kappa,gamma,alpha_xi,beta_xi,xi_max,x_max,alpha,beta])
            elif len(args.hyperparameters)==8:
                hyperparameters=np.array(args.hyperparameters)
            else:
                raise ValueError('Not enough true hyperparameters={} provided for {} data model'.format(args.hyperparameters, args.data_model))
    else:
        # Default hyperparameters
        if args.data_model == 'load':
            # will be read from file, so
            hyperparameters=None
        elif args.data_model == 'real':
            hyperparameters={
                'clean':False,
                'regular_proportion':None,
                'random':False
                }
        elif args.data_model == 'toy':
            hyperparameters={
                'user_type':'regular',
                'mean':30,
                'variance':1
                }
        elif args.data_model == 'poisson':
            # Gamma distribution
            true_kappa=180.
            true_gamma=6.
            # Beta distribution
            true_alpha=2.
            true_beta=20.
            # All hyperparameters
            hyperparameters=[true_kappa, true_gamma, true_alpha, true_beta]
        elif args.data_model == 'generalized_poisson':
            # Gamma distribution for lambda
            true_kappa=180.
            true_gamma=4.
            # Xi distribution for pi
            true_alpha_xi=2.
            true_beta_xi=20.
            xi_max=1
            x_max=float('inf')
            # Beta distribution for pi
            true_alpha=2.
            true_beta=20.
            # All hyperparameters
            hyperparameters=[true_kappa, true_gamma, true_alpha_xi, true_beta_xi, xi_max, x_max, true_alpha, true_beta]
        else:
            raise ValueError('Unknown {} data model'.format(args.data_model))
    
    # Reasonable number of individuals and cycles
    assert args.I>=-1 and args.C>0, 'Not enough individuals {} and cycles {}'.format(args.I, args.C)
    if args.prediction_type == 'online':
        if args.C_init_online is not None:
            assert args.C_init_online>0, 'Not enough cycles to start with {}'.format(args.C_init_online)
        if args.C_step_online is not None:
            assert args.C_step_online>0, 'Not enough cycles to range with {}'.format(args.C_step_online)
        if args.I_init_online is not None:
            assert args.I_init_online>0, 'Not enough individuals to start with {}'.format(args.I_init_online)
        if args.I_step_online is not None:
            assert args.I_step_online>0, 'Not enough individuals to range with {}'.format(args.I_step_online)
    
    # train/test
    assert args.train_test_ratio>0 and args.train_test_ratio<=1, 'Invalid train/test ratio'
    assert args.train_test_splits>0, 'At least 1 train/test split is required'
    
    # Process exec_mode config file
    exec_mode=load_exec_mode_config(args.exec_mode_file)
    
    # Call main function
    main(args.data_model, args.save_data_dir, hyperparameters, args.I, args.C, args.train_test_ratio, args.train_test_splits, args.prediction_type, args.C_init_online, args.C_step_online, args.I_init_online, args.I_step_online, args.model_names, exec_mode, args.exec_stamp, args.data_stamp, args.shuffle_C)
