# Introduction

This repo is the codebase for implementing generative, predictive models for menstrual cycle length predictions based on self-tracked mobile health data.

Cycle lengths are computed based on user reports of when bleeding occurred.

The codebase contains code to

1. load cycle-length data or generate simulated cycle length data, of size I (number of users) x C (number of cycle lengths per user),
2. define and fit a predictive model using PyTorch and learn population-wide model hyperparameters, given simulated or real data
3. generate predictions that are updated on each day of the next cycle
4. evaluate model performance (fitting, inference, prediction accuracy and calibration), via different metrics and plots

# 1) Data description and utilities

The input data for all models is an array/tensor of cycle-lengths per user:
    i.e., C cycle lengths for each user in the cohort of I users.
    That is, the dataset is an array of size I x C, where each (i,c) entry is cycle length c for user i.

Helper function is [get_data](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/data_functions.py#L119 "get_data function")
    
    - Input
        - Data_model: whether to “load” or “simulate data”
    - Output
        - Loaded data information: I,C, cycle_lengths

## 1.a) Loading data: *data_model == 'load'*

Key input is *save_data_dir*

    - The function expects data path to be '{}/cycle_lengths.npz'.format(save_data_dir).

    - The cycle_lengths.npz object is a pickled dictionary with the following key elements:
        - loaded_data['I']: number of users
        - loaded_data['C']: number of cycles per-user
        - loaded_data['cycle_lengths']: I times C array with cycle lengths

## 1.b) Generating simulated data: *data_model != 'load'*

Key input parameters are *hyperparameters* and *save_data_dir*
    
    - Hyperparameters of each generative model (see details below)
        E.g., if drawing cycle lengths from a Poisson, kappa and gamma for a Gamma distribution, and alpha and beta for a Beta distribution

    - *save_data_dir* is the directory where to save simulated data

# 2) Model description

We implement generative statistical models for menstrual cycle lengths, where we hypothesize that observed cycle lengths for each individual are a function of true cycle behavior and the existence of errors in self-tracking (see model details in doc and publications).

The parameters that determine these individual-level factors (expected cycle length, self-tracking probability) are drawn from probability distributions that are described by population-level hyperparameters.

These population-level hyperparameters are learned when we fit the model by minimizing the negative log-likelihood of the observed data (inference details are model specific, see publications for details)

## 2.a) Poisson generative model

In this [model](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/poisson_with_skipped_cycles_models.py "Poisson generative model with skipped cycles"), cycle lengths are drawn from a Poisson distribution.

- Hyperparameters (to be learned)
    
    - [kappa and gamma for Poisson parameter gamma prior](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/poisson_with_skipped_cycles_models.py#L52)
    
    - [alpha and beta for Beta prior on skipping probabilities](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/poisson_with_skipped_cycles_models.py#L55)
    
- [Class initialization](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/poisson_with_skipped_cycles_models.py#L39)
    
    - Input:
        - hyperparameters kappa, gamma, alpha, beta (if not interested in defaults)

- Call to [fit the generative process](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/poisson_with_skipped_cycles_models.py#L661)
    
    - Input variables
        - X is the data to fit to, numpy array or torch Tensor of size I by C
        - Optimizer and criterion are 2 key inputs for optimization
        - Batch_size can be used if dataset is too big to optimize without using mini-batching

    - Output variables
        - None
        - Fit directly changes class hyperparameters attributes

- Call to [predict with the generative process](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/poisson_with_skipped_cycles_models.py#L795)
    
    - Prediction will be done based on class hyperparameter attributes, i.e., this should have been either initialized to values of interest or learned by fitting
    
    - Input variables
        - X is the data used to compute per-user posteriors, i.e., X is of shape I times C.
            Recall that these X might or might not be the same data used for fitting the model: the input data X for predictions can be from a different set of users, from which data will be used to compute predictive posteriors based on these users' data, based on fitted hyperparameters. 
        - posterior_type: whether only mean predictions or full predictive posterior predictions are computed (i.e., full predictive probability mass function)
        - day_range: days after last cycle for which predictions need to be computed

    - Output variables
        - Predictive: the model's predicted posterior, per-user and per-days (day_range) we want predictions at

# 2.b) Generalized Poisson

In this [model](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/generalized_poisson_with_skipped_cycles_models.py "Generalized Poisson generative model with skipped cycles"), cycle lengths are drawn from a [Generalized Poisson distribution](https://www.jstor.org/stable/1267389), which has 2 degrees of freedom allowing different mean and variances for cycle-lengths.

- Hyperparameters (to be learned), all defined in log-space
    
    - [kappa and gamma for Generalized Poisson's $\lambda$ parameter's gamma prior](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/generalized_poisson_with_skipped_cycles_models.py#L76)
    - [alpha and beta for Generalized Poisson's $\xi$ parameter's beta prior](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/generalized_poisson_with_skipped_cycles_models.py#L79)
    - [alpha and beta for Beta prior on skipping probabilities](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/src/prediction/generalized_poisson_with_skipped_cycles_models.py#L85)
        
- Init, fit and predict functions operate as for Poisson model above, with Generalized Poisson model specific computations of log-normalizing constant


# 3) Script for basic generative Poisson predictive model execution

This is a [basic script](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py "Fit and predicti with generative model") that illustrates how to load cycle-length data to fit a generative Poisson cycle length model and make predictions using such model (in the same set of users)

Example usage:
```bash
python3 poisson_model_fit_predict.py \
    -data_dir ../data/cycle_length_data \
        # Load data from ../data/cycle_length_data/cycle_lengths.npz
    -model_name generative_poisson_with_skipped_cycles
        # Fit and predict using the generative poisson model, with parameters in ./generative_model_config/generative_poisson_with_skipped_cycles
```

Summary of script

- [Data-loading](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L38)
    - Loads pickled ../data/cycle_length_data/cycle_lengths.npz dataset
    - And gets dataset details I,C, and X

- [Model instantiation and configuration](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L48), from input config file
    - [Instantiate model](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L54), as determined by information in ./generative_model_config/generative_poisson_with_skipped_cycles

    - Load model [fitting](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L65), [optimization](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L96), and [prediction](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L111) parameters from ./generative_model_config/generative_poisson_with_skipped_cycles

- [Fitting](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L147)
    - Fit the model to data X, given optimizer, criterion, monte carlo samples M and other_fitting_args, all loaded in model configuration step

- [Prediction](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L154)
    - Leverage the fitted model to predict, for which we use the same dataset: i.e., use the same data to compute per-user posteriors, then predict next cycle length
    - Prediction is done for the specified *day_range* (0,30) with only *'mean'*, i.e., expected cycle-length predictions
    - The output of the call is a set of model predictions for each day specified in *day_range*:
        i.e., a dictionary, with only 'mean' as key, with a numpy array of size *I* by length of *day_range* as values:
            my_model_predictions['mean'][i,d] is the model's predicted cycle length for user i in day d

- The script prints the model's cycle-length predictions [at day 0](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L167) and [at day 20](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/poisson_model_fit_predict.py#L170) of the next cycle 
    
# 4) Script for predictive model execution and evaluation

This is a wrapper, [general script](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/evaluate_predictive_models.py "evaluate predictive models") that fits, predicts, and evaluates a set of models of interest (baseline, generative and neural network based) for a given cycle-length dataset.

The script operates based on script input parameters (see description below) and execution and model config files:

- Input execution config file:
    - [Example execution config files](https://github.com/iurteaga/menstrual_cycle_analysis/tree/master/scripts/execution_config)
    - The config file specifies which parts of the script to run (fit, inference, prediction) and their configuration parameters, as well as which evaluations to compute and plot
        
        For instance, for prediction task:
            - 'prediction_eval_metrics' is the list of metrics to use for evaluation
            - 'day_range' specifies number of days to predict into the future
            - 'predictive_posterior' determines whether to make point predictions for cycle length ('mean') or to predict the whole probability distribution for cycle length ('pmf') 

- Input model config file, as specified by model-name argument:
    - [Example generative model config files](https://github.com/iurteaga/menstrual_cycle_analysis/tree/master/scripts/generative_model_config)
        - e.g., for [a generative Poisson model](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/generative_model_config/generative_poisson_with_skipped_cycles)
    - The config files specify:
        - Model hyperparameters
        - Maximum number of skipped cycles to consider in fitting (s_max)
        - Model fitting criterion and parameters, including max number of epochs and convergence criteria
        - Optimizer type and learning rate to use
        - Model prediction criterion, including number of Monte Carlo samples and number of maximum skipped cycles to consider in prediction (s_predict)

- Script description
    - [Fit_model function](https://github.com/iurteaga/menstrual_cycle_analysis/blob/fc050fe77def5076f6a32d400c7cca193c9a6ba4/scripts/evaluation_utils.py#L300)
        - General fit function, fits specified model (as determined by 'model_name' argument)
            
            - [fit] section in 'exec_mode' execution config file specifies model fitting parameters 
            - 'I_train' and 'C_train' specify size of training data
            - If using simulated data, can specify 'true_params' for comparison
            - 'X' and 'Y' specify input and output data
                Unnecessary for generative models, useful for neural network fitting
            - 'fit_model_dir' specifies where to save fitted model and 'stamp' is a filename stamp to distinguish fitted model from others 

    -[Predict function](https://github.com/iurteaga/menstrual_cycle_analysis/blob/fc050fe77def5076f6a32d400c7cca193c9a6ba4/scripts/evaluation_utils.py#L836)
        - General prediction function, makes predictions for specified model (as determined by 'model_name' argument)
            
            - [prediction] section in 'exec_mode' execution config file specifies prediction parameters 
            - 'fitted_model' object to use for predictions
            - 'X_train' and 'X_test' passed as input
                X_train and X_test will be used to compute model posteriors
                These 2 inputs are used to evaluate predictions for the users that were used for fitting (X_train) Vs new unseen users (X_test)
            - 'day_range' specifies number of days to predict into the future
            - 'prediction_plot_dir' and 'prediction_posterior_dir' specify save directories for prediction plots and predictions 
            - 'stamp' specifies filename stamp for prediction results

    - Evaluation functions, as different aspects of the model training and prediction can be evaluated
        - [Fitting evaluation](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/evaluate_predictive_models.py#L164)
        - [Inference evaluation](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/evaluate_predictive_models.py#L177)
        - [Prediction accuracy evaluation](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/evaluate_predictive_models.py#L228)
        - [Prediction calibration evaluation](https://github.com/iurteaga/menstrual_cycle_analysis/blob/master/scripts/evaluate_predictive_models.py#L242)
        
- Example usage of the script:
```bash
python3 evaluate_predictive_models.py \
    -data_model load -save_data_dir ./your_favorite_directory \
        #Load data from ./your_favorite_directory/cycle_lengths.npz
    -I 5000 -C 11 \ 
        # Information about data: 5000 users, 11 cycles available. If want to use full size of dataset, once can use I=-1
    -train_test_ratio 1.0 -train_test_splits 1.0 \
        # Do not split users in train/test sets (i.e., use C-1 cycles from each user to fit model, predict the next cycle length for each user)
    -model_names generative_poisson_with_skipped_cycles \
        # Fit and predict using the generative poisson model, with parameters in ./generative_model_config/generative_poisson_with_skipped_cycles
    -exec_mode_file exec_fit_plot_data_predict_no_plots_no_save
        # Execute based on config file ./execution_config/exec_fit_plot_data_predict_no_plots_no_save (Fit model and predict, no intermediary plots and only save fitted model)
    -prediction_type
        # Predict based on batch (whole batch of I) or online (update based on increased amount of data in I or C)
    -C_init_online C_step_online I_init_online I_step_online
        # If evaluating in 'online' mode, specificies starting point for I, C and step size; note that this is not 'true' online evaluation, since the full dataset is provided (and then subsets are selected based on these parameters)
    -exec_stamp
        # File stamp to distinguish results from other runs
    -data_stamp
        #File stamp to distinguish results from other runs if difference is based on data (e.g., if data was shuffled)
```

