# Basic example of how to load cycle-length data to fit a generative Poisson cycle length model and make predictions using such model

```bash
python3 -m pdb poisson_model_fit_predict.py -data_dir ../data/cycle_length_data -model_name generative_poisson_with_skipped_cycles
```

# Examples of how to run and evaluate different models

## Simulated data

### Poisson generative model, evaluate generative models for I=5000 users

```bash
python3 evaluate_predictive_models.py -data_model poisson -hyperparameters 180 6 2 20 -I 5000 -C 11 -train_test_ratio 1.0 -train_test_splits 5 -prediction_type batch -model_names generative_poisson_with_skipped_cycles generative_generalized_poisson_with_skipped_cycles -exec_mode exec_fit_plot_data_predict_no_plots_no_save -exec_stamp sim_data_poisson_I5000
```

### Generative Poisson generative model, evaluate generative models for I=5000 users

```bash
python3 evaluate_predictive_models.py -data_model generalized_poisson -hyperparameters 160 4 2 20 2 20 -I 5000 -C 11 -train_test_ratio 1.0 -train_test_splits 5 -prediction_type batch -model_names generative_poisson_with_skipped_cycles generative_generalized_poisson_with_skipped_cycles -exec_mode exec_fit_plot_data_predict_no_plots_no_save -exec_stamp sim_data_gpoisson_I5000
```
## Real data

Example where generative and neural network models are evaluated in real data (contained in ../cycle_length_data/cycle_lengths.npz) for I=5000 users

```bash
python3 evaluate_predictive_models.py -data_model load -save_data_dir ../data/cycle_length_data -I 5000 -C 11 \
    -train_test_ratio 1.0 -train_test_splits 5 -prediction_type batch \
    -model_names generative_poisson_with_skipped_cycles generative_generalized_poisson_with_skipped_cycles cnn_dropout_nnet cnn_dropout_nnet_2 lstm_dropout_nnet lstm_dropout_nnet_2 rnn_dropout_nnet rnn_dropout_nnet_2 \
    -exec_mode exec_fit_plot_data_predict_no_plots_no_save -exec_stamp real_data_I5000
```
