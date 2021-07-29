#!/usr/bin/python

# Imports
import sys, os, re, time
import pdb
import pickle
# Science
import numpy as np
import scipy.stats as stats
import pandas as pd
from scipy.special import rel_entr
import sklearn
# Plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

# Plotting default colors
default_colors=[
                colors.cnames['blue'],
                colors.cnames['green'],
                colors.cnames['red'],
                colors.cnames['purple'],
                colors.cnames['yellow'],
                colors.cnames['black'],
                colors.cnames['grey'],
                colors.cnames['skyblue'],
                colors.cnames['cyan'],
                colors.cnames['palegreen'],
                colors.cnames['lime'],
                colors.cnames['orange'],
                colors.cnames['fuchsia'],
                colors.cnames['pink'],
                colors.cnames['saddlebrown'],
                colors.cnames['chocolate'],
                colors.cnames['burlywood']
                ]

# Hyperparameter distribution and samples
def plot_generative_model_hyperparams(model, hyperparameters, drawn_parameters, plot_dir=False):
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
            drawn_parameters, if not None, dictionary with
                lambda
                xi (if generalized poisson)
                pi
            plot_dir: if False, show, if not, where to plot hyperparameter distributions
        Output:
            None
    '''
    
    # Expand hyperparameters
    if model == 'poisson':
        kappa,gamma,alpha,beta=hyperparameters
    elif model == 'generalized_poisson':
        kappa,gamma,alpha_xi,beta_xi,xi_max,x_max,alpha,beta=hyperparameters    

    # Gamma and drawn lambdas
    x_gamma = np.linspace(0, 100, 1000)
    gamma_pdf=stats.gamma.pdf(x_gamma, kappa, loc=0, scale=1/gamma)
    plt.plot(
                x_gamma, gamma_pdf,
                'r-', lw=5, alpha=0.6,
                label='Gamma pdf'
            )
    plt.stem(
                [(kappa-1)/gamma], [np.max(gamma_pdf)],
                'b', linefmt='-', markerfmt='x',use_line_collection=True,
                label='Mode'
            )
    if drawn_parameters is not None:
        plt.stem(
                    drawn_parameters['lambda'], np.zeros(drawn_parameters['lambda'].shape),
                    'g', linefmt='-', markerfmt='x', use_line_collection=True,
                    label=r'$\lambda$ samples'
                )
    plt.autoscale(enable=True, tight=True)
    plt.legend()
    if not plot_dir:
        plt.show()
    else:
        plt.savefig('{}/lambda_gamma_and_drawn_lambdas.pdf'.format(plot_dir), format='pdf', bbox_inches='tight')
        plt.close()
    
    if model == 'generalized_poisson':
        # Set by the underdispersed truncation limit
        if x_max < float('inf'):
            # Xi is set to the biggest value that matches x_max, no randomness
            plt.stem(
                        drawn_parameters['xi'], np.zeros(I),
                        'g', linefmt='-', markerfmt='x', use_line_collection=True,
                        label=r'$\xi$ samples'
                    )
        else:
            # Xi drawn from shifted beta prior distribution
            xi_min=-1*np.ones(drawn_parameters['xi'].shape)
            x_xi = np.linspace(0, 1, 1000)
            xi_beta_pdf=stats.beta.pdf(x_xi, alpha_xi, beta_xi)
            plt.plot(
                    -1+(xi_max+1)*x_xi, xi_beta_pdf,
                    'r-', lw=5, alpha=0.6,
                    label='Beta pdf'
                    )
            plt.stem(
                        [-1+(xi_max+1)*(alpha_xi-1)/(alpha_xi+beta_xi-2)], [np.max(xi_beta_pdf)],
                        'b', linefmt='-', markerfmt='x', use_line_collection=True,
                        label='Mode'
                    )
            if drawn_parameters is not None:
                plt.stem(
                            drawn_parameters['xi'], np.zeros(drawn_parameters['xi'].shape),
                            'g', linefmt='-', markerfmt='x', use_line_collection=True,
                            label=r'$\xi$ samples'
                        )
        # Tighten and close
        plt.autoscale(enable=True, tight=True)
        plt.legend()
        if not plot_dir:
            plt.show()
        else:
            plt.savefig('{}/xi_beta_and_drawn_xis.pdf'.format(plot_dir), format='pdf', bbox_inches='tight')
            plt.close()
    
    # Beta and drawn pis
    x_beta = np.linspace(0, 1, 1000)
    beta_pdf=stats.beta.pdf(x_beta, alpha, beta)
    plt.plot(
                x_beta, beta_pdf,
                'r-', lw=5, alpha=0.6,
                label='Beta pdf'
            )
    plt.stem(
                [(alpha-1)/(alpha+beta-2)], [np.max(beta_pdf)],
                'b', linefmt='-', markerfmt='x', use_line_collection=True,
                label='Mode'
            )
    if drawn_parameters is not None:
        plt.stem(
                    drawn_parameters['pi'],  np.zeros(drawn_parameters['pi'].shape), 
                    'g', linefmt='-', markerfmt='x', use_line_collection=True,
                    label=r'$\pi$ samples'
                )
    plt.autoscale(enable=True, tight=True)
    plt.legend()
    if not plot_dir:
        plt.show()
    else:
        plt.savefig('{}/pi_beta_and_drawn_pis.pdf'.format(plot_dir), format='pdf', bbox_inches='tight')
        plt.close()

# Evaluate hyperparameter estimation
def plot_hyperparameter_estimation(model, prior_hyperparameters, true_hyperparameters, estimated_hyperparameters, estimated_parameters, plot_dir=False, stamp='nostamp'):
    '''
        Input:
            model: one of the following strings
                poisson
                generalized_poisson
            prior_hyperparameters of the model:
                for poisson model:
                    kappa, gamma, alpha, beta
                for generalized_poisson model:
                    kappa, gamma, alpha_xi, beta_xi, xi_max, x_max, alpha, beta
            true_hyperparameters of the model:
                for poisson model:
                    kappa, gamma, alpha, beta
                for generalized_poisson model:
                    kappa, gamma, alpha_xi, beta_xi, xi_max, x_max, alpha, beta
            estimated_hyperparameters of the model:
                for poisson model:
                    kappa, gamma, alpha, beta
                for generalized_poisson model:
                    kappa, gamma, alpha_xi, beta_xi, xi_max, x_max, alpha, beta
            estimated_parameters, if not None, dictionary with
                lambda
                xi (if generalized poisson)
                pi
            plot_dir: if False, show, if not, where to plot hyperparameter distributions
            stamp: string to append to filenames
        Output:
            None
    '''
    # TODO: this only works without model mismatch between true and estimated
    
    # Expand hyperparameters
    if 'generalized_poisson' in model:
        # True hyperparameters
        if true_hyperparameters is not None:
            kappa,gamma,alpha_xi,beta_xi,xi_max,x_max,alpha,beta=true_hyperparameters
        
        # Prior hyperparameters
        kappa_prior,gamma_prior,alpha_xi_prior,beta_xi_prior,alpha_prior,beta_prior,x_max_prior,xi_max_prior,s_max=prior_hyperparameters
        
        # Estimated hyperparameters
        if len(estimated_hyperparameters) == 4:
            kappa_hat,gamma_hat,alpha_hat,beta_hat=estimated_hyperparameters
        else:
            kappa_hat,gamma_hat,alpha_xi_hat,beta_xi_hat,xi_max_hat,x_max_hat,alpha_hat,beta_hat=estimated_hyperparameters
    elif 'poisson' in model:
        # True hyperparameters
        if true_hyperparameters is not None:
            kappa,gamma,alpha,beta=true_hyperparameters        
        
        # Prior hyperparameters
        kappa_prior,gamma_prior,alpha_prior,beta_prior,s_max=prior_hyperparameters
        # Estimated hyperparameters
        kappa_hat,gamma_hat,alpha_hat,beta_hat=estimated_hyperparameters
    
    # Gamma and drawn lambdas
    x_gamma = np.linspace(0, 100, 1000)
    # True gamma
    if true_hyperparameters is not None:
        gamma_pdf=stats.gamma.pdf(x_gamma, kappa, loc=0, scale=1/gamma)
        plt.plot(
                    x_gamma, gamma_pdf,
                    'r-', lw=5, alpha=0.6,
                    label='True gamma pdf'
                )
        plt.stem(
                    [(kappa-1)/gamma], [np.max(gamma_pdf)],
                    'b', linefmt='-', markerfmt='*',use_line_collection=True,
                    label='True Mode'
                )
    # Estimated
    gamma_hat_pdf=stats.gamma.pdf(x_gamma, kappa_hat, loc=0, scale=1/gamma_hat)
    # Label
    estimation_pdf_label=r'Estimated $\lambda$ gamma pdf' if true_hyperparameters is None else r'Estimated $\lambda$ gamma pdf with kl={}'.format(rel_entr(gamma_hat_pdf, gamma_pdf).sum())
    plt.plot(
                x_gamma, gamma_hat_pdf,
                'm-', lw=5, alpha=0.4,
                label=estimation_pdf_label
            )
    plt.stem(
                [(kappa_hat-1)/gamma_hat], [np.max(gamma_hat_pdf)],
                'b', linefmt='-', markerfmt='x',use_line_collection=True,
                label='Estimated Mode'
            )

    # prior
    gamma_prior_pdf=stats.gamma.pdf(x_gamma, kappa_prior, loc=0, scale=1/gamma_prior)
    # Label
    prior_pdf_label=r'Prior $\lambda$ gamma pdf' if true_hyperparameters is None else r'Prior $\lambda$ gamma pdf with kl={}'.format(rel_entr(gamma_prior_pdf, gamma_pdf).sum())
    plt.plot(
                x_gamma, gamma_prior_pdf,
                'c-', lw=5, alpha=0.4,
                label=prior_pdf_label
            )
    plt.stem(
                [(kappa_prior-1)/gamma_prior], [np.max(gamma_prior_pdf)],
                'k', linefmt='-', markerfmt='x',use_line_collection=True,
                label='Prior Mode'
            )


    if estimated_parameters is not None:
        plt.stem(
                    estimated_parameters['lambda']['mean'], np.zeros(estimated_parameters['lambda']['mean'].shape),
                    'g', linefmt='-', markerfmt='x', use_line_collection=True,
                    label=r'$\lambda$ estimates'
                )
    plt.autoscale(enable=True, tight=True)
    plt.legend()
    if not plot_dir:
        plt.show()
    else:
        plt.savefig('{}/lambda_gamma_and_drawn_lambdas_{}_{}.pdf'.format(plot_dir, model, stamp), format='pdf', bbox_inches='tight')
        plt.close()
    
    # Xis, form generalized poisson
    if 'generalized_poisson' in model:
        # Set by the underdispersed truncation limit
        if len(estimated_hyperparameters) == 4:
        #if x_max_hat < float('inf'):
            # Xi is set to the biggest value that matches x_max, no randomness
            plt.stem(
                        estimated_parameters['xi']['mean'], np.zeros(estimated_parameters['xi']['mean'].shape),
                        'g', linefmt='-', markerfmt='x', use_line_collection=True,
                        label=r'$\xi$ samples'
                    )
        else:
            # Xi drawn from shifted beta prior distribution
            x_xi = np.linspace(0, 1, 1000)
            # True beta
            if true_hyperparameters is not None:
                xi_beta_pdf=stats.beta.pdf(x_xi, alpha_xi, beta_xi)
                plt.plot(
                        -1+(xi_max+1)*x_xi, xi_beta_pdf,
                        'r-', lw=5, alpha=0.6,
                        label='True Beta pdf'
                        )
                plt.stem(
                            [-1+(xi_max+1)*(alpha_xi-1)/(alpha_xi+beta_xi-2)], [np.max(xi_beta_pdf)],
                            'b', linefmt='-', markerfmt='*', use_line_collection=True,
                            label='True Mode'
                        )
            
            # Estimated
            xi_beta_hat_pdf=stats.beta.pdf(x_xi, alpha_xi_hat, beta_xi_hat)
            # Label
            estimation_pdf_label=r'Estimated $\xi$ beta pdf' if true_hyperparameters is None else r'Estimated $\xi$ beta pdf with kl={}'.format(rel_entr(xi_beta_hat_pdf, xi_beta_pdf).sum())

            plt.plot(
                    -1+(xi_max_hat+1)*x_xi, xi_beta_hat_pdf,
                    'm-', lw=5, alpha=0.4,
                    label=estimation_pdf_label
                    )
            plt.stem(
                        [-1+(xi_max_hat+1)*(alpha_xi_hat-1)/(alpha_xi_hat+beta_xi_hat-2)], [np.max(xi_beta_hat_pdf)],
                        'b', linefmt='-', markerfmt='x', use_line_collection=True,
                        label='Estimated Mode'
                    )

            # Prior
            xi_beta_prior_pdf=stats.beta.pdf(x_xi, alpha_xi_prior, beta_xi_prior)

            # Label
            prior_pdf_label=r'Prior $\xi$ beta pdf' if true_hyperparameters is None else r'Prior $\xi$ beta pdf with kl={}'.format(rel_entr(xi_beta_prior_pdf, xi_beta_pdf).sum())
            
            plt.plot(
                    -1+(xi_max_prior+1)*x_xi, xi_beta_prior_pdf,
                    'c-', lw=5, alpha=0.4,
                    label=prior_pdf_label
                    )
            if alpha_xi_prior+beta_xi_prior-2>0:
                plt.stem(
                            [-1+(xi_max_prior+1)*(alpha_xi_prior-1)/(alpha_xi_prior+beta_xi_prior-2)], [np.max(xi_beta_prior_pdf)],
                            'k', linefmt='-', markerfmt='x', use_line_collection=True,
                            label='Prior Mode'
                        )

            if estimated_parameters is not None:
                plt.stem(
                            estimated_parameters['xi']['mean'], np.zeros(estimated_parameters['xi']['mean'].shape),
                            'g', linefmt='-', markerfmt='x', use_line_collection=True,
                            label=r'$\xi$ samples'
                        )
        # Tighten and close
        plt.autoscale(enable=True, tight=True)
        plt.legend()
        if not plot_dir:
            plt.show()
        else:
            plt.savefig('{}/xi_beta_and_drawn_xis_{}_{}.pdf'.format(plot_dir, model, stamp), format='pdf', bbox_inches='tight')
            plt.close()
    
    # Beta and drawn pis
    x_beta = np.linspace(0, 1, 1000)
    # True beta
    if true_hyperparameters is not None:
        beta_pdf=stats.beta.pdf(x_beta, alpha, beta)
        plt.plot(
                    x_beta, beta_pdf,
                    'r-', lw=5, alpha=0.6,
                    label='True Beta pdf'
                )
        if alpha+beta-2>0:
            plt.stem(
                        [(alpha-1)/(alpha+beta-2)], [np.max(beta_pdf)],
                        'b', linefmt='-', markerfmt='*', use_line_collection=True,
                        label='True Mode'
                    )
    
    # Estimated
    beta_hat_pdf=stats.beta.pdf(x_beta, alpha_hat, beta_hat)
    # Label
    estimation_pdf_label=r'Estimated $\pi$ beta pdf' if true_hyperparameters is None else r'Estimated $\pi$ beta pdf with kl={}'.format(rel_entr(beta_hat_pdf, beta_pdf).sum())

    plt.plot(
                x_beta, beta_hat_pdf,
                'm-', lw=5, alpha=0.4,
                label=estimation_pdf_label
            )
    plt.stem(
                [(alpha_hat-1)/(alpha_hat+beta_hat-2)], [np.max(beta_hat_pdf)],
                'b', linefmt='-', markerfmt='x', use_line_collection=True,
                label='Estimated Mode'
            )

    # Prior
    beta_prior_pdf=stats.beta.pdf(x_beta, alpha_prior, beta_prior)
    # Label
    prior_pdf_label=r'Prior $\pi$ beta pdf' if true_hyperparameters is None else r'Prior $\pi$ beta pdf with kl={}'.format(rel_entr(beta_prior_pdf, beta_pdf).sum())

    plt.plot(
                x_beta, beta_prior_pdf,
                'c-', lw=5, alpha=0.4,
                label=prior_pdf_label
            )
    plt.stem(
                [(alpha_prior-1)/(alpha_prior+beta_prior-2)], [np.max(beta_prior_pdf)],
                'k', linefmt='-', markerfmt='x', use_line_collection=True,
                label='Prior Mode'
            )

    if estimated_parameters is not None:
        plt.stem(
                    estimated_parameters['pi']['mean'],  np.zeros(estimated_parameters['pi']['mean'].shape), 
                    'g', linefmt='-', markerfmt='x', use_line_collection=True,
                    label=r'$\pi$ estimates'
                )
    plt.autoscale(enable=True, tight=True)
    plt.legend()
    if not plot_dir:
        plt.show()
    else:
        plt.savefig('{}/pi_beta_and_drawn_pis_{}_{}.pdf'.format(plot_dir, model, stamp), format='pdf', bbox_inches='tight')
        plt.close()
    
# Plot learned posteriors per individual
def plot_parameter_estimation(model, estimated_parameters, true_parameters, plot_dir=None, stamp='nostamp', colors=default_colors):
    '''
        Input:
            model: one of the following strings
                poisson
                generalized_poisson
            estimated_parameters: dictionary with
                lambda
                xi (if generalized poisson)
                pi
            true_parameters: dictionary with
                lambda
                xi (if generalized poisson)
                pi
            plot_dir: if False, show, if not, where to plot hyperparameter distributions
            stamp: string to append to filenames
        Output:
            None
    '''
    # TODO: this only works without model mismatch between true and estimated
    
    # Make sure directory is ready
    os.makedirs(plot_dir, exist_ok = True)
    if true_parameters is not None:
        # Collect all parameters:
        parameters=[*estimated_parameters]
        # Then iterate
        for parameter in parameters:
            if parameter in true_parameters:
                # true Vs estimated
                plt.scatter(
                            true_parameters[parameter][:,None], 
                            estimated_parameters[parameter]['mean'], 
                            c=colors[0],
                            label='True Vs Estimated $\{}$'.format(parameter),
                            alpha=0.6
                            )
                legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                filename = '{}/plot_{}_learned_{}_posterior_scatterplot_{}.pdf'.format(plot_dir, model, parameter, stamp)
                plt.savefig(filename, format='pdf', bbox_inches='tight')
                plt.close()

# Plot learned posteriors per individual
def plot_data_statistics_estimation(model, data_mean, data_std, estimated_data_mean, true_data_mean, estimated_parameters, plot_dir=None, stamp='nostamp', colors=default_colors):
    '''
        Input:
            model: one of the following strings
                poisson
                generalized_poisson
            data_mean: mean of data used for inference
            estimated_data_mean: expectation of data, based on infered parameters
            true_data_mean: expectation of data, as per data generating model
            estimated_parameters: dictionary with
                lambda
                xi (if generalized poisson)
                pi
            plot_dir: if False, show, if not, where to plot hyperparameter distributions
            stamp: string to append to filenames
        Output:
            None
    '''
    # Make sure directory is ready
    os.makedirs(plot_dir, exist_ok = True)
    
    # get max of possible means as plot lim
    min_plot_lims = []
    max_plot_lims = []

    if true_data_mean is not None:
        for means in [estimated_data_mean, data_mean, true_data_mean[:,None]]:
            max_plot_lims.append(np.max(means))
            min_plot_lims.append(np.min(means))
    else:
        for means in [estimated_data_mean, data_mean]:
            max_plot_lims.append(np.max(means))
            min_plot_lims.append(np.min(means))

    min_plot_lim = np.min(min_plot_lims)
    max_plot_lim = np.max(max_plot_lims)

    # Compare to data sufficient statistics
    
    ### SCATTER-plot
    # Data mean Vs estimated data mean
    plt.scatter(
                data_mean,
                estimated_data_mean,
                c=colors[2],
                label='Empirical data mean Vs Estimated data mean, R^2={}'.format(
                                                                sklearn.metrics.r2_score(data_mean, estimated_data_mean) 
                                                                ),
                alpha=0.6
                )
    plt.xlim([min_plot_lim, max_plot_lim])
    plt.ylim([min_plot_lim, max_plot_lim])
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('Observed data average')
    plt.ylabel('Estimated data mean')
    #add x = y line
    plt.plot(np.linspace(min_plot_lim, max_plot_lim, 10), np.linspace(min_plot_lim, max_plot_lim, 10), linestyle='dashed', color='k')
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    filename = '{}/plot_{}_posterior_data_mean_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Data mean Vs estimated data mean (size based on data variance)
    my_scatter_plot=plt.scatter(
                data_mean,
                estimated_data_mean,
                c=data_std,
                cmap='Reds',
                label='Empirical data mean Vs Estimated data mean, R^2={}'.format(
                                                                sklearn.metrics.r2_score(data_mean, estimated_data_mean) 
                                                                ),
                alpha=0.6
                )
    plt.xlim([min_plot_lim, max_plot_lim])
    plt.ylim([min_plot_lim, max_plot_lim])
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('Observed data average')
    plt.ylabel('Estimated data mean')
    plt.colorbar(my_scatter_plot)
    #add x = y line
    plt.plot(np.linspace(min_plot_lim, max_plot_lim, 10), np.linspace(min_plot_lim, max_plot_lim, 10), linestyle='dashed', color='k')
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    filename = '{}/plot_{}_posterior_data_mean_var_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
               
    if true_data_mean is not None:
        # true Vs estimated
        plt.scatter(
                    true_data_mean[:,None], 
                    estimated_data_mean,
                    c=colors[2],
                    label='Theoretical (generative) data mean Vs Estimated data mean, R^2={}'.format(
                                                                sklearn.metrics.r2_score(true_data_mean[:,None], estimated_data_mean)
                                                                ),
                    alpha=0.6
                    )
        plt.xlim([min_plot_lim, max_plot_lim])
        plt.ylim([min_plot_lim, max_plot_lim])
        plt.autoscale(enable=True, tight=True)
        plt.xlabel('Theoretical data mean')
        plt.ylabel('Estimated data mean')
        #add x = y line
        plt.plot(np.linspace(min_plot_lim, max_plot_lim, 10), np.linspace(min_plot_lim, max_plot_lim, 10), linestyle='dashed', color='k')
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        filename = '{}/plot_{}_posterior_true_data_mean_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
    
    # Estimated lambda Vs estimated data mean
    my_scatter_plot=plt.scatter(
                estimated_data_mean,
                estimated_parameters['lambda']['mean'], 
                c=data_std,
                cmap='Reds',
                label='Estimated data mean Vs Estimated $\lambda$',
                alpha=0.6
                )
    plt.xlim([min_plot_lim, max_plot_lim])
    plt.ylim([np.min(estimated_parameters['lambda']['mean']), np.max(estimated_parameters['lambda']['mean'])])
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('Estimated data mean')
    plt.ylabel('Estimated $\lambda$')
    plt.colorbar(my_scatter_plot)
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    filename = '{}/plot_{}_learned_data_posterior_lambda_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Estimated lambda Vs data mean
    my_scatter_plot=plt.scatter(
                data_mean,
                estimated_parameters['lambda']['mean'], 
                c=data_std,
                cmap='Reds',
                label='Empirical data mean Vs Estimated $\lambda$',
                alpha=0.6
                )
    plt.xlim([min_plot_lim, max_plot_lim])
    plt.ylim([np.min(estimated_parameters['lambda']['mean']), np.max(estimated_parameters['lambda']['mean'])])
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('Observed data average')
    plt.ylabel('Estimated $\lambda$')
    plt.colorbar(my_scatter_plot)
    legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
    filename = '{}/plot_{}_data_mean_lambda_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    
    # For skipping models
    if 'skipped' in model:
        # Estimated skipping probability Vs estimated data mean
        my_scatter_plot=plt.scatter(
                    estimated_data_mean,
                    estimated_parameters['pi']['mean'], 
                    c=data_std,
                    cmap='Reds',
                    label='Estimated data mean Vs Estimated $\pi$',
                    alpha=0.6
                    )
        plt.xlim([min_plot_lim, max_plot_lim])
        plt.ylim([np.min(estimated_parameters['pi']['mean']), np.max(estimated_parameters['pi']['mean'])])
        plt.autoscale(enable=True, tight=True)
        plt.xlabel('Estimated data mean')
        plt.ylabel('Estimated $\pi')
        plt.colorbar(my_scatter_plot)
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        filename = '{}/plot_{}_learned_data_posterior_skipped_prob_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
        
        # Estimated skipping probability Vs data mean
        my_scatter_plot=plt.scatter(
                    data_mean,
                    estimated_parameters['pi']['mean'], 
                    c=data_std,
                    cmap='Reds',
                    label='Empirical data mean Vs Estimated $\pi$',
                    alpha=0.6
                    )
        plt.xlim([min_plot_lim, max_plot_lim])
        plt.ylim([np.min(estimated_parameters['pi']['mean']), np.max(estimated_parameters['pi']['mean'])])
        plt.autoscale(enable=True, tight=True)
        plt.xlabel('Observed data average')
        plt.ylabel('Estimated $\pi$')
        plt.colorbar(my_scatter_plot)
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        filename = '{}/plot_{}_data_mean_skipped_prob_scatterplot_{}.pdf'.format(plot_dir, model, stamp)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()
    

# Plot fitting cost
def plot_fitting_cost(training_loss, save_dir, stamp=''):
    plt.plot(training_loss)
    plt.xlabel('Iteration over optimization')
    plt.ylabel('Cost function')
    plt.savefig('{}/cost_function{}.pdf'.format(save_dir, stamp), format='pdf', bbox_inches='tight')
    plt.close()

# Plot computed losses over training cycles
def plot_prediction_loss_over_training_cycles(loss, I_range, C_range, evaluation_metrics=None, save_dir=None, stamp='nostamp', colors=default_colors, day_range=None):
    
    if save_dir is not None:
        # Make sure directory is ready
        os.makedirs(save_dir, exist_ok = True)
    
    # Collect prediction_models:
    prediction_models=[*loss]

    # fix prediction model names
    prediction_model_labels = prediction_models.copy()
    for i in range(len(prediction_model_labels)):
        if 'generative' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('generative_','')
        if 'cohort' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('_per_cohort_s_predict','')
        if 'individual' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('_per_individual_s_predict','')

    # If evaluation metric is not specified, then figure out
    if evaluation_metrics is None:
        evaluation_metrics=[*loss[prediction_models[0]]]
    
    # Day range
    if day_range is None:
        day_range = [0]
    else:
        day_range = [0, 14, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 40, 50, 60, 90]
        
    # Figure per evaluation metric
    for evaluation_metric in evaluation_metrics:
        for current_day in day_range:
            # For each I, range C
            if C_range.size>1:
                C_idx=np.arange(C_range.size)
                for this_I_idx, this_I in enumerate(I_range):
                    # only plot for max I for now
                    if this_I == np.max(I_range):
                        # Figure with all prediction models
                        plt.figure()
                        for model_idx, prediction_model in enumerate(prediction_models):
                            print(prediction_model)
                            mean_loss_for_model = np.nanmean(
                                                      loss[prediction_model][evaluation_metric][:,this_I_idx,C_idx,current_day],
                                                      axis=0
                                                      )
                            std_loss_for_model = np.nanstd(
                                                      loss[prediction_model][evaluation_metric][:,this_I_idx,C_idx,current_day],
                                                      axis=0
                                                      )
                            if mean_loss_for_model.any() > 0:
                                plt.plot(C_range, 
                                    mean_loss_for_model,
                                    colors[model_idx%len(colors)],
                                    label='{}'.format(prediction_model_labels[model_idx])
                                    )
                                plt.fill_between(x=C_range,
                                                    y1=mean_loss_for_model - std_loss_for_model,
                                                    y2=mean_loss_for_model + std_loss_for_model,
                                                    color=colors[model_idx%len(colors)],
                                                    alpha=0.4
                                                )
                        plt.xlabel('Number of training cycles')
                        plt.xlim(C_range[0],C_range[-1])
                        plt.ylabel('{}'.format(evaluation_metric))
                        plt.title('{} for I={} over training cycles'.format(evaluation_metric, this_I))
                        #plt.legend()
                        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                        filename = '{}/plot_prediction_{}_I{}_vs_C_day_{}_{}.pdf'.format(save_dir, evaluation_metric, this_I, current_day, stamp)
                        plt.savefig(filename, format='pdf', bbox_inches='tight')
                        plt.close()
                
            # For each C, range I
            if I_range.size>1:
                I_idx=np.arange(I_range.size)
                for this_C_idx, this_C in enumerate(C_range):
                    # only plot for max C for now
                    if this_C == np.max(C_range):
                        # Figure with all prediction models
                        plt.figure()
                        for model_idx, prediction_model in enumerate(prediction_models):
                            mean_loss_for_model = np.nanmean(
                                                    loss[prediction_model][evaluation_metric][:,I_idx,this_C_idx,current_day],
                                                    axis=0
                                                    )
                            std_loss_for_model = np.nanstd(
                                                    loss[prediction_model][evaluation_metric][:,I_idx,this_C_idx,current_day],
                                                    axis=0
                                                    )
                            if mean_loss_for_model.any() > 0:
                                plt.plot(I_range,
                                        mean_loss_for_model,
                                        colors[model_idx%len(colors)],
                                        label='{}'.format(prediction_model_labels[model_idx])
                                    )
                                plt.fill_between(x=I_range,
                                                    y1=mean_loss_for_model - std_loss_for_model,
                                                    y2=mean_loss_for_model + std_loss_for_model,
                                                    color=colors[model_idx%len(colors)],
                                                    alpha=0.4
                                                )
                        plt.xlabel('Number of training individuals')
                        plt.xlim(I_range[0],I_range[-1])
                        plt.ylabel('{}'.format(evaluation_metric))
                        plt.title('{} for C={} over training individuals'.format(evaluation_metric, this_C))
                        #plt.legend()
                        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                        filename = '{}/plot_prediction_{}_I_vs_C{}_day_{}_{}.pdf'.format(save_dir, evaluation_metric, this_C, current_day, stamp)
                        plt.savefig(filename, format='pdf', bbox_inches='tight')
                        plt.close()
                
            
# Plot computed losses by day
def plot_predictive_posterior_mean_by_day_loss_over_models(loss, day_range, I_range, C_range, evaluation_metrics=None, save_dir=None, stamp='nostamp', colors=default_colors):
    # I
    # assert y_true.shape[0] == y_hat.shape[0]
    # # day_range
    # assert day_range.shape[0] == y_hat.shape[1]
    
    if save_dir is not None:
        # Make sure directory is ready
        os.makedirs(save_dir, exist_ok = True)
    
    # Collect prediction_models:
    prediction_models=[*loss]
    
    # fix prediction model names
    prediction_model_labels = prediction_models.copy()
    for i in range(len(prediction_model_labels)):
        if 'generative' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('generative_','')
        if 'cohort' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('_per_cohort_s_predict','')
        if 'individual' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('_per_individual_s_predict','')

    I_idx=np.arange(I_range.size)
    max_day=day_range[-1]
    
    # If evaluation metric is not specified, then figure out
    if evaluation_metrics is None:
        evaluation_metrics=[*loss[prediction_models[0]]]
        
    # Figure per evaluation metric
    for evaluation_metric in evaluation_metrics:
        for this_I_idx in I_idx:
            for this_C_idx, this_C in enumerate(C_range):
                # only plot for max C for now
                # if this_C == np.max(C_range):
                    # Figure with all prediction models
                    plt.figure()
                    for model_idx, prediction_model in enumerate(prediction_models):
                        #average across runs
                        mean_loss_for_model = np.nanmean(loss[prediction_model][evaluation_metric][:,this_I_idx,this_C_idx,:max_day+1], axis=0)
                        if mean_loss_for_model.any() > 0:
                            plt.plot(day_range[:max_day+1],
                                    mean_loss_for_model,
                                    colors[model_idx%len(colors)],
                                    label='{}'.format(prediction_model_labels[model_idx])
                                )
                    plt.xlabel('Current day') #of next cycle ($d_{current}$)')
                    plt.xlim(day_range[0],max_day)
                    plt.ylabel('{}'.format(evaluation_metric))
                    plt.autoscale(enable=True, tight=True)
                    legend = plt.legend(loc='upper left')#plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                    #plt.title('Prediction by day {} for I={}'.format(evaluation_metric, I_range[this_I_idx]))
                    #plt.legend()
                    #legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                    filename = '{}/plot_prediction_by_day_{}_I{}_vs_C_{}.pdf'.format(save_dir, evaluation_metric, I_range[this_I_idx], stamp)
                    if save_dir is None:
                        plt.show()
                    else:
                        plt.savefig(filename, format='pdf', bbox_inches='tight')
                        plt.close()
    

# Plot predictive posterior at day
def plot_predictive_posterior_at_day(predictive_posterior, at_day=0, x_predict_max=120, y_true=None, save_dir=None, stamp='nostamp', colors=default_colors):
    # I
    if y_true is not None:
        assert y_true.shape[0] == predictive_posterior.shape[0]        
    else:
        assert predictive_posterior.shape[0] >0
    
    # day_range
    #assert day_range.shape[0] == predictive_posterior.shape[1]
    # x_predict_max
    assert x_predict_max+1 == predictive_posterior.shape[2]
    
    if save_dir is not None:
        # Make sure directory is ready
        os.makedirs(save_dir, exist_ok = True)
    
    #for i in np.arange(predictive_posterior.shape[0]):
    for i in np.arange(predictive_posterior.shape[0])[:5]:
        # Per-day plotting
        plt.plot(
                    np.arange(x_predict_max+1), predictive_posterior[i][at_day],
                    colors[at_day],
                    label='Predictive posterior at day={}'.format(at_day)
                )
        if y_true is not None:
            plt.stem(
                        y_true[i], [1],
                        'b', linefmt='-', markerfmt='*', use_line_collection=True,
                        label='True cycle-length'
                    )
        plt.autoscale(enable=True, tight=True)
        legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
        filename = '{}/predictive_posterior_at_day_{}_{}_i{}.pdf'.format(save_dir, at_day, stamp, i)
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(filename, format='pdf', bbox_inches='tight')  
            plt.close()
        
# Plot predictive posterior by day
def plot_predictive_posterior_by_day(predictive_posterior, day_range, x_predict_max, y_true=None, save_dir=None, stamp='nostamp', colors=default_colors):
    # I
    if y_true is not None:
        assert y_true.shape[0] == predictive_posterior.shape[0]        
    else:
        assert predictive_posterior.shape[0] >0
    # day_range
    assert day_range.shape[0] == predictive_posterior.shape[1]
    # x_predict_max
    assert x_predict_max+1 == predictive_posterior.shape[2]
    
    if save_dir is not None:
        # Make sure directory is ready
        os.makedirs(save_dir, exist_ok = True)
    
    #for i in np.arange(predictive_posterior.shape[0]):
    for i in np.arange(predictive_posterior.shape[0])[:5]:
        # Per-day plotting
        '''
        for day in day_range:
            plot_predictive_posterior_at_day(predictive_posterior, at_day=day, x_predict_max=x_predict_max, y_true=y_true, save_dir=save_dir, stamp=stamp, colors=colors)
        '''
        
        # Surface plotting
        # Create x, day mesh
        x_range_grid, day_range_grid = np.meshgrid(np.arange(x_predict_max+1), day_range)
        # New Figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # Plot the surface
        surf = ax.plot_surface(
                                x_range_grid, day_range_grid,
                                predictive_posterior[i],
                                cmap=cm.coolwarm, linewidth=0, antialiased=False
                                )
        if y_true is not None:
            raise ValueError('Need to add true value in surface plot')
            
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('Prediction day')
        ax.set_xlim(0,x_predict_max)
        ax.set_ylabel('Current day of next cycle ($d_{current}$)')
        ax.set_ylim(day_range[0],day_range[-1])
        ax.set_zlabel('Probability')
        # Customize the z axis?
        plt.title('Prediction by day surface')
        filename = '{}/predictive_posterior_by_day_surface_{}_i{}.pdf'.format(save_dir, stamp, i)
        # Rotate for better viewing?
        #ax.view_init(30, 30)
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()

# Plot predictive posterior by day
def plot_skipping_posterior_by_day(predictive_posterior, day_range, s_predict, cycle_mean_predictions=None, y_true=None, save_dir=None, stamp='nostamp', colors=default_colors):
    # I
    if y_true is not None:
        assert y_true.shape[0] == predictive_posterior.shape[0]        
    else:
        assert predictive_posterior.shape[0] >0
    # day_range
    assert day_range.shape[0] == predictive_posterior.shape[1]
    # x_predict_max
    assert s_predict == predictive_posterior.shape[2]
    
    if save_dir is not None:
        # Make sure directory is ready
        os.makedirs(save_dir, exist_ok = True)
    
    #for i in np.arange(predictive_posterior.shape[0]):
    for i in np.arange(predictive_posterior.shape[0])[:5]:
        # Per-day plotting
        '''
        for day in day_range:
            plot_predictive_posterior_at_day(predictive_posterior, at_day=day, x_predict_max=x_predict_max, y_true=y_true, save_dir=save_dir, stamp=stamp, colors=colors)
        '''
        
        # Surface plotting
        # Create x, day mesh
        s_grid, day_range_grid = np.meshgrid(np.arange(6), day_range)
        # New Figure
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # Plot the surface
        surf = ax.plot_surface(
                                s_grid, day_range_grid,
                                predictive_posterior[i][:,:6],
                                cmap=cm.coolwarm, linewidth=0, antialiased=False
                                )
        if y_true is not None:
            raise ValueError('Need to add true value in surface plot')
            
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
        ax.set_xlabel('Number of skipped cycles')
        ax.set_xlim(0,5)
        ax.set_ylabel('Current day of next cycle ($d_{current}$)')
        ax.set_ylim(day_range[0],day_range[-1])
        ax.set_zlabel('Probability')
        # Customize the z axis?
        plt.title('Prediction by day surface')
        filename = '{}/skipping_posterior_by_day_surface_{}_i{}.pdf'.format(save_dir, stamp, i)
        # Rotate for better viewing?
        #ax.view_init(30, 30)
        if save_dir is None:
            plt.show()
        else:
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()

        # make stem plots
        prob_s_0 = predictive_posterior[i][:,0]
        prob_s_1 = predictive_posterior[i][:,1]
        prob_s_2 = predictive_posterior[i][:,2]
        if cycle_mean_predictions is not None:
            cycle_pred_day_0 = cycle_mean_predictions[i][0]
            cycle_pred_day_30 = cycle_mean_predictions[i][30]
            cycle_pred_day_60 = cycle_mean_predictions[i][60]

        # plot current day vs. prediction
        if cycle_mean_predictions is not None:
            plt.scatter(day_range, cycle_mean_predictions[i], s=3)
            plt.plot(day_range, cycle_mean_predictions[i])
            plt.xlabel('Current day')
            plt.ylabel('$E_i[d^*|d_{current}]$')
            plt.autoscale(enable=True, tight=True, axis='x')
            filename = '{}/cycle_posterior_by_day_scatter_{}_i{}.pdf'.format(save_dir, stamp, i)
            plt.savefig(filename, format='pdf', bbox_inches='tight')
            plt.close()

        # scatter version
        # plt.scatter(day_range, prob_s_0, color='tab:blue', label='$p_i(s^*=0|d_{current})$', s=3)
        # plt.scatter(day_range, prob_s_1, color='tab:orange', label='$p_i(s^*=1|d_{current})$', s=3)
        # plt.plot(day_range, prob_s_0, color='tab:blue')
        # plt.plot(day_range, prob_s_1, color='tab:orange')

        # plt.scatter(day_range, prob_s_2, color='tab:green', label='$p_i(s^*=2)$', s=3)
        # if cycle_mean_predictions is not None:
        #     plt.axvline(x=30, color='k')
        #     plt.axvline(x=40, color='k')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
        #     # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
        #     # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        #     # plt.axvline(x=cycle_pred_day_0, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_60, color='k', linestyle='dotted')
        # plt.xlabel('Current day of next cycle ($d_{current}$)')
        # plt.ylabel('$p_i(s^*|d^*>d_{current})$')
        # plt.autoscale(enable=True, tight=True, axis='x')
        # plt.ylim((0,1))
        # plt.legend()
        # filename = '{}/skipping_posterior_by_day_line_{}_i{}.pdf'.format(save_dir, stamp, i)
        # plt.savefig(filename, format='pdf', bbox_inches='tight')
        # plt.close()

        # # scatter version with markers
        # plt.scatter(day_range, prob_s_0, color='tab:blue', label='$p_i(s^*=0|d_{current})$', s=3)
        # plt.scatter(day_range, prob_s_1, color='tab:orange', label='$p_i(s^*=1|d_{current})$', s=3)
        # plt.plot(day_range, prob_s_0, color='tab:blue')
        # plt.plot(day_range, prob_s_1, color='tab:orange')

        # # plt.scatter(day_range, prob_s_2, color='tab:green', label='$p_i(s^*=2)$', s=3)
        # if cycle_mean_predictions is not None:
        #     plt.scatter(30, prob_s_0[30], marker='X', s=120, color='tab:blue',label='$p(s^*=0|d_{current}=30]$')
        #     plt.scatter(30, prob_s_1[30], marker='X', s=120, color='tab:orange',label='$p(s^*=1|d_{current}=30]$')
        #     plt.scatter(40, prob_s_0[40], marker='^', s=120, color='tab:blue',label='$p(s^*=0|d_{current}=40]$')
        #     plt.scatter(40, prob_s_1[40], marker='^', s=120, color='tab:orange',label='$p(s^*=1|d_{current}=40]$')
        #     plt.axvline(x=30, color='k')
        #     plt.axvline(x=40, color='k')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
        #     # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
        #     # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        #     # plt.axvline(x=cycle_pred_day_0, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_60, color='k', linestyle='dotted')
        # plt.xlabel('Current day of next cycle ($d_{current}$)')
        # plt.ylabel('$p_i(s^*|d^*>d_{current})$')
        # plt.autoscale(enable=True, tight=True, axis='x')
        # plt.ylim((0,1))
        # plt.legend()
        # filename = '{}/skipping_posterior_by_day_line_with_markers_{}_i{}.pdf'.format(save_dir, stamp, i)
        # plt.savefig(filename, format='pdf', bbox_inches='tight')
        # plt.close()

        # stem version
        # markerline, stemlines, baseline = plt.stem(day_range, prob_s_0, label='$p_i(s^*=0|d_{current})$', markerfmt='C0o', linefmt='C0--', basefmt='k-')
        # plt.setp(stemlines, 'linewidth', 0.8)
        # plt.setp(stemlines, 'alpha', 0.5)
        # plt.setp(markerline, 'markersize', 3)
        # markerline, stemlines, baseline = plt.stem(day_range, prob_s_1, label='$p_i(s^*=1|d_{current})$', markerfmt='C1o', linefmt='C1--', basefmt='k-')
        # plt.setp(stemlines, 'linewidth', 0.8)
        # plt.setp(stemlines, 'alpha', 0.5)
        # plt.setp(markerline, 'markersize', 3)
        # # markerline, stemlines, baseline = plt.stem(day_range, prob_s_2, label='$p_i(s^*=2|d_{current})$', markerfmt='C2o', linefmt='C2--', basefmt='k-')
        # # plt.setp(stemlines, 'linewidth', 0.8)
        # # plt.setp(stemlines, 'alpha', 0.5)
        # # plt.setp(markerline, 'markersize', 3)
        # if cycle_mean_predictions is not None:
        #     plt.axvline(x=30, color='k')
        #     plt.axvline(x=40, color='k')
        #     # plt.axvline(x=cycle_pred_day_0, color='tab:blue', linestyle='dotted', label='$E_i[d^*|d_{current}=0]$')
        #     # plt.axvline(x=cycle_pred_day_30, color='tab:orange',linestyle='dotted', label='$E_i[d^*|d_{current}=30]$')
        #     # plt.axvline(x=cycle_pred_day_60, color='tab:green',linestyle='dotted', label='$E_i[d^*|d_{current}=60]$')
        #     # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue')
        #     # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange')
        #     # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green')

        #     # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
        #     # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
        #     # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        # plt.xlabel('Current day of next cycle')
        # plt.ylabel('$p_i(s^*|d^*>d_{current})$')
        # plt.autoscale(enable=True, tight=True, axis='x')
        # plt.ylim((0,1))
        # plt.legend()
        # filename = '{}/skipping_posterior_by_day_stem_{}_i{}.pdf'.format(save_dir, stamp, i)
        # plt.savefig(filename, format='pdf', bbox_inches='tight')
        # plt.close()

        # stem version with markers
        markerline, stemlines, baseline = plt.stem(day_range, prob_s_0, label='$p_i(s^*=0|d_{current})$', markerfmt='C0o', linefmt='C0--', basefmt='k-')
        plt.setp(stemlines, 'linewidth', 0.8)
        plt.setp(stemlines, 'alpha', 0.5)
        plt.setp(markerline, 'markersize', 3)
        markerline, stemlines, baseline = plt.stem(day_range, prob_s_1, label='$p_i(s^*=1|d_{current})$', markerfmt='C1o', linefmt='C1--', basefmt='k-')
        plt.setp(stemlines, 'linewidth', 0.8)
        plt.setp(stemlines, 'alpha', 0.5)
        plt.setp(markerline, 'markersize', 3)
        # markerline, stemlines, baseline = plt.stem(day_range, prob_s_2, label='$p_i(s^*=2|d_{current})$', markerfmt='C2o', linefmt='C2--', basefmt='k-')
        # plt.setp(stemlines, 'linewidth', 0.8)
        # plt.setp(stemlines, 'alpha', 0.5)
        # plt.setp(markerline, 'markersize', 3)
        if cycle_mean_predictions is not None:
            plt.scatter(30, prob_s_0[30], marker='X', s=120, color='tab:blue',label='$p(s^*=0|d_{current}=30]$', zorder=2)
            plt.scatter(30, prob_s_1[30], marker='X', s=120, color='tab:orange',label='$p(s^*=1|d_{current}=30]$', zorder=2)
            plt.scatter(40, prob_s_0[40], marker='^', s=120, color='tab:blue',label='$p(s^*=0|d_{current}=40]$', zorder=2)
            plt.scatter(40, prob_s_1[40], marker='^', s=120, color='tab:orange',label='$p(s^*=1|d_{current}=40]$', zorder=2)
            plt.axvline(x=30, color='k', zorder=1)
            plt.axvline(x=40, color='k', zorder=1)
            # plt.axvline(x=cycle_pred_day_0, color='tab:blue', linestyle='dotted', label='$E_i[d^*|d_{current}=0]$')
            # plt.axvline(x=cycle_pred_day_30, color='tab:orange',linestyle='dotted', label='$E_i[d^*|d_{current}=30]$')
            # plt.axvline(x=cycle_pred_day_60, color='tab:green',linestyle='dotted', label='$E_i[d^*|d_{current}=60]$')
            # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue')
            # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange')
            # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green')

            # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
            # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
            # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        plt.xlabel('Current day')
        plt.ylabel('$p_i(s^*|d_{current})$')
        plt.autoscale(enable=True, tight=True, axis='x')
        plt.ylim((0,1))
        plt.legend(loc = 'right')
        filename = '{}/skipping_posterior_by_day_stem_with_markers_{}_i{}.pdf'.format(save_dir, stamp, i)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

        # stem version with markers (simplified)
        markerline, stemlines, baseline = plt.stem(day_range, prob_s_1, markerfmt='C1o', linefmt='C1--', basefmt='k-')
        plt.setp(stemlines, 'linewidth', 0.8)
        plt.setp(stemlines, 'alpha', 0.5)
        plt.setp(markerline, 'markersize', 3)
        # markerline, stemlines, baseline = plt.stem(day_range, prob_s_2, label='$p_i(s^*=2|d_{current})$', markerfmt='C2o', linefmt='C2--', basefmt='k-')
        # plt.setp(stemlines, 'linewidth', 0.8)
        # plt.setp(stemlines, 'alpha', 0.5)
        # plt.setp(markerline, 'markersize', 3)
        if cycle_mean_predictions is not None:
            plt.scatter(30, prob_s_1[30], marker='X', s=120, color='tab:orange',label='Probability of user having skipped one cycle (on day 30)', zorder=2)
            plt.scatter(40, prob_s_1[40], marker='^', s=120, color='tab:orange',label='Probability of user having skipped one cycle (on day 40)', zorder=2)
            plt.axvline(x=30, color='k', zorder=1)
            plt.axvline(x=40, color='k', zorder=1)
            # plt.axvline(x=cycle_pred_day_0, color='tab:blue', linestyle='dotted', label='$E_i[d^*|d_{current}=0]$')
            # plt.axvline(x=cycle_pred_day_30, color='tab:orange',linestyle='dotted', label='$E_i[d^*|d_{current}=30]$')
            # plt.axvline(x=cycle_pred_day_60, color='tab:green',linestyle='dotted', label='$E_i[d^*|d_{current}=60]$')
            # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue')
            # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange')
            # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green')

            # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
            # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
            # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        plt.xlabel('Current day')
        plt.ylabel('Probability of user having skipped one cycle')
        plt.autoscale(enable=True, tight=True, axis='x')
        plt.ylim((0,1))
        plt.legend(loc = 'right')
        filename = '{}/skipping_posterior_by_day_stem_with_markers_simplified_{}_i{}.pdf'.format(save_dir, stamp, i)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

        # stem version with markers (simplified) (legend loc diff)
        markerline, stemlines, baseline = plt.stem(day_range, prob_s_1, markerfmt='C1o', linefmt='C1--', basefmt='k-')
        plt.setp(stemlines, 'linewidth', 0.8)
        plt.setp(stemlines, 'alpha', 0.5)
        plt.setp(markerline, 'markersize', 3)
        # markerline, stemlines, baseline = plt.stem(day_range, prob_s_2, label='$p_i(s^*=2|d_{current})$', markerfmt='C2o', linefmt='C2--', basefmt='k-')
        # plt.setp(stemlines, 'linewidth', 0.8)
        # plt.setp(stemlines, 'alpha', 0.5)
        # plt.setp(markerline, 'markersize', 3)
        if cycle_mean_predictions is not None:
            plt.scatter(30, prob_s_1[30], marker='X', s=120, color='tab:orange',label='Probability of user having skipped one cycle (on day 30)', zorder=2)
            plt.scatter(40, prob_s_1[40], marker='^', s=120, color='tab:orange',label='Probability of user having skipped one cycle (on day 40)', zorder=2)
            plt.axvline(x=30, color='k', zorder=1)
            plt.axvline(x=40, color='k', zorder=1)
            # plt.axvline(x=cycle_pred_day_0, color='tab:blue', linestyle='dotted', label='$E_i[d^*|d_{current}=0]$')
            # plt.axvline(x=cycle_pred_day_30, color='tab:orange',linestyle='dotted', label='$E_i[d^*|d_{current}=30]$')
            # plt.axvline(x=cycle_pred_day_60, color='tab:green',linestyle='dotted', label='$E_i[d^*|d_{current}=60]$')
            # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue')
            # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange')
            # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green')

            # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
            # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
            # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        plt.xlabel('Current day')
        plt.ylabel('Probability of user having skipped one cycle')
        plt.autoscale(enable=True, tight=True, axis='x')
        plt.ylim((0,1))
        plt.legend(loc = 'upper right')
        filename = '{}/skipping_posterior_by_day_stem_with_markers_simplified_legend_{}_i{}.pdf'.format(save_dir, stamp, i)
        plt.savefig(filename, format='pdf', bbox_inches='tight')
        plt.close()

        # p(s^*=1) curves only, on same plot (separately for now)
        # plt.scatter(day_range, prob_s_1, color='tab:orange', label='$p_i(s^*=1|d_{current})$', s=3)
        # plt.plot(day_range, prob_s_1, color='tab:orange', label='$p_i(s^*=1|d_{current})$')

        # if cycle_mean_predictions is not None:
        #     plt.axvline(x=30, color='k')
        #     plt.axvline(x=40, color='k')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
        #     # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
        #     # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        #     # plt.axvline(x=cycle_pred_day_0, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_60, color='k', linestyle='dotted')
        # plt.xlabel('Current day of next cycle ($d_{current}$)')
        # plt.ylabel('$p_i(s^*|d^*>d_{current})$')
        # plt.autoscale(enable=True, tight=True, axis='x')
        # plt.ylim((0,1))
        # plt.legend()
        # filename = '{}/skipping_posterior_by_day_line_s_1_only_{}_i{}.pdf'.format(save_dir, stamp, i)
        # plt.savefig(filename, format='pdf', bbox_inches='tight')
        # plt.close()

        # # with markers
        # plt.scatter(day_range, prob_s_1, color='tab:orange', label='$p_i(s^*=1|d_{current})$', s=3)
        # plt.plot(day_range, prob_s_1, color='tab:orange', label='$p_i(s^*=1|d_{current})$')

        # if cycle_mean_predictions is not None:
        #     plt.scatter(30, prob_s_1[30], marker='X', s=120, color='tab:orange',label='$p(s^*=1|d_{current}=30]$')
        #     plt.scatter(40, prob_s_1[40], marker='^', s=120, color='tab:orange',label='$p(s^*=1|d_{current}=40]$')
        #     plt.axvline(x=30, color='k')
        #     plt.axvline(x=40, color='k')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.scatter(cycle_pred_day_0, prob_s_0[round(cycle_pred_day_0)], marker='X', s=120, color='tab:blue',label='$E_i[d^*|d^*>0]$')
        #     # plt.scatter(cycle_pred_day_30, prob_s_1[round(cycle_pred_day_30)], marker='X', s=120, color='tab:orange',label='$E_i[d^*|d^*>30]$')
        #     # plt.scatter(cycle_pred_day_60, prob_s_2[round(cycle_pred_day_60)], marker='X', s=120, color='tab:green',label='$E_i[d^*|d^*>60]$')
        #     # plt.axvline(x=cycle_pred_day_0, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_30, color='k', linestyle='dotted')
        #     # plt.axvline(x=cycle_pred_day_60, color='k', linestyle='dotted')
        # plt.xlabel('Current day of next cycle ($d_{current}$)')
        # plt.ylabel('$p_i(s^*|d^*>d_{current})$')
        # plt.autoscale(enable=True, tight=True, axis='x')
        # plt.ylim((0,1))
        # plt.legend(loc = 'lower right')
        # filename = '{}/skipping_posterior_by_day_line_s_1_only_with_markers_{}_i{}.pdf'.format(save_dir, stamp, i)
        # plt.savefig(filename, format='pdf', bbox_inches='tight')
        # plt.close()

        # fig = plt.figure()
        # ax = fig.add_subplot(1, 1, 1, projection='3d')
        # s_range = np.arange(6)
        # pdb.set_trace()
        # for xi, yi, zi in zip(s_range, day_range, predictive_posterior[i][:,:6]):        
        #     line=art3d.Line3D(*zip((xi, yi, 0), (xi, yi, zi)), marker='o', markevery=(1, 1))
        #     ax.add_line(line)
        # ax.set_xlabel('Number of skipped cycles')
        # ax.set_xlim(0,5)
        # ax.set_ylabel('Current day')
        # ax.set_ylim(day_range[0],day_range[-1])
        # ax.set_zlabel('Probability')
        # # Customize the z axis?
        # plt.title('Prediction by day stem plot')
        # filename = '{}/skipping_posterior_by_day_stem_{}_i{}.pdf'.format(save_dir, stamp, i)
        # # Rotate for better viewing?
        # #ax.view_init(30, 30)
        # if save_dir is None:
        #     plt.show()
        # else:
        #     plt.savefig(filename, format='pdf', bbox_inches='tight')
        #     plt.close()


def plot_reg_vs_pred(X_train, Y_pred, Y_true, model_name, main_dir, stamp):
    '''Plots measure of regularity vs. prediction accuracy'''
    # compute abs diff btw true and pred (length I) - note that this is only for day 0
    pred_abs_err = np.abs(Y_pred - Y_true)

    print('Y pred for {}: {}'.format(model_name, Y_pred))
    print('avg Y pred for {}: {}'.format(model_name, np.mean(Y_pred)))
    print('avg abs error for {}: {}'.format(model_name, np.mean(pred_abs_err)))

    # save dir
    save_dir = '{}/reg_vs_pred'.format(main_dir)
    os.makedirs(save_dir, exist_ok=True)

    # compute median CLD and std of cycles (length I)
    median_CLD = np.median(np.abs(X_train[:,:-1] - X_train[:,1:]), axis=1)
    median_CLD = median_CLD.astype(int)
    cycle_std = np.std(X_train, axis=1)

    # make dataframe with cycle std, median CLD, and abs error
    pred_df = pd.DataFrame(data={'median_CLD':median_CLD.flatten(), 'cycle_std':cycle_std.flatten(), 'Y_pred': Y_pred.flatten(), 'Y_true': Y_true.flatten(), 'abs_error':pred_abs_err.flatten()})

    # restrict to median CLD <= 25
    print('Proportion of users with median CLD > 25: {}'.format(len(pred_df[pred_df['median_CLD']>25])/len(pred_df)))
    pred_df = pred_df[pred_df['median_CLD']<=25]

    # print users' cycles whose abs error > 45 with median CLD 0
    print('Users with abs error > 45 and median CLD = 0: {}'.format(pred_df[(pred_df['median_CLD']==0) & (pred_df['abs_error']>=45)]))
    
    #violin plot
    sns.violinplot(x="median_CLD", y="Y_pred", data=pred_df)
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.ylabel('Predicted next cycle length')
    plt.savefig('{}/median_CLD_vs_pred_violin_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    sns.violinplot(x="median_CLD", y="abs_error", data=pred_df)
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.ylabel('Absolute error of predicted next cycle')
    plt.savefig('{}/median_CLD_vs_abs_error_violin_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    #box plot
    sns.boxplot(x="median_CLD", y="Y_pred", data=pred_df)
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.ylabel('Predicted next cycle length')
    plt.savefig('{}/median_CLD_vs_pred_box_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    sns.boxplot(x="median_CLD", y="abs_error", data=pred_df)
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.ylabel('Absolute error of predicted next cycle')
    plt.savefig('{}/median_CLD_vs_abs_error_box_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    # summary statistic plots
    median_abs_error = pred_df.groupby('median_CLD').median()['abs_error']
    mean_abs_error = pred_df.groupby('median_CLD').mean()['abs_error']
    median_CLD_0 = pred_df[pred_df['median_CLD']==0]
    RMSE_median_CLD_0 = np.sqrt(np.sum((median_CLD_0['Y_true'] - median_CLD_0['Y_pred'])**2) / len(median_CLD_0))

    print('median abs error for median CLD 0: '+str(median_abs_error[0]))
    print('mean abs error for median CLD 0: '+str(mean_abs_error[0]))
    print('RMSE for median CLD 0: '+str(RMSE_median_CLD_0))

    plt.scatter(np.unique(pred_df['median_CLD']), median_abs_error)
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.ylabel('Median absolute error of predicted next cycle')
    plt.savefig('{}/median_CLD_vs_median_abs_error_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    plt.scatter(np.unique(pred_df['median_CLD']), mean_abs_error)
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.ylabel('Mean absolute error of predicted next cycle')
    plt.savefig('{}/median_CLD_vs_mean_abs_error_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()


    #lm plot
    # sns.lmplot(x="median_CLD", y="abs_error", data=pred_df)
    # plt.xticks(rotation=90)
    # plt.xlabel('Median CLD')
    # plt.ylabel('Absolute error of predicted next cycle')
    # plt.savefig('{}/median_CLD_vs_abs_error_lr_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    # plt.close()

    # sns.lmplot(x="median_CLD", y="Y_pred", data=pred_df)
    # plt.xticks(rotation=90)
    # plt.xlabel('Median CLD')
    # plt.ylabel('Predicted next cycle length')
    # plt.savefig('{}/median_CLD_vs_pred_lr_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    # plt.close()

    # sns.lmplot(x="cycle_std", y="abs_error", data=pred_df)
    # plt.xticks(rotation=90)
    # plt.xlabel('Standard deviation of cycles')
    # plt.ylabel('Absolute error of predicted next cycle')
    # plt.savefig('{}/cycle_std_vs_abs_error_lr_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    # plt.close()

    # sns.lmplot(x="cycle_std", y="Y_pred", data=pred_df)
    # plt.xticks(rotation=90)
    # plt.xlabel('Standard deviation of cycles')
    # plt.ylabel('Predicted next cycle length')
    # plt.savefig('{}/cycle_std_vs_pred_lr_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    # plt.close()

    # make histogram of median CLD, Y_pred, abs error
    binwidth=1
    data = pred_df['median_CLD']
    plt.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth))
    plt.xticks(rotation=90)
    plt.xlabel('Median CLD')
    plt.savefig('{}/median_CLD_hist_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    data = pred_df['Y_pred']
    plt.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth))
    plt.xticks(rotation=90)
    plt.xlabel('Predicted next cycle length')
    plt.savefig('{}/pred_hist_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

    data = pred_df['abs_error']
    plt.hist(data, bins=range(int(min(data)), int(max(data)) + binwidth, binwidth))
    plt.ylabel('Absolute error')
    plt.xticks(rotation=90)
    plt.xlabel('Absolute error of predicted next cycle')
    plt.savefig('{}/abs_error_hist_{}_{}.pdf'.format(save_dir, model_name, stamp), format='pdf', bbox_inches='tight')
    plt.close()

#### Calibration plots
def plot_pit_plot(pit, day_range, y_range, plot_filename=None):
    # Histogram over instances
    #for day in day_range:
    for day in day_range[:1]:
        plt.figure()
        # Bins are in p_range
        plt.hist(pit[:,day], bins='auto', density='True', label='PIT', color='blue', histtype='bar')
        # Axis
        plt.xlabel('$p$')
        plt.xlim(0,1)
        plt.ylabel('PIT')
        plt.title('PIT at day {}'.format(day))
        if plot_filename is None:
            plt.show()
        else:
            plt.savefig('{}.pdf'.format(plot_filename), format='pdf', bbox_inches='tight')
            plt.close()
    
def plot_mcp_plot(mcp, day_range, y_range, plot_filename=None):    
    # Useful to plot over y_range
    #for day in day_range:
    for day in day_range[:1]:
        plt.figure()
        # Bins are y_range
        plt.plot(mcp[day,:], label='MCP', color='blue')
        # Axis
        plt.xlabel('$x_{C+1}$')
        plt.xlim(y_range[0],y_range[-1])
        plt.ylabel('MCP')
        plt.title('MCP at day {}'.format(day))
        if plot_filename is None:
            plt.show()
        else:
            plt.savefig('{}.pdf'.format(plot_filename), format='pdf', bbox_inches='tight')
            plt.close()

# Plot computed calibration metrics over models
def plot_predictive_posterior_calibration_by_day_over_models(calib_results, day_range, I_range, C_range, calibration_metrics=None, save_dir=None, stamp='nostamp', colors=default_colors):    
    if save_dir is not None:
        # Make sure directory is ready
        os.makedirs(save_dir, exist_ok = True)
    
    # Collect prediction_models:
    prediction_models=[*calib_results]
    
    # fix prediction model names
    prediction_model_labels = prediction_models.copy()
    for i in range(len(prediction_model_labels)):
        if 'generative' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('generative_','')
        if 'kathy_prior' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('kathy_prior_','')
        if 'cohort' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('_per_cohort_s_predict','')
        if 'individual' in prediction_model_labels[i]:
            prediction_model_labels[i] = prediction_model_labels[i].replace('_per_individual_s_predict','')

    # If evaluation metric is not specified, then figure out
    if calibration_metrics is None:
        calibration_metrics=[*calib_results[prediction_models[0]]]
        
    # Figure per evaluation metric
    for calibration_metric in filter(lambda m: 'plot' not in m, calibration_metrics):
        for this_I_idx, this_I in enumerate(I_range):
            for this_C_idx, this_C in enumerate(C_range):
                # Figure with all prediction models
                plt.figure()
                for model_idx, prediction_model in enumerate(prediction_models):
                    mean_calib_metric_for_model = np.nanmean(calib_results[prediction_model][calibration_metric][:,this_I_idx,this_C_idx,:], axis=0)
                    if mean_calib_metric_for_model.any() > 0:
                        plt.plot(day_range,
                                mean_calib_metric_for_model,
                                colors[model_idx%len(colors)],
                                label='{}'.format(prediction_model_labels[model_idx])
                            )
                plt.xlabel('Current day')
                plt.xlim(day_range[0],day_range[-1])
                plt.ylabel('{}'.format(calibration_metric))
                plt.autoscale(enable=True, tight=True)
                legend = plt.legend(bbox_to_anchor=(1.05,1.05), loc='upper left', ncol=1, shadow=True)
                plt.legend()
                filename = '{}/plot_{}_by_day_I{}_C{}_{}.pdf'.format(save_dir, calibration_metric, this_I, this_C, stamp)
                if save_dir is None:
                    plt.show()
                else:
                    plt.savefig(filename, format='pdf', bbox_inches='tight')
                    plt.close()

