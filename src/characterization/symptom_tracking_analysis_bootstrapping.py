#!/usr/bin/python

# Imports
import sys, os, re, time
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np
import scipy.stats as stats
import pandas as pd
# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D

################################## FUNCTIONS ############################
def tracking_analysis_for_attribute_cutoff_bootstrapped(cycle_stats_df, category_stats, attribute, cutoff_criteria, cutoff, n_bootstrapping, save_dir):
    '''
        Function that computes tracking analysis per group, based on bootstrapping
            It computes the Kolmogorov-Smirnov tests between group distributions
            It computes the likelihood in low, mid and high extremes of the metric

        Input:
            cycle_stats_df: pandas dataframe, with information about user's cycle statistics
            category_stats: pandas dataframe, with information about user's tracking statistics for a given category
            attribute: what specific tracking attribute to study: i.e., concatenation of the metric and the symptom to analyze
            cutoff_criteria: what statistic to use for separating users into groups ('cycle_lengths' for paper)
            cutoff: what statistic cutoff value to use for separating users into groups (9 for paper)
            n_bootstrapping: Number of bootstrapped samples to use for the analysis
            save_dir: path where to save plot
        Output:
            true_statistics: Dictionary with statistics for the observed cohort:
                {'KS', 'p_val', 'prob_values_high', 'prob_values_low', 'ratios'}
            bootstrap_statistics: Dictionary with statistics for the bootstrapped cohort:
                {'KS': mean of the bootstrapped KS values, 'KS_0025': 2.5 percentile of the bootstrapped KS values, 'KS_0975': 97.5 percentile of the bootstrapped KS values,
                'p_val': mean of the bootstrapped p_val values, 'p_val_0025': 2.5 percentile of the bootstrapped p_val values, 'p_val_0975': 97.5 percentile of the bootstrapped p_val values,
                'prob_values_high': mean of the boostrapped probability values for the high volatility group,
                'prob_values_high_0025': 2.5 percentile of the boostrapped probability values for the high volatility group,
                'prob_values_high_0975': 97.5 percentile of the boostrapped probability values for the high volatility group,
                'prob_values_low': mean of the boostrapped probability values for the low volatility group,
                'prob_values_low_0025': 2.5 percentile of the boostrapped probability values for the low volatility group,
                'prob_values_low_0975': 97.5 percentile of the boostrapped probability values for the low volatility group,
                'ratios': mean of the bootstrapped ratios for the high to low volability groups
                'ratios_0025': 2.5 percentile of the bootstrapped ratios for the high to low volability groups
                'ratios_0975': 97.5 percentile of the bootstrapped ratios for the high to low volability groups}
    '''
    ### Define
    # Extreme likelihood ranges
    extreme_bins=np.array([0,0.05,0.95,1])
    # Histogram type, color and labels
    hist_type='step'
    colors = ['orange', 'c']
    labels=['Highly variable', 'NOT highly variable']
    
    # True separation of users into groups
    all_users=np.unique(cycle_stats_df['user_id'])
    true_users_greater_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] > cutoff]['user_id'])
    true_users_less_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] <= cutoff]['user_id'])
    n_users_greater_than_cutoff=true_users_greater_than_cutoff.size
    n_users_less_than_cutoff=true_users_less_than_cutoff.size
    true_category_stats_users_greater_than_cutoff = category_stats[category_stats['user_id'].isin(true_users_greater_than_cutoff)]
    true_category_stats_users_less_than_cutoff = category_stats[category_stats['user_id'].isin(true_users_less_than_cutoff)]    
    
    # Analysis for proportion of cycles metric
    if attribute.startswith('proportion_cycles_'):
        ########### TRUE OBSERVERD STATISTICS ##########
        # KS
        true_KS, true_p_val = stats.ks_2samp(true_category_stats_users_greater_than_cutoff[attribute].dropna(), true_category_stats_users_less_than_cutoff[attribute].dropna())

        # Counts on extremes
        true_extreme_counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(true_category_stats_users_greater_than_cutoff[attribute].dropna(), bins=extreme_bins, density=True)
        true_extreme_counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(true_category_stats_users_less_than_cutoff[attribute].dropna(), bins=extreme_bins, density=True)
        # Probability values
        true_prob_values_high=np.array([(true_extreme_counts_greater_than_cutoff[0]*0.05), (true_extreme_counts_greater_than_cutoff[1]*0.9), (true_extreme_counts_greater_than_cutoff[2]*0.05)])
        true_prob_values_low=np.array([(true_extreme_counts_less_than_cutoff[0]*0.05), (true_extreme_counts_less_than_cutoff[1]*0.9), (true_extreme_counts_less_than_cutoff[2]*0.05)])
        # Ratios
        true_ratios=np.array([true_prob_values_high[0]/true_prob_values_low[0], true_prob_values_high[1]/true_prob_values_low[1], true_prob_values_high[2]/true_prob_values_low[2]])
        
        # CDF
        # Auto bins based on integer range of values
        counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(true_category_stats_users_greater_than_cutoff[attribute].dropna(), bins='auto', density=True)
        counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(true_category_stats_users_less_than_cutoff[attribute].dropna(), bins='auto', density=True)
        all_bins=np.setdiff1d(bins_less_than_cutoff,bins_greater_than_cutoff)
        true_counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(true_category_stats_users_greater_than_cutoff[attribute].dropna(), bins=all_bins, density=True)
        true_counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(true_category_stats_users_less_than_cutoff[attribute].dropna(), bins=all_bins, density=True)
                
        ########### BOOTSTRAP BASED STATISTICS ##########
        # Computed suff statistics
        bootstrapped_KS=np.zeros(n_bootstrapping)
        bootstrapped_p_val=np.zeros(n_bootstrapping)
        bootstrapped_prob_values_high=np.zeros((n_bootstrapping, extreme_bins.size-1))
        bootstrapped_prob_values_low=np.zeros((n_bootstrapping, extreme_bins.size-1))
        bootstrapped_ratios=np.zeros((n_bootstrapping, extreme_bins.size-1))
        bootstrapped_counts_greater_than_cutoff=np.zeros((n_bootstrapping, all_bins.size-1))
        bootstrapped_counts_less_than_cutoff=np.zeros((n_bootstrapping, all_bins.size-1))
    
        for n_bootstrap in np.arange(n_bootstrapping):
            #print('Sample={}/{}'.format(n_bootstrap,n_bootstrapping))
            # Bootstrapped sample indicators
            users_greater_than_cutoff=np.random.choice(true_users_greater_than_cutoff,n_bootstrapping)
            users_less_than_cutoff=np.random.choice(true_users_less_than_cutoff,n_bootstrapping)
            # Bootstrapped data
            category_stats_users_greater_than_cutoff = category_stats[category_stats['user_id'].isin(users_greater_than_cutoff)]
            category_stats_users_less_than_cutoff = category_stats[category_stats['user_id'].isin(users_less_than_cutoff)]    
            
            # KS
            bootstrapped_KS[n_bootstrap], bootstrapped_p_val[n_bootstrap] = stats.ks_2samp(category_stats_users_greater_than_cutoff[attribute].dropna(), category_stats_users_less_than_cutoff[attribute].dropna())
            # Counts on extremes
            counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(category_stats_users_greater_than_cutoff[attribute].dropna(), bins=extreme_bins, density=True)
            counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(category_stats_users_less_than_cutoff[attribute].dropna(), bins=extreme_bins, density=True)
            # Probability values
            bootstrapped_prob_values_high[n_bootstrap]=np.array([(counts_greater_than_cutoff[0]*0.05), (counts_greater_than_cutoff[1]*0.9), (counts_greater_than_cutoff[2]*0.05)])
            bootstrapped_prob_values_low[n_bootstrap]=np.array([(counts_less_than_cutoff[0]*0.05), (counts_less_than_cutoff[1]*0.9), (counts_less_than_cutoff[2]*0.05)])
            # Ratios
            bootstrapped_ratios[n_bootstrap]=bootstrapped_prob_values_high[n_bootstrap]/bootstrapped_prob_values_low[n_bootstrap]
            
            # CDF, based on same bins as for true CDF
            bootstrapped_counts_greater_than_cutoff[n_bootstrap], bins_greater_than_cutoff = np.histogram(category_stats_users_greater_than_cutoff[attribute].dropna(), bins=all_bins, density=True)
            bootstrapped_counts_less_than_cutoff[n_bootstrap], bins_less_than_cutoff = np.histogram(category_stats_users_less_than_cutoff[attribute].dropna(), bins=all_bins, density=True)
    else:
        raise ValueError('Analysis for attribute {} not implemented'.format(attribute))

    # Print bootstrap results
    print('*************************************************************************')
    print('******** {0} KS={1:.3f} (p={2}) ***********'.format(
            attribute, true_KS, true_p_val
            ))
    print('******** {0} Bootstrapped KS={1:.3f}+/-{2:.3f} (p={3} (+/-{4}))***********'.format(
            attribute, bootstrapped_KS.mean(), bootstrapped_KS.std(), bootstrapped_p_val.mean(), bootstrapped_p_val.std()
            ))
    print('******** {0} Bootstrapped KS={1:.3f}({2:.3f},{3:.3f}) p={4} ({5},{6}))***********'.format(
            attribute, bootstrapped_KS.mean(), np.percentile(bootstrapped_KS, 2.5, axis=0), np.percentile(bootstrapped_KS, 97.5, axis=0), bootstrapped_p_val.mean(), np.percentile(bootstrapped_p_val, 2.5, axis=0), np.percentile(bootstrapped_p_val, 97.5, axis=0)
            ))
    print('Bins \t\t\t & p < 0.05 \t\t & 0.05 \leq p < 0.95 \t & 0.95 \leq 1')
    print('True ratio \t\t & {0:.3f} \t\t & {1:.3f} \t\t & {2:.3f}'.format(true_ratios[0],true_ratios[1],true_ratios[2]))
    print('Bootstrapped ratio \t & {0:.3f}+/-{1:.3f} \t & {2:.3f}+/-{3:.3f} \t & {4:.3f}+/-{5:.3f}'.format(
        bootstrapped_ratios.mean(axis=0)[0],bootstrapped_ratios.std(axis=0)[0],
        bootstrapped_ratios.mean(axis=0)[1],bootstrapped_ratios.std(axis=0)[1],
        bootstrapped_ratios.mean(axis=0)[2],bootstrapped_ratios.std(axis=0)[2]
        ))
    print('Bootstrapped ratio \t & {0:.3f} ({1:.3f}, {2:.3f}) \t & {3:.3f} ({4:.3f}, {5:.3f}) \t & {6:.3f} ({7:.3f}, {8:.3f})'.format(
        bootstrapped_ratios.mean(axis=0)[0], np.percentile(bootstrapped_ratios[:,0], 2.5, axis=0), np.percentile(bootstrapped_ratios[:,0], 97.5, axis=0),
        bootstrapped_ratios.mean(axis=0)[1], np.percentile(bootstrapped_ratios[:,1], 2.5, axis=0), np.percentile(bootstrapped_ratios[:,1], 97.5, axis=0),
        bootstrapped_ratios.mean(axis=0)[2], np.percentile(bootstrapped_ratios[:,2], 2.5, axis=0), np.percentile(bootstrapped_ratios[:,2], 97.5, axis=0)
        ))

    # CDF plotting
    # First normalize counts
    bootstrapped_pdf_greater_than_cutoff=bootstrapped_counts_greater_than_cutoff/bootstrapped_counts_greater_than_cutoff.sum(axis=1, keepdims=True)
    bootstrapped_pdf_less_than_cutoff=bootstrapped_counts_less_than_cutoff/bootstrapped_counts_less_than_cutoff.sum(axis=1, keepdims=True)
    # Plot (no bootstrap)
    # True
    plt.hist(all_bins[:-1], all_bins, weights=true_counts_greater_than_cutoff, density=True, cumulative=True, color=colors[0], histtype=hist_type)
    plt.hist(all_bins[:-1], all_bins, weights=true_counts_less_than_cutoff, density=True, cumulative=True, color=colors[1], histtype=hist_type)
    # Polish and close
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('$\lambda_s$')
    plt.ylabel('$P(\lambda_s \leq \lambda)$')
    # Custom legend
    custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color='w', lw=4)]
    plt.legend(custom_lines, ['Highly variable', 'NOT highly variable', 'KS = {:.3f}'.format(true_KS)], loc='upper left')
    filename = '{}/{}_cdf.pdf'.format(save_dir, attribute)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()
    
    # Plot (with bootstrap)
    # True
    plt.hist(all_bins[:-1], all_bins, weights=true_counts_greater_than_cutoff, density=True, cumulative=True, color=colors[0], histtype=hist_type)
    plt.hist(all_bins[:-1], all_bins, weights=true_counts_less_than_cutoff, density=True, cumulative=True, color=colors[1], histtype=hist_type)
    # Bootstrapped
    plt.plot(all_bins[:-1], bootstrapped_pdf_greater_than_cutoff.cumsum(axis=1).mean(axis=0), ':', color=colors[0])
    plt.plot(all_bins[:-1], bootstrapped_pdf_less_than_cutoff.cumsum(axis=1).mean(axis=0), ':', color=colors[1])
    plt.fill_between(all_bins[:-1],
        np.percentile(bootstrapped_pdf_greater_than_cutoff.cumsum(axis=1), 2.5, axis=0),
        np.percentile(bootstrapped_pdf_greater_than_cutoff.cumsum(axis=1), 97.5, axis=0),
        alpha=0.3, facecolor=colors[0])
    plt.fill_between(all_bins[:-1],
        np.percentile(bootstrapped_pdf_less_than_cutoff.cumsum(axis=1), 2.5, axis=0),
        np.percentile(bootstrapped_pdf_less_than_cutoff.cumsum(axis=1), 97.5, axis=0),
        alpha=0.3, facecolor=colors[1])
    # Polish and close
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('$\lambda_s$')
    plt.ylabel('$P(\lambda_s \leq \lambda)$')
    # Custom legend
    custom_lines = [Line2D([0], [0], color=colors[0], lw=2),
                Line2D([0], [0], color=colors[1], lw=2),
                Line2D([0], [0], color='w', lw=4)]
    plt.legend(custom_lines, ['Highly variable', 'NOT highly variable', 'KS = {:.3f}'.format(true_KS)], loc='upper left')
    filename = '{}/{}_bcdf.pdf'.format(save_dir, attribute)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()    
    
    # True statistics
    true_statistics={'KS':true_KS, 'p_val':true_p_val, 'prob_values_high':true_prob_values_high, 'prob_values_low':true_prob_values_low, 'ratios':true_ratios}
    bootstrap_statistics={
        'KS':bootstrapped_KS.mean(), 'KS_0025':np.percentile(bootstrapped_KS, 2.5, axis=0), 'KS_0975':np.percentile(bootstrapped_KS, 97.5, axis=0),
        'p_val':bootstrapped_p_val.mean(), 'p_val_0025':np.percentile(bootstrapped_p_val, 2.5, axis=0), 'p_val_0975':np.percentile(bootstrapped_p_val, 97.5, axis=0),
        'prob_values_high':bootstrapped_prob_values_high.mean(axis=0),
        'prob_values_high_0025':np.percentile(bootstrapped_prob_values_high, 2.5, axis=0), 'prob_values_high_0975':np.percentile(bootstrapped_prob_values_high, 97.5, axis=0),
        'prob_values_low':bootstrapped_prob_values_low.mean(axis=0),
        'prob_values_low_0025':np.percentile(bootstrapped_prob_values_low, 2.5, axis=0), 'prob_values_low_0975':np.percentile(bootstrapped_prob_values_low, 97.5, axis=0),
        'ratios':bootstrapped_ratios.mean(axis=0), 'ratios_0025':np.percentile(bootstrapped_ratios, 2.5, axis=0), 'ratios_0975':np.percentile(bootstrapped_ratios, 97.5, axis=0)
        }
    return true_statistics, bootstrap_statistics
    
################################## MAIN ############################
def main(n_bootstrapping):
    '''
        Main function of the script that runs the symptom tracking analysis, based on bootstrapping

        Input:
            n_bootstrapping: Number of bootstrapped samples to use for the analysis
        Output:
            None
    '''
    
    ### Directories
    data_dir='../data'
    preprocessed_data_dir='../preprocessed_data'
    results_dir = '../results/characterizing_cycle_and_symptoms/symptom_tracking_analysis_bootstrapping_{}'.format(n_bootstrapping)
    os.makedirs(results_dir, exist_ok = True)
    
    ################# SYMPTOMS TRACKED #################
    # Tracking
    with open('{}/tracking_enriched.pickle'.format(data_dir), 'rb') as f:
	    tracking = pickle.load(f)

    print('Tracking-data loaded')

    ################# CYCLES #################    
    # Cycles stats
    with open('{}/cohort_clean_cycle_stats.pickle'.format(preprocessed_data_dir), 'rb') as f:
        cohort_clean_cycle_stats = pickle.load(f)

    print('Cycles-data loaded')

    ################# SYMPTOM TRACKING ANALYSIS #################    
    # Categories of interest
    categories =  list(tracking.category.cat.categories)
    # HBC related
    categories.remove('injection_hbc')
    categories.remove('iud')
    categories.remove('patch_hbc')
    categories.remove('pill_hbc')
    categories.remove('ring_hbc')

    # Metric of interest
    metric='proportion_cycles_out_of_cycles_with_category_for_'
    
    # Initialize dataframe for analysis
    symptom_metric_extreme_ratios=pd.DataFrame(columns=['category','symptom',
        'ks', 'ks_mean', 'ks_0025', 'ks_0975',
        'p_val', 'p_val_mean', 'p_val_0025', 'p_val_0975',
        # Low extreme
        'low_extreme_prob_high_var', 'low_extreme_prob_high_var_mean', 'low_extreme_prob_high_var_0025', 'low_extreme_prob_high_var_0975',
        'low_extreme_prob_low_var', 'low_extreme_prob_low_var_mean', 'low_extreme_prob_low_var_0025', 'low_extreme_prob_low_var_0975',
        'low_extreme_ratios', 'low_extreme_ratios_mean', 'low_extreme_ratios_0025', 'low_extreme_ratios_0975',
        # Mid range
        'mid_extreme_prob_high_var', 'mid_extreme_prob_high_var_mean', 'mid_extreme_prob_high_var_0025', 'mid_extreme_prob_high_var_0975',
        'mid_extreme_prob_low_var', 'mid_extreme_prob_low_var_mean', 'mid_extreme_prob_low_var_0025', 'mid_extreme_prob_low_var_0975',
        'mid_extreme_ratios', 'mid_extreme_ratios_mean', 'mid_extreme_ratios_0025', 'mid_extreme_ratios_0975',
        # High extreme
        'high_extreme_prob_high_var', 'high_extreme_prob_high_var_mean', 'high_extreme_prob_high_var_0025', 'high_extreme_prob_high_var_0975',
        'high_extreme_prob_low_var', 'high_extreme_prob_low_var_mean', 'high_extreme_prob_low_var_0025', 'high_extreme_prob_low_var_0975',
        'high_extreme_ratios', 'high_extreme_ratios_mean', 'high_extreme_ratios_0025', 'high_extreme_ratios_0975'
        ])

    ################# Perform analysis for all categories ##############
    for category in categories:
        # Load symptom tracking for this category 
        with open('{}/cohort_clean_symptom_tracking_stats_{}.pickle'.format(preprocessed_data_dir, category), 'rb') as f:
            category_stats=pickle.load(f)

        # Filter out data for symptoms within category
        tracking_for_category = tracking[tracking['category']==category]
        symptoms_for_category = tracking_for_category['type'].value_counts()
        symptoms_for_category = symptoms_for_category[symptoms_for_category > 0].index.tolist()
        
        # For all symptoms within category
        for symptom in symptoms_for_category:
            # Attribute to study
            attribute = metric+str(symptom)
            
            # Actually run the bootstrapped analysis
            true_statistics, bootstrap_statistics=tracking_analysis_for_attribute_cutoff_bootstrapped(
                                                                                                        cohort_clean_cycle_stats,
                                                                                                        category_stats,
                                                                                                        attribute,
                                                                                                        'median_inter_cycle_length',
                                                                                                        9,
                                                                                                        n_bootstrapping,
                                                                                                        results_dir
                                                                                                        )
                                                                                                        
            # Process results of interest
            symptom_metric_extreme_ratios=symptom_metric_extreme_ratios.append({
            'category':category,'symptom':symptom,
            'ks':true_statistics['KS'], 'ks_mean':bootstrap_statistics['KS'], 'ks_0025':bootstrap_statistics['KS_0025'], 'ks_0975':bootstrap_statistics['KS_0975'],
            'p_val':true_statistics['p_val'], 'p_val_mean':bootstrap_statistics['p_val'], 'p_val_0025':bootstrap_statistics['p_val_0025'], 'p_val_0975':bootstrap_statistics['p_val_0975'],
            # Low extreme
            'low_extreme_prob_high_var':true_statistics['prob_values_high'][0], 'low_extreme_prob_high_var_mean':bootstrap_statistics['prob_values_high'][0],
            'low_extreme_prob_high_var_0025':bootstrap_statistics['prob_values_high_0025'][0], 'low_extreme_prob_high_var_0975':bootstrap_statistics['prob_values_high_0975'][0],
            'low_extreme_prob_low_var':true_statistics['prob_values_low'][0], 'low_extreme_prob_low_var_mean':bootstrap_statistics['prob_values_low'][0],
            'low_extreme_prob_low_var_0025':bootstrap_statistics['prob_values_low_0025'][0], 'low_extreme_prob_low_var_0975':bootstrap_statistics['prob_values_low_0975'][0],
            'low_extreme_ratios':true_statistics['ratios'][0], 'low_extreme_ratios_mean':bootstrap_statistics['ratios'][0],
            'low_extreme_ratios_0025':bootstrap_statistics['ratios_0025'][0], 'low_extreme_ratios_0975':bootstrap_statistics['ratios_0975'][0],
            # Mid range
            'mid_extreme_prob_high_var':true_statistics['prob_values_high'][1], 'mid_extreme_prob_high_var_mean':bootstrap_statistics['prob_values_high'][1],
            'mid_extreme_prob_high_var_0025':bootstrap_statistics['prob_values_high_0025'][1], 'mid_extreme_prob_high_var_0975':bootstrap_statistics['prob_values_high_0975'][1],
            'mid_extreme_prob_low_var':true_statistics['prob_values_low'][1], 'mid_extreme_prob_low_var_mean':bootstrap_statistics['prob_values_low'][1],
            'mid_extreme_prob_low_var_0025':bootstrap_statistics['prob_values_low_0025'][1], 'mid_extreme_prob_low_var_0975':bootstrap_statistics['prob_values_low_0975'][1],
            'mid_extreme_ratios':true_statistics['ratios'][1], 'mid_extreme_ratios_mean':bootstrap_statistics['ratios'][1],
            'mid_extreme_ratios_0025':bootstrap_statistics['ratios_0025'][1], 'mid_extreme_ratios_0975':bootstrap_statistics['ratios_0975'][1],
            # High extreme
            'high_extreme_prob_high_var':true_statistics['prob_values_high'][2], 'high_extreme_prob_high_var_mean':bootstrap_statistics['prob_values_high'][2],
            'high_extreme_prob_high_var_0025':bootstrap_statistics['prob_values_high_0025'][2], 'high_extreme_prob_high_var_0975':bootstrap_statistics['prob_values_high_0975'][2],
            'high_extreme_prob_low_var':true_statistics['prob_values_low'][2], 'high_extreme_prob_low_var_mean':bootstrap_statistics['prob_values_low'][2],
            'high_extreme_prob_low_var_0025':bootstrap_statistics['prob_values_low_0025'][2], 'high_extreme_prob_low_var_0975':bootstrap_statistics['prob_values_low_0975'][2],
            'high_extreme_ratios':true_statistics['ratios'][2], 'high_extreme_ratios_mean':bootstrap_statistics['ratios'][2],
            'high_extreme_ratios_0025':bootstrap_statistics['ratios_0025'][2], 'high_extreme_ratios_0975':bootstrap_statistics['ratios_0975'][2]
            }, ignore_index=True)
            
    # recast dataframe
    symptom_metric_extreme_ratios=symptom_metric_extreme_ratios.astype({'category': pd.api.types.CategoricalDtype(), 'symptom':pd.api.types.CategoricalDtype()})
    # Save
    with open('{}/symptom_results.gz'.format(results_dir), 'wb') as f:
        symptom_metric_extreme_ratios.to_pickle(f, compression='gzip')

    ################# KS results, ordered ##############
    # Print out
    print('*********************KS **************************')
    print('Category & Symptom & KS (%95 CI) & p value')
    for index, row in symptom_metric_extreme_ratios.sort_values(by='ks', ascending=False).iterrows():
        print('{} \t & {} & {:.3f} ({:.3f},{:.3f}) & {:.6f} \\\\ '.format(
            row['category'], row['symptom'],
            row['ks'], row['ks_0025'],row['ks_0975'],
            row['p_val'],
            ))
    print('********************************************************************')

    # Save as csv
    with open('{}/results_ks'.format(results_dir), 'w') as f:
        symptom_metric_extreme_ratios.sort_values(by='ks', ascending=False)[[
            'category', 'symptom',
            'ks', 'ks_mean', 'ks_0025', 'ks_0975', 'p_val' 
            ]].to_csv(f, sep='&', index=False, float_format='%.3f')

    ################# Extremes analysis ##############
    ### Low extreme
    # Print out
    print('********************* Low extreme ratios **************************')
    print('Category & Symptom & Highly Variable Likelihood (%95 CI) & Not Highly Variable Likelihood (%95 CI) & Odds ratio (%95 CI)')
    for index, row in symptom_metric_extreme_ratios.sort_values(by=['low_extreme_ratios'], ascending=False).iterrows():
        print('{} \t & {} & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f})  \\\\ '.format(
            row['category'], row['symptom'],
            row['low_extreme_prob_high_var'], row['low_extreme_prob_high_var_0025'],row['low_extreme_prob_high_var_0975'],
            row['low_extreme_prob_low_var'], row['low_extreme_prob_low_var_0025'],row['low_extreme_prob_low_var_0975'],
            row['low_extreme_ratios'], row['low_extreme_ratios_0025'],row['low_extreme_ratios_0975'],
            ))
    print('********************************************************************')

    # Save as csv
    with open('{}/results_low_extreme'.format(results_dir), 'w') as f:
        symptom_metric_extreme_ratios.sort_values(by=['low_extreme_ratios'], ascending=False)[[
            'category', 'symptom',
            'low_extreme_prob_high_var', 'low_extreme_prob_high_var_mean', 'low_extreme_prob_high_var_0025', 'low_extreme_prob_high_var_0975',
            'low_extreme_prob_low_var', 'low_extreme_prob_low_var_mean', 'low_extreme_prob_low_var_0025', 'low_extreme_prob_low_var_0975',
            'low_extreme_ratios', 'low_extreme_ratios_mean', 'low_extreme_ratios_0025', 'low_extreme_ratios_0975',
            ]].to_csv(f, sep='&', index=False, float_format='%.3f')

    ### High extreme
    # Print out
    print('********************* High extreme ratios **************************')
    print('Category & Symptom & Highly Variable Likelihood (%95 CI) & Not Highly Variable Likelihood (%95 CI) & Odds ratio (%95 CI)')
    for index, row in symptom_metric_extreme_ratios.sort_values(by=['high_extreme_ratios'], ascending=False).iterrows():
        print('{} \t & {} & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f})  \\\\ '.format(
            row['category'], row['symptom'],
            row['high_extreme_prob_high_var'], row['high_extreme_prob_high_var_0025'],row['high_extreme_prob_high_var_0975'],
            row['high_extreme_prob_low_var'], row['high_extreme_prob_low_var_0025'],row['high_extreme_prob_low_var_0975'],
            row['high_extreme_ratios'], row['high_extreme_ratios_0025'],row['high_extreme_ratios_0975'],
            ))
    print('********************************************************************')

    # Save as csv
    with open('{}/results_high_extreme'.format(results_dir), 'w') as f:
        symptom_metric_extreme_ratios.sort_values(by=['high_extreme_ratios'], ascending=False)[[
            'category', 'symptom',
            'high_extreme_prob_high_var', 'high_extreme_prob_high_var_mean', 'high_extreme_prob_high_var_0025', 'high_extreme_prob_high_var_0975',
            'high_extreme_prob_low_var', 'high_extreme_prob_low_var_mean', 'high_extreme_prob_low_var_0025', 'high_extreme_prob_low_var_0975',
            'high_extreme_ratios', 'high_extreme_ratios_mean', 'high_extreme_ratios_0025', 'high_extreme_ratios_0975'
            ]].to_csv(f, sep='&', index=False, float_format='%.3f')

    # Only interesting categories
    interesting_categories=['period', 'pain', 'emotion']
    interesting_symptom_metric_extreme_ratios=symptom_metric_extreme_ratios[symptom_metric_extreme_ratios.category.isin(interesting_categories)].reset_index(drop=True)
    # KS, ordered
    print('*********************KS **************************')
    print('Category & Symptom & KS (%95 CI) & p value')
    for index, row in interesting_symptom_metric_extreme_ratios.sort_values(by='ks', ascending=False).iterrows():
        print('{} \t & {} & {:.3f} ({:.3f},{:.3f}) & {:.6f} \\ '.format(
            row['category'], row['symptom'],
            row['ks'], row['ks_0025'],row['ks_0975'],
            row['p_val'],
            ))
    print('********************************************************************')

    # Save as csv     
    with open('{}/results_ks_interesting'.format(results_dir), 'w') as f:
        interesting_symptom_metric_extreme_ratios.sort_values(by='ks', ascending=False)[[
            'category', 'symptom',
            'ks', 'ks_mean', 'ks_0025', 'ks_0975', 'p_val'
            ]].to_csv(f, sep='&', index=False, float_format='%.3f')

    ################# Extremes analysis ##############
    ### Low extreme
    # Print out
    print('********************* Low extreme ratios **************************')
    print('Category & Symptom & Highly Variable Likelihood (%95 CI) & Not Highly Variable Likelihood (%95 CI) & Odds ratio (%95 CI)')
    for index, row in interesting_symptom_metric_extreme_ratios.sort_values(by=['low_extreme_ratios'], ascending=False).iterrows():
        print('{} \t & {} & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f})  \\\\ '.format(
            row['category'], row['symptom'],
            row['low_extreme_prob_high_var'], row['low_extreme_prob_high_var_0025'],row['low_extreme_prob_high_var_0975'],
            row['low_extreme_prob_low_var'], row['low_extreme_prob_low_var_0025'],row['low_extreme_prob_low_var_0975'],
            row['low_extreme_ratios'], row['low_extreme_ratios_0025'],row['low_extreme_ratios_0975'],
            ))
    print('********************************************************************')

    # Save as csv
    with open('{}/results_low_extreme_interesting'.format(results_dir), 'w') as f:
        interesting_symptom_metric_extreme_ratios.sort_values(by=['low_extreme_ratios'], ascending=False)[[
            'category', 'symptom',
            'low_extreme_prob_high_var', 'low_extreme_prob_high_var_mean', 'low_extreme_prob_high_var_0025', 'low_extreme_prob_high_var_0975',
            'low_extreme_prob_low_var', 'low_extreme_prob_low_var_mean', 'low_extreme_prob_low_var_0025', 'low_extreme_prob_low_var_0975',
            'low_extreme_ratios', 'low_extreme_ratios_mean', 'low_extreme_ratios_0025', 'low_extreme_ratios_0975',
            ]].to_csv(f, sep='&', index=False, float_format='%.3f')

    ### High extreme
    # Print out
    print('********************* High extreme ratios **************************')
    print('Category & Symptom & Highly Variable Likelihood (%95 CI) & Not Highly Variable Likelihood (%95 CI) & Odds ratio (%95 CI)')
    for index, row in interesting_symptom_metric_extreme_ratios.sort_values(by=['high_extreme_ratios'], ascending=False).iterrows():
        print('{} \t & {} & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f}) & {:.3f} ({:.3f},{:.3f})  \\\\ '.format(
            row['category'], row['symptom'],
            row['high_extreme_prob_high_var'], row['high_extreme_prob_high_var_0025'],row['high_extreme_prob_high_var_0975'],
            row['high_extreme_prob_low_var'], row['high_extreme_prob_low_var_0025'],row['high_extreme_prob_low_var_0975'],
            row['high_extreme_ratios'], row['high_extreme_ratios_0025'],row['high_extreme_ratios_0975'],
            ))
    print('********************************************************************')

    # Save as csv
    with open('{}/results_high_extreme_interesting'.format(results_dir), 'w') as f:
        interesting_symptom_metric_extreme_ratios.sort_values(by=['high_extreme_ratios'], ascending=False)[[
            'category', 'symptom',
            'high_extreme_prob_high_var', 'high_extreme_prob_high_var_mean', 'high_extreme_prob_high_var_0025', 'high_extreme_prob_high_var_0975',
            'high_extreme_prob_low_var', 'high_extreme_prob_low_var_mean', 'high_extreme_prob_low_var_0025', 'high_extreme_prob_low_var_0975',
            'high_extreme_ratios', 'high_extreme_ratios_mean', 'high_extreme_ratios_0025', 'high_extreme_ratios_0975'
            ]].to_csv(f, sep='&', index=False, float_format='%.3f')

    ##################################################

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run symptom tracking analysis with bootstrapping')
    parser.add_argument('-n_bootstrapping', type=int, default=1000, help='Number of bootstrapped samples to use')
    
    # Get arguments
    args = parser.parse_args()

    # Just run the main
    main(args.n_bootstrapping)
