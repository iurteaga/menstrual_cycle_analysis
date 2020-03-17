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
from mpl_toolkits.mplot3d import Axes3D

################################## FUNCTIONS ############################
# Population time-series
def population_time_series_embedding_lengths(cycle_stats_df, attribute, cutoff_criteria, cutoff, sample_style, save_dir):
    '''
        Function that plots a population level time series embedding of cycle and period lengths
            In plot:
                x axis is length_attribute for cycle 1,
                y axis is length attribute for cycle 2,
                z is for cycle 3
        Input:
            cycle_stats_df: pandas dataframe, with information about user's cycle statistics
            attribute: whether to consider 'cycle_lengths' or 'period_lengths'
            cutoff_criteria: what statistic to use for separating users into groups ('cycle_lengths' for paper)
            cutoff: what statistic cutoff value to use for separating users into groups (9 for paper)
            sample_style: whether to pick 3 consecutive 'random' or 'first' cycles per-user
            save_dir: path where to save plot
        Output:
            None
    '''
    #get users with color by attribute > cutoff, and <= cutoff
    cycle_stats_df_greater_than = cycle_stats_df[cycle_stats_df[cutoff_criteria] > cutoff]
    cycle_stats_df_less_than = cycle_stats_df[cycle_stats_df[cutoff_criteria] <= cutoff]
    cycle_lengths_greater_than = cycle_stats_df_greater_than[attribute]
    cycle_lengths_less_than = cycle_stats_df_less_than[attribute]
    
    # Filename
    if sample_style == 'first':
        filename = '{}/population_time_series_embedding_for_{}_split_by_{}_{}_first_3.pdf'.format(save_dir, attribute, cutoff_criteria, cutoff)
    if sample_style == 'random':
        filename = '{}/population_time_series_embedding_for_{}_split_by_{}_{}_sample_3.pdf'.format(save_dir, attribute, cutoff_criteria, cutoff)
    
    # Plot
    colors = ['orange', 'c']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for index, cycle_lengths in enumerate([cycle_lengths_greater_than, cycle_lengths_less_than]):
        print('Start selecting cycles for one group')
        if sample_style=='first':
            sample_cycle_lengths = [cycle_length[:3] for cycle_length in cycle_lengths if len(cycle_length) >= 3]
        if sample_style=='random':
            sample_cycle_lengths = []
            for cycle_length in cycle_lengths:
                if len(cycle_length) >= 3:
                    num_cycles_array = np.linspace(0, len(cycle_length)-3, len(cycle_length)-2)
                    start_index = np.random.choice(num_cycles_array, size=1).astype(int)[0]
                    sample_cycle_lengths.append(cycle_length[start_index:start_index+3])
        print('Finished selecting cycles for one group')
        
        print('Start plotting one group')
        for i in range(len(sample_cycle_lengths)):
            xs = sample_cycle_lengths[i][0]
            ys = sample_cycle_lengths[i][1]
            zs = sample_cycle_lengths[i][2]
            # Plot this point
            ax.scatter(xs, ys, zs, color = colors[index], s=1, alpha=0.3)
        print('Finished plotting one group')

    ax.set_xlabel(attribute+ '[i]')
    ax.set_ylabel(attribute+ '[i+1]')
    ax.set_zlabel(attribute+ '[i+2]')
    if attribute == 'cycle_lengths':
        #ref_line_points = np.linspace(10, 90, 10)
        #ax.plot(ref_line_points, ref_line_points, ref_line_points, color='red', linestyle='dashed', linewidth=4, markersize=4)#, alpha=0.8)
        ax.set_xlim3d(10,90)
        ax.set_ylim3d(10,90)
        ax.set_zlim3d(10,90)
    elif attribute == 'period_lengths':
        max_period_days=28
        #ref_line_points = np.linspace(1, max_period_days, 4)
        #ax.plot(ref_line_points, ref_line_points, ref_line_points, color='red', linestyle='dashed', linewidth=4, markersize=4)#, alpha=0.8)
        ax.set_xlim3d(1,max_period_days)
        ax.set_ylim3d(1,max_period_days)
        ax.set_zlim3d(1,max_period_days)
        ax.set_xticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        ax.set_yticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        ax.set_zticks(np.append([1],np.arange(4,max_period_days+1, 4)))
    
    plt.savefig(filename.format(save_dir), format='pdf', bbox_inches='tight')
    print('Finished one view')
    # With angles
    for angle in [30, 60, 90, 180]:
        print('Start one view')
        filename_angle = filename[:-4]+'_'+str(angle)+'.pdf'
        ax.view_init(elev=None, azim=angle)
        # Add (a)/(b) labels for paper
        ax.text2D(12, 7,'(a)', fontsize=14, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=None)
        plt.savefig(filename_angle.format(save_dir), format='pdf', bbox_inches='tight')
        print('Finished one view')

    plt.close()

# Time series embedding for a randomly chosen user
def random_time_series_embedding_lengths(cycle_stats_df, attribute, cutoff_criteria, cutoff, save_dir):
    '''
        Function that plots a time series embedding of cycle and period lengths for a randomly chosen user per group
            In plot:
                x axis is length_attribute for cycle i,
                y axis is length attribute for cycle i+1,
                z is for cycle i+2
        Input:
            cycle_stats_df: pandas dataframe, with information about user's cycle statistics
            attribute: whether to consider 'cycle_lengths' or 'period_lengths'
            cutoff_criteria: what statistic to use for separating users into groups ('cycle_lengths' for paper)
            cutoff: what statistic cutoff value to use for separating users into groups (9 for paper)
            save_dir: path where to save plot
        Output:
            None
    '''
    # Select users with median number of cycles tracked
    cycle_stats_df_median = cycle_stats_df[cycle_stats_df['num_cycles_tracked'] == 11]
    filename = '{}/random_time_series_embedding_for_{}_split_by_{}_{}.pdf'.format(save_dir, attribute, cutoff_criteria, cutoff)

    #get users with color by attribute > cutoff, and <= cutoff
    cycle_stats_df_greater_than = cycle_stats_df_median[cycle_stats_df_median[cutoff_criteria] > cutoff]
    cycle_stats_df_less_than = cycle_stats_df_median[cycle_stats_df_median[cutoff_criteria] <= cutoff]
    cycle_lengths_greater_than = cycle_stats_df_greater_than[attribute]
    cycle_lengths_less_than = cycle_stats_df_less_than[attribute]
    
    # Randomly pick a user from each group
    cycle_lengths_greater_than_user = np.random.choice(cycle_lengths_greater_than, size=1, replace=False)
    cycle_lengths_less_than_user = np.random.choice(cycle_lengths_less_than, size=1, replace=False)

    # Plot
    colors = ['orange', 'c']
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #plot each user, color by median intercycle length
    xs = list(cycle_lengths_greater_than_user[0][0:-2])
    ys = list(cycle_lengths_greater_than_user[0][1:-1])
    zs = list(cycle_lengths_greater_than_user[0][2:])
    ax.scatter(xs, ys, zs, color = 'orange')
    ax.plot(xs, ys, zs, color='orange', linestyle='dashed', alpha=0.8)

    xs = list(cycle_lengths_less_than_user[0][0:-2])
    ys = list(cycle_lengths_less_than_user[0][1:-1])
    zs = list(cycle_lengths_less_than_user[0][2:])
    ax.scatter(xs, ys, zs, color = 'c')
    ax.plot(xs, ys, zs, color='c', linestyle='dashed', alpha=0.8)
    
    ax.set_xlabel(attribute+ '[i]')
    ax.set_ylabel(attribute+ '[i+1]')
    ax.set_zlabel(attribute+ '[i+2]')
    if attribute == 'cycle_lengths':
        #ref_line_points = np.linspace(10, 90, 10)
        #ax.plot(ref_line_points, ref_line_points, ref_line_points, color='red', linestyle='dashed', linewidth=4, markersize=4)#, alpha=0.8)
        ax.set_xlim3d(10,90)
        ax.set_ylim3d(10,90)
        ax.set_zlim3d(10,90)
    elif attribute == 'period_lengths':
        max_period_days=28
        #ref_line_points = np.linspace(1, max_period_days, 4)
        #ax.plot(ref_line_points, ref_line_points, ref_line_points, color='red', linestyle='dashed', linewidth=4, markersize=4)#, alpha=0.8)
        ax.set_xlim3d(1,max_period_days)
        ax.set_ylim3d(1,max_period_days)
        ax.set_zlim3d(1,max_period_days)
        ax.set_xticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        ax.set_yticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        ax.set_zticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        
    plt.savefig(filename.format(save_dir), format='pdf', bbox_inches='tight')
    print('Finished one view')
    # With angles
    for angle in [30, 60, 90, 180]:
        print('Start one view')
        filename_angle = filename[:-4]+'_'+str(angle)+'.pdf'
        ax.view_init(elev=None, azim=angle)
        plt.savefig(filename_angle.format(save_dir), format='pdf', bbox_inches='tight')
        print('Finished one view')

    plt.close()

# Plot period and cycle length distributions per group
def plot_lengths_hist_by_attribute_cutoff(cycle_stats_df, cycle_df, attribute, cutoff_criteria, cutoff, pdf_or_cdf, save_dir):
    '''
        Function that plots cycle and period length distributions across groups
        Input:
            cycle_stats_df: pandas dataframe, with information about user's cycle statistics
            cycle_df: pandas dataframe, with information about each user's cycle
            attribute: whether to consider 'cycle_lengths' or 'period_lengths'
            cutoff_criteria: what statistic to use for separating users into groups ('cycle_lengths' for paper)
            cutoff: what statistic cutoff value to use for separating users into groups (9 for paper)
            pdf_or_cdf: whether to plot 'pdf's or 'cdf's
            save_dir: path where to save plot
        Output:
            None
    '''
    # Identify groups per cutoff criteria
    users_greater_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] > cutoff]['user_id'])
    users_less_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] <= cutoff]['user_id'])
    cycles_users_greater_than_cutoff = cycle_df[cycle_df['user_id'].isin(users_greater_than_cutoff)]
    cycles_users_less_than_cutoff = cycle_df[cycle_df['user_id'].isin(users_less_than_cutoff)]

    colors = ['orange', 'c']
    labels=['Highly variable', 'NOT highly variable']

    if attribute == 'cycle_length':
        # Compute histogram
        # Bins based on integer range of values
        my_bins=np.arange(
            np.min([cycles_users_greater_than_cutoff[attribute].dropna().min(), cycles_users_less_than_cutoff[attribute].dropna().min()]),
            np.max([cycles_users_greater_than_cutoff[attribute].dropna().max(), cycles_users_less_than_cutoff[attribute].dropna().max()])+1)
        all_counts, all_bins = np.histogram(cycle_df[attribute].dropna(), bins=my_bins, density=True)
        counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(cycles_users_greater_than_cutoff[attribute].dropna(), bins=my_bins, density=True)
        counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(cycles_users_less_than_cutoff[attribute].dropna(), bins=my_bins, density=True)

        # Separate PDF/CDF plots
        if pdf_or_cdf=='pdf':
            # PDF
            hist_type='stepfilled'
            cumulative=False
            y_label='P(Cycle length = n)'
            cohort_filename = '{}/{}_pdf_cohort.pdf'.format(save_dir, attribute)
            per_group_filename = '{}/{}_pdf_per_group.pdf'.format(save_dir, attribute)
        elif pdf_or_cdf=='cdf':
            # CDF
            hist_type='step'
            cumulative=True
            y_label='P(Cycle length $\leq$ n)'
            cohort_filename = '{}/{}_cdf_cohort.pdf'.format(save_dir, attribute)
            per_group_filename = '{}/{}_cdf_per_group.pdf'.format(save_dir, attribute)
        else:
            raise ValueError('Can only plot pdf or cdf, not {}'.format(pdf_or_cdf))
            
        # Population
        plt.hist(all_bins[:-1], all_bins, weights=all_counts, density=True, cumulative=cumulative, color='slateblue', alpha=0.5, histtype=hist_type)
        plt.autoscale(enable=True, tight=True)
        plt.xticks(np.arange(my_bins.min(), my_bins.max()+1, 10))
        plt.xlabel('Cycle length in days')
        plt.ylabel(y_label)
        plt.savefig(cohort_filename, format='pdf', bbox_inches='tight')
        plt.close()
        
        # Per-group
        plt.hist(bins_greater_than_cutoff[:-1], bins_greater_than_cutoff, weights=counts_greater_than_cutoff, density=True, cumulative=cumulative, color=colors[0], alpha=0.5, label=labels[0], histtype=hist_type)
        plt.hist(bins_less_than_cutoff[:-1], bins_less_than_cutoff, weights=counts_less_than_cutoff, density=True, cumulative=cumulative, color=colors[1], alpha=0.5, label=labels[1], histtype=hist_type)
        plt.autoscale(enable=True, tight=True)
        plt.xticks(np.arange(my_bins.min(), my_bins.max()+1, 10))
        plt.xlabel('Cycle length in days')
        plt.ylabel(y_label)
        # Add (a)/(b) labels for paper
        plt.text(12, 7, '(b)', fontsize=14, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=None)
        plt.savefig(per_group_filename, format='pdf', bbox_inches='tight')
        plt.close()
        
    elif attribute == 'period_length':
        # Compute histogram
        # Bins based on integer range of values
        my_bins=np.arange(
            np.min([cycles_users_greater_than_cutoff[attribute].dropna().min(), cycles_users_less_than_cutoff[attribute].dropna().min()]),
            np.max([cycles_users_greater_than_cutoff[attribute].dropna().max(), cycles_users_less_than_cutoff[attribute].dropna().max()])+1)
        all_counts, all_bins = np.histogram(cycle_df[attribute].dropna(), bins=my_bins, density=True)
        counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(cycles_users_greater_than_cutoff[attribute].dropna(), bins=my_bins, density=True)
        counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(cycles_users_less_than_cutoff[attribute].dropna(), bins=my_bins, density=True)
            
        # Separate PDF/CDF plots
        max_period_days=28
        if pdf_or_cdf=='pdf':
            # PDF
            hist_type='stepfilled'
            cumulative=False
            y_label='P(Period length = n)'
            cohort_filename = '{}/{}_pdf_cohort.pdf'.format(save_dir, attribute)
            per_group_filename = '{}/{}_pdf_per_group.pdf'.format(save_dir, attribute)
        elif pdf_or_cdf=='cdf':
            # CDF
            hist_type='step'
            cumulative=True
            y_label='P(Period length $\leq$ n)'
            cohort_filename = '{}/{}_cdf_cohort.pdf'.format(save_dir, attribute)
            per_group_filename = '{}/{}_cdf_per_group.pdf'.format(save_dir, attribute)
        else:
            raise ValueError('Can only plot pdf or cdf, not {}'.format(pdf_or_cdf))
        
        # Population
        plt.hist(all_bins[:-1], all_bins, weights=all_counts, density=True, cumulative=cumulative, color='slateblue', alpha=0.5, histtype=hist_type)
        plt.autoscale(enable=True, tight=True)
        plt.xticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        plt.xlim(1,max_period_days)
        plt.xlabel('Period length in days')
        plt.ylabel(y_label)
        plt.savefig(cohort_filename, format='pdf', bbox_inches='tight')
        plt.close()
        
        # Per-group
        plt.hist(bins_greater_than_cutoff[:-1], bins_greater_than_cutoff, weights=counts_greater_than_cutoff, density=True, cumulative=cumulative, color=colors[0], alpha=0.5, label=labels[0], histtype=hist_type)
        plt.hist(bins_less_than_cutoff[:-1], bins_less_than_cutoff, weights=counts_less_than_cutoff, density=True, cumulative=cumulative, color=colors[1], alpha=0.5, label=labels[1], histtype=hist_type)
        plt.autoscale(enable=True, tight=True)
        plt.xticks(np.append([1],np.arange(4,max_period_days+1, 4)))
        plt.xlim(1,max_period_days)
        plt.xlabel('Period length in days')
        plt.ylabel(y_label)
        # Add (a)/(b) labels for paper
        plt.text(12, 7, '(b)', fontsize=14, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=None)
        plt.savefig(per_group_filename, format='pdf', bbox_inches='tight')
        plt.close()
        
    else:
        raise ValueError('Unknown attribute {}'.format(attribute))

# Bootstrapped-KS for cycle and period length
def bootstrapped_cycle_period_lengths_KS(cycle_stats_df, cycle_df, cutoff_criteria, cutoff, n_bootstrapping, results_dir):
    '''
        Function that computes cycle and period length Kolmogorov-Smirnov tests between group distributions, based on bootstrapping
        Input:
            cycle_stats_df: pandas dataframe, with information about user's cycle statistics
            cycle_df: pandas dataframe, with information about user's cycle
            cutoff_criteria: what statistic to use for separating users into groups ('cycle_lengths' for paper)
            cutoff: what statistic cutoff value to use for separating users into groups (9 for paper)
            n_bootstrapping: Number of bootstrapped samples to use for the analysis
            save_dir: path where to save plot
        Output:
            None
    '''
    # True separation of users into groups
    true_users_greater_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] > cutoff]['user_id'])
    true_users_less_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] <= cutoff]['user_id'])
    n_users_greater_than_cutoff=true_users_greater_than_cutoff.size
    n_users_less_than_cutoff=true_users_less_than_cutoff.size
    
    ########### TRUE OBSERVERD STATISTICS ##########
    # Cycles per-group
    true_cycles_users_greater_than_cutoff = cycle_df[cycle_df['user_id'].isin(true_users_greater_than_cutoff)]
    true_cycles_users_less_than_cutoff = cycle_df[cycle_df['user_id'].isin(true_users_less_than_cutoff)]
    # KS cycle_length
    true_KS_cycle_length, true_p_val_cycle_length = stats.ks_2samp(true_cycles_users_greater_than_cutoff['cycle_length'].dropna(), true_cycles_users_less_than_cutoff['cycle_length'].dropna())
    # KS period_length
    true_KS_period_length, true_p_val_period_length = stats.ks_2samp(true_cycles_users_greater_than_cutoff['period_length'].dropna(), true_cycles_users_less_than_cutoff['period_length'].dropna())
    
    ########### BOOTSTRAP BASED STATISTICS ##########
    # Computed suff statistics
    bootstrapped_KS_cycle_length=np.zeros(n_bootstrapping)
    bootstrapped_p_val_cycle_length=np.zeros(n_bootstrapping)
    bootstrapped_KS_period_length=np.zeros(n_bootstrapping)
    bootstrapped_p_val_period_length=np.zeros(n_bootstrapping)

    for n_bootstrap in np.arange(n_bootstrapping):
        #print('Sample={}/{}'.format(n_bootstrap,n_bootstrapping))
        # Bootstrapped sample indicators
        bootstrapped_users_greater_than_cutoff=np.random.choice(true_users_greater_than_cutoff,n_bootstrapping)
        bootstrapped_users_less_than_cutoff=np.random.choice(true_users_less_than_cutoff,n_bootstrapping)
        # Cycles per-group
        bootstrapped_cycles_users_greater_than_cutoff = cycle_df[cycle_df['user_id'].isin(bootstrapped_users_greater_than_cutoff)]
        bootstrapped_cycles_users_less_than_cutoff = cycle_df[cycle_df['user_id'].isin(bootstrapped_users_less_than_cutoff)]
        # KS cycle_length
        bootstrapped_KS_cycle_length[n_bootstrap], bootstrapped_p_val_cycle_length[n_bootstrap] = stats.ks_2samp(bootstrapped_cycles_users_greater_than_cutoff['cycle_length'].dropna(), bootstrapped_cycles_users_less_than_cutoff['cycle_length'].dropna())
        # KS period_length
        bootstrapped_KS_period_length[n_bootstrap], bootstrapped_p_val_period_length[n_bootstrap] = stats.ks_2samp(bootstrapped_cycles_users_greater_than_cutoff['period_length'].dropna(), bootstrapped_cycles_users_less_than_cutoff['period_length'].dropna())

    # Print bootstrap results
    print('*************************************************************************')
    print('******** Cycle-length KS={} (p={}) ***********'.format(true_KS_cycle_length, true_p_val_cycle_length))
    print('******** Cycle-length Bootstrapped KS={}+/-{} (p={} (+/-{}))***********'.format(
            bootstrapped_KS_cycle_length.mean(), bootstrapped_KS_cycle_length.std(), bootstrapped_p_val_cycle_length.mean(), bootstrapped_p_val_cycle_length.std()
            ))
    print('******** Cycle-length Bootstrapped KS={}({},{}) p={} ({},{}))***********'.format(
            bootstrapped_KS_cycle_length.mean(), np.percentile(bootstrapped_KS_cycle_length, 2.5, axis=0), np.percentile(bootstrapped_KS_cycle_length, 97.5, axis=0),
            bootstrapped_p_val_cycle_length.mean(), np.percentile(bootstrapped_p_val_cycle_length, 2.5, axis=0), np.percentile(bootstrapped_p_val_cycle_length, 97.5, axis=0)
            ))
    print('*************************************************************************')
    print('******** Period-length KS={} (p={}) ***********'.format(true_KS_period_length, true_p_val_period_length))
    print('******** Period-length Bootstrapped KS={}+/-{} (p={} (+/-{}))***********'.format(
            bootstrapped_KS_period_length.mean(), bootstrapped_KS_period_length.std(), bootstrapped_p_val_period_length.mean(), bootstrapped_p_val_period_length.std()
            ))
    print('******** Period-length Bootstrapped KS={}({},{}) p={} ({},{}))***********'.format(
            bootstrapped_KS_period_length.mean(), np.percentile(bootstrapped_KS_period_length, 2.5, axis=0), np.percentile(bootstrapped_KS_period_length, 97.5, axis=0),
            bootstrapped_p_val_period_length.mean(), np.percentile(bootstrapped_p_val_period_length, 2.5, axis=0), np.percentile(bootstrapped_p_val_period_length, 97.5, axis=0)
            ))
    print('*************************************************************************')

# Average statistics over cycle-id
def plot_avg_lengths_by_attribute_cutoff(cycle_stats_df, cycle_df, attribute, cutoff_criteria, cutoff, save_dir):
    '''
        Function that plots cycle and period length average and standard deviation across user's timeline (i.e., by cycle-id) across groups
        Input:
            cycle_stats_df: pandas dataframe, with information about user's cycle statistics
            cycle_df: pandas dataframe, with information about each user's cycle
            attribute: whether to consider 'cycle_lengths' or 'period_lengths'
            cutoff_criteria: what statistic to use for separating users into groups ('cycle_lengths' for paper)
            cutoff: what statistic cutoff value to use for separating users into groups (9 for paper)
            save_dir: path where to save plot
        Output:
            None
    '''
    # Identify groups per cutoff criteria
    users_greater_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] > cutoff]['user_id'])
    users_less_than_cutoff = np.unique(cycle_stats_df[cycle_stats_df[cutoff_criteria] <= cutoff]['user_id'])
    cycles_users_greater_than_cutoff = cycle_df[cycle_df['user_id'].isin(users_greater_than_cutoff)]
    cycles_users_less_than_cutoff = cycle_df[cycle_df['user_id'].isin(users_less_than_cutoff)]
    
    # Plotting
    colors = ['slateblue', 'c', 'orange']
    max_cycle_id=20
    
    if attribute == 'cycle_length':
        fig, axes = plt.subplots(3, 1, sharex='all', sharey='all', figsize = (15,15))
        
        for index, dataset in enumerate([cycle_df, cycles_users_less_than_cutoff, cycles_users_greater_than_cutoff]):
            means = dataset.groupby(['cycle_id'])[attribute].mean()[:max_cycle_id]
            std = dataset.groupby(['cycle_id'])[attribute].std()[:max_cycle_id]
            # Plot
            axes[index].plot(np.unique(dataset['cycle_id'])[:20], means, color = colors[index])
            axes[index].autoscale(enable=True, tight=True, axis='x')
            axes[index].fill_between(np.unique(dataset['cycle_id'])[:max_cycle_id], means - std, means + std, alpha=0.4, color=colors[index])
            axes[index].set_xticks(np.append([1],np.arange(2,max_cycle_id+1,2)))
            axes[index].set_xlabel('Cycle ID')
            axes[index].set_ylabel('Cycle length')
            axes[index].set_ylim(20,55)
        
        # Add (a)/(b) labels for paper
        plt.text(12, 7, '(a)', fontsize=14, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=None)
        # Save and close
        plt.savefig('{}/avg_{}_per_cycle_id.pdf'.format(save_dir,attribute), format='pdf', bbox_inches='tight')
        plt.close()
            
    elif attribute == 'period_length':
        fig, axes = plt.subplots(3, 1, sharex='all', sharey='all', figsize = (15,15))
        
        for index, dataset in enumerate([cycle_df, cycles_users_less_than_cutoff, cycles_users_greater_than_cutoff]):
            means = dataset.groupby(['cycle_id'])[attribute].mean()[:max_cycle_id]
            std = dataset.groupby(['cycle_id'])[attribute].std()[:max_cycle_id]
            # Plot
            axes[index].plot(np.unique(dataset['cycle_id'])[:20], means, color = colors[index])
            axes[index].autoscale(enable=True, tight=True, axis='x')
            axes[index].fill_between(np.unique(dataset['cycle_id'])[:max_cycle_id], means - std, means + std, alpha=0.4, color=colors[index])
            axes[index].set_xticks(np.append([1],np.arange(2,max_cycle_id+1,2)))
            axes[index].set_xlabel('Cycle ID')
            axes[index].set_ylabel('Period length')
            axes[index].set_ylim(1,9)
        
        # Add (a)/(b) labels for paper
        plt.text(12, 7, '(b)', fontsize=14, fontweight='bold', horizontalalignment='center', verticalalignment='center', transform=None)
        # Save and close
        plt.savefig('{}/avg_{}_per_cycle_id.pdf'.format(save_dir,attribute), format='pdf', bbox_inches='tight')
        plt.close()

    else:
        raise ValueError('Unknown attribute {}'.format(attribute))

# Plot for max intercycle length (i.e., CLD) histogram   
def plot_max_intercycle_length_hists(cycle_stats, cycle_stats_exclude_flagged, save_dir):
    '''
        Function that plots max inter cycle length (max CLD) histograms with and without excluded cycles
        Input:
            cycle_stats: pandas dataframe, with information about user's cycle statistics
            cycle_stats_exclude_flagged: pandas dataframe for users after removing excluded flags, with information about user's cycle statistics 
            save_dir: path where to save plot
        Output:
            None
    '''
    my_bins=np.arange(min(cycle_stats['max_inter_cycle_length']), max(cycle_stats['max_inter_cycle_length']) + 1)
    plt.hist(cycle_stats['max_inter_cycle_length'], bins=my_bins, label='With behaviorally-tainted cycles', color='blue', histtype='step')
    plt.hist(cycle_stats_exclude_flagged['max_inter_cycle_length'], bins=my_bins, label='Excluding behaviorally-tainted cycles', color='red', histtype='step')
    plt.autoscale(enable=True, tight=True, axis='x')
    plt.ylim(0,38000)
    plt.xlabel('Maximum CLD in days')
    plt.ylabel('User count with maximum CLD')
    plt.savefig('{}/hist_max_inter_cycle_length_with_and_without_excluded_flags.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()

# Plot for median Vs max intercycle length (i.e., CLD) histogram   
def plot_median_vs_max_intercycle_length(cycle_stats, save_dir):
    '''
        Function that plots median Vs max inter cycle length (CLD) 2D scatter histogram
        Input:
            cycle_stats: pandas dataframe, with information about user's cycle statistics
            save_dir: path where to save plot
        Output:
            None
    '''
    plt.hist2d(cycle_stats['median_inter_cycle_length'], cycle_stats['max_inter_cycle_length'], bins=(75, 75), cmap='jet', norm=colors.LogNorm())
    plt.autoscale(enable=True, tight=True)
    range_vals_median = np.linspace(min(cycle_stats['median_inter_cycle_length']), max(cycle_stats['median_inter_cycle_length']), 100)
    plt.plot(range_vals_median, range_vals_median+10, label='Median CLD + 10', color='red')
    plt.xlabel('Median CLD')
    plt.ylabel('Maximum CLD')
    plt.xlim((0,75))
    plt.ylim((0, 75))
    plt.colorbar()
    plt.savefig('{}/median_vs_max_scatter_2d_hist.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()

# Plot for median intercycle length (i.e., CLD) histogram   
def plot_median_CLD_hist(cycle_stats, pdf_or_cdf, save_dir):
    '''
        Function that plots median CLD histograms 
        Input:
            cycle_stats: pandas dataframe, with information about user's cycle statistics
            pdf_or_cdf: whether to plot 'pdf's or 'cdf's
            save_dir: path where to save plot
        Output:
            None
    '''
    
    # Median CLD histogram
    my_bins=np.arange(cycle_stats['median_inter_cycle_length'].dropna().min(),cycle_stats['median_inter_cycle_length'].dropna().max()+1)
    all_counts, all_bins = np.histogram(cycle_stats['median_inter_cycle_length'].dropna(), bins=my_bins, density=True)    
    
    # Separate PDF/CDF plots
    if pdf_or_cdf=='pdf':
        # PDF
        hist_type='stepfilled'
        cumulative=False
        y_label='P(Median CLD = n)'
        cohort_filename = '{}/median_CLD_pdf_cohort.pdf'.format(save_dir)
    elif pdf_or_cdf=='cdf':
        # CDF
        hist_type='step'
        cumulative=True
        y_label='P(Median CLD $\leq$ n)'
        cohort_filename = '{}/median_CLD_cdf_cohort.pdf'.format(save_dir)
    else:
        raise ValueError('Can only plot pdf or cdf, not {}'.format(pdf_or_cdf))
        
    # Actual plot
    plt.hist(all_bins[:-1], all_bins, weights=all_counts, density=True, cumulative=cumulative, color='slateblue', alpha=0.5, histtype=hist_type)
    plt.autoscale(enable=True, tight=True)
    plt.xlabel('Median CLD in days')
    plt.ylabel('P(Median CLD $\leq$ n)')
    plt.grid(True)
    plt.savefig('{}/median_CLD_cdf.pdf'.format(save_dir), format='pdf', bbox_inches='tight')
    plt.close()

################################## MAIN ############################
def main():
    '''
        Main function of the script that runs the cycle and period length related analysis

        Input:
            None
        Output:
            None
    '''
    
    ### Directories
    data_dir='../data'
    preprocessed_data_dir='../preprocessed_data'
    results_dir = '../results/characterizing_cycle_and_symptoms/cycle_period_length_analysis'
    os.makedirs(results_dir, exist_ok = True)
    
    ################# SYMPTOMS TRACKED #################
    # Tracking
    with open('{}/tracking_enriched.pickle'.format(data_dir), 'rb') as f:
	    tracking = pickle.load(f)

    print('Tracking-data loaded')

    ################# CYCLES #################
    with open('{}/cohort_cycle_stats.pickle'.format(preprocessed_data_dir), 'rb') as f:
      cohort_cycle_stats = pickle.load(f)

    # Cycles flagged
    with open('{}/cohort_cycles_flagged.pickle'.format(preprocessed_data_dir), 'rb') as f:
        cohort_cycles_flagged = pickle.load(f)

    # Exclude cycles flagged as badly tracked 
    cohort_cycles = cohort_cycles_flagged[cohort_cycles_flagged['badly_tracked_cycle'] == 'f']
    
    # Cycles stats
    with open('{}/cohort_clean_cycle_stats.pickle'.format(preprocessed_data_dir), 'rb') as f:
        cohort_clean_cycle_stats = pickle.load(f)

    print('Cycles-data loaded')
    
    ################# PLOTTING #################
    #### PLOT histogram of max intercycle length, with and without excluding flagged cycles
    plot_max_intercycle_length_hists(cohort_cycle_stats, cohort_clean_cycle_stats, results_dir)
    #### PLOT Median Vs Max CLD 2D histogram 
    plot_median_vs_max_intercycle_length(cohort_clean_cycle_stats, results_dir)
    #### PLOT Median CLD histogram 
    plot_median_CLD_hist(cohort_clean_cycle_stats, 'cdf', results_dir)
    
    #### PLOT cycle and period length histograms: pdf
    plot_lengths_hist_by_attribute_cutoff(cohort_clean_cycle_stats, cohort_cycles, 'cycle_length', 'median_inter_cycle_length', 9, 'pdf', results_dir)
    plot_lengths_hist_by_attribute_cutoff(cohort_clean_cycle_stats, cohort_cycles, 'period_length', 'median_inter_cycle_length', 9, 'pdf', results_dir)
    
    #### Bootstrapped-KS cycle and period length
    bootstrapped_cycle_period_lengths_KS(cohort_clean_cycle_stats, cohort_cycles, 'median_inter_cycle_length', 9, 100000, results_dir)
    
    #### PLOT average cycle and average length over cycle-id
    plot_avg_lengths_by_attribute_cutoff(cohort_clean_cycle_stats, cohort_cycles, 'cycle_length', 'median_inter_cycle_length', 9, results_dir)
    plot_avg_lengths_by_attribute_cutoff(cohort_clean_cycle_stats, cohort_cycles, 'period_length', 'median_inter_cycle_length', 9, results_dir)

    #### PLOT random cycle length time-series
    random_time_series_embedding_lengths(cohort_clean_cycle_stats, 'cycle_lengths', 'median_inter_cycle_length', 9, results_dir)
    
    #### PLOT population level cycle and period length time-series
    population_time_series_embedding_lengths(cohort_clean_cycle_stats, 'cycle_lengths', 'median_inter_cycle_length', 9, 'random', results_dir)
    population_time_series_embedding_lengths(cohort_clean_cycle_stats, 'period_lengths', 'median_inter_cycle_length', 9, 'random', results_dir)
    
# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Just run the main
    main()
