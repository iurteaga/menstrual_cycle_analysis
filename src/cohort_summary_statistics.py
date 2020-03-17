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

################################## MAIN ############################
def main():
    '''
        Main function of the script that computes all the summary statistics of the cohort

        Input:
            None
        Output:
            None
    '''
    ### Directories
    data_dir='../data'
    preprocessed_data_dir='../preprocessed_data'
    results_dir = '../results/characterizing_cycle_and_symptoms/cohort_summary_statistics'
    os.makedirs(results_dir, exist_ok = True)

    ################# CYCLES #################
    # Cycles flagged
    with open('{}/cohort_cycles_flagged.pickle'.format(preprocessed_data_dir), 'rb') as f:
        cohort_cycles_flagged = pickle.load(f)

    # Exclude cycles flagged as badly tracked 
    cohort_cycles = cohort_cycles_flagged[cohort_cycles_flagged['badly_tracked_cycle'] == 'f']
    print('Cycles-data loaded')

    # CLDs
    cutoff=9
    all_clds=cohort_cycles.groupby(['user_id'])['cycle_length'].apply(lambda x: np.abs(np.diff(x)))
    all_median_clds=all_clds.apply(np.median)
    all_max_clds=all_clds.apply(np.max)
    users_greater_than_cutoff_clds=all_clds[all_clds.apply(np.median)>cutoff]
    users_greater_than_cutoff_median_clds=users_greater_than_cutoff_clds.apply(np.median)
    users_greater_than_cutoff_max_clds=users_greater_than_cutoff_clds.apply(np.max)
    users_less_than_cutoff_clds=all_clds[all_clds.apply(np.median)<=cutoff]
    users_less_than_cutoff_median_clds=users_less_than_cutoff_clds.apply(np.median)
    users_less_than_cutoff_max_clds=users_less_than_cutoff_clds.apply(np.max)

    # Number of users
    users_all=cohort_cycles_flagged['user_id'].unique()
    n_users_all=users_all.size
    users_greater_than_cutoff=users_greater_than_cutoff_clds.index.values
    n_users_greater_than_cutoff=users_greater_than_cutoff.size
    users_less_than_cutoff=users_less_than_cutoff_clds.index.values
    n_users_less_than_cutoff=users_less_than_cutoff.size

    # cycles tracked, based on cohort_cycles
    cycles_users_greater_than_cutoff = cohort_cycles[cohort_cycles['user_id'].isin(users_greater_than_cutoff)]
    cycles_users_less_than_cutoff = cohort_cycles[cohort_cycles['user_id'].isin(users_less_than_cutoff)]

    # Number of cycles tracked
    n_cycles_all=cohort_cycles.groupby(['user_id'])['cycle_id'].count()
    total_n_cycles_all=n_cycles_all.sum()
    n_cycles_greater_than_cutoff=cycles_users_greater_than_cutoff.groupby(['user_id'])['cycle_id'].count()
    total_n_cycles_greater_than_cutoff=n_cycles_greater_than_cutoff.sum()
    n_cycles_less_than_cutoff=cycles_users_less_than_cutoff.groupby(['user_id'])['cycle_id'].count()
    total_n_cycles_less_than_cutoff=n_cycles_less_than_cutoff.sum()

    ################# SYMPTOMS TRACKED #################
    # Tracking
    with open('{}/tracking_enriched.pickle'.format(data_dir), 'rb') as f:
	    tracking = pickle.load(f)

    print('Tracking-data loaded')
    # Remove sensity categories
    categories =  list(tracking.category.cat.categories)
    # HBC related
    categories.remove('injection_hbc')
    categories.remove('iud')
    categories.remove('patch_hbc')
    categories.remove('pill_hbc')
    categories.remove('ring_hbc')
    
    # Observations tracked by cohort of interest
    all_tracking=tracking[tracking['user_id'].isin(users_all)]
    # Observations tracked in categories of interest
    all_tracking=all_tracking[all_tracking.category.isin(categories)]
    # Remove tracking events from cycles that are excluded
    all_tracking_excluded=pd.merge(all_tracking, cohort_cycles, on=['user_id','cycle_id'])
    del tracking
    
    # Separate tracking per group
    tracking_users_greater_than_cutoff=all_tracking_excluded[all_tracking_excluded['user_id'].isin(users_greater_than_cutoff)]
    tracking_users_less_than_cutoff=all_tracking_excluded[all_tracking_excluded['user_id'].isin(users_less_than_cutoff)]

    # Number of observations tracked
    n_observations_all=all_tracking_excluded.shape[0]
    n_observations_greater_than_cutoff=tracking_users_greater_than_cutoff.shape[0]
    n_observations_less_than_cutoff=tracking_users_less_than_cutoff.shape[0]

    # Number of distinct days tracked
    n_observation_days_all=all_tracking_excluded.groupby(['user_id','date']).count().shape[0]
    n_observation_days_greater_than_cutoff=tracking_users_greater_than_cutoff.groupby(['user_id','date']).count().shape[0]
    n_observation_days_less_than_cutoff=tracking_users_less_than_cutoff.groupby(['user_id','date']).count().shape[0]

    # Total, %
    print('********************************************************************************************')
    print('\t All cohort \t Low variability \t High variability \t [58] \t [59]')
    print('# of users \t & {} (%{:.2f}) \t & {} (%{:.2f}) \t & {} (%{:.2f}) ***********'.format(
                n_users_all, n_users_all/n_users_all*100,
                n_users_less_than_cutoff, n_users_less_than_cutoff/n_users_all*100,
                n_users_greater_than_cutoff, n_users_greater_than_cutoff/n_users_all*100,
                ))
    print('# of observations \t & {} (%{:.2f}) \t & {} (%{:.2f}) \t & {} (%{:.2f}) ***********'.format(
                n_observations_all, n_observations_all/n_observations_all*100,
                n_observations_less_than_cutoff, n_observations_less_than_cutoff/n_observations_all*100,
                n_observations_greater_than_cutoff, n_observations_greater_than_cutoff/n_observations_all*100,
                ))
    print('# of days of observations \t & {} (%{:.2f}) \t & {} (%{:.2f}) \t & {} (%{:.2f}) ***********'.format(
                n_observation_days_all, n_observation_days_all/n_observation_days_all*100,
                n_observation_days_less_than_cutoff, n_observation_days_less_than_cutoff/n_observation_days_all*100,
                n_observation_days_greater_than_cutoff, n_observation_days_greater_than_cutoff/n_observation_days_all*100,
                ))
    print('# of cycles \t & {} (%{:.2f}) \t & {} (%{:.2f}) \t & {} (%{:.2f}) ***********'.format(
                total_n_cycles_all, total_n_cycles_all/total_n_cycles_all*100,
                total_n_cycles_less_than_cutoff, total_n_cycles_less_than_cutoff/total_n_cycles_all*100,
                total_n_cycles_greater_than_cutoff, total_n_cycles_greater_than_cutoff/total_n_cycles_all*100,            
                ))
    print('********************************************************************************************')
    
    ################# PRINT OUT  #################
    # Cycle statistics
    print('********************************** Cycle statistics ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('\t \t & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median &')
    print('# of cycles \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
        # Number of cycles tracked: ALL
        n_cycles_all.mean(), n_cycles_all.std() ,
        np.percentile(n_cycles_all,2.5), np.percentile(n_cycles_all,97.5),
        n_cycles_all.median(),
        # Number of cycles tracked: Low
        n_cycles_less_than_cutoff.mean(), n_cycles_less_than_cutoff.std() ,
        np.percentile(n_cycles_less_than_cutoff,2.5), np.percentile(n_cycles_less_than_cutoff,97.5),
        n_cycles_less_than_cutoff.median(),
        # Number of cycles tracked: High
        n_cycles_greater_than_cutoff.mean(), n_cycles_greater_than_cutoff.std() ,
        np.percentile(n_cycles_greater_than_cutoff,2.5), np.percentile(n_cycles_greater_than_cutoff,97.5),
        n_cycles_greater_than_cutoff.median(),
        ))
    print('Cycle-length \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
        # Cycle-length: ALL
        cohort_cycles['cycle_length'].mean(), cohort_cycles['cycle_length'].std() ,
        np.percentile(cohort_cycles['cycle_length'],2.5), np.percentile(cohort_cycles['cycle_length'],97.5),
        cohort_cycles['cycle_length'].median(),
        # Cycle-length: Low
        cycles_users_less_than_cutoff['cycle_length'].mean(), cycles_users_less_than_cutoff['cycle_length'].std() ,
        np.percentile(cycles_users_less_than_cutoff['cycle_length'],2.5), np.percentile(cycles_users_less_than_cutoff['cycle_length'],97.5),
        cycles_users_less_than_cutoff['cycle_length'].median(),
        # Cycle-length: High
        cycles_users_greater_than_cutoff['cycle_length'].mean(), cycles_users_greater_than_cutoff['cycle_length'].std() ,
        np.percentile(cycles_users_greater_than_cutoff['cycle_length'],2.5), np.percentile(cycles_users_greater_than_cutoff['cycle_length'],97.5),
        cycles_users_greater_than_cutoff['cycle_length'].median(),
        ))
    print('Period-length \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
        # Period-length: ALL
        cohort_cycles['period_length'].mean(), cohort_cycles['period_length'].std() ,
        np.percentile(cohort_cycles['period_length'],2.5), np.percentile(cohort_cycles['period_length'],97.5),
        cohort_cycles['period_length'].median(),
        # Period-length: Low
        cycles_users_less_than_cutoff['period_length'].mean(), cycles_users_less_than_cutoff['period_length'].std() ,
        np.percentile(cycles_users_less_than_cutoff['period_length'],2.5), np.percentile(cycles_users_less_than_cutoff['period_length'],97.5),
        cycles_users_less_than_cutoff['period_length'].median(),
        # Period-length: High
        cycles_users_greater_than_cutoff['period_length'].mean(), cycles_users_greater_than_cutoff['period_length'].std() ,
        np.percentile(cycles_users_greater_than_cutoff['period_length'],2.5), np.percentile(cycles_users_greater_than_cutoff['period_length'],97.5),
        cycles_users_greater_than_cutoff['period_length'].median(),
        ))
    print('Median-CLD \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
        # Median-CLD: ALL
        all_median_clds.mean(), all_median_clds.std() ,
        np.percentile(all_median_clds,2.5), np.percentile(all_median_clds,97.5),
        all_median_clds.median(),
        # Median-CLD: Low
        users_less_than_cutoff_median_clds.mean(), users_less_than_cutoff_median_clds.std() ,
        np.percentile(users_less_than_cutoff_median_clds,2.5), np.percentile(users_less_than_cutoff_median_clds,97.5),
        users_less_than_cutoff_median_clds.median(),
        # Median-CLD: High
        users_greater_than_cutoff_median_clds.mean(), users_greater_than_cutoff_median_clds.std() ,
        np.percentile(users_greater_than_cutoff_median_clds,2.5), np.percentile(users_greater_than_cutoff_median_clds,97.5),
        users_greater_than_cutoff_median_clds.median(),
        ))
    print('Max-CLD \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
        # Max-CLD: ALL
        all_max_clds.mean(), all_max_clds.std() ,
        np.percentile(all_max_clds,2.5), np.percentile(all_max_clds,97.5),
        all_max_clds.median(),
        # Max-CLD: Low
        users_less_than_cutoff_max_clds.mean(), users_less_than_cutoff_max_clds.std() ,
        np.percentile(users_less_than_cutoff_max_clds,2.5), np.percentile(users_less_than_cutoff_max_clds,97.5),
        users_less_than_cutoff_max_clds.median(),
        # Max-CLD: High
        users_greater_than_cutoff_max_clds.mean(), users_greater_than_cutoff_max_clds.std() ,
        np.percentile(users_greater_than_cutoff_max_clds,2.5), np.percentile(users_greater_than_cutoff_max_clds,97.5),
        users_greater_than_cutoff_max_clds.median(),
        ))

    print('********************************************************************************************')
    
    ################# Observations for all categories ##############
    # Per-category observations of interest
    print('********************************************************************************************')
    for category in categories:
        # Tracking events within this category per-group
        category_tracking_users_greater_than_cutoff=tracking_users_greater_than_cutoff[tracking_users_greater_than_cutoff['category']==category]
        category_tracking_users_less_than_cutoff=tracking_users_less_than_cutoff[tracking_users_less_than_cutoff['category']==category]
        print('{} & {} ({:.2f}%) & {} ({:.2f}%)'.format(
                                            category,
                                            category_tracking_users_less_than_cutoff.shape[0], category_tracking_users_less_than_cutoff.shape[0]/n_observations_less_than_cutoff*100,
                                            category_tracking_users_greater_than_cutoff.shape[0], category_tracking_users_greater_than_cutoff.shape[0]/n_observations_greater_than_cutoff*100,
                                            )
                        )
    print('********************************************************************************************')
    
    ################# Cycle-exclusion analysis #################
    # Number of users in each group with only 2 cycles
    print('Number of users with only two cycles in the consistently highly variable group = {:.2f}'.format((np.sum(n_cycles_greater_than_cutoff.values==2)/n_cycles_greater_than_cutoff.size)*100))
    print('Number of users with only two cycles in the consistently NOT highly variable group = {:.2f}'.format((np.sum(n_cycles_less_than_cutoff.values==2)/n_cycles_less_than_cutoff.size)*100))    
    
    # Analysis of excluded flagged cycles
    # Flagged as badly tracked
    cycles_flagged=cohort_cycles_flagged.badly_tracked_cycle=='t'
    # Cycles per user
    cycles_per_user=cohort_cycles_flagged.groupby('user_id')['cycle_id'].count()
    # Excluded cycles per user
    excluded_cycles_per_user=cohort_cycles_flagged[cohort_cycles_flagged['badly_tracked_cycle']=='t'].groupby('user_id')['cycle_id'].count()
    print('Number of excluded cycles {}'.format(excluded_cycles_per_user.sum()))
    print('Overall average number of cycles excluded {:.2f}'.format(excluded_cycles_per_user.sum()/cycles_per_user.sum()))
    print('Overall average number of cycles excluded for all users {:.2f}'.format(excluded_cycles_per_user.sum()/cycles_per_user.shape[0]))
    # Proportion of excluded per user
    prop_excluded_cycles_per_user=excluded_cycles_per_user/cycles_per_user
    print('Proportion of users with NO excluded cycle {:.2f}'.format(prop_excluded_cycles_per_user.isna().sum()/prop_excluded_cycles_per_user.shape[0]))
    print('Proportion of users with at least one excluded cycle {:.2f}'.format(prop_excluded_cycles_per_user.dropna().shape[0]/prop_excluded_cycles_per_user.shape[0]))
    print('Overall average number of cycles excluded for those users with removed cycles {:.2f}'.format(excluded_cycles_per_user.sum()/excluded_cycles_per_user.shape[0]))
    
    # Load cycle stats (before excluding flagged ones)
    with open('{}/cohort_cycle_stats.pickle'.format(preprocessed_data_dir), 'rb') as f:
        cohort_cycle_stats = pickle.load(f)
    
    # Add new statistics of interest, to be able to compute expected median period time-interval
    cohort_cycle_stats['median_period_length']=cohort_cycle_stats.period_lengths.apply(np.median)
    cohort_cycle_stats['median_cycle_length']=cohort_cycle_stats.cycle_lengths.apply(np.median)
    
    # Period Tracking events for those user and cycle_ids that have been flagged
    cycles_flagged_period_tracking=pd.merge(cohort_cycles_flagged[cycles_flagged][['user_id', 'cycle_id','cycle_start','cycle_length']], all_tracking[all_tracking.category=='period'], how='left', on=['user_id', 'cycle_id'])
    cycles_stats_flagged_period_tracking=pd.merge(cycles_flagged_period_tracking, cohort_cycle_stats, how='left', on=['user_id'])
    # Fix inconsistencies
    cycles_stats_flagged_period_tracking=cycles_stats_flagged_period_tracking[
    (cycles_stats_flagged_period_tracking.cycle_day.isna()) | (cycles_stats_flagged_period_tracking.cycle_day<=cycles_stats_flagged_period_tracking.cycle_length)
    ]
    # Add tracking indicator within median expected period
    cycles_stats_flagged_period_tracking['within_median_expected_period']=(
                cycles_stats_flagged_period_tracking.cycle_day>(cycles_stats_flagged_period_tracking.median_cycle_length-cycles_stats_flagged_period_tracking.median_period_length)
                ) & (
                cycles_stats_flagged_period_tracking.cycle_day>(cycles_stats_flagged_period_tracking.median_cycle_length+cycles_stats_flagged_period_tracking.median_period_length)
                )
                
    # Overall excluded cycle info
    excluded_cycle_info=cycles_stats_flagged_period_tracking.groupby(['user_id','cycle_id'])
    events_within_cycle=excluded_cycle_info.sum()    
    # Events within median expected period
    print('Total excluded cycles={}, with {} cycles that have tracking within median expected period: %{:.2f} of excluded cycles, %{:.2f} of total analyzed cycles'.format(
        events_within_cycle.shape[0],
        (events_within_cycle.within_median_expected_period>0).sum(),
        100*(events_within_cycle.within_median_expected_period>0).sum()/events_within_cycle.shape[0],
        100*(events_within_cycle.within_median_expected_period>0).sum()/total_n_cycles_all
        ))        
    
    ################# Statistics per-age #################
    # Users
    all_users_per_age=cohort_cycles.groupby('user_id')[['age_at_cycle']].min()
    users_greater_than_cutoff_per_age=cycles_users_greater_than_cutoff.groupby('user_id')[['age_at_cycle']].min()
    users_less_than_cutoff_per_age=cycles_users_less_than_cutoff.groupby('user_id')[['age_at_cycle']].min()
    # Print out
    print('********************************** Users age ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median')
    print('{:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
            # Number of users: ALL
            all_users_per_age['age_at_cycle'].mean(), all_users_per_age['age_at_cycle'].std(),
            np.nanpercentile(all_users_per_age['age_at_cycle'], 2.5), np.nanpercentile(all_users_per_age['age_at_cycle'], 97.5),
            all_users_per_age['age_at_cycle'].median(),
            # Number of users: Low
            users_less_than_cutoff_per_age['age_at_cycle'].mean(), users_less_than_cutoff_per_age['age_at_cycle'].std(),
            np.nanpercentile(users_less_than_cutoff_per_age['age_at_cycle'], 2.5), np.nanpercentile(users_less_than_cutoff_per_age['age_at_cycle'], 97.5),
            users_less_than_cutoff_per_age['age_at_cycle'].median(),
            # Number of users: High
            users_greater_than_cutoff_per_age['age_at_cycle'].mean(), users_greater_than_cutoff_per_age['age_at_cycle'].std(),
            np.nanpercentile(users_greater_than_cutoff_per_age['age_at_cycle'], 2.5), np.nanpercentile(users_greater_than_cutoff_per_age['age_at_cycle'], 97.5),
            users_greater_than_cutoff_per_age['age_at_cycle'].median(),
            ))

    # Age histograms
    my_bins=np.arange(all_users_per_age['age_at_cycle'].min(),all_users_per_age['age_at_cycle'].max()+1)
    all_counts, all_bins = np.histogram(all_users_per_age['age_at_cycle'], bins=my_bins, density=False)
    counts_greater_than_cutoff, bins_greater_than_cutoff = np.histogram(users_greater_than_cutoff_per_age['age_at_cycle'], bins=my_bins, density=False)
    counts_less_than_cutoff, bins_less_than_cutoff = np.histogram(users_less_than_cutoff_per_age['age_at_cycle'], bins=my_bins, density=False)

    # cycles tracked
    all_cycles_per_age=cohort_cycles.groupby(['age_at_cycle'])
    cycles_users_greater_than_cutoff_per_age=cycles_users_greater_than_cutoff.groupby(['age_at_cycle'])
    cycles_users_less_than_cutoff_per_age=cycles_users_less_than_cutoff.groupby(['age_at_cycle'])

    n_cycles_all_per_age=cohort_cycles.groupby(['age_at_cycle','user_id'])['cycle_id'].count().groupby('age_at_cycle')
    n_cycles_users_greater_than_cutoff_per_age=cycles_users_greater_than_cutoff.groupby(['age_at_cycle','user_id'])['cycle_id'].count().groupby('age_at_cycle')
    n_cycles_users_less_than_cutoff_per_age=cycles_users_less_than_cutoff.groupby(['age_at_cycle','user_id'])['cycle_id'].count().groupby('age_at_cycle')

    # Number of cycles
    assert np.all(all_cycles_per_age.count()['cycle_length'] == n_cycles_all_per_age.sum())

    # CLDs
    all_clds_per_age=cohort_cycles.groupby(['age_at_cycle','user_id'])['cycle_length'].apply(lambda x: np.abs(np.diff(x)))
    all_median_clds_per_age=all_clds_per_age.apply(lambda x: np.nanmedian(x) if len(x)>0 else np.nan).groupby('age_at_cycle')
    all_max_clds_per_age=all_clds_per_age.apply(lambda x: np.nanmax(x) if len(x)>0 else np.nan).groupby('age_at_cycle')
    users_greater_than_cutoff_clds_per_age=cycles_users_greater_than_cutoff.groupby(['age_at_cycle','user_id'])['cycle_length'].apply(lambda x: np.abs(np.diff(x)))
    users_greater_than_cutoff_median_clds_per_age=users_greater_than_cutoff_clds_per_age.apply(lambda x: np.nanmedian(x) if len(x)>0 else np.nan).groupby('age_at_cycle')
    users_greater_than_cutoff_max_clds_per_age=users_greater_than_cutoff_clds_per_age.apply(lambda x: np.nanmax(x) if len(x)>0 else np.nan).groupby('age_at_cycle')
    users_less_than_cutoff_clds_per_age=cycles_users_less_than_cutoff.groupby(['age_at_cycle','user_id'])['cycle_length'].apply(lambda x: np.abs(np.diff(x)))
    users_less_than_cutoff_median_clds_per_age=users_less_than_cutoff_clds_per_age.apply(lambda x: np.nanmedian(x) if len(x)>0 else np.nan).groupby('age_at_cycle')
    users_less_than_cutoff_max_clds_per_age=users_less_than_cutoff_clds_per_age.apply(lambda x: np.nanmax(x) if len(x)>0 else np.nan).groupby('age_at_cycle')

    ################# PRINT OUT #################
    # Cycle statistics
    print('********************************** Number of users/cycles ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Age & User Number & Cycle Number & User Number & Cycle Number & User Number & Cycle Number ')
    for age in n_cycles_users_less_than_cutoff_per_age.groups.keys():
        print('{:d} \t & {} & {} & {} & {} & {} & {} & \t & \t & \hline  ***********'.format(
            int(age),
            # Number of users: ALL
            (all_users_per_age['age_at_cycle']==age).sum(),
            # Number of cycles tracked: ALL
            n_cycles_all_per_age.sum()[age],
            # Number of users: Low
            (users_less_than_cutoff_per_age['age_at_cycle']==age).sum(),
            # Number of cycles tracked: Low
            n_cycles_users_less_than_cutoff_per_age.sum()[age],
            # Number of users: High
            (users_greater_than_cutoff_per_age['age_at_cycle']==age).sum(),
            # Number of cycles tracked: High
            n_cycles_users_greater_than_cutoff_per_age.sum()[age],
            ))
    print('********************************** Number of cycles ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Age & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median')
    for age in n_cycles_users_less_than_cutoff_per_age.groups.keys():
        print('{:d} \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
            int(age),
            # Number of cycles tracked: ALL
            n_cycles_all_per_age.mean()[age], n_cycles_all_per_age.std()[age],
            n_cycles_all_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], n_cycles_all_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            n_cycles_all_per_age.median()[age],
            # Number of cycles tracked: Low
            n_cycles_users_less_than_cutoff_per_age.mean()[age], n_cycles_users_less_than_cutoff_per_age.std()[age],
            n_cycles_users_less_than_cutoff_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], n_cycles_users_less_than_cutoff_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            n_cycles_users_less_than_cutoff_per_age.median()[age],
            # Number of cycles tracked: High
            n_cycles_users_greater_than_cutoff_per_age.mean()[age], n_cycles_users_greater_than_cutoff_per_age.std()[age],
            n_cycles_users_greater_than_cutoff_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], n_cycles_users_greater_than_cutoff_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            n_cycles_users_greater_than_cutoff_per_age.median()[age],
            ))
    print('********************************** Cycle-length ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Age & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median')
    for age in n_cycles_users_less_than_cutoff_per_age.groups.keys():
        print('{:d} \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
            int(age),
            # Cycle-length: ALL
            all_cycles_per_age['cycle_length'].mean()[age], all_cycles_per_age['cycle_length'].std()[age],
            all_cycles_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 2.5))[age], all_cycles_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 97.5))[age],
            all_cycles_per_age['cycle_length'].median()[age],
            # Cycle-length: Low
            cycles_users_less_than_cutoff_per_age['cycle_length'].mean()[age], cycles_users_less_than_cutoff_per_age['cycle_length'].std()[age],
            cycles_users_less_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 2.5))[age], cycles_users_less_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 97.5))[age],
            cycles_users_less_than_cutoff_per_age['cycle_length'].median()[age],
            # Cycle-length: High
            cycles_users_greater_than_cutoff_per_age['cycle_length'].mean()[age], cycles_users_greater_than_cutoff_per_age['cycle_length'].std()[age],
            cycles_users_greater_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 2.5))[age], cycles_users_greater_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 97.5))[age],
            cycles_users_greater_than_cutoff_per_age['cycle_length'].median()[age],
            ))
    print('********************************** Period-length ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Age & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median')
    for age in n_cycles_users_less_than_cutoff_per_age.groups.keys():
        print('{:d} \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
            int(age),
            # Period-length: ALL
            all_cycles_per_age['period_length'].mean()[age], all_cycles_per_age['period_length'].std()[age],
            all_cycles_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 2.5))[age], all_cycles_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 97.5))[age],
            all_cycles_per_age['period_length'].median()[age],
            # Period-length: Low
            cycles_users_less_than_cutoff_per_age['period_length'].mean()[age], cycles_users_less_than_cutoff_per_age['period_length'].std()[age],
            cycles_users_less_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 2.5))[age], cycles_users_less_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 97.5))[age],
            cycles_users_less_than_cutoff_per_age['period_length'].median()[age],
            # Period-length: High
            cycles_users_greater_than_cutoff_per_age['period_length'].mean()[age], cycles_users_greater_than_cutoff_per_age['period_length'].std()[age],
            cycles_users_greater_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 2.5))[age], cycles_users_greater_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 97.5))[age],
            cycles_users_greater_than_cutoff_per_age['period_length'].median()[age],
            ))
    print('********************************** Median-CLD ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Age & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median')
    for age in n_cycles_users_less_than_cutoff_per_age.groups.keys():
        print('{:d} \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
            int(age),
            # Median-CLD: ALL
            all_median_clds_per_age.mean()[age], all_median_clds_per_age.std()[age],
            all_median_clds_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], all_median_clds_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            all_median_clds_per_age.median()[age],
            # Median-CLD: Low
            users_less_than_cutoff_median_clds_per_age.mean()[age], users_less_than_cutoff_median_clds_per_age.std()[age],
            users_less_than_cutoff_median_clds_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], users_less_than_cutoff_median_clds_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            users_less_than_cutoff_median_clds_per_age.median()[age],
            # Median-CLD: High
            users_greater_than_cutoff_median_clds_per_age.mean()[age], users_greater_than_cutoff_median_clds_per_age.std()[age],
            users_greater_than_cutoff_median_clds_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], users_greater_than_cutoff_median_clds_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            users_greater_than_cutoff_median_clds_per_age.median()[age],
            ))
    print('********************************** Maximum-CLD ****************************************')
    print('\t \t \t All cohort \t \t \t Low variability  \t \t \t High variability \t \t \t [58] \t \t \t [59]')
    print('Age & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median & Mean \pm std () & Median')
    for age in n_cycles_users_less_than_cutoff_per_age.groups.keys():
        print('{:d} \t & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & {:.2f} $\pm$ {:.2f} ({:.2f},{:.2f}) & {:.2f} & \t & \t & \hline  ***********'.format(
            int(age),
            # Max-CLD: ALL
            all_max_clds_per_age.mean()[age], all_max_clds_per_age.std()[age],
            all_max_clds_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], all_max_clds_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            all_max_clds_per_age.median()[age],
            # Max-CLD: Low
            users_less_than_cutoff_max_clds_per_age.mean()[age], users_less_than_cutoff_max_clds_per_age.std()[age],
            users_less_than_cutoff_max_clds_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], users_less_than_cutoff_max_clds_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            users_less_than_cutoff_max_clds_per_age.median()[age],
            # Max-CLD: High
            users_greater_than_cutoff_max_clds_per_age.mean()[age], users_greater_than_cutoff_max_clds_per_age.std()[age],
            users_greater_than_cutoff_max_clds_per_age.apply(lambda x: np.nanpercentile(x, 2.5))[age], users_greater_than_cutoff_max_clds_per_age.apply(lambda x: np.nanpercentile(x, 97.5))[age],
            users_greater_than_cutoff_max_clds_per_age.median()[age],
            ))

    ################# PLOTTING #################
    print('Plotting')
    # Plots
    colors = ['slateblue', 'c', 'orange']
    labels=['Full cohort', 'NOT highly variable', 'Highly variable']
    age_range=np.fromiter(all_cycles_per_age.groups.keys(), dtype=int)
    
    # Cycle-length per age
    fig, axes = plt.subplots(3, 1, sharex='all', sharey='all', figsize = (15,15))
    # Plot Cohort
    axes[0].plot(age_range, all_cycles_per_age['cycle_length'].mean(), color=colors[0])
    axes[0].fill_between(age_range,
        all_cycles_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 2.5)),
        all_cycles_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 97.5)),
        alpha=0.3, facecolor=colors[0])
    axes[0].autoscale(enable=True, tight=True, axis='x')
    axes[0].set_xticks(age_range)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Cycle length')
    axes[0].set_ylim([10,90])
    # Plot Low
    axes[1].plot(age_range, cycles_users_less_than_cutoff_per_age['cycle_length'].mean(), color=colors[1])
    axes[1].fill_between(age_range,
        cycles_users_less_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 2.5)),
        cycles_users_less_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 97.5)),
        alpha=0.3, facecolor=colors[1])
    axes[1].autoscale(enable=True, tight=True, axis='x')
    axes[1].set_xticks(age_range)
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Cycle length')
    axes[1].set_ylim([10,90])
    # Plot High
    axes[2].plot(age_range, cycles_users_greater_than_cutoff_per_age['cycle_length'].mean(), color=colors[2])
    axes[2].fill_between(age_range,
        cycles_users_greater_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 2.5)),
        cycles_users_greater_than_cutoff_per_age['cycle_length'].apply(lambda x: np.nanpercentile(x, 97.5)),
        alpha=0.3, facecolor=colors[2])
    axes[2].autoscale(enable=True, tight=True, axis='x')
    axes[2].set_xticks(age_range)
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Cycle length')
    axes[2].set_ylim([10,90])
    # Save
    filename = '{}/cycle_length_per_age_subplot.pdf'.format(results_dir)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

    # Plot period-length per age
    fig, axes = plt.subplots(3, 1, sharex='all', sharey='all', figsize = (15,15))
    # Plot Cohort
    axes[0].plot(age_range, all_cycles_per_age['period_length'].mean(), color=colors[0])
    axes[0].fill_between(age_range,
        all_cycles_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 2.5)),
        all_cycles_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 97.5)),
        alpha=0.3, facecolor=colors[0])
    axes[0].autoscale(enable=True, tight=True, axis='x')
    axes[0].set_xticks(age_range)
    axes[0].set_xlabel('Age')
    axes[0].set_ylabel('Period length')
    axes[0].set_ylim([1,20])
    # Plot Low
    axes[1].plot(age_range, cycles_users_less_than_cutoff_per_age['period_length'].mean(), color=colors[1])
    axes[1].fill_between(age_range,
        cycles_users_less_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 2.5)),
        cycles_users_less_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 97.5)),
        alpha=0.3, facecolor=colors[1])
    axes[1].autoscale(enable=True, tight=True, axis='x')
    axes[1].set_xticks(age_range)
    axes[1].set_xlabel('Age')
    axes[1].set_ylabel('Period length')
    axes[1].set_ylim([1,20])
    # Plot High
    axes[2].plot(age_range, cycles_users_greater_than_cutoff_per_age['period_length'].mean(), color=colors[2])
    axes[2].fill_between(age_range,
        cycles_users_greater_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 2.5)),
        cycles_users_greater_than_cutoff_per_age['period_length'].apply(lambda x: np.nanpercentile(x, 97.5)),
        alpha=0.3, facecolor=colors[2])
    axes[2].autoscale(enable=True, tight=True, axis='x')
    axes[2].set_xticks(age_range)
    axes[2].set_xlabel('Age')
    axes[2].set_ylabel('Period length')
    axes[2].set_ylim([1,20])
    # Save
    filename = '{}/period_length_per_age_subplot.pdf'.format(results_dir)
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

    ################# PER-COUNTRY users #################
    # Country
    all_countries=cohort_cycles.groupby(['country'])['user_id'].apply(lambda x: np.unique(x).size).sort_values(ascending=False)
    high_countries=cycles_users_greater_than_cutoff.groupby(['country'])['user_id'].apply(lambda x: np.unique(x).size).sort_values(ascending=False)
    low_countries=cycles_users_less_than_cutoff.groupby(['country'])['user_id'].apply(lambda x: np.unique(x).size).sort_values(ascending=False)

    full_countries=pd.DataFrame(all_countries).merge(pd.DataFrame(high_countries).merge(pd.DataFrame(low_countries), on='country', suffixes=('_high', '_low')), on='country')
    # Print out
    print(full_countries)
    # Save as CSV, with '&' separator for latex tables
    with open('{}/per_country_users'.format(results_dir), 'w') as f:
        full_countries.to_csv(f, sep='&', index=True)

# Making sure the main program is not executed when the module is imported
if __name__ == '__main__':
    # Just run the main
    main()
