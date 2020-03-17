#imports
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use('Agg')
import sys, os, re, time
import argparse
import pdb
import pickle
from itertools import *
# Science
import numpy as np
import scipy.stats as stats
import pandas as pd
from collections import Counter
from datetime import datetime
from datetime import timedelta
import multiprocessing
from multiprocessing import Pool
# Plotting
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

#### Load enriched dataframes
# Users
with open('../data/users.pickle', 'rb') as f:
    users=pickle.load(f)

# Cycles
with open('../data/cycles_enriched.pickle', 'rb') as f:
    cycles = pickle.load(f)

# Tracking
with open('../data/tracking_enriched.pickle', 'rb') as f:
    tracking = pickle.load(f)

def get_cycle_df_for_cohort(cycles, min_cycles):
    '''
    Returns cycle dataframe for cohort - natural cycles, aged 21-33, exclude self-excluded cycles, 
    exclude cycles > 90 days, exclude cycles that belong to users who tracked min_cycles
    
    Input:
        cycles (pandas dataframe): cycle dataframe where each row is cycle information for a user, 
        including user ID, cycle ID, and cycle length
        min_cycles (int): user must have tracked more than min_cycles to be included in cohort
    Output:
        cycles_natural_middle_exclude_90_exclude_single (pandas dataframe): cycle dataframe for desired 
        user cohort - natural cycles only, aged 21-33, <= 90 days, excluding self-excluded cycles and users who tracked min_cycles cycles
    '''

    #get natural cycles
    cycles_natural = cycles[cycles['natural_cycle']==1]
    #get middle age group
    cycles_natural_middle = cycles_natural[(cycles_natural['age_at_cycle']<=33) & (cycles_natural['age_at_cycle']>=21)]
    #exclude self-excluded cycles
    cycles_natural_middle_excluded = cycles_natural_middle[cycles_natural_middle['cycle_excluded']=='f']
    #exclude cycles > 90 days
    cycles_natural_middle_exclude_90 = cycles_natural_middle_excluded[cycles_natural_middle_excluded['cycle_length'] <= 90]
    #remove cycles that belong to users who only tracked one (remaining) cycle
    cycle_counts = cycles_natural_middle_exclude_90.groupby('user_id')['cycle_id'].count()
    user_ids_mult_cycles = cycle_counts[cycle_counts > min_cycles].index.tolist()
    cycles_natural_middle_exclude_90_exclude_single = cycles_natural_middle_exclude_90[cycles_natural_middle_exclude_90['user_id'].isin(user_ids_mult_cycles)]
    return(cycles_natural_middle_exclude_90_exclude_single)

#### GET CYCLE DF FOR COHORT 
cycle_df = get_cycle_df_for_cohort(cycles, 2)

def compute_cycle_stats_fast(cycle_df, filename):
    '''
    Compute cycle stats for desired cycle_df, save under filename; stats include cycle and period lengths, intercycle lengths (CLDs), 
    and summary stats (mean, variance, standard deviation, max, median)
    
    Input: 
        cycle_df (pandas dataframe): dataframe of user cycles, indexed by user ID and cycle ID
        filename (string): desired filename for cycle stats dataframe
    Output:
        cycle_stats (pandas dataframe): cycle stats dataframe computed from input cycle dataframe
    '''

    #preallocate dataframe
    cycle_stats = pd.DataFrame(index=range(len(np.unique(cycle_df['user_id']))), columns = ['user_id', 'cycle_lengths', 'period_lengths', 'inter_cycle_lengths'])
    cycle_stats['user_id'] = np.unique(cycle_df['user_id'])

    for index, user_id in enumerate(np.unique(cycle_df['user_id'])):

        #compute cycle lengths, period lengths, intercycle lengths for each user
        cycle_df_for_user = cycle_df[cycle_df['user_id'] == user_id]
        cycle_lengths_for_user = np.array(cycle_df_for_user['cycle_length'])
        period_lengths_for_user = np.array(cycle_df_for_user['period_length'])
        inter_cycle_lengths_for_user = np.abs(cycle_lengths_for_user[:-1] - cycle_lengths_for_user[1:])

        #add to dataframe
        cycle_stats.at[index, ['cycle_lengths', 'period_lengths', 'inter_cycle_lengths']] = [cycle_lengths_for_user, period_lengths_for_user, inter_cycle_lengths_for_user]

        print(index)

    #compute summary stats after
    num_cycles_tracked_per_user = np.array(cycle_df.groupby('user_id')['cycle_length'].count())
    cycle_stats['num_cycles_tracked'] = num_cycles_tracked_per_user
    avg_cycle_lengths = np.array(cycle_df.groupby('user_id')['cycle_length'].mean())
    cycle_stats['avg_cycle_length'] = avg_cycle_lengths
    var_cycle_lengths = np.array(cycle_df.groupby('user_id')['cycle_length'].var())
    cycle_stats['var_cycle_length'] = var_cycle_lengths
    cycle_stats['std_cycle_length'] = np.sqrt(var_cycle_lengths)
    cycle_stats['max_cycle_length'] = [np.max(cycle_stats.iloc[i]['cycle_lengths']) for i in range(len(cycle_stats))]
    cycle_stats['max_period_length'] = [np.max(cycle_stats.iloc[i]['period_lengths']) for i in range(len(cycle_stats))]
    cycle_stats['median_inter_cycle_length'] = [np.median(cycle_stats.iloc[i]['inter_cycle_lengths']) for i in range(len(cycle_stats))]
    cycle_stats['max_inter_cycle_length'] = [np.max(cycle_stats.iloc[i]['inter_cycle_lengths']) for i in range(len(cycle_stats))]

    with open(filename, 'wb') as f:
        pickle.dump(cycle_stats, f)
    print(cycle_stats.iloc[0])
    return(cycle_stats)

#### COMPUTE CYCLE STATS FOR COHORT
cohort_cycle_stats = compute_cycle_stats_fast(cycle_df, '../preprocessed_data/cohort_cycle_stats.pickle')

print('computed cycle stats')

def flag_badly_tracked_cycles(cycle_stats_df, cycle_df, inter_cycle_threshold, filename):
    '''
    Flag badly tracked cycles in cycle_df, based on users where max intercycle length - median intercycle length > inter_cycle_threshold
    
    Input:
        cycle_stats_df (pandas dataframe): cycle stats dataframe
        cycle_df (pandas dataframe): dataframe of user cycles, indexed by user ID and cycle ID
        inter_cycle_threshold (int): cutoff for where CLD exceeds median (i.e., flag cycles where CLD > median + cutoff
        filename (str): desired filename for cycle dataframe with flagged cycles
    Output:
        cycle_df_with_flagged_bad_cycles (pandas dataframe): cycle dataframe with artificially long cycles flagged
    '''

    cycle_df_with_flagged_bad_cycles = cycle_df.copy()

    index_users_with_badly_tracked_cycles = np.argwhere(cycle_stats_df['max_inter_cycle_length'] - cycle_stats_df['median_inter_cycle_length'] > inter_cycle_threshold).flatten()
    user_ids_with_badly_tracked_cycles = cycle_stats_df['user_id'][index_users_with_badly_tracked_cycles]

    cycles_for_users_with_badly_tracked_cycles = cycle_df_with_flagged_bad_cycles[cycle_df_with_flagged_bad_cycles['user_id'].isin(user_ids_with_badly_tracked_cycles)]
    flags = pd.DataFrame(index = cycle_df_with_flagged_bad_cycles.index)
    flags['flag'] = ['f']*len(cycle_df_with_flagged_bad_cycles)

    for index, user_id in enumerate(user_ids_with_badly_tracked_cycles):
        cycle_stats_for_user = cycle_stats_df[cycle_stats_df['user_id'] == user_id]
        index_for_user = cycle_df[cycle_df['user_id'] == user_id].index

        #get intercycle lengths, cycle lengths for user
        inter_cycle_lengths_for_user = cycle_stats_for_user.iloc[0]['inter_cycle_lengths']
        cycle_lengths_for_user = cycle_stats_for_user.iloc[0]['cycle_lengths']

        #get index of intercycle lengths corresponding to long ones, i.e., where intercycle length > median + 10
        index_long_inter_cycle_lengths_for_user = np.argwhere(inter_cycle_lengths_for_user > cycle_stats_for_user.iloc[0]['median_inter_cycle_length']+inter_cycle_threshold).flatten()

        #now go through corresponding cycles and flag badly tracked ones
        for bad_index in index_long_inter_cycle_lengths_for_user:
            cycles_for_index = cycle_lengths_for_user[bad_index:bad_index+2]
            if cycles_for_index[0] > cycles_for_index[1]:
                flags.at[index_for_user[bad_index], 'flag'] = 't'
            else:
                flags.at[index_for_user[bad_index+1], 'flag'] = 't'

    cycle_df_with_flagged_bad_cycles['badly_tracked_cycle'] = flags['flag']

    with open(filename, 'wb') as f:
        pickle.dump(cycle_df_with_flagged_bad_cycles, f)

    return(cycle_df_with_flagged_bad_cycles)

#### FLAG BADLY TRACKED CYCLES
cohort_cycles_flagged = flag_badly_tracked_cycles(cohort_cycle_stats, cycle_df, 10, '../preprocessed_data/cohort_cycles_flagged.pickle')

print('flagged cycles')
