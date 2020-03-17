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

#### LOAD DATAFRAME WITH FLAGGED CYCLES AND REMOVE
filename = '../preprocessed_data/cohort_cycles_flagged.pickle'

with open(filename, 'rb') as f:
    cohort_cycles_flagged=pickle.load(f)

cohort_clean_cycles = cohort_cycles_flagged[cohort_cycles_flagged['badly_tracked_cycle'] == 'f']

print('loaded data, removed flagged cycles')

#### RECOMPUTE CYCLE STATS FOR COHORT - EXCLUDE FLAGGED CYCLES
cohort_clean_cycle_stats = compute_cycle_stats_fast(cohort_clean_cycles, '../preprocessed_data/cohort_clean_cycle_stats.pickle')

print('recomputed cycle stats excluding flagged')
