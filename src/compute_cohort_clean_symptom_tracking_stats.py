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

def compute_tracking_stats_by_symptom_for_cycles_fast(cycle_df, desired_tracking_categories, filename):
    '''
    For each tracking category in desired_tracking_categories, compute stats for each symptom in that category
    Input: 
        cycle_df (pandas dataframe): dataframe of user cycles, indexed by user ID and cycle ID
        desired_tracking_categories (list): names of desired categories to compute tracking stats for
        filename (string): desired filename for tracking stats dataframes
    Output:
        cycle_stats_tracking_info_by_symptom (pandas dataframe): pandas dataframes of symptom tracking stats for desired categories; 
            metric is proportion_cycles_out_of_cycles_with_category_for_
    '''
  #preallocate
    for category in desired_tracking_categories:
        cycle_stats_tracking_info_by_symptom = pd.DataFrame()
        user_id_list = np.unique(cycle_df['user_id'])
        cycle_stats_tracking_info_by_symptom['user_id'] = user_id_list

        num_cycles_tracked = np.array(cycle_df.groupby('user_id')['cycle_id'].count())
        cycle_stats_tracking_info_by_symptom['num_cycles_tracked'] = list(num_cycles_tracked)

        tracking_for_category = tracking[tracking['category'] == category]

        #drop rows where cycle id is nan
        tracking_for_category = tracking_for_category[tracking_for_category['cycle_id'].notnull()]
        #merge on cycle_df to only get the natural, >90, exclude single cycles tracking info
        tracking_for_category = tracking_for_category.merge(cycle_df, on = ['user_id', 'cycle_id'])

        symptoms_for_category = tracking_for_category['type'].value_counts()
        symptoms_for_category = symptoms_for_category[symptoms_for_category > 0].index.tolist()

    for symptom in symptoms_for_category:
	    tracking_for_symptom = tracking_for_category[tracking_for_category['type'] == symptom]
	    tracking_for_symptom = tracking_for_symptom.merge(cycle_df, on = ['user_id', 'cycle_id'])

	    num_cycles_with_category = tracking_for_category.groupby('user_id').cycle_id.nunique(dropna = False)
	    difference_ids = list(set(user_id_list) - set(num_cycles_with_category.keys()))
	    num_cycles_with_category_df = pd.DataFrame({'user_id':np.unique(num_cycles_with_category.keys()), 'num_cycles_with_category':num_cycles_with_category})
	    rows_to_add = [[difference_ids[i], 0] for i in range(len(difference_ids))]
	    rows_to_add = pd.DataFrame(rows_to_add, columns=list(num_cycles_with_category_df))
	    num_cycles_with_category_df = num_cycles_with_category_df.append(rows_to_add)
	    num_cycles_with_category_df = num_cycles_with_category_df.sort_values(by=['user_id'])
	    num_cycles_with_category = np.array(num_cycles_with_category_df['num_cycles_with_category'])

	    num_cycles_with_symptom = tracking_for_symptom.groupby(['user_id']).cycle_id.nunique(dropna = False)
	    difference_ids = list(set(user_id_list) - set(num_cycles_with_symptom.keys()))
	    num_cycles_with_symptom_df = pd.DataFrame({'user_id':np.unique(num_cycles_with_symptom.keys()), 'num_cycles_with_symptom':num_cycles_with_symptom})
	    rows_to_add = [[difference_ids[i], 0] for i in range(len(difference_ids))]
	    rows_to_add = pd.DataFrame(rows_to_add, columns=list(num_cycles_with_symptom_df))
	    num_cycles_with_symptom_df = num_cycles_with_symptom_df.append(rows_to_add)
	    num_cycles_with_symptom_df = num_cycles_with_symptom_df.sort_values(by=['user_id'])
	    num_cycles_with_symptom = np.array(num_cycles_with_symptom_df['num_cycles_with_symptom'])

	    proportion_cycles_with_symptom_out_of_cycles_with_category = num_cycles_with_symptom / num_cycles_with_category

	    cycle_stats_tracking_info_by_symptom['num_cycles_with_'+str(symptom)] = list(num_cycles_with_symptom)
	    cycle_stats_tracking_info_by_symptom['proportion_cycles_out_of_cycles_with_category_for_'+str(symptom)] = list(proportion_cycles_with_symptom_out_of_cycles_with_category)
    
    filename_save = filename+str(category)+'.pickle'

    with open(filename_save, 'wb') as f:
      pickle.dump(cycle_stats_tracking_info_by_symptom, f)

    print('finished '+str(category))

#### LOAD DATAFRAME WITH FLAGGED CYCLES AND REMOVE
filename = '../preprocessed_data/cohort_cycles_flagged.pickle'

with open(filename, 'rb') as f:
    cohort_cycles_flagged=pickle.load(f)

cohort_clean_cycles = cohort_cycles_flagged[cohort_cycles_flagged['badly_tracked_cycle'] == 'f']

print('loaded data, removed flagged cycles')

#remove birth control tracking categories
categories =  list(tracking.category.cat.categories)
categories.remove('injection_hbc')
categories.remove('iud')
categories.remove('patch_hbc')
categories.remove('pill_hbc')
categories.remove('ring_hbc')

#### COMPUTE TRACKING STATS FOR EACH CATEGORY, EXCLUDING FLAGGED CYCLES
compute_tracking_stats_by_symptom_for_cycles_fast(cohort_clean_cycles, categories, '../preprocessed_data/cohort_clean_symptom_tracking_stats_')
