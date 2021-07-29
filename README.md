# Characterization and analysis of self-tracked menstrual cycle data

Work on the characterization and analysis of menstrual cycles using self-tracked mobile health data

## Present directories

### doc

- doc/characterization: manuscripts on "Characterizing physiological and symptomatic variation in menstrual cycles using self-tracked mobile-health data"
    - [Characterizing physiological and symptomatic variation in menstrual cycles using self-tracked mobile health data. Li, K.; Urteaga, I.; Shea, A.; Vitzthum, V.; Wiggins, C. H; and Elhadad, N. Nature Partner Journal Digital Medicine, 3(79), 2020.](https://www.nature.com/articles/s41746-020-0269-8)

- doc/prediction: manuscripts on predictive models:
    - [A generative, predictive model for menstrual cycle lengths that accounts for potential self-tracking artifacts in mobile health data. Li, K.; Urteaga, I.; Shea, A.; Vitzthum, V.; Wiggins, C. H; and Elhadad, N. In NeurIPS 2020 Workshop Machine Learning for Mobile Health, 2020. Contributed Talk.](https://sites.google.com/view/ml4mobilehealth-neurips-2020/home#h.kx5rlc27ssyh)
    - [A generative, predictive model for menstrual cycle lengths that accounts for potential self-tracking artifacts in mobile health data. Li, K.; Urteaga, I.; Shea, A.; Vitzthum, V. J.; Wiggins, C. H.; and Elhadad, N. arXiv e-print:2102.12439.](https://arxiv.org/abs/2102.12439)
    - [A Generative Modeling Approach to Calibrated Predictions:A Use Case on Menstrual Cycle Length Prediction. Urteaga, I.; Li, K.; Wiggins, C.; and Elhadad, N. In Proceedings of the 6th Machine Learning for Healthcare, 2021.](https://www.mlforhc.org/accepted-papers-1)

### src

Main directory with source code utilities.

- src/characterization
    Directory with code for data processing

- src/prediction
    Directory with code for predictive modeling and evaluation.

### scripts

Main directory with scripts to run, evaluate and plot experiments.

## Expected directory structure and content

### data

Original dataframes with cycles and tracking data
- ./data/cycles.pickle  
    Original cycle information (as pandas dataframe)

- ./data/tracking.pickle  
    Original symptom tracking information (as pandas dataframe)

### preprocessed_data

Pre-processed dataframes with cycles and tracking data

- ./preprocessed_data/cohort_cycle_stats.pickle  
    Cycle statistics of the original cohort

- ./preprocessed_data/cohort_cycles_flagged.pickle  
    Cycle information, with flagged cycle indicator, of the original cohort

- ./preprocessed_data/cohort_clean_cycle_stats.pickle  
    Cycle statistics of the clean cohort, after removing flagged cycles

- ./preprocessed_data/cohort_clean_symptom_tracking_stats_{category}.pickle  
    Symptom tracking statistics for the clean cohort, where {category} matches each of the corresponding tracked categories

### results

Directory for plots and results

- ./results/characterizing_cycle_and_symptoms
    Results regarding the initial exploratory analysis to characterize the menstrual cycle and self-tracked symptoms
    
- ./results/characterizing_cycle_and_symptoms/cohort_summary_statistics  
        Summary statistics and plots for the npjDigitalMedicine cohort

- ./results/characterizing_cycle_and_symptoms/cycle_period_length_analysis  
        Summary statistics and plots regarding the npjDigitalMedicine cohort's self-reported cycles

- ./results/characterizing_cycle_and_symptoms/symptom_tracking_analysis_bootstrapping_{nbootstrapped}  
        Results for a bootstrapped analysis (with nbootstrapped samples) of the npjDigitalMedicine cohort's self-tracked symptoms 
