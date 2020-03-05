# Characterization and analysis of self-tracked menstrual cycle data

Work on the characterization and analysis of menstrual cycles using self-tracked mobile health data

## Directories

### doc

- doc/npjDigitalMedicine: manuscript as published in npj Digital Medicine

### src

Directory with code for data processing

- compute_cohort_cycles_flagged.py

- compute_cohort_clean_cycle_stats.py

- compute_cohort_clean_symptom_tracking_stats.py

- cohort_summary_statistics.py

- cycle_period_length_analysis.py

- symptom_tracking_analysis_bootstrapping.py

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
