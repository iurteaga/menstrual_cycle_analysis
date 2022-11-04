# Characterization and analysis of self-tracked menstrual cycle data

Work on the characterization and analysis of menstrual cycles using self-tracked mobile health data

We provide a conda environment file for ease of replication in ./menstrual_cycle_analysis.yml

## Present directories

### doc

- doc/characterization: manuscripts on "Characterizing physiological and symptomatic variation in menstrual cycles using self-tracked mobile-health data"
    - [Characterizing physiological and symptomatic variation in menstrual cycles using self-tracked mobile health data. Li, K.; Urteaga, I.; Shea, A.; Vitzthum, V.; Wiggins, C. H; and Elhadad, N. Nature Partner Journal Digital Medicine, 3(79), 2020.](https://www.nature.com/articles/s41746-020-0269-8)

- doc/prediction: manuscripts on predictive models:
    - [A generative, predictive model for menstrual cycle lengths that accounts for potential self-tracking artifacts in mobile health data. Li, K.; Urteaga, I.; Shea, A.; Vitzthum, V.; Wiggins, C. H; and Elhadad, N. In NeurIPS 2020 Workshop Machine Learning for Mobile Health, 2020. Contributed Talk.](https://sites.google.com/view/ml4mobilehealth-neurips-2020/home#h.kx5rlc27ssyh)
    - [A generative, predictive model for menstrual cycle lengths that accounts for potential self-tracking artifacts in mobile health data. Li, K.; Urteaga, I.; Shea, A.; Vitzthum, V. J.; Wiggins, C. H.; and Elhadad, N. arXiv e-print:2102.12439.](https://arxiv.org/abs/2102.12439)
    - [A predictive model for next cycle start date that accounts for adherence in menstrual self-tracking. Li K.; Urteaga, I.; Shea, A.; Vitzthum, V. J.; Wiggins, C. H.; and Elhadad, N.; Journal of the American Medical Informatics Association, Volume 29, Issue 1, Pages 3–11, January 2022.](https://doi.org/10.1093/jamia/ocab182)
    - [A Generative Modeling Approach to Calibrated Predictions:A Use Case on Menstrual Cycle Length Prediction. Urteaga, I.; Li, K.; Wiggins, C.; and Elhadad, N. In Proceedings of the 6th Machine Learning for Healthcare, 2021.](https://proceedings.mlr.press/v149/urteaga21a.html)

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

Cycle length only information for predictive work
- ./data/cycle_length_data/cycle_lengths.npz  
        Numpy array with I (number of individuals) by C (number of cycles per-individual) information
     
### preprocessed_data

Pre-processed dataframes with cycles and tracking data were used for the [characterization of menstrual cycles using self-tracked mobile health data](https://www.nature.com/articles/s41746-020-0269-8): these are not publicly available.

### results

Directory for plots and results

Characterization outputs for code in src/characterization

- ./results/characterizing_cycle_and_symptoms  
    Results regarding the initial exploratory analysis to characterize the menstrual cycle and self-tracked symptoms
    
- ./results/characterizing_cycle_and_symptoms/cohort_summary_statistics  
        Summary statistics and plots for the npjDigitalMedicine cohort

- ./results/characterizing_cycle_and_symptoms/cycle_period_length_analysis  
        Summary statistics and plots regarding the npjDigitalMedicine cohort's self-reported cycles

- ./results/characterizing_cycle_and_symptoms/symptom_tracking_analysis_bootstrapping_{nbootstrapped}  
        Results for a bootstrapped analysis (with nbootstrapped samples) of the npjDigitalMedicine cohort's self-tracked symptoms 
        
Predictive outputs

- ./results/evaluate_predictive_models/  
        Directory for results per each evaluated cycle length dataset and model
