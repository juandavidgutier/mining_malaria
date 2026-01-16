# Project Title

Illegal Mining as a Driver of Malaria Risk in Colombia: Evidence from Causal Machine Learning at the Municipal Level

## Description

Code and dataset shared to reproduce the results of the paper Illegal Mining as a Driver of Malaria Risk in Colombia: Evidence from Causal Machine Learning at the Municipal Level.
The file data_final.csv is the dataset used for the results presented in the manuscript. 
To reproduce the fine-tuning use the file rscore_mining_malaria.py
To reproduce the results use the file causal_model_malaria.py
To reproduce the results of E-Value use the file EValue_mining.R

## Data Privacy and Anonymization

This dataset has been processed to ensure complete anonymization and contains no personally identifiable information (PII. All data has been:

- Aggregated at appropriate spatial/temporal scales
- Stripped of any individual identifiers
- Processed to remove direct or indirect identifying elements

The dataset is suitable for public sharing and complies with data privacy standards.

## Privacy Statement

This repository contains datasets that have been carefully processed to protect individual privacy:

### What is NOT included:
- Names, addresses, or contact information
- Individual-level identifiers
- Location data below 25 km resolution
- Timestamps more precise than monthly
- Any data that could be used to re-identify individuals

### Data Processing:
- Spatial aggregation to municipality
- Temporal aggregation to monthly averages

### Compliance:
This dataset meets requirements for public data sharing under applicable privacy regulations.

## Author

Juan David Gutiérrez  

## libraries

pandas; dowhy; econml; statsmodels; arviz; matplolib; zepid; scipy; scikit-learn; plotnine
EValue

