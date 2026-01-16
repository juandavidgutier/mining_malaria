import random
import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from econml.dr import SparseLinearDRLearner, ForestDRLearner, LinearDRLearner
from sklearn.preprocessing import PolynomialFeatures
from plotnine import ggplot, aes, geom_line, geom_ribbon, ggtitle, labs, geom_point, geom_hline, theme_linedraw, theme, element_rect, theme_light, element_line, element_text
from zepid.graphics import EffectMeasurePlot
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import expon
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import statsmodels.api as sm
from dowhy import CausalModel
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor, XGBClassifier
from dowhy.causal_estimator import CausalEstimate
from sklearn.preprocessing import StandardScaler
from econml.dr import DRLearner
from sklearn.linear_model import LassoCV
from econml.dml import DML, SparseLinearDML


# Set seeds for reproducibility
def seed_everything(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 123
seed_everything(seed)
warnings.filterwarnings('ignore')

#%%

data_all = pd.read_csv("D:/data_final.csv", encoding='latin-1')

data_all = data_all.dropna()

columns_to_drop = ['Department', 'Year', 'Month', 'Year_month', 'Period', 
                   'DANE_year', 'DANE_Year_Month', 'Total_pob', 
                   'cases', 'sir']

# 1. Label Encoding for DANE
le = LabelEncoder()
data_all['DANE_labeled'] = le.fit_transform(data_all['DANE'])
scaler = MinMaxScaler()
data_all['DANE_normalized'] = scaler.fit_transform(
    data_all[['DANE_labeled']]
)

# 2. Label Encoding for Deparment_DANE
le_year = LabelEncoder()
data_all['Deparment_DANE_labeled'] = le_year.fit_transform(data_all['Deparment_DANE'])
scaler_DDANE = MinMaxScaler()
data_all['Deparment_DANE_normalized'] = scaler_DDANE.fit_transform(
    data_all[['Deparment_DANE_labeled']]
)

# 3. Label Encoding for Deparment_Month
le_year = LabelEncoder()
data_all['Deparment_Month_labeled'] = le_year.fit_transform(data_all['Deparment_Month'])
scaler_DDANE = MinMaxScaler()
data_all['Deparment_Month_normalized'] = scaler_DDANE.fit_transform(
    data_all[['Deparment_Month_labeled']]
)

# 4. Label Encoding for DANE_period
le_year = LabelEncoder()
data_all['DANE_period_labeled'] = le_year.fit_transform(data_all['DANE_period'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_period_normalized'] = scaler_DDANE.fit_transform(
    data_all[['DANE_period_labeled']]
)

# 5. Label Encoding for DANE_Department_Month
le_year = LabelEncoder()
data_all['DANE_Department_Month_labeled'] = le_year.fit_transform(data_all['DANE_Department_Month'])
scaler_DDANE = MinMaxScaler()
data_all['DANE_Department_Month_normalized'] = scaler_DDANE.fit_transform(
    data_all[['DANE_Department_Month_labeled']]
)

data_all.drop(columns=columns_to_drop, inplace=True)

std_mining = data_all['mining'].std()
print(f"std of mining: {std_mining}")

median_mining = data_all['mining'].median()
print(f"median of mining: {median_mining}")

scaler = StandardScaler()
data_all['forest'] = scaler.fit_transform(data_all[['forest']])
data_all['MPI'] = scaler.fit_transform(data_all[['MPI']])
data_all['temperature'] = scaler.fit_transform(data_all[['temperature']])
data_all['rainfall'] = scaler.fit_transform(data_all[['rainfall']])
data_all['deforest'] = scaler.fit_transform(data_all[['deforest']])
data_all['vector'] = scaler.fit_transform(data_all[['vector']])
data_all['fire'] = scaler.fit_transform(data_all[['fire']])
data_all['coca'] = scaler.fit_transform(data_all[['coca']])


#%%

# Standardized dataset
data_std = data_all

data_std['mining'] = scaler.fit_transform(data_std[['mining']])


#%%

data_std = data_std[['DANE_normalized', 'Deparment_DANE_normalized', 'Deparment_Month_normalized', 'DANE_period_normalized', 'DANE_Department_Month_normalized',
                     'deforest', 'fire', 'HFP', 'coca', 'vector',
                     'forest', 'MPI', 'temperature', 'rainfall',
                     'mining', 'excess', 'altitude']]


#%%

# Causal mechanism
model_mining = CausalModel(
    data=data_std,
    treatment=['mining'],
    outcome=['excess'],
    mediator=['deforest'], 
    common_causes=['coca', 'fire'], 
    graph="""graph[directed 1 
                node[id "mining" label "mining"]
                node[id "excess" label "excess"]
                node[id "DANE_normalized" label "DANE_normalized"]
                node[id "Deparment_DANE_normalized" label "Deparment_DANE_normalized"]
                node[id "Deparment_Month_normalized" label "Deparment_Month_normalized"]
                node[id "DANE_period_normalized" label "DANE_period_normalized"]
                node[id "DANE_Department_Month_normalized" label "DANE_Department_Month_normalized"]
                node[id "deforest" label "deforest"]
                node[id "fire" label "fire"]
                node[id "HFP" label "HFP"]
                node[id "coca" label "coca"]
                node[id "vector" label "vector"]
                node[id "forest" label "forest"]
                node[id "MPI" label "MPI"]
                node[id "temperature" label "temperature"]
                node[id "rainfall" label "rainfall"]
                node[id "altitude" label "altitude"]
                
                edge[source "DANE_normalized" target "excess"]
                edge[source "Deparment_DANE_normalized" target "excess"]
                edge[source "Deparment_Month_normalized" target "excess"]
                edge[source "DANE_period_normalized" target "excess"]
                edge[source "DANE_Department_Month_normalized" target "excess"]
                
                edge[source "forest" target "temperature"]
                edge[source "forest" target "rainfall"]
                edge[source "rainfall" target "MPI"]
                
                edge[source "temperature" target "vector"]
                edge[source "HFP" target "deforest"]
                edge[source "HFP" target "vector"]
                edge[source "HFP" target "MPI"]
                edge[source "HFP" target "forest"]
                edge[source "HFP" target "temperature"]
                
                edge[source "MPI" target "vector"]
                edge[source "MPI" target "temperature"]
                
                edge[source "mining" target "deforest"]
                edge[source "mining" target "HFP"]
                edge[source "mining" target "MPI"]
                
                edge[source "deforest" target "temperature"]
                
                edge[source "rainfall" target "vector"]
                
                edge[source "fire" target "mining"]
                edge[source "fire" target "excess"]
                edge[source "coca" target "mining"]
                edge[source "coca" target "excess"]
                
                edge[source "vector" target "excess"]
                edge[source "deforest" target "excess"]
                edge[source "forest" target "excess"]
                edge[source "MPI" target "excess"]
                edge[source "HFP" target "excess"]
                edge[source "altitude" target "temperature"]
                edge[source "mining" target "excess"]
            ]"""
)

#%%

from PIL import Image
import matplotlib.pyplot as plt

# Generate the causal model diagram
model_mining.view_model()

# Load and display the image
img = Image.open("causal_model.png")
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.title("Causal DAG Model")
plt.show()
    
#%% 

# Identifying causal effects
identified_estimand_mining = model_mining.identify_effect(proceed_when_unidentifiable=None)                                                       
print(identified_estimand_mining)


#%%

reg1 = lambda: XGBRegressor(n_estimators=3800, max_depth=30, random_state=123, eta=0.0001, reg_lambda=1.5, alpha=0.001)

# Estimate causal effect using SparseLinearDML from EconML
causal_estimate_std = model_mining.estimate_effect(identified_estimand_mining,
                                        method_name="backdoor.econml.dml.SparseLinearDML",
                                        effect_modifiers=['altitude', 'DANE_normalized', 'Deparment_DANE_normalized', 'Deparment_Month_normalized', 'DANE_period_normalized', 'DANE_Department_Month_normalized'],
                                        confidence_intervals=True,
                                        method_params={
                                            "init_params": {
                                                "model_y": reg1(),
                                                "model_t": reg1(),
                                                "discrete_outcome": True,
                                                "discrete_treatment": False,
                                                "tol": 1e-4,
                                                "alpha": 'auto', 
                                                "max_iter": 50000,
                                                "cv": 5,
                                                "random_state": 123
                                            },
                                        }
                                    )

#%%

# Access the internal estimator
econml_estimator = causal_estimate_std.estimator.estimator

effect_modifiers = ['altitude', 'DANE_normalized', 'Deparment_DANE_normalized', 'Deparment_Month_normalized', 'DANE_period_normalized', 'DANE_Department_Month_normalized']  

# Compute ATE and CI for mining
estimator_mining = causal_estimate_std.estimator.estimator
X_data_mining = data_std[effect_modifiers].dropna()  
ate_mining = estimator_mining.ate(X=X_data_mining)
ate_ci_mining = estimator_mining.ate_interval(X=X_data_mining, alpha=0.05)
ci_lower_mining = ate_ci_mining[0]
ci_upper_mining = ate_ci_mining[1]

print(f"  ATE: {ate_mining}")
print(f"  95% CI of ATE: {ate_ci_mining}")


#%%

# CATE by altitude
alt = data_std['altitude']  # Assuming altitude is in the first column

# Grid for altitude
min_alt = alt.min()
max_alt = alt.max()
delta = (max_alt - min_alt) / 100
alt_grid = np.arange(min_alt, max_alt + delta - 0.001, delta)

# Mean values for other effect modifiers
DANE_encoded_mean = data_std['DANE_normalized'].mean()
DANE_Dept_encoded_mean = data_std['Deparment_DANE_normalized'].mean()
Dept_month_encoded_mean = data_std['Deparment_Month_normalized'].mean()
DANE_month_encoded_mean = data_std['DANE_period_normalized'].mean()
DANE_Dept_month_encoded_mean = data_std['DANE_Department_Month_normalized'].mean()

# Prediction matrix
X_test_grid = np.column_stack([
    alt_grid,
    np.full_like(alt_grid, DANE_encoded_mean),
    np.full_like(alt_grid, DANE_Dept_encoded_mean),
    np.full_like(alt_grid, Dept_month_encoded_mean),
    np.full_like(alt_grid, DANE_month_encoded_mean),
    np.full_like(alt_grid, DANE_Dept_month_encoded_mean)    
])

# Predict treatment effect
treatment_effect = econml_estimator.effect(X_test_grid)

# Confidence intervals
hte_lower2_cons, hte_upper2_cons = econml_estimator.effect_interval(X_test_grid, alpha=0.05)
    
plot_data = pd.DataFrame({
    'alt': alt_grid,
    'treatment_effect': treatment_effect.flatten(),
    'hte_lower2_cons': hte_lower2_cons.flatten(),
    'hte_upper2_cons': hte_upper2_cons.flatten()
})

cate_plot = (
    ggplot(plot_data)
    + aes(x='alt', y='treatment_effect')
    + geom_line(color='black', size=1)
    + geom_ribbon(aes(ymin='hte_lower2_cons', ymax='hte_upper2_cons'), alpha=0.2, fill='blue')
    + labs(x='Altitude (m)', y='Effect of illegal mining on excess malaria cases', title='')
    + geom_hline(yintercept=0, color="red", linetype="dashed", size=0.8)
    + theme(
        plot_title=element_text(hjust=0.5, size=12),
        axis_title_x=element_text(size=10),
        axis_title_y=element_text(size=10)
    )
)

print(cate_plot)    


#%%

# Refutation tests

# Random common cause
random_std = model_mining.refute_estimate(identified_estimand_mining, causal_estimate_std,
                                         method_name="random_common_cause", random_state=123, num_simulations=50)
print(random_std)

# Data subset refuter
subset_std = model_mining.refute_estimate(identified_estimand_mining, causal_estimate_std,
                                          method_name="data_subset_refuter", subset_fraction=0.1, random_state=123, num_simulations=50)
print(subset_std) 
      
# Bootstrap refuter
bootstrap_std = model_mining.refute_estimate(identified_estimand_mining, causal_estimate_std,
                                             method_name="bootstrap_refuter", random_state=123, num_simulations=50)
print(bootstrap_std)

# Placebo treatment refuter
placebo_std = model_mining.refute_estimate(identified_estimand_mining, causal_estimate_std,
                                           method_name="placebo_treatment_refuter", placebo_type="permute", random_state=123, num_simulations=50)
print(placebo_std)    


#%%

# E-value calculation

# Mining values for comparison
mining_std_min = -0.28
mining_std_max = 17.64

# Effect modifiers
effect_modifiers = ['altitude', 'DANE_normalized', 'Deparment_DANE_normalized', 
                    'Deparment_Month_normalized', 'DANE_period_normalized', 
                    'DANE_Department_Month_normalized']

X_data_mining = data_std[effect_modifiers].dropna()

# Compute causal effects (probability differences)
effect_at_min = econml_estimator.effect(X=X_data_mining, T0=0, T1=mining_std_min)
effect_at_max = econml_estimator.effect(X=X_data_mining, T0=0, T1=mining_std_max)

# Average over all units
mean_effect_at_min = np.mean(effect_at_min)
mean_effect_at_max = np.mean(effect_at_max)

# Risk Difference
effect_diff = mean_effect_at_max - mean_effect_at_min

print(f"Average effect at mining_min: {mean_effect_at_min:.6f}")
print(f"Average effect at mining_max: {mean_effect_at_max:.6f}")
print(f"Risk Difference: {effect_diff:.6f}")

# ============================================
# STEP 2: Compute baseline prevalence
# ============================================

baseline_risk = data_std['excess'].mean()
print(f"\nBaseline prevalence of excess: {baseline_risk:.6f}")

# ============================================
# STEP 3: Estimate risks at each exposure level
# ============================================

# Estimated risks
risk_at_min = np.clip(baseline_risk + mean_effect_at_min, 0.0001, 0.9999)
risk_at_max = np.clip(baseline_risk + mean_effect_at_max, 0.0001, 0.9999)

print(f"Estimated risk at mining_min: {risk_at_min:.6f}")
print(f"Estimated risk at mining_max: {risk_at_max:.6f}")

# ============================================
# STEP 4: Compute Risk Ratio on log scale
# ============================================

# Log Risk Ratio
log_RR = np.log(risk_at_max) - np.log(risk_at_min)
RR_point_estimate = np.exp(log_RR)

print(f"\nLog(RR): {log_RR:.6f}")
print(f"Risk Ratio (RR): {RR_point_estimate:.6f}")

# ============================================
# STEP 5: Compute SE of log(RR) using delta method
# ============================================

# Get confidence intervals for the effects
effect_interval_min = econml_estimator.effect_interval(X=X_data_mining, T0=0, T1=mining_std_min, alpha=0.05)
effect_interval_max = econml_estimator.effect_interval(X=X_data_mining, T0=0, T1=mining_std_max, alpha=0.05)

# CI for effects
ci_lower_effect_min = np.mean(effect_interval_min[0])
ci_upper_effect_min = np.mean(effect_interval_min[1])

ci_lower_effect_max = np.mean(effect_interval_max[0])
ci_upper_effect_max = np.mean(effect_interval_max[1])

# Estimate SE of effects
se_effect_min = (ci_upper_effect_min - ci_lower_effect_min) / (2 * 1.96)
se_effect_max = (ci_upper_effect_max - ci_lower_effect_max) / (2 * 1.96)

print(f"\nSE of effect at mining_min: {se_effect_min:.6f}")
print(f"SE of effect at mining_max: {se_effect_max:.6f}")

# ============================================
# STEP 6: Delta method for SE of log(RR)
# ============================================

# log(RR) = log(P_max) - log(P_min)
# where P_max = baseline_risk + effect_max
#       P_min = baseline_risk + effect_min

# Partial derivatives
d_log_P_max = 1 / risk_at_max
d_log_P_min = -1 / risk_at_min  # negative because we subtract

# Variances of effects
var_effect_min = se_effect_min**2
var_effect_max = se_effect_max**2

# Variance of log(RR)
# Assuming independence between the two estimated effects
var_log_RR = (d_log_P_max**2 * var_effect_max) + (d_log_P_min**2 * var_effect_min)

# SE of log(RR)
se_log_RR = np.sqrt(var_log_RR)

print(f"\nSE of log(RR): {se_log_RR:.6f}")

# ============================================
# STEP 7: Compute CI for RR on log scale, then exponentiate
# ============================================

# CI for log(RR)
log_RR_lower = log_RR - 1.96 * se_log_RR
log_RR_upper = log_RR + 1.96 * se_log_RR

# Exponentiate to get CI for RR
RR_lower_CI = np.exp(log_RR_lower)
RR_upper_CI = np.exp(log_RR_upper)

print(f"\n95% CI of log(RR): [{log_RR_lower:.6f}, {log_RR_upper:.6f}]")
print(f"95% CI of RR: [{RR_lower_CI:.6f}, {RR_upper_CI:.6f}]")

# ============================================
# STEP 8: VALIDATION – Check CI order
# ============================================

if RR_lower_CI > RR_upper_CI:
    print("\n⚠️ ERROR: Confidence interval bounds are inverted. Swapping them...")
    RR_lower_CI, RR_upper_CI = RR_upper_CI, RR_lower_CI
    print(f"Corrected 95% CI of RR: [{RR_lower_CI:.6f}, {RR_upper_CI:.6f}]")

# Additional validation
assert RR_lower_CI < RR_point_estimate < RR_upper_CI, \
    "Point estimate must lie within the confidence interval"

# ============================================
# STEP 9: Prepare data for E-value
# ============================================

param_evalue_mining_final = pd.DataFrame({
    'Analysis': ['mining_effect'],
    'RR_point_estimate': [RR_point_estimate],
    'RR_lower_CI': [RR_lower_CI],
    'RR_upper_CI': [RR_upper_CI],
    'log_RR': [log_RR],
    'se_log_RR': [se_log_RR],
    'baseline_risk': [baseline_risk],
    'risk_at_min': [risk_at_min],
    'risk_at_max': [risk_at_max],
    'risk_difference': [effect_diff],
    'mining_min': [mining_std_min],
    'mining_max': [mining_std_max]
})

print("\n" + "="*80)
print("FINAL SUMMARY FOR E-VALUE")
print("="*80)
print(param_evalue_mining_final[['Analysis', 'RR_point_estimate', 'RR_lower_CI', 'RR_upper_CI']].to_string(index=False))
print("="*80)

# Save file
output_path = "D:/param_evalue_mining_final.csv"
param_evalue_mining_final.to_csv(output_path, index=False)

print(f"\n✓ File saved to: {output_path}")

# ============================================
# STEP 10: Additional information for interpretation
# ============================================

print("\n" + "="*80)
print("ADDITIONAL INFORMATION FOR INTERPRETATION")
print("="*80)
print(f"Change in mining: {mining_std_min:.2f} → {mining_std_max:.2f} (standardized)")
print(f"Absolute risk change: {risk_at_min:.4f} → {risk_at_max:.4f}")
print(f"Risk Difference (RD): {effect_diff:.4f} ({effect_diff*100:.2f}%)")
print(f"Risk Ratio (RR): {RR_point_estimate:.4f}")

if RR_point_estimate > 1:
    pct_increase = (RR_point_estimate - 1) * 100
    print(f"\nInterpretation: Increasing mining from {mining_std_min:.2f} to {mining_std_max:.2f}")
    print(f"increases the risk of excess by {pct_increase:.1f}%")
elif RR_point_estimate < 1:
    pct_decrease = (1 - RR_point_estimate) * 100
    print(f"\nInterpretation: Increasing mining from {mining_std_min:.2f} to {mining_std_max:.2f}")
    print(f"decreases the risk of excess by {pct_decrease:.1f}%")
else:
    print("\nInterpretation: No association between mining and excess")

print("="*80)


#%%

# Positivity check
import matplotlib.pyplot as plt

# Distribution of mining
plt.figure(figsize=(10, 6))
plt.hist(data_std['mining'], bins=50, alpha=0.7, edgecolor='black')
plt.axvline(mining_std_min, color='red', linestyle='--', label=f'Min: {mining_std_min}')
plt.axvline(mining_std_max, color='blue', linestyle='--', label=f'Max: {mining_std_max}')
plt.xlabel('Mining (standardized)')
plt.ylabel('Frequency')
plt.title('Distribution of Mining Exposure')
plt.legend()
plt.show()

# Check if enough observations exist across the range
n_in_range = len(data_std[(data_std['mining'] >= mining_std_min) & 
                          (data_std['mining'] <= mining_std_max)])
print(f"Observations in range [{mining_std_min}, {mining_std_max}]: {n_in_range}")
print(f"Proportion of total: {n_in_range/len(data_std)*100:.2f}%")

#%%