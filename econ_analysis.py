#!/usr/bin/env python3
"""
ECON104 Project 1: Real Estate Price Prediction Analysis
Complete econometric analysis following all assignment requirements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.diagnostic import linear_reset
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directory for figures
import os
os.makedirs('/home/ubuntu/figures', exist_ok=True)
os.makedirs('/home/ubuntu/results', exist_ok=True)

print("="*80)
print("ECON104 PROJECT 1: REAL ESTATE VALUATION ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)

df = pd.read_csv('/home/ubuntu/real_estate_data.csv')

# Rename columns for easier handling
df.columns = ['transaction_date', 'house_age', 'distance_mrt', 
              'convenience_stores', 'latitude', 'longitude', 'price']

print(f"\nDataset shape: {df.shape}")
print(f"Number of observations: {len(df)}")
print(f"Number of variables: {len(df.columns)}")

# ============================================================================
# STEP 2: DATA SPLITTING (80/20 train/test)
# ============================================================================
print("\n" + "="*80)
print("STEP 2e: DATA SPLITTING FOR CROSS-VALIDATION")
print("="*80)

# Set random seed for reproducibility
np.random.seed(42)

# Split data
X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Combine for easier analysis
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print(f"\nTraining set size: {len(train_df)} observations ({len(train_df)/len(df)*100:.1f}%)")
print(f"Test set size: {len(test_df)} observations ({len(test_df)/len(df)*100:.1f}%)")

# Save split data
train_df.to_csv('/home/ubuntu/results/train_data.csv', index=False)
test_df.to_csv('/home/ubuntu/results/test_data.csv', index=False)

# ============================================================================
# STEP 2a: DATASET CITATION
# ============================================================================
print("\n" + "="*80)
print("STEP 2a: DATASET CITATION")
print("="*80)

citation = """
Yeh, I-Cheng (2018). Real Estate Valuation [Dataset]. 
UCI Machine Learning Repository. 
https://doi.org/10.24432/C5J30W

Source: Sindian Dist., New Taipei City, Taiwan
Published in: Applied Soft Computing (2018)
"""
print(citation)

# ============================================================================
# STEP 2b: DATASET SUMMARY
# ============================================================================
print("\n" + "="*80)
print("STEP 2b: DATASET SUMMARY")
print("="*80)

summary_text = """
This dataset contains market historical data of real estate valuation from 
Sindian District, New Taipei City, Taiwan. The dataset includes 414 property 
transactions with information about transaction timing, property characteristics, 
location features, and unit prices.

Variables:
- Transaction Date: Time of transaction (2012-2013)
- House Age: Age of the property in years
- Distance to MRT: Distance to nearest Mass Rapid Transit station (meters)
- Convenience Stores: Number of convenience stores in walking distance
- Latitude: Geographic coordinate (degrees)
- Longitude: Geographic coordinate (degrees)
- Price: House price per unit area (10,000 New Taiwan Dollar/Ping)
  where 1 Ping = 3.3 square meters
"""
print(summary_text)

# ============================================================================
# STEP 2c: DESCRIPTIVE ANALYSIS OF VARIABLES
# ============================================================================
print("\n" + "="*80)
print("STEP 2c: DESCRIPTIVE ANALYSIS OF VARIABLES (TRAINING SET)")
print("="*80)

# Five number summary
print("\nFive Number Summary:")
print(train_df.describe().T[['min', '25%', '50%', '75%', 'max']])

print("\nMean and Standard Deviation:")
print(train_df.describe().T[['mean', 'std']])

# Save descriptive statistics
train_df.describe().to_csv('/home/ubuntu/results/descriptive_stats.csv')

# ============================================================================
# VISUALIZATIONS: Histograms with fitted distributions
# ============================================================================
print("\nCreating histograms with fitted distributions...")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Histograms with Fitted Normal Distributions', fontsize=16, y=1.00)

variables = train_df.columns
for idx, var in enumerate(variables):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    # Plot histogram
    data = train_df[var].dropna()
    n, bins, patches = ax.hist(data, bins=30, density=True, alpha=0.7, 
                                color='skyblue', edgecolor='black')
    
    # Fit and plot normal distribution
    mu, sigma = data.mean(), data.std()
    x = np.linspace(data.min(), data.max(), 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
            label=f'Normal(μ={mu:.2f}, σ={sigma:.2f})')
    
    # Calculate skewness and kurtosis
    skew = stats.skew(data)
    kurt = stats.kurtosis(data)
    
    ax.set_title(f'{var}\nSkew={skew:.2f}, Kurt={kurt:.2f}', fontsize=10)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove extra subplot
if len(variables) < 9:
    fig.delaxes(axes[2, 2])

plt.tight_layout()
plt.savefig('/home/ubuntu/figures/histograms_fitted.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Histograms saved")

# ============================================================================
# VISUALIZATIONS: Box plots
# ============================================================================
print("Creating box plots...")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Box Plots for All Variables', fontsize=16, y=1.00)

for idx, var in enumerate(variables):
    row = idx // 3
    col = idx % 3
    ax = axes[row, col]
    
    data = train_df[var].dropna()
    bp = ax.boxplot(data, vert=True, patch_artist=True)
    
    # Color the box
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    # Calculate outlier statistics
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
    
    ax.set_title(f'{var}\nOutliers: {len(outliers)} ({len(outliers)/len(data)*100:.1f}%)', 
                 fontsize=10)
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3, axis='y')

# Remove extra subplot
if len(variables) < 9:
    fig.delaxes(axes[2, 2])

plt.tight_layout()
plt.savefig('/home/ubuntu/figures/boxplots.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Box plots saved")

# ============================================================================
# VISUALIZATIONS: Correlation matrix
# ============================================================================
print("Creating correlation matrix...")

# Calculate correlation matrix
corr_matrix = train_df.corr()

# Create heatmap
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, 
            cbar_kws={"shrink": 0.8}, mask=mask)
plt.title('Correlation Matrix (Training Set)', fontsize=14, pad=20)
plt.tight_layout()
plt.savefig('/home/ubuntu/figures/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("✓ Correlation matrix saved")

# Save correlation matrix
corr_matrix.to_csv('/home/ubuntu/results/correlation_matrix.csv')

# ============================================================================
# STEP 2d: POSSIBLE VIOLATIONS OF REGRESSION ASSUMPTIONS
# ============================================================================
print("\n" + "="*80)
print("STEP 2d: POSSIBLE VIOLATIONS OF REGRESSION ASSUMPTIONS")
print("="*80)

violations = []

# Check for skewness
print("\nSkewness Analysis:")
for var in train_df.columns:
    skew = stats.skew(train_df[var])
    print(f"{var}: {skew:.3f}", end="")
    if abs(skew) > 1:
        print(" - HIGHLY SKEWED")
        violations.append(f"{var} is highly skewed (skew={skew:.3f})")
    elif abs(skew) > 0.5:
        print(" - Moderately skewed")
        violations.append(f"{var} is moderately skewed (skew={skew:.3f})")
    else:
        print(" - Approximately symmetric")

# Check for outliers
print("\nOutlier Analysis:")
for var in train_df.columns:
    Q1 = train_df[var].quantile(0.25)
    Q3 = train_df[var].quantile(0.75)
    IQR = Q3 - Q1
    outliers = train_df[(train_df[var] < Q1 - 1.5*IQR) | 
                        (train_df[var] > Q3 + 1.5*IQR)][var]
    pct = len(outliers) / len(train_df) * 100
    print(f"{var}: {len(outliers)} outliers ({pct:.1f}%)")
    if pct > 5:
        violations.append(f"{var} has {len(outliers)} outliers ({pct:.1f}%)")

# Check for multicollinearity (preliminary)
print("\nHigh Correlations (|r| > 0.7):")
high_corr_found = False
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.7:
            print(f"{corr_matrix.columns[i]} <-> {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
            violations.append(f"High correlation between {corr_matrix.columns[i]} and {corr_matrix.columns[j]} ({corr_matrix.iloc[i, j]:.3f})")
            high_corr_found = True

if not high_corr_found:
    print("No high correlations found (|r| > 0.7)")

# Save violations summary
with open('/home/ubuntu/results/potential_violations.txt', 'w') as f:
    f.write("POTENTIAL REGRESSION ASSUMPTION VIOLATIONS\n")
    f.write("="*60 + "\n\n")
    if violations:
        for v in violations:
            f.write(f"- {v}\n")
    else:
        f.write("No major violations detected in preliminary analysis.\n")

print("\n✓ Descriptive analysis complete")

# ============================================================================
# STEP 3: BASELINE MULTIPLE LINEAR REGRESSION
# ============================================================================
print("\n" + "="*80)
print("STEP 3: BASELINE MULTIPLE LINEAR REGRESSION MODEL")
print("="*80)

# Prepare data for regression
X_train_sm = sm.add_constant(X_train)
X_test_sm = sm.add_constant(X_test)

# Fit baseline model
baseline_model = sm.OLS(y_train, X_train_sm).fit()

print("\n" + "-"*80)
print("BASELINE MODEL SUMMARY")
print("-"*80)
print(baseline_model.summary())

# Save baseline model summary
with open('/home/ubuntu/results/baseline_model_summary.txt', 'w') as f:
    f.write("BASELINE MULTIPLE LINEAR REGRESSION MODEL\n")
    f.write("="*80 + "\n\n")
    f.write(str(baseline_model.summary()))

print("\n✓ Baseline model fitted")

# ============================================================================
# STEP 4: VIF TEST FOR MULTICOLLINEARITY
# ============================================================================
print("\n" + "="*80)
print("STEP 4: VIF TEST FOR MULTICOLLINEARITY")
print("="*80)

# Calculate VIF for each predictor
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train.columns
vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) 
                   for i in range(X_train.shape[1])]

print("\nVariance Inflation Factors:")
print(vif_data.to_string(index=False))

# Identify variables with high VIF
high_vif = vif_data[vif_data['VIF'] > 5]
print(f"\nVariables with VIF > 5:")
if len(high_vif) > 0:
    print(high_vif.to_string(index=False))
else:
    print("None - No multicollinearity issues detected")

# Save VIF results
vif_data.to_csv('/home/ubuntu/results/vif_analysis.csv', index=False)

# Sequential VIF removal if needed
variables_to_remove = []
X_train_vif = X_train.copy()

while True:
    vif_values = [variance_inflation_factor(X_train_vif.values, i) 
                  for i in range(X_train_vif.shape[1])]
    max_vif = max(vif_values)
    
    if max_vif > 5:
        max_vif_idx = vif_values.index(max_vif)
        var_to_remove = X_train_vif.columns[max_vif_idx]
        print(f"\nRemoving {var_to_remove} (VIF = {max_vif:.2f})")
        variables_to_remove.append(var_to_remove)
        X_train_vif = X_train_vif.drop(columns=[var_to_remove])
    else:
        break

if variables_to_remove:
    print(f"\nVariables removed due to multicollinearity: {variables_to_remove}")
    X_train_reduced = X_train.drop(columns=variables_to_remove)
    X_test_reduced = X_test.drop(columns=variables_to_remove)
    
    # Refit model after VIF removal
    X_train_reduced_sm = sm.add_constant(X_train_reduced)
    model_after_vif = sm.OLS(y_train, X_train_reduced_sm).fit()
    
    print("\n" + "-"*80)
    print("MODEL AFTER VIF-BASED VARIABLE REMOVAL")
    print("-"*80)
    print(model_after_vif.summary())
    
    with open('/home/ubuntu/results/model_after_vif_summary.txt', 'w') as f:
        f.write("MODEL AFTER VIF-BASED VARIABLE REMOVAL\n")
        f.write("="*80 + "\n\n")
        f.write(str(model_after_vif.summary()))
else:
    print("\nNo variables removed - proceeding with all predictors")
    X_train_reduced = X_train.copy()
    X_test_reduced = X_test.copy()
    model_after_vif = baseline_model

print("\n✓ VIF analysis complete")

print("\n" + "="*80)
print("ANALYSIS PART 1 COMPLETE")
print("="*80)
print("\nNext: Run econ_analysis_part2.py for Steps 5-11")

