# ECON104 Project 1: Real Estate Valuation Analysis

**Author**: Benjamin Kirk

## Project Completion Summary

This project presents a comprehensive econometric analysis of real estate prices in the Sindian District of New Taipei City, Taiwan. The analysis follows all 11 steps required by the assignment and includes rigorous statistical testing, model diagnostics, and validation.

## Main Deliverable

**ECON104_Project_1_Report.pdf** (31 pages, 2.6 MB)

This is the primary submission file containing:
- Complete analysis following all 11 required steps
- Professional formatting with clear section headings
- All visualizations (histograms, box plots, correlation matrix, residual diagnostics, prediction plots)
- Statistical test results (VIF, RESET, heteroskedasticity tests)
- Model summaries and interpretations
- Complete Python code in appendix for reproducibility

## Project Highlights

### Dataset
- **Source**: UCI Machine Learning Repository
- **Citation**: Yeh, I-Cheng (2018). Real Estate Valuation [Dataset]
- **Size**: 414 observations, 6 predictors, 1 response variable
- **Split**: 80% training (331 obs), 20% testing (83 obs)

### Key Findings

1. **Multicollinearity Resolution**: Removed 3 highly collinear variables (longitude, transaction_date, latitude) with VIF > 5 million

2. **Final Model Variables**:
   - house_age (linear and squared)
   - distance_mrt (linear and squared)
   - convenience_stores (linear)
   - distance_stores (interaction term)

3. **Model Performance**:
   - Training R²: 0.593
   - Test R²: 0.702
   - Test MAE: 5.06 (approximately 50,600 TWD/Ping)
   - Test RMSE: 7.07

4. **Statistical Tests Conducted**:
   - VIF test for multicollinearity ✓
   - AIC/BIC model selection ✓
   - RESET test for functional form ✓
   - Breusch-Pagan & White tests for heteroskedasticity ✓
   - Residual diagnostics ✓

### Model Interpretation

The final model reveals non-linear relationships between property characteristics and prices:

- **Distance to MRT**: Strong negative effect with diminishing impact at greater distances
- **House Age**: U-shaped relationship (prices decrease with age but at a decreasing rate)
- **Convenience Stores**: Positive effect, but less pronounced for properties far from MRT
- All predictors are statistically significant (p < 0.05)

## Supporting Files

### Analysis Scripts
- `econ_analysis.py` - Steps 1-4 (data loading, descriptive analysis, baseline model, VIF)
- `econ_analysis_part2.py` - Steps 5-11 (model selection, diagnostics, final model, validation)
- `download_data.py` - Dataset acquisition from UCI repository

### Data Files
- `real_estate_data.csv` - Complete dataset (414 observations)
- `results/train_data.csv` - Training set (331 observations)
- `results/test_data.csv` - Test set (83 observations)

### Results and Outputs
- `results/baseline_model_summary.txt`
- `results/model_after_vif_summary.txt`
- `results/selected_model_summary.txt`
- `results/final_model_summary.txt`
- `results/vif_analysis.csv`
- `results/model_selection_results.csv`
- `results/performance_metrics.csv`
- `results/descriptive_stats.csv`
- `results/correlation_matrix.csv`

### Visualizations
- `figures/histograms_fitted.png` - Distribution analysis with fitted normal curves
- `figures/boxplots.png` - Outlier detection
- `figures/correlation_matrix.png` - Variable relationships
- `figures/residual_diagnostics.png` - 4-panel diagnostic plots
- `figures/out_of_sample_predictions.png` - Actual vs predicted prices

## Compliance with Assignment Requirements

✓ All 11 steps completed as specified
✓ Dataset with >5 predictors (6 predictors used)
✓ Proper train/test split (80/20)
✓ All required statistical tests performed
✓ Complete code included in appendix
✓ Professional PDF report format
✓ No errors, warnings, or messages
✓ Comprehensive interpretation and analysis
✓ Economic and statistical significance discussed
✓ Model limitations acknowledged

## Technical Details

**Software**: Python 3.11
**Key Libraries**: pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, statsmodels
**Random Seed**: 42 (for reproducibility)
**Analysis Date**: October 12, 2025

## Conclusion

This project successfully demonstrates the application of econometric techniques to real-world data. The final model provides robust predictions with strong out-of-sample performance (R² = 0.70) and reveals important insights about the drivers of real estate prices in the Taiwanese market. The analysis is comprehensive, methodologically sound, and ready for submission.

