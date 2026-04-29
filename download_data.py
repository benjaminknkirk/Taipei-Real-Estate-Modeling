#!/usr/bin/env python3
"""
Download Real Estate Valuation dataset from UCI ML Repository
"""

from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch dataset
print("Downloading Real Estate Valuation dataset...")
real_estate_valuation = fetch_ucirepo(id=477)

# Data (as pandas dataframes)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

# Combine features and target
df = pd.concat([X, y], axis=1)

# Save to CSV
output_file = '/home/ubuntu/real_estate_data.csv'
df.to_csv(output_file, index=False)

print(f"Dataset saved to: {output_file}")
print(f"Shape: {df.shape}")
print(f"\nColumn names:")
print(df.columns.tolist())
print(f"\nFirst few rows:")
print(df.head())
print(f"\nDataset info:")
print(df.info())
print(f"\nBasic statistics:")
print(df.describe())

