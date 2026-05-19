import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


# Week 1: Load & Combine

sold_files = sorted(glob.glob('raw/CRMLSSold*.csv'))
print(f"Sold files found: {len(sold_files)}")

print("\n--- Individual file row counts (Sold) ---")
sold_dfs = []
for f in sold_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"{f}: {len(df)} rows")
    sold_dfs.append(df)

sold_raw = pd.concat(sold_dfs, ignore_index=True)
print(f"\nRows after combining (sold): {len(sold_raw)}")


# Week 1: Filter to Residential

print("\n--- PropertyType frequency before filter (Sold) ---")
print(sold_raw['PropertyType'].value_counts())

sold = sold_raw[sold_raw['PropertyType'] == 'Residential'].copy()
print(f"\nRows after filtering to Residential (sold): {len(sold)}")

print("\n--- PropertyType frequency after filter (Sold) ---")
print(sold['PropertyType'].value_counts())