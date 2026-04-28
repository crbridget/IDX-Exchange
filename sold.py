import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# ── 1. Load & Combine ─────────────────────────────────────────────────────────

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

# ── 2. Filter to Residential ──────────────────────────────────────────────────

print("\n--- PropertyType frequency before filter (Sold) ---")
print(sold_raw['PropertyType'].value_counts())

sold = sold_raw[sold_raw['PropertyType'] == 'Residential'].copy()
print(f"\nRows after filtering to Residential (sold): {len(sold)}")

print("\n--- PropertyType frequency after filter (Sold) ---")
print(sold['PropertyType'].value_counts())

# ── 3. Dataset Understanding ──────────────────────────────────────────────────

print("\n--- Sold Columns ---")
print(sold.columns.tolist())

print("\n--- Sold Head ---")
print(sold.head())

print(f"\nSold shape: {sold.shape}")

print("\n--- Sold Data Types ---")
print(sold.dtypes)

# ── 4. Missing Value Analysis ─────────────────────────────────────────────────

def missing_summary(df, name):
    """Calculate missing counts and percentages per column, flag columns >90% missing"""
    missing = pd.DataFrame({
        'missing_count': df.isnull().sum(),
        'missing_percent': (df.isnull().sum() / len(df) * 100).round(2)
    }).sort_values('missing_percent', ascending=False)

    print(f"\n--- {name} Missing Value Summary ---")
    print(missing)

    flagged = missing[missing['missing_percent'] > 90]
    print(f"\n--- {name} Columns Flagged >90% Missing ---")
    print(flagged)

    return flagged.index.tolist()

sold_flagged = missing_summary(sold, 'Sold')

# ── 5. Drop High-Missing Columns ──────────────────────────────────────────────

core_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
    'CloseDate', 'ListingContractDate', 'PurchaseContractDate',
    'CountyOrParish', 'City', 'PostalCode', 'PropertyType',
    'PropertySubType', 'Latitude', 'Longitude'
]

to_drop = [col for col in sold_flagged if col not in core_fields]
print(f"\n--- Sold Columns Dropped ---")
print(to_drop)

sold_filtered = sold.drop(columns=to_drop)
print(f"\nSold columns before: {len(sold.columns)}, after: {len(sold_filtered.columns)}")

# ── 6. EDA Questions ──────────────────────────────────────────────────────────

# Property type share already shown above (from raw data before filter)

# Median and average close price
print("\n--- Close Price Stats ---")
print(f"Median ClosePrice: ${sold_filtered['ClosePrice'].median():,.0f}")
print(f"Mean ClosePrice: ${sold_filtered['ClosePrice'].mean():,.0f}")

# Days on Market distribution
print("\n--- Days on Market Distribution ---")
print(sold_filtered['DaysOnMarket'].describe(percentiles=[.25, .50, .75, .90, .95]))

# % sold above vs below list price
sold_filtered['above_list'] = sold_filtered['ClosePrice'] > sold_filtered['ListPrice']
above = sold_filtered['above_list'].sum()
below = (~sold_filtered['above_list']).sum()
total = len(sold_filtered)
print(f"\n--- Sold Above vs Below List Price ---")
print(f"Above list: {above / total * 100:.1f}%")
print(f"Below list: {below / total * 100:.1f}%")

# Date consistency issues
sold_filtered['CloseDate'] = pd.to_datetime(sold_filtered['CloseDate'])
sold_filtered['ListingContractDate'] = pd.to_datetime(sold_filtered['ListingContractDate'])
close_before_listing = (sold_filtered['CloseDate'] < sold_filtered['ListingContractDate']).sum()
print(f"\n--- Date Consistency ---")
print(f"Records where CloseDate before ListingContractDate: {close_before_listing}")

# Top 10 counties by median close price
print("\n--- Top 10 Counties by Median ClosePrice ---")
print(sold_filtered.groupby('CountyOrParish')['ClosePrice']
      .median()
      .sort_values(ascending=False)
      .head(10)
      .apply(lambda x: f"${x:,.0f}"))

# ── 7. Numeric Distribution Review ───────────────────────────────────────────

os.makedirs('plots', exist_ok=True)

numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt'
]

for field in numeric_fields:
    if field not in sold_filtered.columns:
        print(f"\n--- Sold: {field} --- MISSING FROM DATASET")
        continue

    col = sold_filtered[field].dropna()

    print(f"\n--- Sold: {field} ---")
    print(col.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

    # IQR outlier count
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((col < lower) | (col > upper)).sum()
    print(f"Outliers (IQR method): {outliers} ({outliers / len(col) * 100:.2f}%)")
    print(f"Lower bound: {lower:.2f} | Upper bound: {upper:.2f}")

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Sold: {field}", fontsize=13)

    # Histogram - capped at 99th percentile for readability
    q99 = col.quantile(0.99)
    axes[0].hist(col[col <= q99], bins=50, edgecolor='black', color='steelblue')
    axes[0].set_title('Histogram (capped at 99th percentile)')
    axes[0].set_xlabel(field)
    axes[0].set_ylabel('Frequency')

    # Boxplot - capped at 99th percentile
    col_capped = col[col <= q99]
    axes[1].boxplot(col_capped, vert=False)
    axes[1].set_title('Boxplot (capped at 99th percentile)')
    axes[1].set_xlabel(field)

    plt.tight_layout()
    plt.savefig(f"plots/Sold_{field}_distribution.png")
    plt.show()

# ── 8. Save Filtered CSV ──────────────────────────────────────────────────────

os.makedirs('data', exist_ok=True)
sold_filtered.to_csv('data/sold_filtered.csv', index=False)
print("\nSold filtered file saved!")

# ── 9. Mortgage Rate Merge ────────────────────────────────────────────────────

# Fetch 30-year fixed mortgage rate data from FRED
url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=MORTGAGE30US"
mortgage = pd.read_csv(url, parse_dates=['observation_date'])
mortgage.columns = ['date', 'rate_30yr_fixed']

# Resample weekly rates to monthly averages
mortgage['year_month'] = mortgage['date'].dt.to_period('M')
mortgage_monthly = (
    mortgage.groupby('year_month')['rate_30yr_fixed']
    .mean()
    .reset_index()
)

# Key off CloseDate for sold dataset
sold_filtered['CloseDate'] = pd.to_datetime(sold_filtered['CloseDate'])
sold_filtered['year_month'] = sold_filtered['CloseDate'].dt.to_period('M')

# Merge
sold_with_rates = sold_filtered.merge(mortgage_monthly, on='year_month', how='left')

# Validate merge
print(f"\nUnmatched rows (rate is null): {sold_with_rates['rate_30yr_fixed'].isnull().sum()}")

# Preview
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head())

# Save
sold_with_rates.to_csv('data/sold_with_rates.csv', index=False)
print("\nSold with rates file saved!")

# ── 10. Data Cleaning ─────────────────────────────────────────────────────────

print(f"\nRows before cleaning: {len(sold_with_rates)}")
print(f"Columns before cleaning: {len(sold_with_rates.columns)}")

cleaning = sold_with_rates.copy()

# ── 10a. Convert Date Fields to Datetime ──────────────────────────────────────

date_fields = ['CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate']

for col in date_fields:
    if col in cleaning.columns:
        cleaning[col] = pd.to_datetime(cleaning[col], errors='coerce')

print("\n--- Date Fields Converted to Datetime ---")
print(cleaning[date_fields].dtypes)

# ── 10b. Remove Unnecessary or Redundant Columns ─────────────────────────────

cols_before = len(cleaning.columns)

cols_to_drop = []

# year_month was a helper column for the mortgage rate merge, not needed for analysis
if 'year_month' in cleaning.columns:
    cols_to_drop.append('year_month')

# above_list is a derived boolean we created during EDA, ClosePrice/ListPrice already in dataset
if 'above_list' in cleaning.columns:
    cols_to_drop.append('above_list')

cleaning = cleaning.drop(columns=cols_to_drop)

print(f"\n--- Columns Dropped (Redundant/Helper) ---")
for col in cols_to_drop:
    print(f"  {col}")
print(f"\nColumns before: {cols_before} | after: {len(cleaning.columns)}")

# ── 10c. Ensure Numeric Fields Are Properly Typed ─────────────────────────────

numeric_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
    'DaysOnMarket', 'YearBuilt', 'Latitude', 'Longitude'
]

for col in numeric_fields:
    if col in cleaning.columns:
        cleaning[col] = pd.to_numeric(cleaning[col], errors='coerce')

print("\n--- Numeric Fields dtype check ---")
print(cleaning[numeric_fields].dtypes)

# ── 10d. Remove / Flag Invalid Numeric Values ─────────────────────────────────

rows_before = len(cleaning)

# ClosePrice: $0 is not a valid sale
invalid_close_price = cleaning['ClosePrice'] <= 0
print(f"\nClosePrice <= 0: {invalid_close_price.sum()} rows removed")
cleaning = cleaning[~invalid_close_price]

# ListPrice: $0 or negative is not valid
invalid_list_price = cleaning['ListPrice'] <= 0
print(f"ListPrice <= 0: {invalid_list_price.sum()} rows removed")
cleaning = cleaning[~invalid_list_price]

# DaysOnMarket: negative is impossible
invalid_dom = cleaning['DaysOnMarket'] < 0
print(f"DaysOnMarket < 0: {invalid_dom.sum()} rows removed")
cleaning = cleaning[~invalid_dom]

# LivingArea: 0 sqft is not a valid home
invalid_living = cleaning['LivingArea'] <= 0
print(f"LivingArea <= 0: {invalid_living.sum()} rows removed")
cleaning = cleaning[~invalid_living]

# YearBuilt: before 1800 is likely a data entry error
invalid_year = cleaning['YearBuilt'] < 1800
print(f"YearBuilt < 1800: {invalid_year.sum()} rows removed")
cleaning = cleaning[~invalid_year]

# YearBuilt: after current year is impossible
invalid_future_year = cleaning['YearBuilt'] > 2026
print(f"YearBuilt > 2026: {invalid_future_year.sum()} rows removed")
cleaning = cleaning[~invalid_future_year]

# BedroomsTotal: negative is not valid
invalid_beds = cleaning['BedroomsTotal'] < 0
print(f"BedroomsTotal < 0: {invalid_beds.sum()} rows removed")
cleaning = cleaning[~invalid_beds]

# BathroomsTotalInteger: negative is not valid
invalid_baths = cleaning['BathroomsTotalInteger'] < 0
print(f"BathroomsTotalInteger < 0: {invalid_baths.sum()} rows removed")
cleaning = cleaning[~invalid_baths]

print(f"\nRows before invalid value removal: {rows_before}")
print(f"Rows after invalid value removal: {len(cleaning)}")
print(f"Rows removed: {rows_before - len(cleaning)}")

# ── 10e. Handle Missing Values ────────────────────────────────────────────────

rows_before = len(cleaning)

# Drop rows missing core fields that are essential for any analysis
core_required = ['ClosePrice', 'ListPrice', 'CloseDate', 'LivingArea']
cleaning = cleaning.dropna(subset=core_required)

print(f"\nRows dropped for missing core fields {core_required}: {rows_before - len(cleaning)}")
print(f"Rows remaining: {len(cleaning)}")

# Fill missing BedroomsTotal and BathroomsTotalInteger with median (minor missing, reasonable imputation)
for col in ['BedroomsTotal', 'BathroomsTotalInteger']:
    if col in cleaning.columns:
        median_val = cleaning[col].median()
        missing_count = cleaning[col].isnull().sum()
        cleaning[col] = cleaning[col].fillna(median_val)
        print(f"{col}: {missing_count} missing values filled with median ({median_val})")

# All other fields: leave as NaN — dropping would lose too many rows
# and these fields are not required for core analysis
print("\nAll other missing fields left as NaN (not required for core analysis)")

# ── 10f. Final Shape ──────────────────────────────────────────────────────────

print(f"\n--- Cleaning Summary ---")
print(f"Rows before: {len(sold_with_rates)} | Rows after: {len(cleaning)}")
print(f"Columns before: {len(sold_with_rates.columns)} | Columns after: {len(cleaning.columns)}")

# ── 10g. Save ─────────────────────────────────────────────────────────────────

cleaning.to_csv('data/sold_cleaned.csv', index=False)
print("\nSold cleaned file saved!")