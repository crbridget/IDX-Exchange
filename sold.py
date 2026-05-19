import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


# Week 1: Load & Combine

# load sold files
sold_files = sorted(glob.glob('raw/CRMLSSold*.csv'))
print(f"Sold files found: {len(sold_files)}")

# print number of rows in each file
print("\n--- Individual file row counts (Sold) ---")
sold_dfs = []
for f in sold_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"{f}: {len(df)} rows")
    sold_dfs.append(df)

# combine all sold files and print rows
sold_raw = pd.concat(sold_dfs, ignore_index=True)
print(f"\nRows after combining (sold): {len(sold_raw)}")


# Week 1: Filter to Residential

# print number of each distinct property types before filtering
print("\n--- PropertyType frequency before filter (Sold) ---")
print(sold_raw['PropertyType'].value_counts())

# print number of rows after filtering by residential 
sold = sold_raw[sold_raw['PropertyType'] == 'Residential'].copy()
print(f"\nRows after filtering to Residential (sold): {len(sold)}")

# verify that there is only one property type
print("\n--- PropertyType frequency after filter (Sold) ---")
print(sold['PropertyType'].value_counts())


# Week 2: Dataset Understanding

print("\n--- Sold Columns ---")
print(sold.columns.tolist())

print("\n--- Sold Head ---")
print(sold.head())

print(f"\nSold shape: {sold.shape}")

print("\n--- Sold Data Types ---")
print(sold.dtypes)


# Week 2: Missing Value Analysis

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


# Week 2: Drop High-Missing Columns

core_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
    'CloseDate',  'ListingContractDate', 'PurchaseContractDate',
    'CountyOrParish', 'City', 'PostalCode', 'PropertyType',
    'PropertySubType', 'Latitude', 'Longitude'
]

to_drop = [col for col in sold_flagged if col not in core_fields]
print(f"\n--- Sold Columns Dropped ---")
print(to_drop)

sold_filtered = sold.drop(columns=to_drop)
print(f"\nSold columns before: {len(sold.columns)}, after: {len(sold_filtered.columns)}")


# Week 2: EDA Questions

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


# Week 2: Numeric Distribution Review

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

    q99 = col.quantile(0.99)
    axes[0].hist(col[col <= q99], bins=50, edgecolor='black', color='steelblue')
    axes[0].set_title('Histogram (capped at 99th percentile)')
    axes[0].set_xlabel(field)
    axes[0].set_ylabel('Frequency')

    col_capped = col[col <= q99]
    axes[1].boxplot(col_capped, vert=False)
    axes[1].set_title('Boxplot (capped at 99th percentile)')
    axes[1].set_xlabel(field)

    plt.tight_layout()
    plt.savefig(f"plots/Sold_{field}_distribution.png")
    # plt.show()


# Week 2: Save Filtered CSV

os.makedirs('data', exist_ok=True)
sold_filtered.to_csv('data/sold_filtered.csv', index=False)
print("\nSold filtered file saved!")


# Week 3: Mortgage Rate Merge

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
print(sold_with_rates[['CloseDate', 'year_month', 'ClosePrice', 'rate_30yr_fixed']].head())

sold_with_rates.to_csv('data/sold_with_rates.csv', index=False)
print("\nSold with rates file saved!")


# Week 4: Data Cleaning

print(f"\nRows before cleaning: {len(sold_with_rates)}")
print(f"Columns before cleaning: {len(sold_with_rates.columns)}")

cleaning = sold_with_rates.copy()


# Week 4: Convert Date Fields to Datetime

date_fields = ['CloseDate', 'ListingContractDate', 'PurchaseContractDate', 'ContractStatusChangeDate']

for col in date_fields:
    if col in cleaning.columns:
        cleaning[col] = pd.to_datetime(cleaning[col], errors='coerce')

print("\n--- Date Fields Converted to Datetime ---")
print(cleaning[date_fields].dtypes)


# Week 4: Remove Unnecessary or Redundant Columns

cols_before = len(cleaning.columns)

cols_to_drop = {
    # Helper columns from earlier steps
    'year_month':               'mortgage rate merge helper, not needed for analysis',
    'above_list':               'EDA derived boolean, ClosePrice and ListPrice already in dataset',

    # Duplicate key fields
    'ListingKeyNumeric':        'duplicate of ListingKey in numeric format',
    'LotSizeSquareFeet':        'duplicate of LotSizeAcres in different units',

    # Duplicate lat/lon fields
    'latfilled':                'duplicate of Latitude, filled version from prior processing step',
    'lonfilled':                'duplicate of Longitude, filled version from prior processing step',

    # Agent metadata (not useful for market analysis)
    'ListAgentFirstName':       'agent metadata, not relevant to market analysis',
    'ListAgentLastName':        'agent metadata, not relevant to market analysis',
    'ListAgentFullName':        'agent metadata, not relevant to market analysis',
    'ListAgentEmail':           'agent metadata, not relevant to market analysis',
    'ListAgentAOR':             'agent metadata, not relevant to market analysis',
    'CoListAgentFirstName':     'agent metadata, not relevant to market analysis',
    'CoListAgentLastName':      'agent metadata, not relevant to market analysis',
    'BuyerAgentFirstName':      'agent metadata, not relevant to market analysis',
    'BuyerAgentLastName':       'agent metadata, not relevant to market analysis',
    'BuyerAgentMlsId':          'agent metadata, not relevant to market analysis',
    'BuyerAgentAOR':            'agent metadata, not relevant to market analysis',

    # Office metadata — ListOfficeName and BuyerOfficeName kept for Week 6 competitive intelligence
    'CoListOfficeName':         'office metadata, not relevant to market analysis',
    'BuyerOfficeAOR':           'office metadata, not relevant to market analysis',

    # Single-value columns after filtering
    'MlsStatus':                'all rows are Sold after filtering',
}

existing_drops = {col: reason for col, reason in cols_to_drop.items() if col in cleaning.columns}
cleaning = cleaning.drop(columns=list(existing_drops.keys()))

print(f"\n--- Columns Dropped (Redundant/Helper) ---")
for col, reason in existing_drops.items():
    print(f"  {col}: {reason}")
print(f"\nColumns before: {cols_before} | after: {len(cleaning.columns)}")


# Week 4: Ensure Numeric Fields Are Properly Typed

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


# Week 4: Remove / Flag Invalid Numeric Values

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


# Week 4: Handle Missing Values

rows_before = len(cleaning)

# Drop rows missing core fields that are essential for any analysis
core_required = ['ClosePrice', 'ListPrice', 'CloseDate', 'LivingArea']
cleaning = cleaning.dropna(subset=core_required)

print(f"\nRows dropped for missing core fields {core_required}: {rows_before - len(cleaning)}")
print(f"Rows remaining: {len(cleaning)}")

# Fill missing BedroomsTotal and BathroomsTotalInteger with median
for col in ['BedroomsTotal', 'BathroomsTotalInteger']:
    if col in cleaning.columns:
        median_val = cleaning[col].median()
        missing_count = cleaning[col].isnull().sum()
        cleaning[col] = cleaning[col].fillna(median_val)
        print(f"{col}: {missing_count} missing values filled with median ({median_val})")

print("\nAll other missing fields left as NaN (not required for core analysis)")

print(f"\n--- Week 4 Cleaning Summary ---")
print(f"Rows before: {len(sold_with_rates)} | Rows after: {len(cleaning)}")
print(f"Columns before: {len(sold_with_rates.columns)} | Columns after: {len(cleaning.columns)}")

cleaning.to_csv('data/sold_cleaned.csv', index=False)
print("\nSold cleaned file saved!")


# Week 5: Date Consistency Flags
# Rule: ListingContractDate < PurchaseContractDate < CloseDate
# Records are FLAGGED, not dropped — analysts decide how to handle them

print("\n" + "─" * 65)
print("Week 5: Date Consistency Flags")
print("─" * 65)

# Confirm datetime types after CSV round-trip
for col in ['CloseDate', 'ListingContractDate', 'PurchaseContractDate']:
    if col in cleaning.columns:
        cleaning[col] = pd.to_datetime(cleaning[col], errors='coerce')
        print(f"  {col}: dtype confirmed → {cleaning[col].dtype}")

# listing_after_close_flag
# A property cannot close before it was listed — indicates a data entry error
if 'ListingContractDate' in cleaning.columns and 'CloseDate' in cleaning.columns:
    cleaning['listing_after_close_flag'] = (
        cleaning['ListingContractDate'] >= cleaning['CloseDate']
    )
    count = cleaning['listing_after_close_flag'].sum()
    pct   = count / len(cleaning) * 100
    print(f"\n  listing_after_close_flag   : {count:,} records ({pct:.2f}%)")
    print(f"    - ListingContractDate >= CloseDate (impossible timeline)")
else:
    cleaning['listing_after_close_flag'] = np.nan
    print("\n  listing_after_close_flag   : SKIPPED (required columns missing)")

# purchase_after_close_flag
# The contract must be signed before the sale can close
if 'PurchaseContractDate' in cleaning.columns and 'CloseDate' in cleaning.columns:
    both_present = cleaning['PurchaseContractDate'].notna() & cleaning['CloseDate'].notna()
    cleaning['purchase_after_close_flag'] = False
    cleaning.loc[both_present, 'purchase_after_close_flag'] = (
        cleaning.loc[both_present, 'PurchaseContractDate']
        >= cleaning.loc[both_present, 'CloseDate']
    )
    count = cleaning['purchase_after_close_flag'].sum()
    pct   = count / len(cleaning) * 100
    print(f"\n  purchase_after_close_flag  : {count:,} records ({pct:.2f}%)")
    print(f"    - PurchaseContractDate >= CloseDate (impossible timeline)")
else:
    cleaning['purchase_after_close_flag'] = np.nan
    print("\n  purchase_after_close_flag  : SKIPPED (required columns missing)")

# negative_timeline_flag
# Union of all date ordering violations — single filter for any date issue
cleaning['negative_timeline_flag'] = (
    cleaning.get('listing_after_close_flag',  False) |
    cleaning.get('purchase_after_close_flag', False)
)
count = cleaning['negative_timeline_flag'].sum()
pct   = count / len(cleaning) * 100
print(f"\n  negative_timeline_flag     : {count:,} records ({pct:.2f}%)")
print(f"    - Any date ordering violation (union of both flags above)")


# Week 5: Geographic Data Checks
# California bounding box: Lat 32.5–42.0 | Lon -124.5 to -114.0
# Records are FLAGGED, not dropped

print("\n" + "─" * 65)
print("Week 5: Geographic Data Checks")
print("─" * 65)

CA_LAT_MIN, CA_LAT_MAX =  32.5,   42.0
CA_LON_MIN, CA_LON_MAX = -124.5, -114.0

# missing_coords_flag
# Records with null Latitude or Longitude cannot be mapped or used spatially
cleaning['missing_coords_flag'] = (
    cleaning['Latitude'].isna() | cleaning['Longitude'].isna()
)
count = cleaning['missing_coords_flag'].sum()
pct   = count / len(cleaning) * 100
print(f"\n  missing_coords_flag        : {count:,} records ({pct:.2f}%)")
print(f"    - Latitude or Longitude is null")

# zero_coords_flag
# Zero is a sentinel placeholder written by some MLS export tools — not a valid CA location
cleaning['zero_coords_flag'] = (
    (cleaning['Latitude']  == 0) |
    (cleaning['Longitude'] == 0)
)
count = cleaning['zero_coords_flag'].sum()
pct   = count / len(cleaning) * 100
print(f"\n  zero_coords_flag           : {count:,} records ({pct:.2f}%)")
print(f"    - Latitude = 0 or Longitude = 0 (sentinel null value)")

# positive_longitude_flag
# All CA longitudes are negative (west of prime meridian); positive means sign was dropped
cleaning['positive_longitude_flag'] = cleaning['Longitude'] > 0
count = cleaning['positive_longitude_flag'].sum()
pct   = count / len(cleaning) * 100
print(f"\n  positive_longitude_flag    : {count:,} records ({pct:.2f}%)")
print(f"    - Longitude > 0 (California coordinates must be negative)")

# out_of_state_flag
# Coordinates outside CA bounding box; only evaluated on non-null, non-zero values
valid_coords = (
    cleaning['Latitude'].notna()  & (cleaning['Latitude']  != 0) &
    cleaning['Longitude'].notna() & (cleaning['Longitude'] != 0)
)
cleaning['out_of_state_flag'] = False
cleaning.loc[valid_coords, 'out_of_state_flag'] = (
    (cleaning.loc[valid_coords, 'Latitude']  < CA_LAT_MIN) |
    (cleaning.loc[valid_coords, 'Latitude']  > CA_LAT_MAX) |
    (cleaning.loc[valid_coords, 'Longitude'] < CA_LON_MIN) |
    (cleaning.loc[valid_coords, 'Longitude'] > CA_LON_MAX)
)
count = cleaning['out_of_state_flag'].sum()
pct   = count / len(cleaning) * 100
print(f"\n  out_of_state_flag          : {count:,} records ({pct:.2f}%)")
print(f"    - Coordinates outside CA bounding box")
print(f"       Lat [{CA_LAT_MIN}, {CA_LAT_MAX}] | Lon [{CA_LON_MIN}, {CA_LON_MAX}]")

# invalid_coords_flag
# Master geographic flag — union of all coordinate issues
cleaning['invalid_coords_flag'] = (
    cleaning['missing_coords_flag']     |
    cleaning['zero_coords_flag']        |
    cleaning['positive_longitude_flag'] |
    cleaning['out_of_state_flag']
)
count = cleaning['invalid_coords_flag'].sum()
pct   = count / len(cleaning) * 100
print(f"\n  invalid_coords_flag        : {count:,} records ({pct:.2f}%)")
print(f"    - Any geographic issue (union of all geo flags)")


# Week 5: Final Summary & Save

print("\n" + "=" * 65)
print("Week 5: Final Summary")
print("=" * 65)

print(f"\n  DATE CONSISTENCY SUMMARY:")
print(f"    listing_after_close_flag   : {cleaning['listing_after_close_flag'].sum():,}")
print(f"    purchase_after_close_flag  : {cleaning['purchase_after_close_flag'].sum():,}")
print(f"    negative_timeline_flag     : {cleaning['negative_timeline_flag'].sum():,}")

print(f"\n  GEOGRAPHIC DATA QUALITY SUMMARY:")
print(f"    missing_coords_flag        : {cleaning['missing_coords_flag'].sum():,}")
print(f"    zero_coords_flag           : {cleaning['zero_coords_flag'].sum():,}")
print(f"    positive_longitude_flag    : {cleaning['positive_longitude_flag'].sum():,}")
print(f"    out_of_state_flag          : {cleaning['out_of_state_flag'].sum():,}")
print(f"    invalid_coords_flag        : {cleaning['invalid_coords_flag'].sum():,}")

clean_geo_pct  = (1 - cleaning['invalid_coords_flag'].mean()) * 100
clean_date_pct = (1 - cleaning['negative_timeline_flag'].mean()) * 100
print(f"\n  Records with clean coordinates : {clean_geo_pct:.1f}%")
print(f"  Records with clean date order  : {clean_date_pct:.1f}%")
print(f"\n  Final shape: {len(cleaning):,} rows × {len(cleaning.columns)} columns")

cleaning.to_csv('data/sold_final.csv', index=False)
print("\n  Saved: data/sold_final.csv")
print("=" * 65)


# Week 6: Feature Engineering and Market Metrics

# ensure date fields are datetime
for col in ['ListingContractDate', 'PurchaseContractDate', 'CloseDate']:
    if col in cleaning.columns:
        cleaning[col] = pd.to_datetime(cleaning[col], errors='coerce')

# Price Ratio
cleaning['price_ratio'] = cleaning['ClosePrice'] / cleaning['OriginalListPrice']

# Price Per Sq Ft
cleaning['price_per_sqft'] = cleaning['ClosePrice'] / cleaning['LivingArea']

# Year / Month / YrMo
cleaning['year'] = cleaning['CloseDate'].dt.year
cleaning['month'] = cleaning['CloseDate'].dt.month
cleaning['YrMo'] = cleaning['CloseDate'].dt.to_period('M')

# Close to Original List Ratio
cleaning['close_to_og_list'] = cleaning['ClosePrice'] / cleaning['OriginalListPrice']

# Days from Listing to Contract
cleaning['days_on_market'] = (cleaning['PurchaseContractDate'] - cleaning['ListingContractDate']).dt.days

# Days from Contract to Close
cleaning['contract_to_close'] = (cleaning['CloseDate'] - cleaning['PurchaseContractDate']).dt.days

# Sample output showing all engineered columns populated
print("Engineered Metrics Sample")
engineered_cols = [
    'ClosePrice', 'OriginalListPrice', 'LivingArea',
    'price_ratio', 'close_to_og_list', 'price_per_sqft',
    'days_on_market', 'contract_to_close',
    'year', 'month', 'YrMo'
]
pd.set_option('display.max_columns', None)
print(cleaning[engineered_cols].head(10).to_string(index=False))

# Segment: PropertyType and PropertySubType
prop_summary = (
    cleaning
    .groupby(['PropertyType', 'PropertySubType'], observed=True)
    .agg(
        median_close_price = ('ClosePrice',    'median'),
        avg_price_per_sqft = ('price_per_sqft','mean'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ClosePrice',    'count'),
    )
    .reset_index()
    .sort_values('count', ascending=False)
)
print("\nSegment: PropertyType and PropertySubType")
print(prop_summary.head(10).to_string(index=False))
prop_summary.to_csv('data/property_type_summary.csv', index=False)

# Segment: CountyOrParish and MLSAreaMajor
geo_summary = (
    cleaning
    .groupby(['CountyOrParish', 'MLSAreaMajor'], observed=True)
    .agg(
        median_close_price = ('ClosePrice',    'median'),
        avg_price_per_sqft = ('price_per_sqft','mean'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ClosePrice',    'count'),
    )
    .reset_index()
    .sort_values('median_close_price', ascending=False)
)
print("\nSegment: CountyOrParish and MLSAreaMajor")
print(geo_summary.head(10).to_string(index=False))
geo_summary.to_csv('data/county_mls_summary.csv', index=False)

# Segment: ListOfficeName and BuyerOfficeName
office_summary = (
    cleaning
    .groupby(['ListOfficeName', 'BuyerOfficeName'], observed=True)
    .agg(
        median_close_price = ('ClosePrice',    'median'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ClosePrice',    'count'),
    )
    .reset_index()
    .sort_values('count', ascending=False)
)
print("\nSegment: ListOfficeName and BuyerOfficeName")
print(office_summary.head(10).to_string(index=False))
office_summary.to_csv('data/office_summary.csv', index=False)

# Save
cleaning.to_csv('data/sold_features.csv', index=False)
print("\nFeatures saved, data/sold_features.csv")


# Week 7: Outlier Detection and Data Quality

df = cleaning.copy()

# stats before filtering
before_size = len(df)
before_medians = {field: df[field].median() for field in ["ClosePrice", "LivingArea", "DaysOnMarket"]}

print(f"\nRows entering Week 7: {before_size:,}")

# flag extreme values using IQR + percentiles
# ClosePrice - 99th percentile as hard upper bound (CA has very high prices)
# Living Are - 99th percentile as hard upper bound; IQR too aggressive for large but legitimate homes (flagged 4.4% at 1.5x)
# DaysOnMarket - looser 3.0 multiplier (long market times are real, not errors)

iqr_config = {
        "ClosePrice":   {"multiplier": 1.5, "use_percentile_upper": True},
        "LivingArea":   {"multiplier": 1.5, "use_percentile_upper": True},
        "DaysOnMarket": {"multiplier": 3.0, "use_percentile_upper": False},
}

for field, config in iqr_config.items():
    q1 = df[field].quantile(0.25)
    q3 = df[field].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - config['multiplier'] * iqr
    p99 = df[field].quantile(0.99)
    p01 = df[field].quantile(0.01)

    # ClosePrice: use 99th percentile as upper bound instead of IQR
    # IQR is too agressive for skewed CA home prices
    if config['use_percentile_upper']:
        upper = p99
    else:
        upper = q3 + config["multiplier"] * iqr


    df[f"{field}_outlier"] = ~df[field].between(lower, upper)
    n_flagged = df[f"{field}_outlier"].sum()

    print(f"\n  {field}")
    print(f"    IQR multiplier : {config['multiplier']}  |  upper bound: {'99th percentile' if config['use_percentile_upper'] else 'IQR'}")
    print(f"    IQR bounds   : lower={lower:,.2f}  upper={upper:,.2f}")
    print(f"    Percentiles  : 1st={p01:,.2f}  99th={p99:,.2f}")
    print(f"    Flagged      : {n_flagged:,} records ({n_flagged / len(df) * 100:.1f}%)")
print()

# Apply business rules as additional flag columns

# ClosePrice <= 0: not a valid sale
df['invalid_close_price_flag'] = df['ClosePrice'] <= 0
print(f"  invalid_close_price_flag  : {df['invalid_close_price_flag'].sum():,} records (ClosePrice <= 0)")

# LivingArea <= 0: not a valid home
df['invalid_living_area_flag'] = df['LivingArea'] <= 0
print(f"  invalid_living_area_flag  : {df['invalid_living_area_flag'].sum():,} records (LivingArea <= 0)")

# DaysOnMarket < 0: impossible
df['invalid_dom_flag'] = df['DaysOnMarket'] < 0
print(f"  invalid_dom_flag          : {df['invalid_dom_flag'].sum():,} records (DaysOnMarket < 0)")

# price_per_sqft below $10 or above $10,000 
df['invalid_price_per_sqft_flag'] = (
    df['price_per_sqft'] < 10) | (df['price_per_sqft'] > 10000
)
print(f"  invalid_price_per_sqft_flag: {df['invalid_price_per_sqft_flag'].sum():,} records (price_per_sqft < $10 or > $10,000)")
 
# price_ratio below 0.5 or above 2.0
# close price was less than half or more than double the original list price, extremely rare and likely an error
df['invalid_price_ratio_flag'] = (
    df['price_ratio'] < 0.5) | (df['price_ratio'] > 2.0
)
print(f"  invalid_price_ratio_flag  : {df['invalid_price_ratio_flag'].sum():,} records (price_ratio < 0.5 or > 2.0)")


# save full flagged dataset
df.to_csv('data/sold_flagged.csv', index=False)
print(f"\nFlagged dataset saved, data/sold_flagged.csv  ({len(df):,} rows)")

# build clean filtered dataset
# exclude rows flagged by any IQR or business rule flag
iqr_flag_cols      = [f"{field}_outlier" for field in iqr_config.keys()]
biz_rule_flag_cols = [
    'invalid_close_price_flag',
    'invalid_living_area_flag',
    'invalid_dom_flag',
    'invalid_price_per_sqft_flag',
    'invalid_price_ratio_flag',
]

is_any_flag = df[iqr_flag_cols + biz_rule_flag_cols].any(axis=1)
clean_df    = df[~is_any_flag].copy()

clean_df.to_csv('data/sold_clean.csv', index=False)
print(f"Clean dataset saved, data/sold_clean.csv  ({len(clean_df):,} rows)")


# Print comparison

after_size = len(clean_df)
removed    = before_size - after_size
pct        = removed / before_size * 100

print(f"""
Dataset Size
------------
Before : {before_size:,} records
After  : {after_size:,} records
Removed: {removed:,} ({pct:.1f}%)
 
Median Values Before vs After
------------------------------""")

for field in iqr_config.keys():
    b = before_medians[field]
    a = clean_df[field].median()
    change    = a - b
    direction = "up" if change > 0 else "down"
    print(f"  {field:<20}  Before: {b:>12,.2f}   After: {a:>12,.2f}   ({direction} {abs(change):,.2f})")



