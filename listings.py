import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os


# Week 1: Load & Combine 

listing_files = sorted(glob.glob('raw/CRMLSListing*.csv'))
print(f"Listing files found: {len(listing_files)}")

print("\n--- Individual file row counts (Listings) ---")
listing_dfs = []
for f in listing_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"{f}: {len(df)} rows")
    listing_dfs.append(df)

listings_raw = pd.concat(listing_dfs, ignore_index=True)
print(f"\nRows after combining (listings): {len(listings_raw)}")


# Week 1: Filter to Residential

print("\n--- PropertyType frequency before filter (Listings) ---")
print(listings_raw['PropertyType'].value_counts())

listings = listings_raw[listings_raw['PropertyType'] == 'Residential'].copy()
print(f"\nRows after filtering to Residential (listings): {len(listings)}")

print("\n--- PropertyType frequency after filter (Listings) ---")
print(listings['PropertyType'].value_counts())


# Week 2: Dataset Understanding

print("\n--- Listings Columns ---")
print(listings.columns.tolist())

print("\n--- Listings Head ---")
print(listings.head())

print(f"\nListings shape: {listings.shape}")

print("\n--- Listings Data Types ---")
print(listings.dtypes)


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

listings_flagged = missing_summary(listings, 'Listings')


# Week 2: Drop High-Missing Columns

core_fields = [
    'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
    'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
    'CloseDate', 'ListingContractDate', 'PurchaseContractDate',
    'CountyOrParish', 'City', 'PostalCode', 'PropertyType',
    'PropertySubType', 'Latitude', 'Longitude'
]

to_drop = [col for col in listings_flagged if col not in core_fields]
print(f"\n--- Listings Columns Dropped ---")
print(to_drop)

listings_filtered = listings.drop(columns=to_drop)
print(f"\nListings columns before: {len(listings.columns)}, after: {len(listings_filtered.columns)}")


# Week 2: EDA Questions 

# Median and average list price
print("\n--- List Price Stats ---")
print(f"Median ListPrice: ${listings_filtered['ListPrice'].median():,.0f}")
print(f"Mean ListPrice: ${listings_filtered['ListPrice'].mean():,.0f}")

# Days on Market distribution
print("\n--- Days on Market Distribution ---")
print(listings_filtered['DaysOnMarket'].describe(percentiles=[.25, .50, .75, .90, .95]))

# Date consistency issues
listings_filtered['ListingContractDate'] = pd.to_datetime(listings_filtered['ListingContractDate'])

if 'CloseDate' in listings_filtered.columns:
    listings_filtered['CloseDate'] = pd.to_datetime(listings_filtered['CloseDate'])
    close_before_listing = (listings_filtered['CloseDate'] < listings_filtered['ListingContractDate']).sum()
    print(f"\n--- Date Consistency ---")
    print(f"Records where CloseDate before ListingContractDate: {close_before_listing}")

# Top 10 counties by median list price
print("\n--- Top 10 Counties by Median ListPrice ---")
print(listings_filtered.groupby('CountyOrParish')['ListPrice']
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
    if field not in listings_filtered.columns:
        print(f"\n--- Listings: {field} --- MISSING FROM DATASET")
        continue

    col = listings_filtered[field].dropna()

    print(f"\n--- Listings: {field} ---")
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
    fig.suptitle(f"Listings: {field}", fontsize=13)

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
    plt.savefig(f"plots/Listings_{field}_distribution.png")
    # plt.show()


# Week 2: Save Filtered CSV

os.makedirs('data', exist_ok=True)
listings_filtered.to_csv('data/listings_filtered.csv', index=False)
print("\nListings filtered file saved!")


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

# Key off ListingContractDate for listings dataset
listings_filtered['ListingContractDate'] = pd.to_datetime(listings_filtered['ListingContractDate'])
listings_filtered['year_month'] = listings_filtered['ListingContractDate'].dt.to_period('M')

# Merge
listings_with_rates = listings_filtered.merge(mortgage_monthly, on='year_month', how='left')

# Validate merge
print(f"\nUnmatched rows (rate is null): {listings_with_rates['rate_30yr_fixed'].isnull().sum()}")
print(listings_with_rates[['ListingContractDate', 'year_month', 'ListPrice', 'rate_30yr_fixed']].head())

listings_with_rates.to_csv('data/listings_with_rates.csv', index=False)
print("\nListings with rates file saved!")


# Week 4: Data Cleaning

print(f"\nRows before cleaning: {len(listings_with_rates)}")
print(f"Columns before cleaning: {len(listings_with_rates.columns)}")

cleaning = listings_with_rates.copy()


# Week 4: Convert Date Fields to Datetime

date_fields = ['ListingContractDate', 'PurchaseContractDate', 'CloseDate', 'ContractStatusChangeDate']

for col in date_fields:
    if col in cleaning.columns:
        cleaning[col] = pd.to_datetime(cleaning[col], errors='coerce')

print("\n--- Date Fields Converted to Datetime ---")
print(cleaning[[c for c in date_fields if c in cleaning.columns]].dtypes)


# Week 4: Remove Unnecessary or Redundant Columns

cols_before = len(cleaning.columns)

cols_to_drop = {
    # Helper columns from earlier steps
    'year_month':               'mortgage rate merge helper, not needed for analysis',

    # Duplicate key fields
    'ListingKeyNumeric':        'duplicate of ListingKey in numeric format',
    'LotSizeSquareFeet':        'duplicate of LotSizeAcres in different units',

    # Duplicate lat/lon fields
    'latfilled':                'duplicate of Latitude, filled version from prior processing step',
    'lonfilled':                'duplicate of Longitude, filled version from prior processing step',

    # .1 suffix columns — pandas auto-renames duplicate column headers from raw CSV
    'PropertyType.1':           'duplicate of PropertyType',
    'DaysOnMarket.1':           'duplicate of DaysOnMarket',
    'LivingArea.1':             'duplicate of LivingArea',
    'Longitude.1':              'duplicate of Longitude',
    'Latitude.1':               'duplicate of Latitude',
    'ListPrice.1':              'duplicate of ListPrice',
    'CloseDate.1':              'duplicate of CloseDate',
    'BuyerOfficeName.1':        'duplicate of BuyerOfficeName',
    'UnparsedAddress.1':        'duplicate of UnparsedAddress',
    'ListAgentFirstName.1':     'duplicate of ListAgentFirstName',
    'ListAgentLastName.1':      'duplicate of ListAgentLastName',

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
print(cleaning[[c for c in numeric_fields if c in cleaning.columns]].dtypes)


# Week 4: Remove / Flag Invalid Numeric Values

rows_before = len(cleaning)

# ListPrice: $0 or negative is not valid
invalid_list_price = cleaning['ListPrice'] <= 0
print(f"\nListPrice <= 0: {invalid_list_price.sum()} rows removed")
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

# ClosePrice not required — many listings may not have sold yet
core_required = ['ListPrice', 'ListingContractDate', 'LivingArea']
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
print(f"Rows before: {len(listings_with_rates)} | Rows after: {len(cleaning)}")
print(f"Columns before: {len(listings_with_rates.columns)} | Columns after: {len(cleaning.columns)}")

cleaning.to_csv('data/listings_cleaned.csv', index=False)
print("\nListings cleaned file saved!")


# =============================================================================
# Week 5: Date Consistency Flags
# Rule: ListingContractDate < PurchaseContractDate < CloseDate
# Records are FLAGGED, not dropped — analysts decide how to handle them
# Note: many listings won't have a CloseDate (not yet sold); those rows
# are left as False since no violation can be detected without a close date
# =============================================================================

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
# Only evaluated when both dates are present
if 'ListingContractDate' in cleaning.columns and 'CloseDate' in cleaning.columns:
    both_present = (
        cleaning['ListingContractDate'].notna() &
        cleaning['CloseDate'].notna()
    )
    cleaning['listing_after_close_flag'] = False
    cleaning.loc[both_present, 'listing_after_close_flag'] = (
        cleaning.loc[both_present, 'ListingContractDate']
        >= cleaning.loc[both_present, 'CloseDate']
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
# Only evaluated when both dates are present
if 'PurchaseContractDate' in cleaning.columns and 'CloseDate' in cleaning.columns:
    both_present = (
        cleaning['PurchaseContractDate'].notna() &
        cleaning['CloseDate'].notna()
    )
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


# =============================================================================
# Week 5: Final Summary & Save
# =============================================================================

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

cleaning.to_csv('data/listings_final.csv', index=False)
print("\n  Saved: data/listings_final.csv")
print("=" * 65)


# Week 6: Feature Engineering and Market Metrics

# ensure date fields are datetime
for col in ['ListingContractDate', 'PurchaseContractDate', 'CloseDate']:
    if col in cleaning.columns:
        cleaning[col] = pd.to_datetime(cleaning[col], errors='coerce')

# Price Ratio (sold listings only — NaN for active/unsold)
cleaning['price_ratio'] = cleaning['ClosePrice'] / cleaning['OriginalListPrice']

# Price Per Sq Ft (use ClosePrice if sold, else ListPrice)
price_col = cleaning['ClosePrice'].where(cleaning['ClosePrice'].notna(), cleaning['ListPrice'])
cleaning['price_per_sqft'] = price_col / cleaning['LivingArea']

# Year / Month / YrMo (keyed off ListingContractDate)
cleaning['year'] = cleaning['ListingContractDate'].dt.year
cleaning['month'] = cleaning['ListingContractDate'].dt.month
cleaning['YrMo'] = cleaning['ListingContractDate'].dt.to_period('M')

# Close to Original List Ratio (sold listings only — NaN for active/unsold)
cleaning['close_to_og_list'] = cleaning['ClosePrice'] / cleaning['OriginalListPrice']

# Days from Listing to Contract (NaN if not yet under contract)
cleaning['days_on_market'] = (cleaning['PurchaseContractDate'] - cleaning['ListingContractDate']).dt.days

# Days from Contract to Close (NaN if not yet closed)
cleaning['contract_to_close'] = (cleaning['CloseDate'] - cleaning['PurchaseContractDate']).dt.days

# Sample output showing all engineered columns populated
print("Engineered Metrics Sample")
engineered_cols = [
    'ListPrice', 'OriginalListPrice', 'LivingArea',
    'price_ratio', 'close_to_og_list', 'price_per_sqft',
    'days_on_market', 'contract_to_close',
    'year', 'month', 'YrMo'
]
pd.set_option('display.max_columns', None)
print(cleaning[engineered_cols].head(10).to_string(index=False))

# Summary statistics by PropertyType and PropertySubType
prop_summary = (
    cleaning
    .groupby(['PropertyType', 'PropertySubType'], observed=True)
    .agg(
        avg_list_price     = ('ListPrice',     'mean'),
        median_list_price  = ('ListPrice',     'median'),
        avg_price_per_sqft = ('price_per_sqft','mean'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ListPrice',     'count'),
    )
    .reset_index()
    .sort_values('count', ascending=False)
)
print(prop_summary.to_string(index=False))
prop_summary.to_csv('data/listings_property_type_summary.csv', index=False)

# Summary statistics by CountyOrParish and MLSAreaMajor
geo_summary = (
    cleaning
    .groupby(['CountyOrParish', 'MLSAreaMajor'], observed=True)
    .agg(
        avg_list_price     = ('ListPrice',     'mean'),
        median_list_price  = ('ListPrice',     'median'),
        avg_price_per_sqft = ('price_per_sqft','mean'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ListPrice',     'count'),
    )
    .reset_index()
    .sort_values('median_list_price', ascending=False)
)
print(geo_summary.head(20).to_string(index=False))
geo_summary.to_csv('data/listings_county_mls_summary.csv', index=False)

# Top 10 counties by listing volume
print("\n--- Top 10 Counties by Listing Volume ---")
print(
    cleaning['CountyOrParish']
    .value_counts()
    .head(10)
    .to_string()
)

# Top 10 counties by median list price
print("\n--- Top 10 Counties by Median List Price ---")
print(
    cleaning.groupby('CountyOrParish')['ListPrice']
    .median()
    .sort_values(ascending=False)
    .head(10)
    .apply(lambda x: f"${x:,.0f}")
    .to_string()
)

# Segment Analysis: Office competitive intelligence

# Top listing offices by volume
print("\n--- Top 20 Listing Offices by Volume ---")
list_office = (
    cleaning
    .groupby('ListOfficeName', observed=True)
    .agg(
        median_list_price  = ('ListPrice',     'median'),
        avg_price_per_sqft = ('price_per_sqft','mean'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ListPrice',     'count'),
    )
    .sort_values('count', ascending=False)
)
print(list_office.head(20).to_string())

# Top listing offices by median list price (min 10 listings)
print("\n--- Top 10 Listing Offices by Median List Price (min 10 listings) ---")
print(
    list_office[list_office['count'] >= 10]
    .sort_values('median_list_price', ascending=False)
    .head(10)[['median_list_price', 'count']]
    .to_string()
)

# Top buyer offices by volume
print("\n--- Top 20 Buyer Offices by Volume ---")
buyer_office = (
    cleaning
    .groupby('BuyerOfficeName', observed=True)
    .agg(
        median_list_price  = ('ListPrice',     'median'),
        avg_price_per_sqft = ('price_per_sqft','mean'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ListPrice',     'count'),
    )
    .sort_values('count', ascending=False)
)
print(buyer_office.head(20).to_string())

office_summary = (
    cleaning
    .groupby(['ListOfficeName', 'BuyerOfficeName'], observed=True)
    .agg(
        median_list_price  = ('ListPrice',     'median'),
        avg_price_ratio    = ('price_ratio',   'mean'),
        avg_dom            = ('days_on_market','mean'),
        count              = ('ListPrice',     'count'),
    )
    .reset_index()
    .sort_values('count', ascending=False)
)
office_summary.to_csv('data/listings_office_summary.csv', index=False)

# Time-Based Analysis
# Median list price by year
print("\n--- Median List Price by Year ---")
print(
    cleaning.groupby('year')['ListPrice']
    .median()
    .apply(lambda x: f"${x:,.0f}")
    .to_string()
)

# Median price per sqft by year and county (top 10 counties by volume)
top_counties = cleaning['CountyOrParish'].value_counts().head(10).index
yearly_county = (
    cleaning[cleaning['CountyOrParish'].isin(top_counties)]
    .groupby(['year', 'CountyOrParish'])['price_per_sqft']
    .median()
    .unstack()
)
print("\n--- Median Price per Sqft by Year (Top 10 Counties) ---")
print(yearly_county.to_string())

# Median price ratio by year and property subtype (sold listings only)
yearly_ratio = (
    cleaning
    .groupby(['year', 'PropertySubType'], observed=True)['price_ratio']
    .median()
    .unstack()
)
print("\n--- Median Price Ratio by Year and PropertySubType (Sold Listings Only) ---")
print(yearly_ratio.to_string())

# summary stats
print("\n" + "=" * 65)
print("Quick Summary Statistics")
print("=" * 65)

print(f"""
  Total listings        : {len(cleaning):,}
  Total list volume     : ${cleaning['ListPrice'].sum():,.0f}
  Median list price     : ${cleaning['ListPrice'].median():,.0f}
  Mean list price       : ${cleaning['ListPrice'].mean():,.0f}
  Median price per sqft : ${cleaning['price_per_sqft'].median():,.0f}
  Median price ratio    : {cleaning['price_ratio'].median():.3f}
  Median days on market : {cleaning['days_on_market'].median():.0f} days
  Median contract->close: {cleaning['contract_to_close'].median():.0f} days
  Date range            : {cleaning['ListingContractDate'].min().date()} to {cleaning['ListingContractDate'].max().date()}
""")

# Save
cleaning.to_csv('data/listings_features.csv', index=False)
print("Features saved, data/listings_features.csv")

# Week 7: Outlier Detection and Data Quality (Listings)

df = cleaning.copy()

# stats before filtering
before_size = len(df)
before_medians = {field: df[field].median() for field in ["ListPrice", "LivingArea", "DaysOnMarket"]}

print(f"\nRows entering Week 7 (Listings): {before_size:,}")

# flag extreme values using IQR + percentiles
# ListPrice    - 99th percentile as hard upper bound (CA has very high prices)
# LivingArea   - 99th percentile as hard upper bound; IQR too aggressive for large but legitimate homes (flagged 4.4% at 1.5x)
# DaysOnMarket - looser 3.0 multiplier (long market times are real, not errors)

iqr_config = {
    "ListPrice":    {"multiplier": 1.5, "use_percentile_upper": True},
    "LivingArea":   {"multiplier": 1.5, "use_percentile_upper": True},
    "DaysOnMarket": {"multiplier": 3.0, "use_percentile_upper": False},
}

for field, config in iqr_config.items():
    q1    = df[field].quantile(0.25)
    q3    = df[field].quantile(0.75)
    iqr   = q3 - q1
    lower = q1 - config['multiplier'] * iqr
    p99   = df[field].quantile(0.99)
    p01   = df[field].quantile(0.01)

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

# apply business rules as additional flag columns

# ListPrice <= 0: not a valid listing
df['invalid_list_price_flag'] = df['ListPrice'] <= 0
print(f"  invalid_list_price_flag    : {df['invalid_list_price_flag'].sum():,} records (ListPrice <= 0)")

# LivingArea <= 0: not a valid home
df['invalid_living_area_flag'] = df['LivingArea'] <= 0
print(f"  invalid_living_area_flag   : {df['invalid_living_area_flag'].sum():,} records (LivingArea <= 0)")

# DaysOnMarket < 0: impossible
df['invalid_dom_flag'] = df['DaysOnMarket'] < 0
print(f"  invalid_dom_flag           : {df['invalid_dom_flag'].sum():,} records (DaysOnMarket < 0)")

# price_per_sqft below $10 or above $10,000
df['invalid_price_per_sqft_flag'] = (df['price_per_sqft'] < 10) | (df['price_per_sqft'] > 10000)
print(f"  invalid_price_per_sqft_flag: {df['invalid_price_per_sqft_flag'].sum():,} records (price_per_sqft < $10 or > $10,000)")

# price_ratio only exists for sold listings — skip rows where it's NaN
df['invalid_price_ratio_flag'] = (
    df['price_ratio'].notna() &
    ((df['price_ratio'] < 0.5) | (df['price_ratio'] > 2.0))
)
print(f"  invalid_price_ratio_flag   : {df['invalid_price_ratio_flag'].sum():,} records (price_ratio < 0.5 or > 2.0, sold listings only)")


# save full flagged dataset
df.to_csv('data/listings_flagged.csv', index=False)
print(f"\nFlagged dataset saved, data/listings_flagged.csv  ({len(df):,} rows)")

# build clean filtered dataset
# exclude rows flagged by any IQR or business rule flag
iqr_flag_cols      = [f"{field}_outlier" for field in iqr_config.keys()]
biz_rule_flag_cols = [
    'invalid_list_price_flag',
    'invalid_living_area_flag',
    'invalid_dom_flag',
    'invalid_price_per_sqft_flag',
    'invalid_price_ratio_flag',
]

is_any_flag = df[iqr_flag_cols + biz_rule_flag_cols].any(axis=1)
clean_df    = df[~is_any_flag].copy()

clean_df.to_csv('data/listings_clean.csv', index=False)
print(f"Clean dataset saved, data/listings_clean.csv  ({len(clean_df):,} rows)")



# print comparison

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
    b         = before_medians[field]
    a         = clean_df[field].median()
    change    = a - b
    direction = "up" if change > 0 else "down"
    print(f"  {field:<20}  Before: {b:>12,.2f}   After: {a:>12,.2f}   ({direction} {abs(change):,.2f})")
