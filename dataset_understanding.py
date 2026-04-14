import pandas as pd
import os
import glob

# Load data
sold = pd.read_csv('data/sold_combined_residential.csv')
listings = pd.read_csv('data/listings_combined_residential.csv')

# Inspect structure
print(sold.columns.tolist())
print(sold.head())

print(listings.columns.tolist())
print(listings.head())

# Check property categories (should only be Residential since already filtered)
print(sold['PropertyType'].unique())
print(listings['PropertyType'].unique())

# Validate completeness
print(sold.isnull().sum())
print(f"\n{listings.isnull().sum()}\n")

# Shape
print(f"Sold shape: {sold.shape}")
print(f"Listed shape: {listings.shape}\n")

# Data types
print('\n--- Sold data types ---')
print(sold.dtypes)
print('\n--- Listings data types ---')
print(listings.dtypes)

# Missing value analysis
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
listings_flagged = missing_summary(listings, 'Listings')


# Drop high-missing columns
def drop_missing_columns(df, flagged_cols, name):
    """Drop columns with >90% missing values, retain core fields"""
    # Core fields to always keep even if partially missing
    core_fields = [
        'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
        'DaysOnMarket', 'BedroomsTotal', 'BathroomsTotalInteger',
        'CloseDate', 'ListingContractDate', 'PurchaseContractDate',
        'CountyOrParish', 'City', 'PostalCode', 'PropertyType',
        'PropertySubType', 'Latitude', 'Longitude'
    ]
    to_drop = [col for col in flagged_cols if col not in core_fields]

    print(f"\n--- {name} Columns Dropped ---")
    print(to_drop)

    return df.drop(columns=to_drop)

sold_filtered = drop_missing_columns(sold, sold_flagged, 'Sold')
listings_filtered = drop_missing_columns(listings, listings_flagged, 'Listings')

print(f"\nSold columns before: {len(sold.columns)}, after: {len(sold_filtered.columns)}")
print(f"Listings columns before: {len(listings.columns)}, after: {len(listings_filtered.columns)}")

# EDA Questions

# 1. Residential vs other property type share (from raw unfiltered data)
sold_files = sorted(glob.glob('raw/CRMLSSold*.csv'))
sold_raw_all = pd.concat([pd.read_csv(f, low_memory=False) for f in sold_files], ignore_index=True)

print("\n--- Property Type Share (Sold) ---")
print(sold_raw_all['PropertyType'].value_counts(normalize=True).mul(100).round(2))

listing_files = sorted(glob.glob('raw/CRMLSListing*.csv'))
listings_raw_all = pd.concat([pd.read_csv(f, low_memory=False) for f in listing_files], ignore_index=True)

print("\n--- Property Type Share (Listings) ---")
print(listings_raw_all['PropertyType'].value_counts(normalize=True).mul(100).round(2))

# 2. Median and average close price
print("\n--- Close Price Stats ---")
print(f"Median ClosePrice: ${sold_filtered['ClosePrice'].median():,.0f}")
print(f"Mean ClosePrice: ${sold_filtered['ClosePrice'].mean():,.0f}")

# 3. Days on Market distribution
print("\n--- Days on Market Distribution ---")
print(sold_filtered['DaysOnMarket'].describe(percentiles=[.25, .50, .75, .90, .95]))

# 4. % sold above vs below list price
sold_filtered['above_list'] = sold_filtered['ClosePrice'] > sold_filtered['ListPrice']
above = sold_filtered['above_list'].sum()
below = (~sold_filtered['above_list']).sum()
total = len(sold_filtered)
print(f"\n--- Sold Above vs Below List Price ---")
print(f"Above list: {above / total * 100:.1f}%")
print(f"Below list: {below / total * 100:.1f}%")

# 5. Date consistency issues
sold_filtered['CloseDate'] = pd.to_datetime(sold_filtered['CloseDate'])
sold_filtered['ListingContractDate'] = pd.to_datetime(sold_filtered['ListingContractDate'])
close_before_listing = (sold_filtered['CloseDate'] < sold_filtered['ListingContractDate']).sum()
print(f"\n--- Date Consistency ---")
print(f"Records where CloseDate before ListingContractDate: {close_before_listing}")

# 6. Counties with highest median prices
print("\n--- Top 10 Counties by Median ClosePrice ---")
print(sold_filtered.groupby('CountyOrParish')['ClosePrice']
      .median()
      .sort_values(ascending=False)
      .head(10)
      .apply(lambda x: f"${x:,.0f}"))

# Save
os.makedirs('data', exist_ok=True)
sold_filtered.to_csv('data/sold_filtered.csv', index=False)
listings_filtered.to_csv('data/listings_filtered.csv', index=False)