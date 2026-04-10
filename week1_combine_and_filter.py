import pandas as pd
import glob

# Load files
sold_files = sorted(glob.glob('raw/CRMLSSold*.csv'))
listing_files = sorted(glob.glob('raw/CRMLSListing*.csv'))

print(f"Sold files found: {len(sold_files)}")
print(f"Listing files found: {len(listing_files)}")

# Row count per file before append
print("\n--- Individual file row counts (Sold) ---")
sold_dfs = []
for f in sold_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"{f}: {len(df)} rows")
    sold_dfs.append(df)

print("\n--- Individual file row counts (Listings) ---")
listing_dfs = []
for f in listing_files:
    df = pd.read_csv(f, low_memory=False)
    print(f"{f}: {len(df)} rows")
    listing_dfs.append(df)

# Concatenate
sold_raw = pd.concat(sold_dfs, ignore_index=True)
listings_raw = pd.concat(listing_dfs, ignore_index=True)

print(f"\nRows after combining (sold): {len(sold_raw)}")
print(f"Rows after combining (listings): {len(listings_raw)}")

# PropertyType frequency before filter
print("\n--- PropertyType frequency before filter (Sold) ---")
print(sold_raw['PropertyType'].value_counts())

print("\n--- PropertyType frequency before filter (Listings) ---")
print(listings_raw['PropertyType'].value_counts())

# Filter to Residential
sold = sold_raw[sold_raw['PropertyType'] == 'Residential'].copy()
listings = listings_raw[listings_raw['PropertyType'] == 'Residential'].copy()

print(f"\nRows after filtering to Residential (sold): {len(sold)}")
print(f"Rows after filtering to Residential (listings): {len(listings)}")

# PropertyType frequency after filter
print("\n--- PropertyType frequency after filter (Sold) ---")
print(sold['PropertyType'].value_counts())

print("\n--- PropertyType frequency after filter (Listings) ---")
print(listings['PropertyType'].value_counts())

# Save
sold.to_csv('sold_combined_residential.csv', index=False)
listings.to_csv('listings_combined_residential.csv', index=False)

print("\nFiles saved!")