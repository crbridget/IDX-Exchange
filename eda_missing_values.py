import pandas as pd

# Load data
sold = pd.read_csv('sold_combined_residential.csv')
listings = pd.read_csv('listings_combined_residential.csv')

# Inspect structure
print(sold.columns.tolist())
print(sold.head())

# Check property categories (should only be Residential since already filtered)
print(sold['PropertyType'].unique())
print(listings['PropertyType'].unique())

# Validate completeness
print(sold.isnull().sum())
print(f"{listings.isnull().sum()}\n")

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

sold_cleaned = drop_missing_columns(sold, sold_flagged, 'Sold')
listings_cleaned = drop_missing_columns(listings, listings_flagged, 'Listings')

print(f"\nSold columns before: {len(sold.columns)}, after: {len(sold_cleaned.columns)}")
print(f"Listings columns before: {len(listings.columns)}, after: {len(listings_cleaned.columns)}")

# Save
sold_cleaned.to_csv('sold_combined_residential.csv', index=False)
listings_cleaned.to_csv('listings_combined_residential.csv', index=False)