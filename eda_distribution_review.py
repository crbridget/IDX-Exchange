import matplotlib.pyplot as plt
import pandas as pd
import os

# Make plots folder if it doesn't already exist
os.makedirs('plots', exist_ok=True)

# Load data
sold = pd.read_csv('data/sold_filtered.csv')
listings = pd.read_csv('data/listings_filtered.csv')


def analyze_numeric_distribution(df, name):
    """
    For each key numeric field, generate histograms, boxplots,
    and percentile summaries, and flag extreme outliers.
    """
    numeric_fields = [
        'ClosePrice', 'ListPrice', 'OriginalListPrice', 'LivingArea',
        'LotSizeAcres', 'BedroomsTotal', 'BathroomsTotalInteger',
        'DaysOnMarket', 'YearBuilt'
    ]

    for field in numeric_fields:
        if field not in df.columns:
            print(f"{field} not found in {name}, skipping...")
            continue

        col = df[field].dropna()

        # Percentile summary
        print(f"\n--- {name}: {field} ---")
        print(col.describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

        # Outlier flagging via IQR
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = col[(col < lower) | (col > upper)]
        print(f"Outliers (IQR method): {len(outliers)} ({len(outliers)/len(col)*100:.2f}%)")
        print(f"Lower bound: {lower:.2f} | Upper bound: {upper:.2f}")

        # Plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{name}: {field}", fontsize=13)

        # Histogram - with capped x-axis for readability
        q99 = col.quantile(0.99)
        axes[0].hist(col[col <= q99], bins=50, edgecolor='black', color='steelblue')
        axes[0].set_title('Histogram (capped at 99th percentile)')
        axes[0].set_xlabel(field)
        axes[0].set_ylabel('Frequency')

        # Boxplot - with capped data
        col_capped = col[col <= q99]
        axes[1].boxplot(col_capped, vert=False)
        axes[1].set_title('Boxplot (capped at 99th percentile)')
        axes[1].set_xlabel(field)

        plt.tight_layout()
        plt.savefig(f"plots/{name}_{field}_distribution.png")
        plt.show()

# Run for both datasets
analyze_numeric_distribution(sold, 'Sold')
analyze_numeric_distribution(listings, 'Listings')