# IDX Exchange: CRMLS Residential Data Pipeline

A Python pipeline that turns 27 months of raw CRMLS (California Regional MLS) exports into clean, analysis-ready datasets of residential sales and listings. It combines the monthly files, cleans and validates them, adds 30-year mortgage rates from FRED, engineers market metrics, and flags data quality problems and outliers.

The repository has two parallel pipelines:

- **[`sold.py`](sold.py)** processes closed residential sales.
- **[`listings.py`](listings.py)** processes residential listings, whether or not they sold.

Both scripts are organized into labeled sections, one per weekly milestone (Weeks 1–7), and run top to bottom.

## Key findings

From the Week 2 exploratory analysis, computed before cleaning (full write-up in [docs/eda_findings.md](docs/eda_findings.md)):

- **Residential dominates:** 67% of sold transactions and 63% of listings. Residential leases are another 23% and 21%.
- **Prices are right-skewed:** the median close price is \$820,000, but the mean is \$1,185,616, pulled up by luxury sales.
- **Homes sell quickly:** median days on market is 19 (mean 37), and 75% of homes sell within 48 days.
- **Most homes sell at or below list:** 40.1% sold above list price and 59.9% at or below it.
- **Priciest counties:** San Mateo (\$1.7M), Santa Clara (\$1.6M), and Santa Cruz (\$1.2M) have the highest median close prices.

## Data

| Source | Coverage | Used for |
| --- | --- | --- |
| CRMLS monthly exports via IDX Exchange | Sold and listing records, January 2024 – March 2026 (27 files each) | All pipeline stages |
| [FRED `MORTGAGE30US`](https://fred.stlouisfed.org/series/MORTGAGE30US) | Weekly 30-year fixed mortgage rate | Week 3 enrichment |

MLS data is proprietary and includes agent contact details, so the raw exports, the generated datasets, and the fetch scripts (which use private API credentials) aren't committed. See [`.gitignore`](.gitignore).

## Pipeline

| Week | Stage | What happens |
| --- | --- | --- |
| 1 | Load and filter | Combine the monthly CSVs in `raw/` and keep only `Residential` records |
| 2 | Explore | Report shape, types, and missing values; drop columns that are more than 90% empty unless they're core fields; answer the EDA questions; print summary stats and IQR outlier counts, and plot a histogram and boxplot for each numeric field |
| 3 | Enrich | Download the FRED mortgage rate series, average the weekly rates by month, and join them on close month (sold) or listing month (listings) |
| 4 | Clean | Parse dates, drop redundant and agent-detail columns, deduplicate on `ListingKey`, coerce numeric types, remove impossible values, drop rows missing core fields, fill missing beds and baths with the median, and standardize text, ZIP codes, and coordinates |
| 5 | Validate | Flag out-of-order dates and bad coordinates. Rows are flagged, not dropped |
| 6 | Engineer features | Add price ratios, price per square foot, time-period fields, and contract timelines; summarize by property type, county and MLS area, and listing and buyer office |
| 7 | Handle outliers | Flag extreme values and business-rule violations, then save every record with its flags alongside a clean subset without the flagged rows |

### Outputs

The sold pipeline writes these files to `data/`. The listings pipeline writes the same set as `listings_*.csv`.

| File | Week | Contents |
| --- | --- | --- |
| `sold_filtered.csv` | 2 | Residential records without the mostly-empty columns |
| `sold_with_rates.csv` | 3 | Adds `rate_30yr_fixed` |
| `sold_cleaned.csv` | 4 | Cleaned, typed, and deduplicated |
| `sold_final.csv` | 5 | Adds the date and geographic flags |
| `sold_features.csv` | 6 | Adds the engineered features |
| `property_type_summary.csv` | 6 | Market metrics by property type and subtype |
| `county_mls_summary.csv` | 6 | Market metrics by county and MLS area |
| `office_summary.csv` | 6 | Market metrics by listing office and buyer office |
| `sold_flagged.csv` | 7 | Every record, with outlier and business-rule flags |
| `sold_clean.csv` | 7 | **Analysis-ready:** records with any Week 7 flag removed |

The Week 6 summaries report median price, average price ratio, average days from listing to contract, and record count for each segment, plus average price per sq ft for the property type and area summaries. Week 2 also saves a distribution chart for each numeric field to `plots/` (`Sold_<field>_distribution.png` and `Listings_<field>_distribution.png`).

### Engineered features (Week 6)

| Column | Definition |
| --- | --- |
| `price_ratio` | `ClosePrice / ListPrice` |
| `close_to_og_list` | `ClosePrice / OriginalListPrice` |
| `price_per_sqft` | `ClosePrice / LivingArea`; listings use `ListPrice` until the home sells |
| `year`, `month`, `YrMo` | From `CloseDate` (sold) or `ListingContractDate` (listings) |
| `days_on_market` | Days from `ListingContractDate` to `PurchaseContractDate`. Computed, so it can differ from the MLS `DaysOnMarket` field |
| `contract_to_close` | Days from `PurchaseContractDate` to `CloseDate` |

In the listings data, ratios are blank for homes that haven't sold, and negative day counts are blanked out as date errors.

### Data quality flags (Weeks 5 and 7)

The pipeline flags problem records instead of deleting them, so analysts can choose how to handle them.

| Column | Week | `True` when |
| --- | --- | --- |
| `listing_after_close_flag` | 5 | `ListingContractDate` is on or after `CloseDate` |
| `purchase_after_close_flag` | 5 | `PurchaseContractDate` is on or after `CloseDate` |
| `negative_timeline_flag` | 5 | Either date flag is set |
| `missing_coords_flag` | 5 | Latitude or longitude is missing |
| `zero_coords_flag` | 5 | Latitude or longitude is 0 |
| `positive_longitude_flag` | 5 | Longitude is positive (California longitudes are negative) |
| `out_of_state_flag` | 5 | Coordinates fall outside California (latitude 32.5 to 42.0, longitude -124.5 to -114.0) |
| `invalid_coords_flag` | 5 | Any geographic flag is set |
| `ClosePrice_outlier` (sold), `ListPrice_outlier` (listings) | 7 | Above the 99th percentile or below Q1 − 1.5 × IQR |
| `LivingArea_outlier` | 7 | Above the 99th percentile or below Q1 − 1.5 × IQR |
| `DaysOnMarket_outlier` | 7 | Outside Q1 − 3 × IQR to Q3 + 3 × IQR |
| `invalid_price_per_sqft_flag` | 7 | Price per sq ft is under \$10 or over \$10,000 |
| `invalid_price_ratio_flag` | 7 | `price_ratio` is under 0.5 or over 2.0 |
| `invalid_close_price_flag` (sold), `invalid_list_price_flag` (listings), `invalid_living_area_flag`, `invalid_dom_flag` | 7 | Price or living area is 0 or less, or `DaysOnMarket` is negative |

A standard 1.5 × IQR fence would cap sold prices around \$2.4M and living area around 3,673 sq ft, cutting legitimate luxury and large homes, so those fields use the 99th percentile (about \$5.5M for sold prices) as the upper bound instead. Long market times are real signal rather than errors, so days on market gets a wider 3 × IQR fence, which is still stricter than its 99th percentile.

`*_clean.csv` excludes rows with any Week 7 flag. The Week 5 date and geographic flags stay as columns for you to filter on.

## Getting started

You need Python 3.10 or newer (developed on 3.13) and a copy of the raw CRMLS exports.

```bash
git clone https://github.com/crbridget/IDX-Exchange.git
cd IDX-Exchange
pip install -r requirements.txt
```

Put the monthly exports in `raw/`, named by record type and month:

```text
raw/
├── CRMLSSold202401.csv … CRMLSSold202603.csv
└── CRMLSListing202401.csv … CRMLSListing202603.csv
```

Then run each pipeline from the repository root, since paths are relative to the working directory:

```bash
python sold.py
python listings.py
```

Each step prints what it did, such as row counts, dropped columns, and flag totals, so the console output works as a cleaning log. Each script downloads the mortgage rate series from FRED, so it needs an internet connection. Together they write about 3.3 GB of CSVs to `data/`.

## Repository layout

```text
IDX-Exchange/
├── sold.py              # Sold pipeline, Weeks 1–7
├── listings.py          # Listings pipeline, Weeks 1–7
├── demo.py              # Week 1 only: load and filter the sold files
├── docs/
│   └── eda_findings.md  # Week 2 EDA answers and data quality notes
├── requirements.txt
├── raw/                 # Monthly CRMLS exports (not committed)
├── data/                # Generated datasets (not committed)
└── plots/               # Generated charts (not committed)
```
