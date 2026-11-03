# EDA Findings

Answers to the Week 2 exploratory questions, based on the combined CRMLS exports for January 2024 through March 2026. The figures were computed before cleaning, and apart from the property type share they cover residential records only.

## Residential vs. other property types

Residential is the dominant property type. ResidentialLease is the second largest, and every other type (Land, ManufacturedInPark, CommercialSale, etc.) makes up less than 4%.

| Property type | Share of sold | Share of listings |
| --- | ---: | ---: |
| Residential | 67% | 63% |
| ResidentialLease | 23% | 21% |
| Each other type | < 4% | < 4% |

## Median and average close price

The median close price is \$820,000 and the mean is \$1,185,616. The large gap between the two indicates strong right skew: a relatively small number of very high-priced sales pull the average up significantly.

## Days on market

The median is 19 days and the mean is 37, again showing right skew. 75% of homes sell within 48 days, but the 95th percentile is 131 days and the maximum is 12,430, which is clearly a data quality issue worth flagging. There are also negative values (minimum of -288), which are invalid and need to be cleaned.

## Sold above vs. below list price

40.1% of homes sold above list price and 59.9% sold at or below it. That suggests the market slightly favors buyers overall, though a significant share of sellers still get over asking.

## Date consistency

58 records have a `CloseDate` before their `ListingContractDate`, which is logically impossible and flags a data quality issue to address in cleaning.

## Counties with the highest median prices

San Mateo (\$1.7M), Santa Clara (\$1.6M), and Santa Cruz (\$1.2M) are the top legitimate counties. Del Norte (\$2.49M) and "Other County" (\$2.46M) rank highest, but they likely reflect data quality issues given their small sample sizes and unusual values.

## Data quality issues

| Field | Issue | How the pipeline handles it |
| --- | --- | --- |
| `DaysOnMarket` | Minimum of -288; negative days on market is impossible | Week 4 removes negative values |
| `DaysOnMarket` | Maximum of 12,430 days | Week 7 flags values above Q3 + 3 × IQR |
| `ClosePrice` | Minimum of \$0, a home that sold for nothing | Week 4 removes sold records priced at \$0 or less |
| `LivingArea` | Maximum of 17 million sq ft, a clear data entry error | Week 4 removes values of 0 or less; Week 7 flags values above the 99th percentile |
| `LotSizeAcres` | Maximum of 781,069 acres, the same kind of error | Not filtered |
| `BathroomsTotalInteger` | Maximum of 2,208 in listings, obviously wrong | Week 4 removes negative values only |
| `YearBuilt` | Minimum of 1776 is suspicious; maximum of 2028 is in the future | Week 4 removes years before 1800 or after 2026 |
| `CloseDate` | 58 closes dated before the listing date | Week 5 flags them with `listing_after_close_flag` |

## General observations

- Prices are heavily right-skewed: median `ClosePrice` is \$820K but the mean is \$1.18M, pulled up by luxury outliers.
- IQR outlier rates are high across the board (7–16%), so it matters how aggressively to filter. Week 7 settles this by using the 99th percentile as the upper bound for price and living area, and a wider 3 × IQR fence for days on market.
