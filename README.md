# Hotel Booking Analytics: Power BI Dashboard + Cancellation Prediction

An end-to-end analytics project on the **Hotel Booking Demand** dataset: a 6-page interactive Power BI report, plus a Python model that predicts which bookings will be cancelled.

![Home](images/home.png)

## Why this project

Cancellations are one of the biggest revenue risks for a hotel. This project answers four questions:

1. **What is happening?** Bookings, revenue, ADR and cancellations at a glance.
2. **When and where do bookings come from?** Seasonality, channels, countries.
3. **Why are bookings cancelled, and which ones are at risk?** Lead time, deposit type, market segment, and an ML risk score.
4. **How do the two hotels compare?** City Hotel vs Resort Hotel.

## Key findings

| Finding | Evidence |
|---|---|
| **37.1% of bookings are cancelled** | 44,199 of 119,210 bookings; 16.7M of revenue lost to cancelled stays |
| **Lead time is the clearest cancellation signal** | Cancellation rate rises from **9.6%** (booked 0–7 days ahead) to **67.7%** (booked 366+ days ahead) |
| **Cancellations differ sharply by market** | Groups 61.1%, Online TA 36.8%, Offline TA/TO 34.3%, Corporate 18.8%, Direct 15.4% |
| **Portugal drives the volume and the rate** | 48,483 bookings (the largest market), 56.7% cancelled, against 16.7% to 37.4% for the other top-10 countries |
| **City earns more but cancels more** | City: 55.3% of revenue, 41.8% cancelled. Resort: 44.7% of revenue, 27.8% cancelled |
| **Resort is far more seasonal** | Resort ADR is about 174 in August vs 50 in January; City stays between 85 and 122 |
| **Bookings and revenue peak in different seasons** | Average bookings per month peak in May (5.9K); average revenue peaks in August (1.64M) because rates are higher |
| **Resort guests stay longer** | 4.3 nights per booking vs 3.0 at City |

## Dashboard pages

| Page | What it shows |
|---|---|
| **Home** | Project landing page with navigation |
| **Executive Summary** | KPIs, revenue by month, bookings by hotel, cancellation rate by month |
| **Booking Trends** | Seasonality, weekday vs weekend nights, market segments, monthly bookings |
| **Customer Insights** | Guest origin map, customer types, top countries, repeat guests |
| **Cancellation Insights** | Cancellation rate by deposit type, segment, lead time and country; ML risk levels |
| **Hotel Performance** | City vs Resort: bookings, revenue, ADR by month, stay length, room mix |

<p>
  <img src="images/executive_summary.png" width="49%" />
  <img src="images/cancellation_insights.png" width="49%" />
</p>

## Cancellation prediction model

**Goal:** predict `is_canceled` for each booking.

**Approach**
- Cleaned the data (removed 180 bookings with zero guests; filled missing `children`, `country`, `agent`, `company`).
- Engineered `total_nights`, `total_guests`, `is_family` and `room_mismatch` (reserved room differs from assigned room).
- **Removed `reservation_status` and `reservation_status_date`.** They reveal the outcome (a cancelled booking has status "Canceled"), so using them would be target leakage.
- Stratified 80/20 train/test split, and compared three models.

**Results on the held-out test set (20%)**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **XGBoost** | 88.1% | 85.9% | 81.3% | 83.5% | **0.954** |
| Random Forest | 86.9% | 88.2% | 74.7% | 80.9% | 0.946 |
| Logistic Regression | 77.9% | 75.6% | 59.7% | 66.7% | 0.839 |

**Top drivers (XGBoost feature importance):** `deposit_type` (0.51), `room_mismatch` (0.09), `required_car_parking_spaces` (0.09), `previous_cancellations` (0.05), `market_segment` (0.04).

**Risk levels in the dashboard.** Each booking gets a `cancellation_risk_score` (predicted probability), grouped into **Low** (< 0.33), **Medium** (0.33 to 0.66) and **High** (≥ 0.66): 56.8% / 14.6% / 28.6% of bookings.

> **Note:** the risk scores in the report are generated for *all* bookings, including the 80% the model was trained on, so they are descriptive. Model quality should be judged by the held-out metrics above, not by the dashboard scores.

## Data model

Star schema in Power BI:

- **Fact:** `hotel_bookings_with_predictions`
- **Dimensions:** `Dim_Date`, `Dim_Hotel`, `Dim_Customer`, `Dim_Market`, `CountryFlagMap`
- A separate measures table holds all DAX measures.

Examples of DAX used:

```dax
Cancellation Rate % =
DIVIDE ( CALCULATE ( COUNTROWS ( hotel_bookings_with_predictions ), hotel_bookings_with_predictions[is_canceled] = 1 ),
         COUNTROWS ( hotel_bookings_with_predictions ) )

-- The data covers Jul 2015 to Aug 2017, so July and August appear 3 times and other months 2 times.
-- Averaging per year makes months comparable.
Avg Monthly Bookings =
DIVIDE ( [Total Bookings], DISTINCTCOUNT ( hotel_bookings_with_predictions[arrival_date_year] ) )

ADR Weighted =
DIVIDE (
    SUMX ( FILTER ( hotel_bookings_with_predictions, hotel_bookings_with_predictions[is_canceled] = 0 ),
           hotel_bookings_with_predictions[adr] * hotel_bookings_with_predictions[total_nights] ),
    SUMX ( FILTER ( hotel_bookings_with_predictions, hotel_bookings_with_predictions[is_canceled] = 0 ),
           hotel_bookings_with_predictions[total_nights] )
)
```

Custom visuals: a country ranking list with flags built as an HTML measure, and a page background and navigation system designed for the report.

## Methodology notes and limitations

- **Revenue** is `ADR × nights` for non-cancelled bookings (about 25.99M). It is an estimate from the dataset, not billed revenue.
- **Average ADR (101.97)** is the simple average across all bookings. ADR by month uses a night-weighted average over confirmed stays.
- **No RevPAR.** The dataset has no room inventory, so RevPAR cannot be calculated.
- **Partial years.** Data runs from July 2015 to August 2017, so year-over-year comparisons would be misleading and are not used.
- **Non-refundable bookings show a 99.4% cancellation rate** (14,587 bookings). This is unusually high and is reported as an observation in the data, likely a recording quirk, not as proof that the deposit type causes cancellations. Refundable bookings (162) are too few for reliable conclusions.
- **Historical data (2015–2017).** Patterns may not reflect current behaviour.

## Tech stack

Power BI (Power Query, DAX, data modelling) · Python (pandas, scikit-learn, XGBoost) · HTML/CSS inside Power BI visuals

## Repository structure

```
├── README.md
├── requirements.txt
├── data/                     # hotel_bookings.csv (download from Kaggle, see below)
├── python/
│   └── cancellation_model.py
├── outputs/                  # generated by the script
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   ├── best_model_summary.txt
│   └── hotel_bookings_with_predictions.csv
├── powerbi/
│   └── hotel_booking_dashboard.pbix
└── images/                   # dashboard screenshots
```

## How to run

```bash
pip install -r requirements.txt
python python/cancellation_model.py --data data/hotel_bookings.csv --out outputs
```

Then open `powerbi/hotel_booking_dashboard.pbix` and, if needed, point the data source to `outputs/hotel_bookings_with_predictions.csv`.

## Data source

Hotel Booking Demand dataset (Kaggle), originally described in:
Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.* Data in Brief, 22, 41–49.


