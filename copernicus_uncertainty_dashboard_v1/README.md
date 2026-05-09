# Hackathon Plan

## Day 1

-   Data pipeline
-   Indicators
-   Ensemble statistics
-   Fan charts

## Day 2

-   Confidence visualization
-   Maps
-   ML explainability
-   Final storytelling

------------------------------------------------------------------------

# Uncertainty-aware Copernicus Indicators Dashboard

## Overview

This dashboard demonstrates how Copernicus Climate (C3S) and Atmosphere
(CAMS) data can be transformed into **uncertainty-aware indicators**
for: - Health (heat stress, air pollution, compound risk) - Cultural
Heritage (humidity, dust, material degradation)

The key innovation is moving from: \> "Risk is high" to: \> "Risk is
high with low confidence due to high uncertainty"

------------------------------------------------------------------------

## Objectives

-   Integrate environmental data with health and heritage indicators
-   Quantify uncertainty using ensemble statistics
-   Communicate risk + confidence clearly
-   Support decision-making under uncertainty

The dashboard translates environmental data into actionable health and cultural-heritage risk messages while communicating uncertainty clearly.

Instead of saying:

> PM2.5 risk is high.

It says:

> PM2.5 risk is high with low confidence because ensemble spread is large and only part of the ensemble exceeds the threshold.

------------------------------------------------------------------------

## Data Sources

-   C3S (Climate): temperature, humidity → heat stress
-   CAMS (Atmosphere): PM2.5, PM10, NO2, O3, SO2, CO → air quality
-   Synthetic fallback for demo purposes

------------------------------------------------------------------------

## Key Features

-   Risk classification (Low → Extreme)
-   Confidence estimation (High / Medium / Low)
-   Fan charts (uncertainty bands)
-   Geospatial uncertainty maps
-   Compound heat + pollution risk
-   Machine learning explanations (feature importance)
-   Interval prediction (quantile regression)

------------------------------------------------------------------------

## Algorithm (Core Workflow)

### Step 1: Data Input

Collect environmental variables: - Temperature, humidity, wind -
Pollutants (PM2.5, NO2, O3, etc.)

### Step 2: Indicator Calculation

Compute: - Heat index - Pollution risk - Multi-pollutant burden -
Compound heat + pollution risk - Heritage humidity risk

### Step 3: Ensemble Processing

For each timestep: - Mean - Standard deviation - Percentiles (5%, 25%,
75%, 95%)

### Step 4: Uncertainty Estimation

-   Spread = std deviation
-   Probability of exceedance
-   Confidence classification:
    -   Low spread → High confidence
    -   High spread → Low confidence

### Step 5: Risk Classification

Map values to: - Low / Moderate / High / Extreme

### Step 6: Visualization

-   Fan charts (time uncertainty)
-   Map (spatial uncertainty)
-   Traffic-light cards (risk + confidence)

### Step 7: ML Explainability

-   Train Random Forest
-   Extract feature importance
-   Predict uncertainty intervals

------------------------------------------------------------------------

## Color Codes

### Risk

-   Green → Low
-   Yellow → Moderate
-   Orange → High
-   Red → Extreme
-   Blue → Dry stress (heritage)

### Confidence

-   Green → High
-   Amber → Medium
-   Red → Low

------------------------------------------------------------------------

## Health Interpretation

-   Heat + pollution → compound stress
-   Supports alerts, hospital readiness, advisories

------------------------------------------------------------------------

## Cultural Heritage Interpretation

-   Humidity → mold or cracking risk
-   Pollution → corrosion, surface damage
-   Dust → deposition and soiling

------------------------------------------------------------------------

## How to Run

``` bash
pip install -r requirements.txt
python -m streamlit run app.py
```

------------------------------------------------------------------------

## Final Message

This dashboard helps users understand: - What is happening (risk) - How
sure we are (confidence) - Where it matters (map)

# Uncertainty-aware Copernicus Indicators Dashboard

This version makes the dashboard more interactive and easier for non-expert users to understand.

## What changed

- Added layman-friendly explanations for every indicator.
- Added Streamlit help tooltips beside sidebar controls and metrics.
- Added tabs for Forecast, Map, Messages, Health & Heritage, ML, and Data.
- Added hover text in Plotly charts and maps.
- Added country filter for Norway, Sweden and Italy in the synthetic demo grid.
- Added downloadable CSV outputs for current indicator summary and map data.
- Added clearer deterministic vs uncertainty-aware message comparison.
- Added explanations for risk, confidence, ensemble spread, probability of exceedance and uncertainty bands.

## How to read the dashboard

- Risk means how severe the possible impact is.
- Confidence means how much the ensemble members agree.
- A wide uncertainty band means the forecast is less certain.
- Probability above threshold means the chance that the indicator crosses an action level.
- Map color shows risk intensity.
- Map marker size shows uncertainty.

## Color codes

### Risk

- Green: Low risk
- Yellow: Moderate risk
- Orange: High risk
- Red: Extreme risk
- Blue: Dry stress for cultural heritage

### Confidence

- Green: High confidence
- Amber: Medium confidence
- Red: Low confidence

## Dashboard workflow

1. Select an indicator.
2. Select country/location.
3. View the traffic-light risk and confidence summary.
4. Inspect uncertainty bands in the forecast chart.
5. Use the map to identify where uncertainty and risk are high.
6. Compare deterministic and uncertainty-aware messages.
7. Use ML feature importance to understand which environmental variables drive compound risk.
8. Download the summary CSV or map CSV for reporting.

# Datasets Used in the Uncertainty-aware Copernicus Dashboard

## Overview

This dashboard integrates **real Copernicus datasets** and **synthetic
datasets** to demonstrate environmental, health, and cultural heritage
risk under uncertainty.

------------------------------------------------------------------------

## 1. C3S ERA5-HEAT (UTCI)

### What it is

-   Dataset: *derived-utci-historical*
-   Source: Copernicus Climate Change Service (C3S)

### Variables used

-   Universal Thermal Climate Index (UTCI)
-   Temperature proxies

### Why it is useful

-   Represents **human thermal stress**
-   Combines temperature, humidity, wind, and radiation
-   Used to estimate **heat-related health risks**

### In the dashboard

-   Drives **heat stress indicator**
-   Used in **compound heat + pollution risk**
-   Supports **Monte Carlo uncertainty analysis**

------------------------------------------------------------------------

## 2. CAMS Atmospheric Composition Forecasts

### What it is

-   Dataset: *cams-global-atmospheric-composition-forecasts*
-   Source: Copernicus Atmosphere Monitoring Service (CAMS)

### Variables used

-   PM2.5
-   PM10
-   NO₂
-   O₃
-   SO₂
-   CO
-   Aerosol Optical Depth (AOD)
-   Dust

### Why it is useful

-   Represents **air quality and pollution exposure**
-   Critical for **health impact assessment**
-   Captures **short-term environmental variability**

### In the dashboard

-   Drives:
    -   Air pollution indicators
    -   Multi-pollutant health burden
    -   Compound risk with heat

------------------------------------------------------------------------

## 3. ERA5 Meteorological Fallback

### What it is

-   Dataset: *reanalysis-era5-single-levels*

### Variables used

-   Temperature
-   Humidity
-   Wind
-   Surface pressure

### Why it is useful

-   Provides **baseline environmental conditions**
-   Used when CAMS data is unavailable
-   Supports **derived indicators**

------------------------------------------------------------------------

## 4. Synthetic Ensemble Dataset

### What it is

-   Generated in `data_access.py`
-   Simulates Copernicus-like data

### Variables generated

-   Temperature
-   Humidity
-   Wind
-   PM2.5, PM10
-   NO₂, O₃, SO₂, CO
-   AOD
-   Dust deposition
-   Latitude, longitude
-   Country (Norway, Sweden, Italy)

### Why it is useful

-   Enables **instant demo without API delays**
-   Simulates **ensemble forecasts**
-   Allows **uncertainty visualization**

### In the dashboard

-   Provides:
    -   Ensemble spread
    -   Spatial grid for maps
    -   Multi-country comparison

------------------------------------------------------------------------

## 5. Derived Indicators (Computed Data)

### Health indicators

-   Heat index
-   PM exposure levels
-   Multi-pollutant burden
-   Compound heat + pollution risk

### Cultural heritage indicators

-   Humidity risk
-   Dry stress
-   Dust deposition risk

### Why they are useful

-   Translate raw environmental data into:
    -   **Actionable insights**
    -   **Decision-making signals**

------------------------------------------------------------------------

## 6. Uncertainty Metrics

### Computed values

-   Mean
-   Standard deviation
-   Percentiles (5%, 25%, 75%, 95%)
-   Probability of exceedance

### Why they are useful

-   Show **confidence level**
-   Prevent **overconfidence in forecasts**
-   Enable **risk-aware decision-making**

------------------------------------------------------------------------

## 7. Machine Learning Outputs

### Models used

-   Random Forest
-   Gradient Boosting (quantile regression)
-   Neural Network (Keras)

### Outputs

-   Feature importance
-   Interval predictions

### Why they are useful

-   Explain **drivers of risk**
-   Provide **predictive uncertainty**
-   Improve **interpretability**

------------------------------------------------------------------------

## Summary

The dashboard combines: - Climate data (C3S) - Air quality data (CAMS) -
Synthetic ensembles - Derived indicators - Machine learning

to produce:

> **Risk + Confidence + Explanation**

This makes it useful for: - Public health decisions - Environmental
monitoring - Cultural heritage protection

# Datasets & Statistical Methods -- Uncertainty-aware Copernicus Dashboard

## Why this section

This project does not just show data --- it applies statistical methods
to turn environmental signals into **risk + confidence + explanation**
for health and cultural heritage decisions.

------------------------------------------------------------------------

# A. Datasets (recap)

### 1. C3S ERA5-HEAT (UTCI)

-   Human heat stress indicator
-   Used for: heat risk, compound risk

### 2. CAMS Atmospheric Composition

-   PM2.5, PM10, NO2, O3, SO2, CO, aerosols
-   Used for: air quality, health burden

### 3. ERA5 fallback

-   Temperature, humidity, wind
-   Used when CAMS unavailable

### 4. Synthetic ensemble dataset

-   Multi-location + multi-member simulation
-   Used for: uncertainty visualization, maps, demo robustness

------------------------------------------------------------------------

# B. Statistical Approaches Used

## 1. Ensemble Statistics

### What it does

Aggregates multiple model members:

-   Mean → central estimate\
-   Median → robust estimate\
-   Standard deviation → spread\
-   Percentiles (5--95, 25--75) → uncertainty bands

### Why useful

-   Shows **range of possible outcomes**
-   Avoids misleading single-value predictions
-   Core for **fan charts and maps**

------------------------------------------------------------------------

## 2. Uncertainty Quantification

### Methods

-   Standard deviation (spread)
-   Interquartile range (IQR)
-   Coefficient of variation (CV)

### Why useful

-   Quantifies **how uncertain the forecast is**
-   Supports **confidence classification**

------------------------------------------------------------------------

## 3. Probability of Exceedance

### What it does

Calculates:

> \% of ensemble members above a threshold

Example: - 65% exceed PM2.5 limit

### Why useful

-   Directly supports **decision-making**
-   Easy for non-experts to understand

------------------------------------------------------------------------

## 4. Confidence Classification

### Logic

Based on relative spread:

-   Low spread → High confidence\
-   Medium spread → Medium confidence\
-   High spread → Low confidence

### Why useful

-   Converts technical uncertainty into **simple labels**
-   Enables:
    -   Green (confident)
    -   Amber (moderate)
    -   Red (uncertain)

------------------------------------------------------------------------

## 5. Monte Carlo Simulation

### What it does

Randomly samples inputs (temperature, humidity) to compute: -
distribution of outcomes - probability of extreme values

### Why useful

-   Propagates **input uncertainty**
-   Shows **range of possible futures**
-   Supports heat stress analysis

------------------------------------------------------------------------

## 6. Compound Risk Modeling

### What it does

Combines: - Heat stress - Pollution exposure - Interaction effect

### Why useful

-   Captures **real-world complexity**
-   Heat + pollution amplify health risk
-   More realistic than single indicators

------------------------------------------------------------------------

## 7. Machine Learning (Explainability)

### Models used

-   Random Forest → feature importance
-   Gradient Boosting → quantile regression
-   Neural Network → nonlinear relationships

### Outputs

-   Feature importance ranking
-   Prediction intervals

### Why useful

-   Explains **what drives risk**
-   Adds **predictive capability**
-   Enhances **trust and transparency**

------------------------------------------------------------------------

## 8. Interval Prediction (Quantile Regression)

### What it does

Predicts: - lower bound (5%) - median (50%) - upper bound (95%)

### Why useful

-   Shows **future uncertainty range**
-   More informative than point prediction
-   Aligns with climate risk communication

------------------------------------------------------------------------

## 9. Spatial Statistics (Map)

### What it does

Aggregates per location: - mean risk - spread (uncertainty) -
probability of exceedance

### Why useful

-   Answers:
    -   where is risk high?
    -   where is uncertainty high?

------------------------------------------------------------------------

# C. Why these methods matter

Together, these methods allow the dashboard to move from:

> "PM2.5 = 40"

to:

> "PM2.5 is high risk, confidence is low, and 65% of models exceed the
> threshold"

This supports: - Better public-health decisions - Smarter heritage
protection - Transparent communication

------------------------------------------------------------------------

# D. One-line summary

The dashboard combines **Copernicus data + statistical methods + ML** to
turn environmental data into **actionable, uncertainty-aware
decisions**.
