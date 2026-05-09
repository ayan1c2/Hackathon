# Uncertainty-aware Copernicus Indicators Dashboard

## Hackathon Plan

### Day 1

- Data pipeline
- Indicators
- Ensemble statistics
- Fan charts

### Day 2

- Confidence visualization
- Maps
- ML explainability
- Final storytelling

---

## Overview

This dashboard demonstrates how Copernicus Climate Change Service (C3S) and Copernicus Atmosphere Monitoring Service (CAMS) data can be transformed into uncertainty-aware indicators for public health and cultural heritage.

It translates environmental data into actionable risk messages while communicating uncertainty clearly. Instead of only saying:

> PM2.5 risk is high.

It says:

> PM2.5 risk is high with low confidence because ensemble spread is large and only part of the ensemble exceeds the threshold.

The key idea is to move from a single risk statement to:

> Risk + Confidence + Explanation

---

## Objectives

- Integrate environmental data with health and cultural heritage indicators.
- Quantify uncertainty using ensemble statistics.
- Communicate risk and confidence clearly for non-expert users.
- Support decision-making under uncertainty.

---

## Key Features

- Risk classification from Low to Extreme.
- Confidence estimation: High, Medium, or Low.
- Fan charts with uncertainty bands.
- Geospatial uncertainty maps.
- Compound heat and pollution risk.
- Machine learning explanations through feature importance.
- Interval prediction using quantile regression.
- Layman-friendly explanations, help tooltips, hover text, and dashboard tabs.
- Country filter for Norway, Sweden, and Italy in the synthetic demo grid.
- Downloadable CSV outputs for current indicator summaries and map data.
- Deterministic versus uncertainty-aware message comparison.

---

## Data Sources

### 1. C3S ERA5-HEAT (UTCI)

- Dataset: `derived-utci-historical`
- Source: Copernicus Climate Change Service (C3S)
- Main variable: Universal Thermal Climate Index (UTCI)
- Purpose: human heat stress, heat-related health risk, compound heat and pollution risk, and Monte Carlo uncertainty analysis.

### 2. CAMS Atmospheric Composition Forecasts

- Dataset: `cams-global-atmospheric-composition-forecasts`
- Source: Copernicus Atmosphere Monitoring Service (CAMS)
- Variables: PM2.5, PM10, NO2, O3, SO2, CO, Aerosol Optical Depth (AOD), and dust.
- Purpose: air quality, pollution exposure, multi-pollutant health burden, and compound risk with heat.

### 3. ERA5 Meteorological Fallback

- Dataset: `reanalysis-era5-single-levels`
- Variables: temperature, humidity, wind, and surface pressure.
- Purpose: baseline environmental conditions when CAMS data is unavailable.

### 4. Synthetic Ensemble Dataset

- Generated in `data_access.py`.
- Simulates Copernicus-like multi-location and multi-member data.
- Variables: temperature, humidity, wind, PM2.5, PM10, NO2, O3, SO2, CO, AOD, dust deposition, latitude, longitude, and country.
- Countries: Norway, Sweden, and Italy.
- Purpose: instant demo use, ensemble spread, spatial maps, uncertainty visualization, and multi-country comparison.

### 5. Derived Indicators

Health indicators:

- Heat index
- PM exposure levels
- Multi-pollutant burden
- Compound heat and pollution risk

Cultural heritage indicators:

- Humidity risk
- Dry stress
- Dust deposition risk

---

## Dashboard Workflow

1. Select an indicator.
2. Select a country or location.
3. View the traffic-light risk and confidence summary.
4. Inspect uncertainty bands in the forecast chart.
5. Use the map to identify where uncertainty and risk are high.
6. Compare deterministic and uncertainty-aware messages.
7. Use ML feature importance to understand which environmental variables drive compound risk.
8. Download the summary CSV or map CSV for reporting.

---

## Core Algorithm

### Step 1: Data Input

Collect environmental variables:

- Temperature, humidity, and wind
- Pollutants such as PM2.5, NO2, and O3

### Step 2: Indicator Calculation

Compute:

- Heat index
- Pollution risk
- Multi-pollutant burden
- Compound heat and pollution risk
- Heritage humidity and dry-stress risk

### Step 3: Ensemble Processing

For each timestep or location, compute:

- Mean
- Median
- Standard deviation
- Percentiles: 5%, 25%, 75%, and 95%

### Step 4: Uncertainty Estimation

Use:

- Standard deviation as ensemble spread
- Interquartile range (IQR)
- Coefficient of variation (CV)
- Probability of exceedance

### Step 5: Confidence Classification

Classify confidence using relative spread:

- Low spread -> High confidence
- Medium spread -> Medium confidence
- High spread -> Low confidence

### Step 6: Risk Classification

Map indicator values to:

- Low
- Moderate
- High
- Extreme

### Step 7: Visualization

Show:

- Fan charts for time-based uncertainty
- Maps for spatial risk and uncertainty
- Traffic-light cards for risk and confidence

### Step 8: Machine Learning and Explainability

Use:

- Random Forest for feature importance
- Gradient Boosting for quantile regression
- Neural Network models for nonlinear relationships
- Prediction intervals for uncertainty-aware forecasts

---

## Statistical Methods

### Ensemble Statistics

Aggregates multiple model members to estimate the central outcome and uncertainty range.

Outputs:

- Mean: central estimate
- Median: robust estimate
- Standard deviation: spread
- Percentiles: uncertainty bands

Why useful:

- Shows a range of possible outcomes.
- Avoids misleading single-value predictions.
- Supports fan charts and uncertainty maps.

### Probability of Exceedance

Calculates the percentage of ensemble members above a threshold.

Example:

> 65% of ensemble members exceed the PM2.5 limit.

Why useful:

- Directly supports decision-making.
- Makes uncertainty easier for non-experts to understand.

### Monte Carlo Simulation

Randomly samples uncertain inputs, such as temperature and humidity, to estimate the distribution of possible outcomes and the probability of extreme values.

Why useful:

- Propagates input uncertainty.
- Shows the range of possible futures.
- Supports heat stress analysis.

### Compound Risk Modeling

Combines heat stress, pollution exposure, and interaction effects.

Why useful:

- Captures real-world complexity.
- Reflects how heat and pollution can amplify health risk.
- Gives a more realistic picture than single-indicator analysis.

### Interval Prediction

Uses quantile regression to predict:

- Lower bound: 5%
- Median: 50%
- Upper bound: 95%

Why useful:

- Shows the future uncertainty range.
- Is more informative than a point prediction.
- Aligns with climate risk communication.

### Spatial Statistics

Aggregates per location:

- Mean risk
- Spread or uncertainty
- Probability of exceedance

Why useful:

- Shows where risk is high.
- Shows where uncertainty is high.

---

## How to Read the Dashboard

- Risk means how severe the possible impact is.
- Confidence means how much the ensemble members agree.
- A wide uncertainty band means the forecast is less certain.
- Probability above threshold means the chance that an indicator crosses an action level.
- Map color shows risk intensity.
- Map marker size shows uncertainty.

---

## Color Codes

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

---

## Interpretation

### Health

- Heat and pollution together can create compound stress.
- Outputs can support alerts, hospital readiness, and public advisories.

### Cultural Heritage

- Humidity can indicate mold or cracking risk.
- Pollution can contribute to corrosion and surface damage.
- Dust can cause deposition and soiling.

---

## Why These Methods Matter

Together, the dashboard methods move communication from:

> PM2.5 = 40

To:

> PM2.5 is high risk, confidence is low, and 65% of models exceed the threshold.

This supports better public-health decisions, smarter heritage protection, and transparent environmental communication.

---

## How to Run

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

---

## Final Message

This dashboard helps users understand:

- What is happening: risk
- How sure we are: confidence
- Where it matters: map
- Why it matters: explanation


---

## Compound Heat + Pollution Score (Health & Heritage Interpretation)

### Overview

The compound heat + pollution score captures the combined impact of thermal stress and air pollution on human health and cultural heritage.

### Methodology

**Step 1: Heat normalization (0–100)**  
Heat stress is scaled between 27°C (no stress) and 41°C (extreme stress).

heat_component = max((heat_index - 27) / (41 - 27), 0) * 100

**Step 2: Pollution burden (0–100+)**  
Multiple pollutants (PM2.5, PM10, NO2, O3, SO2, CO) are combined relative to health thresholds.

health_burden_score = (
    0.30 * pm25 / 25 +
    0.20 * pm10 / 50 +
    0.18 * no2 / 40 +
    0.17 * o3 / 100 +
    0.08 * so2 / 20 +
    0.07 * co / 4
) * 100

**Step 3: Interaction term**  
Captures amplification when heat and pollution are both high.

interaction = 0.20 * heat_component * pollution_component / 100

**Step 4: Final score**  
Weighted combination of heat, pollution, and interaction.

compound_score = (
    0.45 * heat_component +
    0.45 * pollution_component +
    interaction
)

### Risk Classification

- Low (<65)  
- Moderate (65–100)  
- High (100–160)  
- Extreme (>160)

### Health Interpretation

- Low: safe conditions  
- Moderate: sensitive groups affected  
- High: increased health stress  
- Extreme: severe compound risk  

### Cultural Heritage Interpretation

- Low: stable conditions  
- Moderate: minor stress  
- High: degradation risk  
- Extreme: accelerated damage  

### Key Takeaway

The compound score transforms multiple environmental stressors into a unified, interaction-aware index enabling decision-ready insights.
