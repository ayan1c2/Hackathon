import streamlit as st
import pandas as pd
import numpy as np

from data_access import create_demo_dataset
from indicators import (
    approximate_heat_index, air_health_burden_score, compound_heat_pollution_score,
    heat_risk_category, pm25_risk_category, pm10_risk_category, no2_risk_category,
    o3_risk_category, health_burden_risk_category, compound_risk_category,
    heritage_humidity_risk, color_for_risk, color_for_confidence
)
from uncertainty import ensemble_summary, confidence_from_spread, probability_of_exceedance, monte_carlo_indicator
from visualization import fan_chart, traffic_light_card, interval_prediction_chart, uncertainty_map
from ml_models import train_random_forest_regressor, train_interval_models

st.set_page_config(page_title='Uncertainty-aware Copernicus Indicators', layout='wide')
st.title('Uncertainty-aware Copernicus Indicators Dashboard')
st.caption('ECMWF hackathon prototype for C3S/CAMS-style health and cultural heritage indicators.')

with st.sidebar:
    st.header('Hackathon controls')
    use_case = st.selectbox('Indicator', [
        'Health: heat stress', 'Health: PM2.5 exposure', 'Health: PM10 exposure',
        'Health: NO2 exposure', 'Health: O3 exposure', 'Health: multi-pollutant burden',
        'Health: compound heat + pollution risk',
        'Cultural heritage: humidity risk', 'Cultural heritage: dust deposition'
    ])
    n_days = st.slider('Forecast days', 7, 60, 30)
    n_members = st.slider('Ensemble members', 5, 50, 20)
    show_map = st.checkbox('Show geospatial uncertainty map', value=True)
    show_ml = st.checkbox('Show ML feature importance and interval prediction', value=True)

st.markdown('''
### Challenge
Move from deterministic warnings to uncertainty-aware messages: **risk level + confidence level + reason**.

**Day 1:** Build data pipeline, synthetic fallback data, indicators, ensemble statistics and fan chart.  
**Day 2:** Add traffic-light confidence cards, deterministic vs uncertainty-aware messages, Monte Carlo module, ML feature-importance panel, interval prediction, geospatial uncertainty map and final demo story.
''')

# Synthetic C3S/CAMS-style fallback dataset now includes latitude/longitude.
df = create_demo_dataset(n_days=n_days, n_members=n_members)

# Indicator calculations.
df['heat_index'] = approximate_heat_index(df['temperature_c'], df['relative_humidity'])
df['health_burden_score'] = df.apply(air_health_burden_score, axis=1)
df['compound_health_score'] = df.apply(compound_heat_pollution_score, axis=1)
df['heat_risk'] = df['heat_index'].apply(heat_risk_category)
df['pm25_risk'] = df['pm25'].apply(pm25_risk_category)
df['pm10_risk'] = df['pm10'].apply(pm10_risk_category)
df['no2_risk'] = df['no2'].apply(no2_risk_category)
df['o3_risk'] = df['o3'].apply(o3_risk_category)
df['health_burden_risk'] = df['health_burden_score'].apply(health_burden_risk_category)
df['compound_health_risk'] = df['compound_health_score'].apply(compound_risk_category)
df['heritage_risk'] = df['relative_humidity'].apply(heritage_humidity_risk)

risk_code_map = {'Low': 0, 'Moderate': 1, 'High': 2, 'Extreme': 3, 'Dry stress': 1}
df['risk_code'] = df['compound_health_risk'].map(risk_code_map)

config = {
    'Health: heat stress': ('heat_index', 'Heat index', 'deg C', 32, 'heat_risk'),
    'Health: PM2.5 exposure': ('pm25', 'PM2.5', 'ug/m3', 25, 'pm25_risk'),
    'Health: PM10 exposure': ('pm10', 'PM10', 'ug/m3', 50, 'pm10_risk'),
    'Health: NO2 exposure': ('no2', 'NO2', 'ug/m3', 40, 'no2_risk'),
    'Health: O3 exposure': ('o3', 'O3', 'ug/m3', 100, 'o3_risk'),
    'Health: multi-pollutant burden': ('health_burden_score', 'Multi-pollutant health burden', 'index', 110, 'health_burden_risk'),
    'Health: compound heat + pollution risk': ('compound_health_score', 'Compound heat + pollution risk', 'index', 100, 'compound_health_risk'),
    'Cultural heritage: humidity risk': ('relative_humidity', 'Relative humidity', '%', 75, 'heritage_risk'),
    'Cultural heritage: dust deposition': ('dust_deposition', 'Dust deposition proxy', 'mg/m2/day', 12, 'heritage_risk')
}
variable, label, unit, threshold, risk_column = config[use_case]

# Use a representative location for the main time-series panel.
location_options = sorted(df['location_id'].unique())
selected_location = st.sidebar.selectbox('Representative map location for time-series', location_options, index=len(location_options) // 2)
df_loc = df[df['location_id'] == selected_location].copy()

summary = ensemble_summary(df_loc, variable)
exceedance = probability_of_exceedance(df_loc, variable, threshold)
summary = summary.merge(exceedance, on='date', how='left')
latest = summary.iloc[-1]
latest_members = df_loc[df_loc['date'] == latest['date']]
confidence = confidence_from_spread(latest['std'], latest['mean'])
dominant_risk = latest_members[risk_column].mode().iloc[0]
risk_color = color_for_risk(dominant_risk)
confidence_color = color_for_confidence(confidence)
prob_col = f'prob_{variable}_above_{threshold}'

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(traffic_light_card(label, f"{latest['mean']:.1f} {unit}", dominant_risk, confidence, risk_color, confidence_color), unsafe_allow_html=True)
with col2:
    st.metric('Ensemble spread', f"{latest['std']:.2f} {unit}")
    st.metric('5-95% range', f"{latest['q05']:.1f} - {latest['q95']:.1f} {unit}")
with col3:
    st.metric(f'Probability above {threshold} {unit}', f"{latest[prob_col] * 100:.0f}%")
    st.metric('Interquartile range', f"{latest['iqr']:.2f} {unit}")

st.subheader('Fan chart: forecast with uncertainty bands')
st.plotly_chart(fan_chart(summary, f'{label}: ensemble uncertainty bands at location {selected_location}', f'{label} [{unit}]'), use_container_width=True)

if show_map:
    st.subheader('Geospatial uncertainty map')
    selected_date = st.slider(
        'Map date',
        min_value=df['date'].min().to_pydatetime(),
        max_value=df['date'].max().to_pydatetime(),
        value=df['date'].max().to_pydatetime(),
        format='YYYY-MM-DD'
    )
    map_day = pd.Timestamp(selected_date).normalize()
    df_map_day = df[df['date'] == map_day].copy()
    if df_map_day.empty:
        df_map_day = df[df['date'] == df['date'].max()].copy()

    map_summary = df_map_day.groupby(['location_id', 'latitude', 'longitude'])[variable].agg(
        mean='mean', std='std',
        q05=lambda x: np.percentile(x, 5),
        q95=lambda x: np.percentile(x, 95)
    ).reset_index()
    prob_map = df_map_day.assign(exceed=lambda x: x[variable] > threshold).groupby('location_id')['exceed'].mean().reset_index(name='prob_exceedance')
    map_summary = map_summary.merge(prob_map, on='location_id', how='left')
    map_summary['confidence'] = map_summary.apply(lambda r: confidence_from_spread(r['std'], r['mean']), axis=1)
    # Plotly map sizes cannot be zero; scale spread while retaining uncertainty meaning.
    spread = map_summary['std'].fillna(0)
    if spread.max() > spread.min():
        map_summary['std_scaled'] = 8 + 24 * (spread - spread.min()) / (spread.max() - spread.min())
    else:
        map_summary['std_scaled'] = 14

    st.plotly_chart(
        uncertainty_map(map_summary, f'{label}: spatial risk and uncertainty on {map_day.date()}', f'{label} mean'),
        use_container_width=True
    )
    st.caption('Map interpretation: color shows ensemble mean risk intensity; marker size shows uncertainty/spread; hover shows confidence and probability of exceedance.')

st.subheader('Deterministic vs uncertainty-aware communication')
c1, c2 = st.columns(2)
with c1:
    st.info(f'### Deterministic message\n**{label} risk is {dominant_risk}.**')
with c2:
    msg = f'''### Uncertainty-aware message
**{label} risk is {dominant_risk}, with {confidence.lower()} confidence.**

Mean: **{latest['mean']:.1f} {unit}**  
5-95% interval: **{latest['q05']:.1f}-{latest['q95']:.1f} {unit}**  
Probability above threshold: **{latest[prob_col] * 100:.0f}%**
'''
    if confidence == 'High':
        st.success(msg)
    elif confidence == 'Medium':
        st.warning(msg)
    else:
        st.error(msg)

st.subheader('Compound heat + pollution health perspective')
st.markdown('''
The compound health-risk indicator combines thermal stress with multi-pollutant exposure. This matters because heat and air pollution can amplify each other: during stagnant hot episodes, ozone, PM and nitrogen dioxide can worsen respiratory and cardiovascular stress. The dashboard therefore reports both the individual components and the combined risk, making it easier to prioritise heat-health alerts, air-quality advisories, targeted messages for vulnerable groups and operational readiness for hospitals or care homes.
''')

compound_summary = ensemble_summary(df_loc, 'compound_health_score')
compound_latest = compound_summary.iloc[-1]
compound_conf = confidence_from_spread(compound_latest['std'], compound_latest['mean'])
st.metric('Latest compound heat + pollution score', f"{compound_latest['mean']:.1f}", help=f"Confidence: {compound_conf}; 5-95% interval: {compound_latest['q05']:.1f}-{compound_latest['q95']:.1f}")

st.subheader('Monte Carlo uncertainty propagation')
if variable == 'heat_index':
    mc = monte_carlo_indicator(latest_members['temperature_c'].mean(), latest_members['relative_humidity'].mean(), latest_members['temperature_c'].std(), latest_members['relative_humidity'].std(), indicator_function=approximate_heat_index)
    st.dataframe(pd.DataFrame([mc]), use_container_width=True)
else:
    st.write('For non-heat indicators, ensemble spread and probability of exceedance are shown above. The same Monte Carlo pattern can be applied using pollutant-specific input uncertainty.')

st.subheader('Statistical diagnostics')
st.dataframe(summary[['date', 'mean', 'median', 'std', 'q05', 'q25', 'q75', 'q95', 'iqr', 'range', 'cv']], use_container_width=True)

if show_ml:
    st.subheader('ML: feature importance and interval prediction')
    model, metrics = train_random_forest_regressor(df, target='compound_health_score')
    importance = pd.DataFrame(metrics['feature_importance'].items(), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
    m1, m2 = st.columns(2)
    with m1:
        st.metric('Random Forest MAE', f"{metrics['mae']:.2f}")
        st.bar_chart(importance.set_index('Feature'))
    with m2:
        _, interval_df = train_interval_models(df, target='compound_health_score')
        compound_all_summary = ensemble_summary(df, 'compound_health_score')
        st.plotly_chart(interval_prediction_chart(compound_all_summary, interval_df, 'Compound heat + pollution interval prediction', 'Compound health score'), use_container_width=True)

st.subheader('Final demo story')
st.markdown(f'''
A decision-maker normally sees: **{label} risk is {dominant_risk}.**

This dashboard improves the message: **{label} risk is {dominant_risk}, confidence is {confidence.lower()},
and the probability of exceeding the action threshold is {latest[prob_col] * 100:.0f}%.**

The map adds the operational question: **where is the risk high, and where is the uncertainty also high?**
This makes the forecast more transparent, supports proportional action, and avoids hiding uncertainty from users.
''')
