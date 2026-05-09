import streamlit as st
import pandas as pd
import numpy as np

from data_access import load_copernicus_dataset
from indicators import (
    approximate_heat_index, air_health_burden_score, compound_heat_pollution_score,
    heat_risk_category, pm25_risk_category, pm10_risk_category, no2_risk_category,
    o3_risk_category, health_burden_risk_category, compound_risk_category,
    heritage_humidity_risk, dry_stress_score, dry_stress_risk_category, dust_deposition_proxy, dust_deposition_risk_category, heritage_compound_score, heritage_compound_risk_category, color_for_risk, color_for_confidence
)
from uncertainty import ensemble_summary, confidence_from_spread, probability_of_exceedance, monte_carlo_indicator
from visualization import fan_chart, traffic_light_card, interval_prediction_chart, uncertainty_map
from ml_models import train_random_forest_regressor, train_interval_models

st.set_page_config(page_title='Uncertainty-aware Copernicus Indicators', layout='wide')
st.title('Uncertainty-aware Copernicus Indicators Dashboard')
st.caption('Interactive ECMWF hackathon prototype using real C3S/CAMS files from the local data folder.')

INDICATOR_HELP = {
    'Health: heat stress': 'Shows how hot it feels to the human body when temperature and humidity combine. Useful for heat-health warnings.',
    'Health: PM2.5 exposure': 'Fine particles that can enter deep into the lungs. Higher values can affect people with respiratory or heart conditions.',
    'Health: PM10 exposure': 'Coarser particles such as dust and road particles. Important for breathing problems and surface soiling.',
    'Health: NO2 exposure': 'Nitrogen dioxide, often related to traffic and combustion. Useful as an urban air-quality stress indicator.',
    'Health: O3 exposure': 'Ground-level ozone, often worse during sunny hot periods. Can aggravate asthma and respiratory irritation.',
    'Health: multi-pollutant burden': 'A simple combined score from several pollutants. Higher means several air-quality pressures are occurring together.',
    'Health: compound heat + pollution risk': 'Combines heat stress and air pollution. Useful because hot stagnant weather can amplify pollution-related health stress.',
    'Cultural heritage: humidity risk': 'Shows whether humidity is too dry, stable, or too humid for heritage materials such as wood, paper, stone and paintings.',
    'Cultural heritage: dry stress': 'Quantifies low-humidity stress that can cause shrinkage, cracking and deformation in wood, paper, textiles and paintings.',
    'Cultural heritage: dust deposition': 'Shows potential dust/particle deposition risk, relevant for surface soiling, cleaning needs and outdoor monument exposure.',
    'Cultural heritage: combined material stress': 'Combines humidity, dry stress, dust and reactive pollutants into one preventive-conservation screening score.'
}

PLAIN_LABELS = {
    'mean': 'average forecast',
    'std': 'uncertainty spread',
    'q05': 'low likely value',
    'q95': 'high likely value',
    'cv': 'relative uncertainty',
    'iqr': 'middle half range'
}


def explain_confidence(confidence):
    if confidence == 'High':
        return 'Most ensemble members agree, so the message is relatively stable.'
    if confidence == 'Medium':
        return 'The signal is useful, but there is noticeable disagreement between ensemble members.'
    return 'The ensemble members disagree strongly, so the result should be treated carefully.'


def explain_risk(risk):
    mapping = {
        'Low': 'No immediate concern for most users.',
        'Moderate': 'Worth monitoring, especially for sensitive people or vulnerable heritage materials.',
        'High': 'Action may be needed, especially for health alerts or preventive conservation.',
        'Extreme': 'Strong action is likely needed; conditions may cause serious impacts.',
        'Dry stress': 'Air is too dry for some heritage materials, increasing cracking or shrinkage risk.'
    }
    return mapping.get(risk, 'Interpret together with confidence and local context.')


def risk_sort_key(risk):
    return {'Low': 0, 'Dry stress': 1, 'Moderate': 2, 'High': 3, 'Extreme': 4}.get(risk, 0)




def ensure_spatial_columns(df):
    """
    Make the dashboard robust when an older data_access.py or real-data loader
    returns a dataframe without country/location columns.
    """
    df = df.copy()

    if 'latitude' not in df.columns:
        df['latitude'] = 59.9139
    if 'longitude' not in df.columns:
        df['longitude'] = 10.7522

    if 'country' not in df.columns:
        def infer_region(row):
            lat = row.get('latitude', np.nan)
            lon = row.get('longitude', np.nan)

            try:
                lat = float(lat)
                lon = float(lon)
            except Exception:
                return 'Unknown region'

            if not np.isfinite(lat) or not np.isfinite(lon):
                return 'Unknown region'

            if lat >= 66.5:
                band = 'Arctic'
            elif lat >= 23.5:
                band = 'Northern mid-latitudes'
            elif lat > -23.5:
                band = 'Tropics'
            elif lat > -66.5:
                band = 'Southern mid-latitudes'
            else:
                band = 'Antarctic'

            if -30 <= lon <= 60:
                sector = 'Europe-Africa sector'
            elif 60 < lon <= 150:
                sector = 'Asia-Pacific sector'
            elif lon > 150 or lon <= -120:
                sector = 'Pacific/Americas sector'
            else:
                sector = 'Americas-Atlantic sector'

            return f'{band} / {sector}'

        df['country'] = df.apply(infer_region, axis=1)

    if 'location_id' not in df.columns:
        df['location_id'] = (
            df['country'].astype(str) + '_' +
            df.groupby(['country', 'latitude', 'longitude']).ngroup().astype(str).str.zfill(2)
        )

    if 'member' not in df.columns:
        df['member'] = 0

    return df

with st.sidebar:
    st.header('Interactive controls')
    st.caption('Hover over the small question marks beside controls for plain-language help.')
    use_case = st.selectbox(
        'Indicator',
        list(INDICATOR_HELP.keys()),
        help='Choose the environmental risk indicator to analyse. The dashboard will update charts, maps and messages.'
    )
    st.info(INDICATOR_HELP[use_case])

    st.success('Data source: real C3S/CAMS files from ./data only')
    show_map = st.checkbox(
        'Show geospatial uncertainty map', value=True,
        help='Shows where risk is high and where uncertainty is large. Marker size means uncertainty.'
    )
    show_ml = st.checkbox(
        'Show ML feature importance and interval prediction', value=True,
        help='Adds a simple machine-learning explanation of which variables drive compound risk.'
    )
    show_details = st.checkbox(
        'Show detailed explanation panels', value=True,
        help='Shows extra layman-friendly text for each dashboard section.'
    )

st.markdown('''
### What this dashboard does
It converts climate and air-quality data into **risk + confidence** messages for health and cultural heritage.

Instead of only saying **“risk is high”**, it says **“risk is high, but confidence is low because the forecast spread is large.”**
''')

if show_details:
    with st.expander('Plain-language guide: how to read this dashboard', expanded=True):
        st.markdown('''
        - **Risk** means how severe the possible impact is.
        - **Confidence** means how much the ensemble members agree.
        - **Wide uncertainty band** means the forecast is less certain.
        - **Probability above threshold** means the chance that the indicator crosses an action level.
        - **Map marker color** means risk intensity; **map marker size** means uncertainty.
        ''')

# Real Copernicus dataset only. No synthetic fallback is used.
try:
    df = load_copernicus_dataset()
except Exception as exc:
    st.error('Could not load real C3S/CAMS data from ./data.')
    st.exception(exc)
    st.stop()

df = ensure_spatial_columns(df)

# Useful debug line in the sidebar so users can verify the active schema.
st.sidebar.caption(f'Data columns available: {len(df.columns)}')

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

# Cultural heritage indicators for real C3S/CAMS data.
# Humidity risk: direct RH threshold classification.
# Dry stress: low-RH score from 0 to 100.
# Dust deposition: use CAMS dust variable if available; otherwise estimate from PM10 - PM2.5.
if 'dust_deposition' not in df.columns or df['dust_deposition'].isna().all():
    if 'aod' in df.columns:
        df['dust_deposition'] = dust_deposition_proxy(df['pm10'], df['pm25'], dust_aod=df['aod'])
    else:
        df['dust_deposition'] = dust_deposition_proxy(df['pm10'], df['pm25'])

df['dry_stress_score'] = dry_stress_score(df['relative_humidity'])
df['dry_stress_risk'] = df['dry_stress_score'].apply(dry_stress_risk_category)
df['dust_deposition_risk'] = df['dust_deposition'].apply(dust_deposition_risk_category)
df['heritage_compound_score'] = df.apply(heritage_compound_score, axis=1)
df['heritage_compound_risk'] = df['heritage_compound_score'].apply(heritage_compound_risk_category)

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
    'Cultural heritage: dry stress': ('dry_stress_score', 'Dry stress score', 'index', 40, 'dry_stress_risk'),
    'Cultural heritage: dust deposition': ('dust_deposition', 'Dust deposition proxy', 'mg/m2/day', 12, 'dust_deposition_risk'),
    'Cultural heritage: combined material stress': ('heritage_compound_score', 'Combined heritage material stress', 'index', 60, 'heritage_compound_risk')
}
variable, label, unit, threshold, risk_column = config[use_case]

# Location controls.
st.sidebar.caption('The app is not restricted to Norway, Sweden or Italy; regions are inferred from the loaded grid extent.')
country_options = ['All'] + sorted(df['country'].unique().tolist())
selected_country = st.sidebar.selectbox(
    'Region filter', country_options,
    help='Filter the map and location list by broad spatial region inferred from latitude and longitude.'
)
country_df = df if selected_country == 'All' else df[df['country'] == selected_country]
location_options = sorted(country_df['location_id'].unique())
selected_location = st.sidebar.selectbox(
    'Representative location for time-series',
    location_options,
    index=len(location_options) // 2,
    help='The time-series charts below show one selected grid point. The map shows all points in the selected region.'
)
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

# Summary downloads.
summary_for_download = summary.copy()
summary_for_download['indicator'] = label
summary_for_download['location_id'] = selected_location
summary_for_download['plain_meaning'] = f'{label} at {selected_location}: risk={dominant_risk}, confidence={confidence}'

st.download_button(
    'Download current indicator summary as CSV',
    data=summary_for_download.to_csv(index=False).encode('utf-8'),
    file_name=f'{selected_location}_{variable}_uncertainty_summary.csv',
    mime='text/csv',
    help='Downloads the time-series statistics currently shown in the dashboard.'
)

st.subheader('At-a-glance interpretation')
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        traffic_light_card(
            label,
            f"{latest['mean']:.1f} {unit}",
            dominant_risk,
            confidence,
            risk_color,
            confidence_color,
            explanation=f"{explain_risk(dominant_risk)} {explain_confidence(confidence)}"
        ),
        unsafe_allow_html=True
    )
with col2:
    st.metric(
        'Ensemble spread',
        f"{latest['std']:.2f} {unit}",
        help='Spread shows how far apart the ensemble members are. A larger spread means lower certainty.'
    )
    st.metric(
        '5-95% range',
        f"{latest['q05']:.1f} - {latest['q95']:.1f} {unit}",
        help='Most possible outcomes are expected inside this range. A wider range means more uncertainty.'
    )
with col3:
    st.metric(
        f'Probability above {threshold} {unit}',
        f"{latest[prob_col] * 100:.0f}%",
        help='Percent of ensemble members that cross the chosen warning/action threshold.'
    )
    st.metric(
        'Interquartile range',
        f"{latest['iqr']:.2f} {unit}",
        help='The middle 50% range. This is a robust view of uncertainty.'
    )

if show_details:
    st.info(f"For lay users: {explain_risk(dominant_risk)} Confidence is {confidence.lower()}: {explain_confidence(confidence)}")

tab_forecast, tab_map, tab_messages, tab_health_heritage, tab_ml, tab_data = st.tabs([
    'Forecast bands', 'Map', 'Messages', 'Health & heritage meaning', 'ML explanation', 'Data table'
])

with tab_forecast:
    st.subheader('Forecast with uncertainty bands')
    st.caption('Hover over the chart to see what the average, likely range and spread mean.')
    st.plotly_chart(
        fan_chart(summary, f'{label}: ensemble uncertainty bands at {selected_location}', f'{label} [{unit}]'),
        use_container_width=True
    )
    if show_details:
        st.markdown('''
        **How to read this:** the central line is the average forecast. The shaded areas show possible values.
        If the shaded area is wide, the forecast is uncertain. If it is narrow, the forecast is more stable.
        ''')

with tab_map:
    if show_map:
        st.subheader('Geospatial uncertainty map')
        selected_date = st.slider(
            'Map date',
            min_value=df['date'].min().to_pydatetime(),
            max_value=df['date'].max().to_pydatetime(),
            value=df['date'].max().to_pydatetime(),
            format='YYYY-MM-DD',
            help='Choose which forecast day to show on the map.'
        )
        map_day = pd.Timestamp(selected_date).normalize()
        df_map_day = country_df[country_df['date'] == map_day].copy()
        if df_map_day.empty:
            df_map_day = country_df[country_df['date'] == country_df['date'].max()].copy()

        map_summary = df_map_day.groupby(['location_id', 'country', 'latitude', 'longitude'])[variable].agg(
            mean='mean', std='std',
            q05=lambda x: np.percentile(x, 5),
            q95=lambda x: np.percentile(x, 95)
        ).reset_index()
        prob_map = df_map_day.assign(exceed=lambda x: x[variable] > threshold).groupby('location_id')['exceed'].mean().reset_index(name='prob_exceedance')
        map_summary = map_summary.merge(prob_map, on='location_id', how='left')
        map_summary['confidence'] = map_summary.apply(lambda r: confidence_from_spread(r['std'], r['mean']), axis=1)
        map_summary['risk_label'] = map_summary['mean'].apply(lambda v: latest_members[risk_column].mode().iloc[0] if variable == 'dust_deposition' else '')
        if variable == 'heat_index':
            map_summary['risk_label'] = map_summary['mean'].apply(heat_risk_category)
        elif variable == 'pm25':
            map_summary['risk_label'] = map_summary['mean'].apply(pm25_risk_category)
        elif variable == 'pm10':
            map_summary['risk_label'] = map_summary['mean'].apply(pm10_risk_category)
        elif variable == 'no2':
            map_summary['risk_label'] = map_summary['mean'].apply(no2_risk_category)
        elif variable == 'o3':
            map_summary['risk_label'] = map_summary['mean'].apply(o3_risk_category)
        elif variable == 'health_burden_score':
            map_summary['risk_label'] = map_summary['mean'].apply(health_burden_risk_category)
        elif variable == 'compound_health_score':
            map_summary['risk_label'] = map_summary['mean'].apply(compound_risk_category)
        elif variable == 'relative_humidity':
            map_summary['risk_label'] = map_summary['mean'].apply(heritage_humidity_risk)
        elif variable == 'dry_stress_score':
            map_summary['risk_label'] = map_summary['mean'].apply(dry_stress_risk_category)
        elif variable == 'dust_deposition':
            map_summary['risk_label'] = map_summary['mean'].apply(dust_deposition_risk_category)
        elif variable == 'heritage_compound_score':
            map_summary['risk_label'] = map_summary['mean'].apply(heritage_compound_risk_category)

        map_summary['plain_language'] = map_summary.apply(
            lambda r: f"{r['risk_label']} risk; {r['confidence'].lower()} confidence; {r['prob_exceedance']:.0%} chance above threshold.",
            axis=1
        )
        spread = map_summary['std'].fillna(0)
        if spread.max() > spread.min():
            map_summary['std_scaled'] = 8 + 24 * (spread - spread.min()) / (spread.max() - spread.min())
        else:
            map_summary['std_scaled'] = 14

        st.plotly_chart(
            uncertainty_map(map_summary, f'{label}: spatial risk and uncertainty on {map_day.date()}', f'{label} mean'),
            use_container_width=True
        )
        st.caption('Map interpretation: color shows risk intensity, marker size shows uncertainty, hover text explains risk, confidence and chance above threshold.')
        st.download_button(
            'Download current map data as CSV',
            data=map_summary.to_csv(index=False).encode('utf-8'),
            file_name=f'{variable}_map_{map_day.date()}.csv',
            mime='text/csv',
            help='Downloads the map-ready table with coordinates, uncertainty and plain-language explanations.'
        )
    else:
        st.info('Enable the map in the sidebar to view geospatial uncertainty.')

with tab_messages:
    st.subheader('Deterministic vs uncertainty-aware communication')
    c1, c2 = st.columns(2)
    with c1:
        st.info(f'### Old style message\n**{label} risk is {dominant_risk}.**\n\nThis is simple, but it hides uncertainty.')
    with c2:
        msg = f'''### Improved message
**{label} risk is {dominant_risk}, with {confidence.lower()} confidence.**

Mean: **{latest['mean']:.1f} {unit}**  
Likely interval: **{latest['q05']:.1f}-{latest['q95']:.1f} {unit}**  
Chance above threshold: **{latest[prob_col] * 100:.0f}%**

Plain meaning: {explain_risk(dominant_risk)} {explain_confidence(confidence)}
'''
        if confidence == 'High':
            st.success(msg)
        elif confidence == 'Medium':
            st.warning(msg)
        else:
            st.error(msg)

with tab_health_heritage:
    st.subheader('Why this matters for health')
    st.markdown('''
    - **Heat stress** can increase dehydration, cardiovascular stress and heat illness.
    - **PM2.5 and PM10** affect lungs and heart, especially for vulnerable groups.
    - **Ozone and NO2** can aggravate asthma and respiratory irritation.
    - **Compound heat + pollution risk** is important because hot, stagnant weather can worsen air pollution impacts.
    ''')
    compound_summary = ensemble_summary(df_loc, 'compound_health_score')
    compound_latest = compound_summary.iloc[-1]
    compound_conf = confidence_from_spread(compound_latest['std'], compound_latest['mean'])
    st.metric(
        'Latest compound heat + pollution score',
        f"{compound_latest['mean']:.1f}",
        help=f"Confidence: {compound_conf}; likely range: {compound_latest['q05']:.1f}-{compound_latest['q95']:.1f}. This combines heat stress and pollutants."
    )

    st.subheader('Why this matters for cultural heritage')
    st.markdown('''
    - **High humidity** can increase mould risk and salt crystallisation.
    - **Low humidity** can cause cracking or shrinkage in wood, paper and paintings.
    - **Dust and particles** can soil surfaces and increase cleaning needs.
    - **NO2, SO2 and ozone** can contribute to corrosion and material degradation.
    ''')

    st.markdown('#### Cultural heritage indicator calculations')
    h1, h2, h3, h4 = st.columns(4)
    with h1:
        rh_mean = latest_members['relative_humidity'].mean()
        st.metric('Humidity risk input', f"{rh_mean:.1f} %", help='Relative humidity is classified into dry stress, low, moderate, high or extreme moisture risk.')
        st.caption(f"Risk: **{heritage_humidity_risk(rh_mean)}**")
    with h2:
        dry_mean = latest_members['dry_stress_score'].mean()
        st.metric('Dry stress score', f"{dry_mean:.1f}", help='0 means no dry stress; 100 means severe low-humidity stress.')
        st.caption(f"Risk: **{dry_stress_risk_category(dry_mean)}**")
    with h3:
        dust_mean = latest_members['dust_deposition'].mean()
        st.metric('Dust deposition proxy', f"{dust_mean:.1f}", help='Uses CAMS dust variable if available; otherwise estimates coarse particles as PM10 - PM2.5.')
        st.caption(f"Risk: **{dust_deposition_risk_category(dust_mean)}**")
    with h4:
        heritage_mean = latest_members['heritage_compound_score'].mean()
        st.metric('Combined material stress', f"{heritage_mean:.1f}", help='Combines humidity, dry stress, dust and reactive pollutants.')
        st.caption(f"Risk: **{heritage_compound_risk_category(heritage_mean)}**")

    with st.expander('How cultural heritage indicators are calculated'):
        st.markdown('''
        **Humidity risk:** relative humidity thresholds identify dry stress, stable conditions, mould risk and high-moisture stress.

        **Dry stress:** low relative humidity is converted to a 0-100 score: RH >= 40% gives 0; RH <= 20% approaches 100.

        **Dust deposition risk:** uses a CAMS dust variable where available; otherwise a coarse-particle proxy is calculated as `PM10 - PM2.5`.

        **Combined material stress:** combines humidity/moisture stress, dry stress, dust deposition and reactive pollutants (`NO2`, `SO2`, `O3`) into one preventive-conservation screening score.
        ''')

    st.subheader('Monte Carlo uncertainty propagation')
    if variable == 'heat_index':
        mc = monte_carlo_indicator(
            latest_members['temperature_c'].mean(),
            latest_members['relative_humidity'].mean(),
            latest_members['temperature_c'].std(),
            latest_members['relative_humidity'].std(),
            indicator_function=approximate_heat_index
        )
        st.dataframe(pd.DataFrame([mc]), use_container_width=True)
        st.caption('Monte Carlo means we perturb temperature and humidity many times to see how much the heat indicator changes.')
    else:
        st.write('For non-heat indicators, ensemble spread and probability of exceedance are shown above. The same Monte Carlo pattern can be applied using pollutant-specific input uncertainty.')

with tab_ml:
    if show_ml:
        st.subheader('ML: feature importance and interval prediction')
        st.caption('This section explains which variables most influence the compound heat + pollution score.')
        model, metrics = train_random_forest_regressor(df, target='compound_health_score')
        importance = pd.DataFrame(metrics['feature_importance'].items(), columns=['Feature', 'Importance']).sort_values('Importance', ascending=False)
        m1, m2 = st.columns(2)
        with m1:
            st.metric('Random Forest MAE', f"{metrics['mae']:.2f}", help='Average prediction error. Smaller means better model fit.')
            st.bar_chart(importance.set_index('Feature'))
            top_feature = importance.iloc[0]['Feature']
            st.info(f'Plain meaning: the model currently relies most on **{top_feature}** to explain compound risk.')
        with m2:
            _, interval_df = train_interval_models(df, target='compound_health_score')
            compound_all_summary = ensemble_summary(df, 'compound_health_score')
            st.plotly_chart(
                interval_prediction_chart(compound_all_summary, interval_df, 'Compound heat + pollution interval prediction', 'Compound health score'),
                use_container_width=True
            )
    else:
        st.info('Enable ML in the sidebar to see feature importance and interval prediction.')

with tab_data:
    st.subheader('Statistical diagnostics')
    diagnostics = summary[['date', 'mean', 'median', 'std', 'q05', 'q25', 'q75', 'q95', 'iqr', 'range', 'cv']].rename(columns=PLAIN_LABELS)
    st.dataframe(diagnostics, use_container_width=True)
    st.caption('These diagnostics are the numeric basis for the charts, confidence score and risk message.')
    with st.expander('What do these columns mean?'):
        st.markdown('''
        - **Average forecast:** the central estimate.
        - **Uncertainty spread:** how far apart ensemble members are.
        - **Low/high likely value:** lower and upper likely bounds.
        - **Relative uncertainty:** spread relative to the average.
        - **Middle half range:** the range containing the central 50% of ensemble members.
        ''')

st.subheader('Final demo story')
st.markdown(f'''
A decision-maker normally sees: **{label} risk is {dominant_risk}.**

This dashboard improves the message: **{label} risk is {dominant_risk}, confidence is {confidence.lower()},
and the probability of exceeding the action threshold is {latest[prob_col] * 100:.0f}%.**

The map adds the operational question: **where is the risk high, and where is the uncertainty also high?**
This makes the forecast more transparent, supports proportional action, and avoids hiding uncertainty from users.
''')
