import numpy as np


def approximate_heat_index(temp_c, rh):
    temp_f = temp_c * 9 / 5 + 32
    hi_f = (-42.379 + 2.04901523 * temp_f + 10.14333127 * rh
            - 0.22475541 * temp_f * rh - 0.00683783 * temp_f ** 2
            - 0.05481717 * rh ** 2 + 0.00122874 * temp_f ** 2 * rh
            + 0.00085282 * temp_f * rh ** 2 - 0.00000199 * temp_f ** 2 * rh ** 2)
    return (hi_f - 32) * 5 / 9


def air_health_burden_score(row):
    # Simple normalized multi-pollutant score. Replace with local health guidance as needed.
    return (
        0.30 * row['pm25'] / 25
        + 0.20 * row['pm10'] / 50
        + 0.18 * row['no2'] / 40
        + 0.17 * row['o3'] / 100
        + 0.08 * row['so2'] / 20
        + 0.07 * row['co'] / 4
    ) * 100


def compound_heat_pollution_score(row):
    """
    Compound health risk screening index combining heat stress and air pollution.

    Interpretation:
    - Heat and pollution both contribute independently.
    - A heat-pollution interaction term increases the score when both are elevated.
    - This is a hackathon screening metric, not a regulatory health index.
    """
    heat_component = max((row['heat_index'] - 27) / (41 - 27), 0) * 100
    pollution_component = row['health_burden_score']
    interaction = 0.20 * heat_component * pollution_component / 100
    return 0.45 * heat_component + 0.45 * pollution_component + interaction


def risk_category(value, thresholds):
    if value < thresholds[0]:
        return 'Low'
    if value < thresholds[1]:
        return 'Moderate'
    if value < thresholds[2]:
        return 'High'
    return 'Extreme'


def heat_risk_category(value):
    return risk_category(value, [27, 32, 41])


def pm25_risk_category(value):
    return risk_category(value, [10, 25, 50])


def pm10_risk_category(value):
    return risk_category(value, [20, 50, 100])


def no2_risk_category(value):
    return risk_category(value, [20, 40, 100])


def o3_risk_category(value):
    return risk_category(value, [60, 100, 180])


def health_burden_risk_category(value):
    return risk_category(value, [70, 110, 180])


def compound_risk_category(value):
    return risk_category(value, [65, 100, 160])


def heritage_humidity_risk(rh):
    if rh < 40:
        return 'Dry stress'
    if rh <= 60:
        return 'Low'
    if rh <= 75:
        return 'Moderate'
    if rh <= 85:
        return 'High'
    return 'Extreme'


def dry_stress_score(rh):
    """
    Cultural heritage dry-stress score from relative humidity.

    0 = no dry stress
    100 = severe dry stress
    """
    return np.where(
        np.asarray(rh) >= 40,
        0,
        np.minimum((40 - np.asarray(rh)) / 20 * 100, 100)
    )


def dry_stress_risk_category(score):
    if score <= 0:
        return 'Low'
    if score < 40:
        return 'Moderate'
    if score < 70:
        return 'High'
    return 'Extreme'


def dust_deposition_proxy(pm10, pm25, dust_aod=None):
    """
    Cultural heritage dust-deposition proxy.

    If dust AOD is available, convert it to a proxy deposition scale.
    Otherwise use coarse particulate matter: max(PM10 - PM2.5, 0).
    """
    if dust_aod is not None:
        return np.maximum(np.asarray(dust_aod), 0) * 100
    return np.maximum(np.asarray(pm10) - np.asarray(pm25), 0)


def dust_deposition_risk_category(value):
    if value < 10:
        return 'Low'
    if value < 25:
        return 'Moderate'
    if value < 50:
        return 'High'
    return 'Extreme'


def heritage_compound_score(row):
    """
    Combined cultural heritage material-stress score.

    Combines humidity/moisture stress, dry stress, dust deposition,
    and reactive pollutants NO2, SO2 and O3.
    """
    rh = row['relative_humidity']

    if rh < 40:
        humidity_component = (40 - rh) / 40 * 100
    elif rh <= 60:
        humidity_component = 0
    else:
        humidity_component = min((rh - 60) / 30 * 100, 100)

    dry_component = float(dry_stress_score(rh))
    dust_component = min(row.get('dust_deposition', 0) / 25 * 100, 100)

    pollutant_component = (
        0.40 * row.get('no2', 0) / 40 +
        0.30 * row.get('so2', 0) / 20 +
        0.30 * row.get('o3', 0) / 100
    ) * 100

    return (
        0.35 * humidity_component +
        0.25 * dry_component +
        0.25 * dust_component +
        0.15 * pollutant_component
    )


def heritage_compound_risk_category(value):
    return risk_category(value, [35, 60, 85])


def color_for_risk(risk):
    colors = {
        'Low': '#2ecc71',
        'Moderate': '#f1c40f',
        'High': '#e67e22',
        'Extreme': '#e74c3c',
        'Dry stress': '#3498db'
    }
    return colors.get(risk, '#95a5a6')


def color_for_confidence(confidence):
    colors = {'High': '#27ae60', 'Medium': '#f39c12', 'Low': '#c0392b'}
    return colors.get(confidence, '#7f8c8d')
