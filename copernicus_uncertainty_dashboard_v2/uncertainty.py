import numpy as np


def ensemble_summary(df, variable):
    grouped = df.groupby('date')[variable]
    summary = grouped.agg(
        mean='mean', median='median', std='std', min='min', max='max',
        q05=lambda x: np.percentile(x, 5),
        q25=lambda x: np.percentile(x, 25),
        q75=lambda x: np.percentile(x, 75),
        q95=lambda x: np.percentile(x, 95)
    ).reset_index()
    summary['iqr'] = summary['q75'] - summary['q25']
    summary['range'] = summary['max'] - summary['min']
    summary['cv'] = summary['std'] / summary['mean'].replace(0, np.nan)
    return summary


def confidence_from_spread(std, mean, low_threshold=0.08, high_threshold=0.18):
    rel_spread = abs(std / mean) if mean != 0 else np.inf
    if rel_spread < low_threshold:
        return 'High'
    if rel_spread < high_threshold:
        return 'Medium'
    return 'Low'


def probability_of_exceedance(df, variable, threshold):
    col = f'prob_{variable}_above_{threshold}'
    return (df.assign(exceed=lambda x: x[variable] > threshold)
              .groupby('date')['exceed'].mean().reset_index()
              .rename(columns={'exceed': col}))


def monte_carlo_indicator(temp_c, rh, temp_sd, rh_sd, n=2000, indicator_function=None, seed=42):
    rng = np.random.default_rng(seed)
    temp_samples = rng.normal(temp_c, temp_sd, n)
    rh_samples = np.clip(rng.normal(rh, rh_sd, n), 0, 100)
    values = indicator_function(temp_samples, rh_samples)
    return {
        'mean': np.mean(values),
        'std': np.std(values),
        'q05': np.percentile(values, 5),
        'q50': np.percentile(values, 50),
        'q95': np.percentile(values, 95),
        'prob_above_heat_warning': np.mean(values > 32)
    }
