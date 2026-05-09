import pandas as pd
import numpy as np


def download_utci_from_cds(year='2023', month='07', day='15', area=None, output_file='utci_era5_heat.nc'):
    if area is None:
        area = [72, -10, 35, 30]
    import cdsapi
    client = cdsapi.Client()
    client.retrieve(
        'derived-utci-historical',
        {
            'variable': 'universal_thermal_climate_index',
            'version': '1_1',
            'product_type': 'consolidated_dataset',
            'year': year,
            'month': month,
            'day': day,
            'time': ['00:00', '06:00', '12:00', '18:00'],
            'area': area,
            'format': 'netcdf'
        },
        output_file
    )
    return output_file


def create_demo_dataset(n_days=30, n_members=20, seed=42, n_locations=25):
    """
    Synthetic C3S/CAMS-like ensemble dataset with latitude/longitude.
    The spatial grid enables uncertainty map visualisation during hackathons.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2026-07-01', periods=n_days, freq='D')
    seasonal = np.linspace(0, np.pi, n_days)
    pollution_wave = np.linspace(0, 4 * np.pi, n_days)
    rows = []

    # Compact Oslo/Northern Europe demonstration grid.
    lat_values = np.linspace(58.6, 61.4, int(np.sqrt(n_locations)))
    lon_values = np.linspace(8.2, 12.4, int(np.sqrt(n_locations)))
    locations = [(lat, lon) for lat in lat_values for lon in lon_values]

    for loc_id, (lat, lon) in enumerate(locations):
        urban_factor = rng.uniform(0.7, 1.3)
        coastal_cooling = (lon - np.mean(lon_values)) * 0.15
        elevation_or_exposure = rng.normal(0, 0.5)

        for member in range(n_members):
            member_bias = rng.normal(0, 0.8)
            temp = 30 + 6 * np.sin(seasonal) + member_bias - coastal_cooling + elevation_or_exposure + rng.normal(0, 1.7, n_days)
            rh = 56 + 16 * np.sin(np.linspace(0, 2.8 * np.pi, n_days)) + rng.normal(0, 6.5, n_days)
            wind = np.clip(rng.normal(2.8, 0.9, n_days), 0.2, 12)
            pm25 = urban_factor * (17 + 8 * np.sin(pollution_wave)) + rng.normal(0, 3.5, n_days)
            pm10 = pm25 * rng.normal(1.7, 0.12, n_days) + rng.normal(4, 3, n_days)
            no2 = urban_factor * (24 + 9 * np.sin(pollution_wave + 0.7)) - 1.3 * wind + rng.normal(0, 4, n_days)
            o3 = 82 + 14 * np.sin(seasonal) - 0.25 * no2 + rng.normal(0, 8, n_days)
            so2 = 4 + 1.2 * np.sin(pollution_wave + 1.4) + rng.normal(0, 0.9, n_days)
            co = 0.35 + 0.08 * np.sin(pollution_wave + 0.3) + rng.normal(0, 0.04, n_days)
            aod = 0.12 + 0.006 * np.clip(pm25, 0, None) + rng.normal(0, 0.025, n_days)
            dust_deposition = np.clip(6 + 0.25 * np.clip(pm10 - pm25, 0, None) + rng.normal(0, 2.0, n_days), 0, None)

            for i, date in enumerate(dates):
                rows.append({
                    'date': date,
                    'member': member,
                    'location_id': loc_id,
                    'latitude': lat,
                    'longitude': lon,
                    'temperature_c': temp[i],
                    'relative_humidity': np.clip(rh[i], 20, 100),
                    'wind_speed': wind[i],
                    'pm25': np.clip(pm25[i], 1, 150),
                    'pm10': np.clip(pm10[i], 2, 250),
                    'no2': np.clip(no2[i], 1, 200),
                    'o3': np.clip(o3[i], 1, 250),
                    'so2': np.clip(so2[i], 0.1, 80),
                    'co': np.clip(co[i], 0.05, 10),
                    'aod': np.clip(aod[i], 0.01, 2),
                    'dust_deposition': dust_deposition[i]
                })

    return pd.DataFrame(rows)
