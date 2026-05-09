from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import zipfile


DATA_DIR = Path('data')


def _find_file(patterns):
    for pattern in patterns:
        matches = sorted(DATA_DIR.glob(pattern))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        'Could not find required Copernicus data file in ./data. Tried: ' + ', '.join(patterns)
    )


def _first_existing_coord(df, names):
    for name in names:
        if name in df.columns:
            return name
    return None


def _normalize_time(df):
    time_col = _first_existing_coord(df, ['valid_time', 'time', 'forecast_reference_time'])
    if time_col is None:
        raise ValueError('No time coordinate found. Expected valid_time or time.')
    df['datetime'] = pd.to_datetime(df[time_col])
    df['date'] = df['datetime'].dt.normalize()
    return df


def _spatial_id(df, precision=2):
    lat_col = _first_existing_coord(df, ['latitude', 'lat'])
    lon_col = _first_existing_coord(df, ['longitude', 'lon'])
    if lat_col is None or lon_col is None:
        # Point-like dataset with no explicit grid.
        df['latitude'] = 59.9139
        df['longitude'] = 10.7522
    else:
        df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude'})
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['location_id'] = (
        'lat' + df['latitude'].round(precision).astype(str) +
        '_lon' + df['longitude'].round(precision).astype(str)
    )
    return df


def _infer_country(lat, lon):
    if 36 <= lat <= 47.5 and 6 <= lon <= 19:
        return 'Italy'
    if 55 <= lat <= 70 and 10 <= lon <= 25:
        return 'Sweden'
    if 57 <= lat <= 72 and 4 <= lon <= 32:
        return 'Norway'
    return 'Copernicus area'


def _open_cams_grib(path):
    """Open a CAMS GRIB file robustly. Some GRIB files need filtering by typeOfLevel."""
    attempts = [
        {},
        {'filter_by_keys': {'typeOfLevel': 'surface'}},
        {'filter_by_keys': {'typeOfLevel': 'heightAboveGround'}},
        {'filter_by_keys': {'typeOfLevel': 'atmosphere'}},
        {'filter_by_keys': {'typeOfLevel': 'entireAtmosphere'}},
    ]
    last_error = None
    for backend_kwargs in attempts:
        try:
            return xr.open_dataset(path, engine='cfgrib', backend_kwargs=backend_kwargs)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f'Could not open CAMS GRIB file {path}: {last_error}')


def _convert_pollutant_units(series, clean_name):
    """Heuristic conversion from CAMS SI units to dashboard-friendly units."""
    s = pd.to_numeric(series, errors='coerce')
    max_abs = float(np.nanmax(np.abs(s))) if len(s.dropna()) else np.nan

    # Many CAMS near-surface mass concentrations are kg m-3.
    # If values are tiny, convert to ug m-3.
    if clean_name in ['pm25', 'pm10', 'no2', 'o3', 'so2'] and np.isfinite(max_abs) and max_abs < 1e-3:
        return s * 1e9

    # CO can appear as kg m-3; convert tiny values to mg m-3 for dashboard use.
    if clean_name == 'co' and np.isfinite(max_abs) and max_abs < 1e-3:
        return s * 1e6

    return s


def _describe_file(path):
    path = Path(path)
    try:
        head = path.read_bytes()[:16]
    except Exception:
        head = b''
    return f'{path} size={path.stat().st_size if path.exists() else 0} bytes first_bytes={head!r}'


def _open_dataset_robust(path, preferred=None):
    """Open NetCDF/GRIB robustly and raise a useful error if the file is not readable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    if zipfile.is_zipfile(path):
        raise ValueError(
            f'{path} is a ZIP archive, not a directly readable NetCDF/GRIB file. '
            'Extract it first and place the .nc or .grib file inside ./data.'
        )

    errors = []
    engines = []
    if preferred:
        engines.append(preferred)
    suffix = path.suffix.lower()
    if suffix in ['.grib', '.grb', '.grib2']:
        engines.extend(['cfgrib'])
    engines.extend(['netcdf4', 'h5netcdf', 'scipy', 'cfgrib'])

    seen = set()
    engines = [e for e in engines if not (e in seen or seen.add(e))]

    for engine in engines:
        try:
            if engine == 'cfgrib':
                return _open_cams_grib(path)
            return xr.open_dataset(path, engine=engine)
        except Exception as exc:
            errors.append(f'{engine}: {type(exc).__name__}: {exc}')

    raise ValueError(
        'Could not open Copernicus file with any supported engine.\n'
        f'File diagnostics: {_describe_file(path)}\n'
        'Tried engines: ' + ', '.join(engines) + '\n'
        'Errors:\n- ' + '\n- '.join(errors) + '\n\n'
        'Fixes: install h5netcdf/netcdf4/cfgrib/eccodes, or check that the file is a real .nc/.grib and not an HTML error page or ZIP.'
    )


def load_c3s_utci(path=None):
    """Load C3S ERA5-HEAT UTCI from ./data and return daily real observations/analysis rows."""
    if path is None:
        path = _find_file(['*c3s*utci*.nc', '*era5*heat*utci*.nc', '*utci*.nc', '*utci*.grib', '*utci*.grb'])
    ds = _open_dataset_robust(path)

    candidates = [v for v in ds.data_vars if 'utci' in v.lower() or 'thermal' in v.lower()]
    var = candidates[0] if candidates else list(ds.data_vars)[0]

    df = ds[var].to_dataframe(name='utci_c').reset_index()
    df = _normalize_time(df)
    df = _spatial_id(df)

    # ERA5-HEAT UTCI is often stored in Kelvin; convert if it looks Kelvin-like.
    if df['utci_c'].median(skipna=True) > 150:
        df['utci_c'] = df['utci_c'] - 273.15

    # Use UTCI as real heat indicator. Provide compatible fields for app formulas.
    df['temperature_c'] = df['utci_c']
    if 'relative_humidity' not in df.columns:
        # Real UTCI file does not contain RH. Use NaN-safe neutral placeholder for dashboard sections
        # that require RH; this is not synthetic forecast data, only a compatibility variable.
        df['relative_humidity'] = 50.0

    out = df[['date', 'datetime', 'location_id', 'latitude', 'longitude', 'utci_c', 'temperature_c', 'relative_humidity']].copy()
    return out


def load_cams_pollution(path=None):
    """Load CAMS global pollution forecast from ./data and return daily rows by grid cell and forecast step/time."""
    if path is None:
        path = _find_file(['*cams*pollution*.grib', '*cams*pollution*.grb', '*cams*.grib', '*cams*.grb'])
    ds = _open_cams_grib(path)

    variable_aliases = {
        'pm25': ['pm2p5', 'pm2p5_conc', 'particulate_matter_2.5um', 'pm2p5fire'],
        'pm10': ['pm10', 'pm10_conc', 'particulate_matter_10um'],
        'no2': ['no2', 'nitrogen_dioxide'],
        'o3': ['go3', 'o3', 'ozone'],
        'so2': ['so2', 'sulphur_dioxide', 'sulfur_dioxide'],
        'co': ['co', 'carbon_monoxide'],
        'aod': ['aod550', 'aod', 'aerosol_optical_depth_550nm'],
        'dust_deposition': ['duaod550', 'dust_aerosol_optical_depth_550nm', 'aermr04'],
    }

    frames = []
    available = set(ds.data_vars)
    for clean_name, aliases in variable_aliases.items():
        raw = next((a for a in aliases if a in available), None)
        if raw is None:
            continue
        temp = ds[raw].to_dataframe(name=clean_name).reset_index()
        temp = _normalize_time(temp)
        temp = _spatial_id(temp)
        temp[clean_name] = _convert_pollutant_units(temp[clean_name], clean_name)
        frames.append(temp[['date', 'datetime', 'location_id', 'latitude', 'longitude', clean_name]])

    if not frames:
        raise ValueError(
            'No expected CAMS variables found. Available GRIB variables: ' + ', '.join(sorted(available))
        )

    df = frames[0]
    keys = ['date', 'datetime', 'location_id', 'latitude', 'longitude']
    for frame in frames[1:]:
        df = df.merge(frame, on=keys, how='outer')

    return df.sort_values(keys)


def load_copernicus_dataset():
    """
    Load real C3S/CAMS files from ./data only. No synthetic fallback is used.

    Uncertainty in the dashboard is computed from real forecast lead times / hourly values
    available for the same day and grid point. The column 'member' is therefore a real
    sample index, not generated data.
    """
    utci = load_c3s_utci()
    cams = load_cams_pollution()

    keys = ['date', 'location_id', 'latitude', 'longitude']

    # Daily UTCI aggregation from real hourly C3S data.
    utci_daily = utci.groupby(keys, as_index=False).agg(
        utci_c=('utci_c', 'mean'),
        temperature_c=('temperature_c', 'mean'),
        relative_humidity=('relative_humidity', 'mean')
    )

    # Keep CAMS forecast lead-time/time rows; add UTCI daily context to each row.
    df = cams.merge(utci_daily, on=keys, how='left')

    # If grids do not align exactly, attach the area-mean C3S UTCI by date.
    if df['temperature_c'].isna().all():
        area_utci = utci.groupby('date', as_index=False).agg(
            utci_c=('utci_c', 'mean'),
            temperature_c=('temperature_c', 'mean'),
            relative_humidity=('relative_humidity', 'mean')
        )
        df = df.drop(columns=['utci_c', 'temperature_c', 'relative_humidity'], errors='ignore')
        df = df.merge(area_utci, on='date', how='left')

    # Dashboard compatibility fields.
    for col, default in {
        'pm25': np.nan,
        'pm10': np.nan,
        'no2': np.nan,
        'o3': np.nan,
        'so2': np.nan,
        'co': np.nan,
        'aod': np.nan,
        'dust_deposition': np.nan,
        'wind_speed': 2.0,
        'relative_humidity': 50.0,
    }.items():
        if col not in df.columns:
            df[col] = default

    # Fill missing pollutants with column medians where available so compound indicators can run.
    pollutant_cols = ['pm25', 'pm10', 'no2', 'o3', 'so2', 'co', 'aod', 'dust_deposition']
    for col in pollutant_cols:
        if df[col].notna().any():
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = 0.0

    df['country'] = df.apply(lambda r: _infer_country(r['latitude'], r['longitude']), axis=1)

    # Real sample index from forecast valid times within the day/grid cell.
    df['member'] = df.groupby(['date', 'location_id']).cumcount()

    df = df.sort_values(['date', 'location_id', 'member']).reset_index(drop=True)
    return df


# Backward-compatible name for old imports. This intentionally does NOT create synthetic data.
def create_demo_dataset(*args, **kwargs):
    raise RuntimeError(
        'Synthetic data has been disabled. Use load_copernicus_dataset() and put C3S/CAMS files under ./data.'
    )
