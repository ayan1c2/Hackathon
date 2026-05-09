"""
Download C3S/CAMS data for the uncertainty-aware indicators dashboard.

Datasets:
1. C3S ERA5-HEAT UTCI from CDS
2. CAMS global atmospheric composition forecasts from ADS

Why this version:
- C3S and CAMS live on different Data Store endpoints.
- C3S UTCI must be requested from the CDS endpoint, not ADS.
- CAMS variables can fail when requested together because of valid-combination
  rules in the ADS form/API. This script downloads CAMS variables one-by-one
  so one bad variable does not stop the whole download.

Run:
    python download_copernicus_data.py
"""

from pathlib import Path
import zipfile
import cdsapi


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


CDS_URL = "https://cds.climate.copernicus.eu/api"
ADS_URL = "https://ads.atmosphere.copernicus.eu/api"


# -------------------------------
# Helpers
# -------------------------------
def extract_zip(zip_path, extract_dir=None):
    zip_path = Path(zip_path)
    if extract_dir is None:
        extract_dir = zip_path.with_suffix("")
    extract_dir = Path(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(zip_path):
        print(f"Extracting {zip_path} -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    return extract_dir


def safe_download(client, dataset, request, output_file):
    """
    Download with the new retrieve(...).download(...) style.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result = client.retrieve(dataset, request)
    result.download(str(output_file))
    return output_file


# -------------------------------
# 1. C3S UTCI from CDS
# -------------------------------
def download_c3s_utci(
    year="2024",
    month="07",
    days=None,
    area=None,
    output_file="data/c3s_era5_heat_utci.zip",
    extract=True,
):
    """
    Download C3S ERA5-HEAT UTCI.

    Important:
    Use the CDS endpoint:
        https://cds.climate.copernicus.eu/api

    The previous error happened because the script tried to retrieve
    derived-utci-historical from ADS, where that dataset does not exist.
    """

    if days is None:
        days = [f"{d:02d}" for d in range(1, 8)]

    if area is None:
        # North, West, South, East
        area = [61.5, 8.0, 58.5, 12.5]

    client = cdsapi.Client(url=CDS_URL)

    dataset = "derived-utci-historical"

    request = {
        "variable": "universal_thermal_climate_index",
        "version": "1_1",
        "product_type": "consolidated_dataset",
        "year": year,
        "month": month,
        "day": days,
        "time": [
            "00:00", "03:00", "06:00", "09:00",
            "12:00", "15:00", "18:00", "21:00",
        ],
        "area": area,
        "format": "netcdf",
    }

    output_file = safe_download(client, dataset, request, output_file)

    if extract and zipfile.is_zipfile(output_file):
        extract_zip(output_file, DATA_DIR / "c3s_era5_heat_utci")

    return str(output_file)


# -------------------------------
# 2. CAMS from ADS
# -------------------------------
CAMS_VARIABLES = {
    # Particles
    "pm25": "particulate_matter_2.5um",
    "pm10": "particulate_matter_10um",

    # Gases
    "no2": "nitrogen_dioxide",
    "o3": "ozone",
    "so2": "sulphur_dioxide",
    "co": "carbon_monoxide",

    # Aerosols
    "aod550": "aerosol_optical_depth_550nm",
    "dust_aod550": "dust_aerosol_optical_depth_550nm",
}


def download_single_cams_variable(
    variable_key,
    variable_name,
    date="2024-07-01",
    area=None,
    leadtime_hours=None,
    time="00:00",
    output_dir="data/cams_global_pollution",
    data_format="netcdf_zip",
):
    """
    Download one CAMS variable.

    Downloading one variable at a time is slower but more robust because
    ADS valid-combination rules may reject mixed variable groups.
    """

    if area is None:
        area = [61.5, 8.0, 58.5, 12.5]

    if leadtime_hours is None:
        leadtime_hours = ["0", "3", "6", "9", "12", "15", "18", "21", "24"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url=ADS_URL)

    dataset = "cams-global-atmospheric-composition-forecasts"

    request = {
        "date": [f"{date}/{date}"],
        "type": ["forecast"],
        "data_format": data_format,
        "variable": [variable_name],
        "leadtime_hour": leadtime_hours,
        "time": [time],
        "area": area,
    }

    suffix = "zip" if data_format == "netcdf_zip" else "grib"
    output_file = output_dir / f"cams_{variable_key}_{date}.{suffix}"

    print(f"\nDownloading CAMS variable: {variable_key} = {variable_name}")

    try:
        downloaded = safe_download(client, dataset, request, output_file)
        print(f"Saved: {downloaded}")

        if data_format == "netcdf_zip" and zipfile.is_zipfile(downloaded):
            extract_zip(downloaded, output_dir / variable_key)

        return str(downloaded), None

    except Exception as exc:
        print(f"FAILED: {variable_key} = {variable_name}")
        print(f"Reason: {exc}")
        return None, str(exc)


def download_cams_global_forecast(
    date="2024-07-01",
    area=None,
    leadtime_hours=None,
    time="00:00",
    output_dir="data/cams_global_pollution",
    data_format="netcdf_zip",
):
    """
    Download CAMS variables one by one.

    Returns:
        successes, failures
    """

    successes = {}
    failures = {}

    for key, variable in CAMS_VARIABLES.items():
        path, error = download_single_cams_variable(
            variable_key=key,
            variable_name=variable,
            date=date,
            area=area,
            leadtime_hours=leadtime_hours,
            time=time,
            output_dir=output_dir,
            data_format=data_format,
        )

        if path:
            successes[key] = path
        else:
            failures[key] = error

    return successes, failures


# -------------------------------
# Inspection utility
# -------------------------------
def inspect_downloaded_netcdf(folder="data/cams_global_pollution"):
    """
    Print variables found in extracted CAMS NetCDF files.
    """
    try:
        import xarray as xr
    except ImportError:
        print("Install xarray first: pip install xarray netCDF4")
        return

    folder = Path(folder)
    nc_files = sorted(folder.rglob("*.nc"))

    if not nc_files:
        print(f"No .nc files found under {folder}")
        return

    print("\nInspecting extracted CAMS NetCDF files:")
    for f in nc_files:
        try:
            ds = xr.open_dataset(f)
            print(f"\nFILE: {f}")
            print("VARIABLES:", list(ds.data_vars))
            print("DIMS:", dict(ds.dims))
        except Exception as exc:
            print(f"Could not open {f}: {exc}")


# -------------------------------
# MAIN
# -------------------------------
def main():
    # Change these as needed.
    date = "2020-01-01"
    area = [61.2, 8.0, 58.8, 12.4]
    leadtime_hours = ["0", "3", "6", "9", "12", "15", "18", "21", "24"]

    print("\nDownloading C3S ERA5-HEAT UTCI...")
    try:
        utci_file = download_c3s_utci(
            year=date[:4],
            month=date[5:7],
            days=[date[8:10]],
            area=area,
            output_file="data/c3s_era5_heat_utci.zip",
            extract=True,
        )
        print(f"Saved UTCI: {utci_file}")
    except Exception as exc:
        print("UTCI download failed:", exc)
        print("\nLikely causes:")
        print("- CDS licence not accepted for derived-utci-historical")
        print("- CDS API token missing or wrong")
        print("- Your .cdsapirc points only to ADS; this script explicitly uses CDS_URL")

    print("\nDownloading CAMS global pollution forecast...")
    successes, failures = download_cams_global_forecast(
        date=date,
        area=area,
        leadtime_hours=leadtime_hours,
        output_dir="data/cams_global_pollution",
        data_format="netcdf_zip",
    )

    print("\nCAMS successes:")
    for key, path in successes.items():
        print(f" - {key}: {path}")

    if failures:
        print("\nCAMS failures:")
        for key, error in failures.items():
            print(f" - {key}: {error[:300]}")

        print("\nNote:")
        print("Some CAMS variables may not be valid for your selected date, format, area,")
        print("lead time, or product combination. Because this script downloads one")
        print("variable at a time, successful variables are still kept.")

    inspect_downloaded_netcdf("data/cams_global_pollution")


if __name__ == "__main__":
    main()
