"""
Download C3S/CAMS data for the uncertainty-aware indicators dashboard.

Datasets:
1. C3S ERA5-HEAT UTCI
2. CAMS global atmospheric composition forecast (ADS)

Run:
    python download_copernicus_data.py
"""

from pathlib import Path
import cdsapi

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# -------------------------------
# 1. C3S UTCI
# -------------------------------
def download_c3s_utci(
    year="2024",
    month="07",
    days=None,
    area=None,
    output_file="data/c3s_era5_heat_utci.nc"
):

    if days is None:
        days = [f"{d:02d}" for d in range(1, 8)]

    if area is None:
        area = [61.5, 8.0, 58.5, 12.5]

    client = cdsapi.Client()

    request = {
        "variable": "universal_thermal_climate_index",
        "version": "1_1",
        "product_type": "consolidated_dataset",
        "year": year,
        "month": month,
        "day": days,
        "time": [
            "00:00", "03:00", "06:00", "09:00",
            "12:00", "15:00", "18:00", "21:00"
        ],
        "area": area,
        "format": "netcdf"
    }

    result = client.retrieve("derived-utci-historical", request)
    result.download(output_file)

    return output_file


# -------------------------------
# 2. CAMS (ADS) – NEW API STYLE
# -------------------------------
def download_cams_global_forecast(
    date="2024-07-01",
    output_file="data/cams_global_pollution.grib"
):
    """
    Uses NEW CDS/ADS API syntax (.download()).
    Requires ADS credentials and accepted licence.
    """

    client = cdsapi.Client(
        url="https://ads.atmosphere.copernicus.eu/api"
    )

    dataset = "cams-global-atmospheric-composition-forecasts"

    request = {
        "date": [f"{date}/{date}"],
        "type": ["forecast"],
        "data_format": "grib",

        # Add useful variables
        "variable": [
            "particulate_matter_2.5um",
            "particulate_matter_10um",
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "carbon_monoxide",
            "aerosol_optical_depth_550nm",
            "dust_aerosol_optical_depth_550nm"
        ],

        # Reduce size for testing
        "leadtime_hour": ["0", "6", "12", "24"],
        "time": ["00:00"],

        # Optional region
        "area": [61.5, 8.0, 58.5, 12.5]
    }

    result = client.retrieve(dataset, request)
    result.download(output_file)

    return output_file


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("\nDownloading C3S ERA5-HEAT UTCI...")
    try:
        utci_file = download_c3s_utci()
        print(f"Saved: {utci_file}")
    except Exception as e:
        print("UTCI download failed:", e)

    print("\nDownloading CAMS global pollution forecast...")
    try:
        cams_file = download_cams_global_forecast()
        print(f"Saved: {cams_file}")
    except Exception as e:
        print("CAMS download failed:", e)
        print("\n👉 Likely causes:")
        print("- ADS licence not accepted")
        print("- Missing ADS API key")
        print("- Dataset access not enabled")


if __name__ == "__main__":
    main()