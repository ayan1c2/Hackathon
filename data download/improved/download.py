"""
Download C3S/CAMS data for the uncertainty-aware indicators dashboard.

Datasets:
1. C3S ERA5-HEAT UTCI
2. CAMS global atmospheric composition forecast (ADS)

Run:
    python download_copernicus_data.py
"""

from pathlib import Path
import zipfile
import cdsapi

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


# -------------------------------
# 1. C3S UTCI
# -------------------------------
def download_c3s_utci(
    year="2020",
    month="01",
    days=None,
    area=None,
    output_file="data/c3s_era5_heat_utci.zip",
):
    if days is None:
        days = [f"{d:02d}" for d in range(1, 8)]

    if area is None:
        area = [61.5, 8.0, 58.5, 12.5]  # North, West, South, East

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
            "12:00", "15:00", "18:00", "21:00",
        ],
        "area": area,
        "format": "netcdf",
    }

    client.retrieve("derived-utci-historical", request).download(output_file)
    return output_file


# -------------------------------
# 2. CAMS ADS - grouped NetCDF ZIP
# -------------------------------
def download_cams_global_forecast(
    date="2020-01-01",
    area=None,
    leadtime_hours=None,
    time="00:00",
    output_dir="data/cams_global_pollution",
):
    """
    Downloads CAMS in grouped NetCDF ZIP files.

    Why grouped?
    CAMS GRIB/NetCDF messages can split variables by level/type/metadata.
    Downloading grouped NetCDF ZIPs avoids the common issue where only
    PM2.5/PM10 appear while gases or AOD are hidden in another group.
    """

    if area is None:
        area = [61.5, 8.0, 58.5, 12.5]

    if leadtime_hours is None:
        leadtime_hours = ["0", "3", "6", "9", "12", "15", "18", "21", "24"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client(url="https://ads.atmosphere.copernicus.eu/api")

    dataset = "cams-global-atmospheric-composition-forecasts"

    variable_groups = {
        "particles": [
            "particulate_matter_2.5um",
            "particulate_matter_10um",
        ],
        "gases": [
            "nitrogen_dioxide",
            "ozone",
            "sulphur_dioxide",
            "carbon_monoxide",
        ],
        "aerosols": [
            "aerosol_optical_depth_550nm",
            "dust_aerosol_optical_depth_550nm",
        ],
    }

    downloaded = []

    for group_name, variables in variable_groups.items():
        zip_file = output_dir / f"cams_{group_name}_{date}.zip"
        extract_dir = output_dir / group_name
        extract_dir.mkdir(exist_ok=True)

        request = {
            "date": [f"{date}/{date}"],
            "type": ["forecast"],
            "data_format": "netcdf_zip",
            "variable": variables,
            "leadtime_hour": leadtime_hours,
            "time": [time],
            "area": area,
        }

        print(f"\nDownloading CAMS group: {group_name}")
        print(f"Variables: {variables}")

        client.retrieve(dataset, request).download(str(zip_file))

        print(f"Extracting: {zip_file}")
        with zipfile.ZipFile(zip_file, "r") as zf:
            zf.extractall(extract_dir)

        downloaded.append(str(zip_file))

    return downloaded


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
        cams_files = download_cams_global_forecast()
        print("Saved CAMS files:")
        print("Saved CAMS files:")
        for f in cams_files:
            print(" -", f)
    except Exception as e:
        print("CAMS download failed:", e)
        print("\nLikely causes:")
        print("- ADS licence not accepted")
        print("- Missing ADS API key")
        print("- Dataset access not enabled")
        print("- Variable name not valid for the selected product/API version")


if __name__ == "__main__":
    main()