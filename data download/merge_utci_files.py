
import xarray as xr
from pathlib import Path
import pandas as pd

def merge_utci_files(input_folder="data", output_file="data/c3s_era5_heat_utci.nc"):
    """
    Merge multiple NetCDF UTCI files into a single dataset.
    """
    folder = Path(input_folder)
    nc_files = sorted(folder.glob("*.nc"))

    if not nc_files:
        raise ValueError(f"No .nc files found in {input_folder}")

    print(f"Found {len(nc_files)} files. Merging...")

    ds = xr.open_mfdataset(nc_files, combine="by_coords")

    ds.to_netcdf(output_file)
    print(f"Merged dataset saved to {output_file}")

    return ds


def convert_to_dataframe(input_file="data/c3s_era5_heat_utci.nc"):
    """
    Convert merged dataset to pandas DataFrame (spatial mean).
    """
    ds = xr.open_dataset(input_file)

    var_name = list(ds.data_vars)[0]

    da = ds[var_name].mean(dim=[d for d in ds[var_name].dims if d in ["latitude", "longitude"]])

    df = da.to_dataframe().reset_index()

    if "time" in df.columns:
        df = df.rename(columns={"time": "date"})

    df = df.rename(columns={var_name: "utci_c"})

    print("Converted to DataFrame")
    return df


if __name__ == "__main__":
    merge_utci_files()
    df = convert_to_dataframe()
    print(df.head())
