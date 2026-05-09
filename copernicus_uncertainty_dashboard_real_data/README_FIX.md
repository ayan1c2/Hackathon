# Real C3S/CAMS dashboard - patched loader

This version does not use synthetic data. It loads files from ./data only.

Expected structure:

```text
copernicus_uncertainty_dashboard_real_data/
  app.py
  data_access.py
  data/
    c3s_era5_heat_utci.nc       # or .grib/.grb
    cams_global_pollution.grib  # or .grb
```

Install dependencies:

```bash
pip install -r requirements.txt
```

On Windows, GRIB support is often easiest with conda:

```bash
conda install -c conda-forge eccodes cfgrib
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

If the UTCI file is actually a ZIP archive, extract it first and place the real .nc or .grib file in ./data.
