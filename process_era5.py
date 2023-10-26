import pandas as pd
import xarray as xr
from pandas import Timestamp
import os

FIRST_BASETIME = Timestamp("2021-09-01T00:00Z")
LAST_BASETIME = Timestamp("2022-09-01T00:00Z")

era5_dir = "/path/to/era5/"

da_arr = []
for month_start in pd.date_range(
    FIRST_BASETIME, LAST_BASETIME, freq="MS", inclusive="left"
):
    print(month_start)
    month_end = month_start + pd.tseries.offsets.MonthEnd(0)
    month_start_formatted = month_start.strftime("%Y%m%d")
    month_end_formatted = month_end.strftime("%Y%m%d")
    filename = f"{era5_dir}/mtpr_era5_oper_sfc_{month_start_formatted}-{month_end_formatted}.nc"
    ds = xr.open_dataset(filename)
    da = ds["mtpr"]
    da_arr.append(da)
da = xr.concat(da_arr, dim="time")
# convert to daily mean precip rate
da = da.resample(time="1D", closed="right", label="right").mean()
# convert to mm; original units are kg/m**2/s = mm/s
da = da * 3600 * 24
da.attrs["units"] = "mm"


output_dir = os.path.join(era5_dir, "processed")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, "20210901_20220901.nc")
da.to_netcdf(output_path)
