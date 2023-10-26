"""Load forecast and obs datasets, format for input into Improver
calibration.dataframe_utilities.forecast_and_truth_dataframes_to_cubes"""

import datetime as dt
import os

import pandas as pd
from raincal.site_obs.observation_config import MAX_OBS_OVER_PERIOD
from raincal.model.train import _get_site_ids, _exclude_large_increment_gauges, _exclude_dew

from improver.calibration.dataframe_utilities import (
    _prepare_dataframes,
    forecast_dataframe_to_cube,
    truth_dataframe_to_cube,
)
from improver.utilities.save import save_netcdf
from multiprocessing import Pool


model = "accessge3"
variable = "precipitation_accumulation"
accum_periods = [24]
# date period used for calibration; start and end dates should be first of month
start_date = dt.datetime(2020, 8, 1)
end_date = dt.datetime(2021, 8, 1)
output_folder = f"/path/to/output/folder/"
base_obs_dir = "/path/to/obs/dir/"
base_fcst_dir = "/path/to/nwp/forecast/sites"

def process_lead_time(lead_time, accum_period):
    print(f"Processing {accum_period}H accumulations for lead time {lead_time}H")

    # obs
    accum_period_string = f"PT{accum_period:02d}H"
    # get obs data from the merged dataframes so it is the same as used in rainforests training
    obs_dir = f"base_obs_dir/{accum_period_string}/{model}"
    months = pd.date_range(start_date, end_date, freq="MS", closed="left")
    df_arr = []
    for d in months:
        month_string = d.strftime("%Y-%m")
        filename = f"{model}-training-{month_string}-{accum_period_string}.parquet"
        curr_df = pd.read_parquet(os.path.join(obs_dir, filename))
        # solar radiation is needed for _exclude_dew
        cols = ["precipitation_accumulation_mm_obs", "valid_time", "site_id", "clearsky_solar_radiation_W_s_m-2"]
        curr_df = curr_df.loc[curr_df["lead_time_hours"] == lead_time, cols]
        df_arr.append(curr_df)
    obs_df = pd.concat(df_arr, axis=0, ignore_index=True)

    # Apply the same filtering as in Rainforests
    # Filter on observation
    MAX_OBS = MAX_OBS_OVER_PERIOD[accum_period]
    obs_df = obs_df.loc[obs_df["precipitation_accumulation_mm_obs"] < MAX_OBS]
    # Remove large-increment gauges
    gauge_site_ids = _get_site_ids("gauge", accum_period)
    obs_df = _exclude_large_increment_gauges(obs_df, gauge_site_ids)
    # exclude radar observations
    obs_df = obs_df.loc[obs_df["site_id"].isin(gauge_site_ids)]
    # exclude dew
    _exclude_dew(obs_df, gauge_site_ids)

    # reformat
    obs_df.rename(
        columns={
            "precipitation_accumulation_mm_obs": "ob_value",
            "valid_time": "time",
            "site_id": "station_id"
        },
        inplace=True,
    )
    obs_df = obs_df.loc[obs_df["ob_value"].notnull()]
    obs_df["ob_value"] *= 0.001
    obs_df["wmo_id"] = 0
    obs_df["diagnostic"] = variable
    obs_df["latitude"] = 0
    obs_df["longitude"] = 0
    obs_df["altitude"] = 0


    # forecast
    accum_period_string = f"PT{accum_period:02d}H"
    fcst_dir = base_fcst_dir
    valid_times = obs_df["time"].unique()
    df_arr = []
    for valid_time in valid_times:
        valid_time_string = valid_time.strftime("%Y%m%dT%H%MZ")
        lead_time_string = f"PT{lead_time:04d}H00M"
        filename = f"{valid_time_string}-{lead_time_string}-{variable}-{accum_period_string}.parquet"
        path = os.path.join(fcst_dir, filename)
        df_arr.append(pd.read_parquet(path))
    fcst_df = pd.concat(df_arr, axis=0, ignore_index=True)
    fcst_df.rename(
        columns={
            "precipitation_accumulation_mm": "forecast",
            "valid_time": "time",
            "site_id": "station_id",
        },
        inplace=True,
    )
    fcst_df = fcst_df.loc[fcst_df["forecast"].notnull()]
    fcst_df["forecast"] *= 0.001
    fcst_df["blend_time"] = fcst_df["time"] - pd.Timedelta(hours=lead_time)
    fcst_df["forecast_period"] = pd.Timedelta(hours=lead_time)
    fcst_df["forecast_reference_time"] = fcst_df["blend_time"]
    fcst_df["wmo_id"] = 0
    fcst_df["diagnostic"] = variable
    fcst_df["latitude"] = 0
    fcst_df["longitude"] = 0
    fcst_df["altitude"] = 0
    fcst_df["period"] = pd.Timedelta(accum_period * 3600)
    fcst_df["height"] = 0
    fcst_df["cf_name"] = "lwe_thickness_of_precipitation_amount"
    fcst_df["units"] = "m"
    fcst_df["experiment"] = model
    tz = dt.timezone.utc
    fcst_df["time"] = fcst_df["time"].dt.tz_localize(tz)

    # convert to cubes
    training_dates = pd.date_range(start_date, end_date, closed="left", tz="UTC")
    forecast_period = lead_time * 3600
    fcst_df, obs_df = _prepare_dataframes(
        fcst_df, obs_df, forecast_period, experiment=model,
    )
    fcst_cube = forecast_dataframe_to_cube(fcst_df, training_dates, forecast_period)
    truth_cube = truth_dataframe_to_cube(obs_df, training_dates)

    # write output
    output_filename = "{data_type}-lead_time_{lead_time:03d}-{accum_period_string}.nc"
    output_path_fcst = os.path.join(
        output_folder,
        output_filename.format(
            data_type="forecast",
            lead_time=lead_time,
            accum_period_string=accum_period_string,
        ),
    )
    output_path_obs = os.path.join(
        output_folder,
        output_filename.format(
            data_type="truth",
            lead_time=lead_time,
            accum_period_string=accum_period_string,
        ),
    )
    save_netcdf(fcst_cube, output_path_fcst)
    save_netcdf(truth_cube, output_path_obs)


if __name__ == "__main__":

    if not (os.path.exists(output_folder)):
        os.makedirs(output_folder, exist_ok=True)

    args = []
    for accum_period in accum_periods:

        lead_times = range(accum_period, 240 + 1, accum_period)
        for lead_time in lead_times:
            args.append([lead_time, accum_period])


    with Pool(5) as p:
        p.starmap(process_lead_time, args)
