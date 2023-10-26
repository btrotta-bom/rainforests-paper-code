"""Function to get input/output files for inference."""

import os
import pandas as pd


def get_file_list(forecast_type, input_folder_site, output_folder_site, input_folder_grid, output_folder_grid, accum_periods, start_date, end_date):
    file_list = []
    if forecast_type == "site":
        if not (os.path.exists(output_folder_site)):
            os.makedirs(output_folder_site, exist_ok=True)
        for accum_period in accum_periods:
            lead_times = range(accum_period, 240 + 1, accum_period)
            for lead_time in lead_times:
                accum_period_string = f"PT{accum_period:02d}H"
                input_path = os.path.join(
                    input_folder_site, accum_period_string, f"PT{lead_time:04d}H00M.nc"
                )
                output_path = os.path.join(
                    output_folder_site,
                    f"calibrated_forecast-lead_time_{lead_time:03d}-{accum_period_string}.nc",
                )
                if os.path.exists(input_path):
                    file_list.append([input_path, output_path, accum_period, lead_time])
    else:
        if not (os.path.exists(output_folder_grid)):
            os.makedirs(output_folder_grid, exist_ok=True)
        for accum_period in accum_periods:
            accum_period_string = f"PT{accum_period:02d}H"
            lead_time = 24
            for basetime in pd.date_range(start_date, end_date, freq="D"):
                formatted_basetime = basetime.strftime("%Y%m%dT%H%MZ")
                valid_time = basetime + pd.Timedelta(hours=lead_time)
                formatted_valid_time = valid_time.strftime("%Y%m%dT%H%MZ")
                filename = (
                    f"{formatted_valid_time}-PT{lead_time:04d}H00M-precipitation_accumulation-PT24H.nc"
                )
                input_path = os.path.join(input_folder_grid, formatted_basetime, filename)
                output_path = os.path.join(output_folder_grid, filename)
                if os.path.exists(input_path):
                    file_list.append([input_path, output_path, accum_period, lead_time])
    return file_list
