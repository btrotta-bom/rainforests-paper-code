import datetime as dt
import json
import os
from multiprocessing import Pool

import iris
import numpy as np
from utilities import get_file_list

from improver.calibration.reliability_calibration import ApplyReliabilityCalibration
from improver.cli import threshold
from improver.utilities.save import save_netcdf

accum_periods = [24]
n_bins = 50
model = "ecmwf"
variable = "precipitation_accumulation"
# hindcast period
start_date = dt.datetime(2021, 9, 1)
end_date = dt.datetime(2022, 9, 1)
forecast_type = "site"  # site or grid

threshold_config_file = "/path/to/threshold/config/"
input_folder_site = f"/path/to/nwp/forecast/sites/"
input_folder_grid = f"/path/to/nwp/forecast/grid/"
reliability_tables_folder = (
    f"/path/to/reliability/tables/"
)
output_folder_site = (
    f"/path/to/output/folder/sites/"
)
output_folder_grid = (
    f"/path/to/output/folder/grid/"
)


def process(input_path, output_path, accum_period, lead_time):

    print(f"Processing {input_path}")

    accum_period_string = f"PT{accum_period:02d}H"
    reliability_tables = iris.load(
        os.path.join(
            reliability_tables_folder,
            f"reliability_table-lead_time_{lead_time:03d}-{accum_period_string}.nc",
        )
    )
    fcst_cube = iris.load_cube(input_path)
    if fcst_cube.coords("realization") == []:
        return

    fcst_cube.convert_units("m")
    with open(threshold_config_file, "r") as f:
        threshold_config = json.load(f)
    fcst_cube_thresholded = threshold.process(
        fcst_cube,
        threshold_config=threshold_config,
        threshold_units="mm",
        comparison_operator=">=",
        collapse_coord="realization",
    )
    output_cube = ApplyReliabilityCalibration().process(
        fcst_cube_thresholded, reliability_tables
    )
    save_netcdf(output_cube, output_path)


if __name__ == "__main__":

    args = get_file_list(
        forecast_type,
        input_folder_site,
        output_folder_site,
        input_folder_grid,
        output_folder_grid,
        accum_periods,
        start_date,
        end_date,
    )

    with Pool(60) as p:
        p.starmap(process, args)
