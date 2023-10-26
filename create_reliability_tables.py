import datetime as dt
import json
import os
from copy import deepcopy
from multiprocessing import Pool

import iris
import numpy as np

from improver.calibration.reliability_calibration import (
    ConstructReliabilityCalibrationTables,
    ManipulateReliabilityTable,
)
from improver.cli import threshold
from improver.utilities.save import save_netcdf

accum_periods = [24]
n_bins = 50
model = "ecmwf"
variable = "precipitation_accumulation"
output_folder = f"/path/to/output/folder/"
fcst_truth_cubes_folder =  f"/path/to/fcst_obs_cubes/"
threshold_config_file = "/path/to/threshold/config/"


def process(accum_period, lead_time):
    accum_period_string = f"PT{accum_period:02d}H"
    output_filename = "{data_type}-lead_time_{lead_time:03d}-{accum_period_string}.nc"
    output_path_fcst = os.path.join(
        fcst_truth_cubes_folder,
        output_filename.format(
            data_type="forecast",
            lead_time=lead_time,
            accum_period_string=accum_period_string,
        ),
    )
    output_path_obs = os.path.join(
        fcst_truth_cubes_folder,
        output_filename.format(
            data_type="truth",
            lead_time=lead_time,
            accum_period_string=accum_period_string,
        ),
    )
    fcst_cube = iris.load_cube(output_path_fcst)
    truth_cube = iris.load_cube(output_path_obs)

    # mask where forecast or obs data is missing or masked
    invalid_data = np.logical_or(
        np.max(np.isnan(fcst_cube.data), axis=0), np.isnan(truth_cube.data)
    )
    fcst_cube.data = np.ma.masked_where(
        np.broadcast_to(invalid_data[np.newaxis, :, :], fcst_cube.data.shape),
        fcst_cube.data,
    )
    truth_cube.data = np.ma.masked_where(invalid_data, truth_cube.data)

    # threshold
    with open(threshold_config_file, "r") as f:
        threshold_config = json.load(f)
    fcst_cube_thresholded = threshold.process(
        fcst_cube,
        threshold_config=deepcopy(threshold_config),
        threshold_units="mm",
        comparison_operator=">=",
        collapse_coord="realization",
    )
    truth_cube_thresholded = threshold.process(
        truth_cube,
        threshold_config=deepcopy(threshold_config),
        threshold_units="mm",
        comparison_operator=">=",
    )

    # create reliability tables
    plugin = ConstructReliabilityCalibrationTables(n_probability_bins=n_bins)
    reliability_table = plugin.process(
        fcst_cube_thresholded, truth_cube_thresholded, aggregate_coords=["spot_index"]
    )

    # manipulate reliability tables
    reliability_table = ManipulateReliabilityTable().process(reliability_table)

    save_netcdf(
        reliability_table,
        os.path.join(
            output_folder,
            f"reliability_table-lead_time_{lead_time:03d}-{accum_period_string}.nc",
        ),
    )


if __name__ == "__main__":

    if not (os.path.exists(output_folder)):
        os.makedirs(output_folder, exist_ok=True)

    args = []
    for accum_period in accum_periods:
        lead_times = range(accum_period, 240 + 1, accum_period)
        for lead_time in lead_times:
            args.append([accum_period, lead_time])

    with Pool(10) as p:
        p.starmap(process, args)
