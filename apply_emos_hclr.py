import datetime as dt
import json
import os
from multiprocessing import Pool

import iris
import numpy as np
import pandas as pd
from calculate_emos_coefficients_hclr import g, inner_arg
from scipy.special import expit
from utilities import get_file_list

from improver.utilities.save import save_netcdf

accum_periods = [24]
model = "ecmwf"
variable = "precipitation_accumulation"
# date period used for calibration; start and end dates should be first of month
start_date = dt.datetime(2021, 9, 1)
end_date = dt.datetime(2022, 9, 1)
forecast_type = "site"  # site or grid

threshold_config_file = "/path/to/threshold/config/"
input_folder_site = f"/path/to/nwp/forecast/sites/"
input_folder_grid = f"/path/to/nwp/forecast/grid/"
emos_coeffs_folder = f"/path/to/emos/coeffs/"
output_folder_site = (
    f"/path/to/output/folder/sites/"
)
output_folder_grid = (
    f"/path/to/output/folder/grid/"
)


def cdf(t, mu, sigma, gamma_0, gamma_1, delta_0, delta_1):
    a = inner_arg(t, mu, sigma, gamma_0, gamma_1, delta_0, delta_1)
    return expit(a)


def process(input_path, output_path, accum_period, lead_time):

    print(f"Processing {input_path}")

    accum_period_string = f"PT{accum_period:02d}H"
    coeffs = pd.read_csv(
        os.path.join(
            emos_coeffs_folder,
            f"emos_coeffs-lead_time_{lead_time:03d}-{accum_period_string}.nc",
        )
    )

    fcst_cube = iris.load_cube(input_path)
    if fcst_cube.coords("realization") == []:
        return
    fcst_cube.convert_units("mm")

    # transform by g
    fcst_transformed_cube = fcst_cube.copy(data=g(fcst_cube.data))

    # extract mean and std dev from forecast
    fcst_mean_cube = fcst_transformed_cube.collapsed("realization", iris.analysis.MEAN)
    fcst_mean_cube.remove_coord("realization")
    fcst_std_cube = fcst_transformed_cube.collapsed(
        "realization", iris.analysis.STD_DEV
    )
    fcst_std_cube.remove_coord("realization")

    # get thresholds (in mm)
    with open(threshold_config_file, "r") as f:
        threshold_config = json.load(f)
    thresholds = np.sort(
        np.array([np.float32(t) for t in list(threshold_config.keys())])
    )

    # get distribution params
    mu = fcst_mean_cube.data.flatten()
    sigma = fcst_std_cube.data.flatten()
    gamma_0, gamma_1, delta_0, delta_1 = coeffs.values[0]

    # predict
    predicted = np.empty((len(thresholds),) + fcst_mean_cube.data.shape)
    for i, t in enumerate(thresholds):
        if t == 0:
            predicted[i, :] = np.ones(fcst_mean_cube.shape)
        else:
            predicted[i, :] = 1 - np.reshape(
                cdf(t, mu, sigma, gamma_0, gamma_1, delta_0, delta_1),
                fcst_mean_cube.data.shape,
            )

    # make output cube
    threshold_dim = iris.coords.DimCoord(
        thresholds * 0.001,
        standard_name="lwe_thickness_of_precipitation_amount",
        units="m",
        var_name="threshold",
        attributes={"spp__relative_to_threshold": "greater_than_or_equal_to"},
    )
    dim_coords_and_dims = [(threshold_dim, 0)] + [
        (coord.copy(), fcst_mean_cube.coord_dims(coord)[0] + 1)
        for coord in fcst_mean_cube.coords(dim_coords=True)
    ]
    aux_coords_and_dims = []
    for coord in getattr(fcst_mean_cube, "aux_coords"):
        coord_dims = fcst_mean_cube.coord_dims(coord)
        if len(coord_dims) == 0:
            aux_coords_and_dims.append((coord.copy(), []))
        else:
            aux_coords_and_dims.append(
                (coord.copy(), fcst_mean_cube.coord_dims(coord)[0] + 1)
            )
    probability_cube = iris.cube.Cube(
        predicted.astype(np.float32),
        long_name=f"probability_of_lwe_thickness_of_precipitation_amount_above_threshold",
        units=1,
        attributes=fcst_mean_cube.attributes,
        dim_coords_and_dims=dim_coords_and_dims,
        aux_coords_and_dims=aux_coords_and_dims,
    )
    save_netcdf(
        probability_cube, output_path,
    )


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

    with Pool(40) as p:
        p.starmap(process, args)
