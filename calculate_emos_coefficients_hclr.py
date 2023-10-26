"""Model precipitation distribution as heteroscedastic censored logistic regression as described in Eqn 11 of 
Messner, J. W., G. J. Mayr, D. S. Wilks, and A. Zeileis, 2014: 
Extending Extended Logistic Regression: Extended versus Separate versus Ordered versus Censored. 
Mon. Wea. Rev., 142, 3003–3014, 
https://doi.org/10.1175/MWR-D-13-00355.1.

The model assumes that the transformed rainfall g(r) follows a logistic distribution on the domain t >= 0.
The model variables are the ensemble mean mu and standard deviation sigma of the transformed data, which are multiplied by 
coefficients gamma_1 and delta_1 respectively.
For any rainfall threshold t >= 0, we assume that 
the probability P(g(r) <= t) is given by the logistic function 
lambda(t) = exp(a(t)) / (1 + exp(a(t))),
where 
a(t) = (t - gamma_0 - gamma_1 * mu) / exp(delta_0 + delta_1 * sigma). 
 
For t > 0, the likelihood P(g(r) = t | gamma, delta) is the derivative of lambda(t) with respect to t, which is
Lambda(t) = exp(-a(t)) / [ exp(delta_0 + delta_1 * sigma) * (1 + exp(-a(t)))**2 ]. 
(Note capital L to distinguish it from lambda above.)
Equivalently, 
P(r = t) = P(g(r) = g(t)) = Lambda(g(t)), 
which is Eqn 10 in the paper.

The distribution of rainfall is not continuous at zero, so the calculation of the likelihood as a derivative 
does not apply. Instead, using our assumption that P(g(r) <= t) = lambda(t), we get
P(r == 0) = P(g(r) <= 0) = lambda(0).

To fit the model, we find the values of gamma and delta that minimise the negative log-likelihood of P.
"""


import os
from itertools import product

import iris
import numpy as np
import pandas as pd
from scipy.optimize import check_grad, minimize

accum_period = 24
model = "ecmwf"
variable = "precipitation_accumulation"
output_folder = f"/path/to/output/folder/"
fcst_truth_cubes_folder = f"/path/to/fcst_obs_cubes/"


def g(t):
    return np.sqrt(t)


def inner_arg(t, mu, sigma, gamma_0, gamma_1, delta_0, delta_1):
    # t: rainfall threshold
    # mu: ensemble mean
    # sigma: ensemble std
    # gamma: location parameter of assumed distribution
    # delta: spread parameter of assumed distribution
    return (g(t) - gamma_0 - gamma_1 * mu) / np.exp(delta_0 + delta_1 * sigma)


def neg_log_likelihood_zero(mu, sigma, gamma_0, gamma_1, delta_0, delta_1):
    # Negative log-likelihood of t given gamma and delta where t = 0,
    # i.e. -log(lambda(0))
    a = inner_arg(0, mu, sigma, gamma_0, gamma_1, delta_0, delta_1)
    return np.log(1 + np.exp(-a))


def neg_log_likelihood_non_zero(t, mu, sigma, gamma_0, gamma_1, delta_0, delta_1):
    # Negative log-likelihood of t given gamma and delta where t > 0,
    # i.e. -log(Lambda(g(t)))
    a = inner_arg(t, mu, sigma, gamma_0, gamma_1, delta_0, delta_1)
    return a + (delta_0 + delta_1 * sigma) + (2 * np.log(1 + np.exp(-a)))


if __name__ == "__main__":

    if not (os.path.exists(output_folder)):
        os.makedirs(output_folder, exist_ok=True)

    accum_period_string = f"PT{accum_period:02d}H"
    lead_times = range(accum_period, 240 + 1, accum_period)

    for lead_time in lead_times:
        print(lead_time)
        output_filename = (
            "{data_type}-lead_time_{lead_time:03d}-{accum_period_string}.nc"
        )
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

        # convert masked data to nans
        fcst_cube.data = np.ma.filled(fcst_cube.data, np.nan)
        truth_cube.data = np.ma.filled(truth_cube.data, np.nan)

        # fill with nans where forecast or obs data is missing
        invalid_data = np.logical_or(
            np.max(np.isnan(fcst_cube.data), axis=0), np.isnan(truth_cube.data)
        )
        fcst_cube.data = np.where(
            np.broadcast_to(invalid_data[np.newaxis, :, :], fcst_cube.data.shape),
            np.nan,
            fcst_cube.data,
        )
        truth_cube.data = np.where(invalid_data, np.nan, truth_cube.data)

        # convert to mm so the optimization converges
        truth_cube.convert_units("mm")
        fcst_cube.convert_units("mm")

        # transform by g
        fcst_transformed_cube = fcst_cube.copy(data=g(fcst_cube.data))

        # extract mean and std dev from forecast
        fcst_mean_cube = fcst_transformed_cube.collapsed(
            "realization", iris.analysis.MEAN
        )
        fcst_std_cube = fcst_transformed_cube.collapsed(
            "realization", iris.analysis.STD_DEV
        )

        # create arrays to pass to optimizer
        X = np.concatenate(
            [
                fcst_mean_cube.data.flatten()[:, np.newaxis],
                fcst_std_cube.data.flatten()[:, np.newaxis],
                truth_cube.data.flatten()[:, np.newaxis],
            ],
            axis=1,
        )
        valid_rows = np.nonzero(np.logical_not(np.max(np.isnan(X), axis=1)))[0]
        X = X[valid_rows, :].astype(np.float64)
        y_zero_rows = np.nonzero(X[:, -1] == 0)[0]
        X_zero = X[y_zero_rows, :]
        y_nonzero_rows = np.nonzero(X[:, -1] > 0)[0]
        X_nonzero = X[y_nonzero_rows, :]
        zero_prop = X_zero.shape[0] / X.shape[0]

        def loss(args):
            # loss function for optimizer
            gamma_0, gamma_1, delta_0, delta_1 = args
            y_zero_loss = neg_log_likelihood_zero(
                X_zero[:, 0], X_zero[:, 1], gamma_0, gamma_1, delta_0, delta_1
            )
            y_nonzero_loss = neg_log_likelihood_non_zero(
                X_nonzero[:, 2],
                X_nonzero[:, 0],
                X_nonzero[:, 1],
                gamma_0,
                gamma_1,
                delta_0,
                delta_1,
            )
            return np.mean(y_zero_loss) * zero_prop + np.mean(y_nonzero_loss) * (
                1 - zero_prop
            )

        x_0 = [0, 0, 0, 0]
        res = minimize(loss, x_0, method="nelder-mead")
        gamma_0, gamma_1, delta_0, delta_1 = res.x
        print(res.message)

        if res.message != "Optimization terminated successfully.":
            raise RuntimeError(
                f"Optimization failed for {accum_period}H accumulation and lead time {lead_time}."
            )

        df = pd.DataFrame(
            [[gamma_0, gamma_1, delta_0, delta_1]],
            columns=["gamma_0", "gamma_1", "delta_0", "delta_1"],
        )
        df.to_csv(
            os.path.join(
                output_folder,
                f"emos_coeffs-lead_time_{lead_time:03d}-{accum_period_string}.nc",
            ),
            index=False,
        )
