"""Code to calculate exact CRPS for a probabilistic forecast expressed 
as an ensemble or thresholded probabilities.

The xarray wrapper function crps_exact is based on the code for crps_ensemble from xskillscore
https://github.com/xarray-contrib/xskillscore/blob/main/xskillscore/core/probabilistic.py
Copyright xskillscore developers 2018-2021, released under the Apache-2.0 License

The vectorisation of crps_at_point follows the example of _crps_ensemble_gufunc from properscoring
https://github.com/properscoring/properscoring/blob/master/properscoring/_gufuncs.py
Copyright 2015 The Climate Corporation, released under the Apache-2.0 License
"""


from typing import Iterable, List, Optional, Union

import numpy as np
import xarray as xr
from numba import float64, guvectorize, jit
from numpy import ndarray
from xarray import DataArray

from xskillscore.core.probabilistic import probabilistic_broadcast

EPSILION = 1e-8


@jit
def integral_below(x0, x1, y0, y1):
    if x1 - x0 < EPSILION:
        return 0
    if abs(y1 - y0) < EPSILION:
        return (x1 - x0) * y0 ** 2
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return (1 / (3 * slope)) * (
        (slope * x1 + intercept) ** 3 - (slope * x0 + intercept) ** 3
    )


@jit
def integral_above(x0, x1, y0, y1):
    if x1 - x0 < EPSILION:
        return 0
    if abs(y1 - y0) < EPSILION:
        return (x1 - x0) * (1 - y0) ** 2
    slope = (y1 - y0) / (x1 - x0)
    intercept = y0 - slope * x0
    return (-1 / (3 * slope)) * (
        (1 - slope * x1 - intercept) ** 3 - (1 - slope * x0 - intercept) ** 3
    )


@guvectorize([(float64, float64[:], float64[:], float64[:])], "(),(n),(n)->()")
def crps_at_point(obs: float, fc: ndarray, thresholds: ndarray, res: ndarray):
    """"CRPRS at a single point for a thresholded probabilistic forecast.
    
    Args:
        obs: observed value
        fc: 1-d array of forecast probabilities
        thresholds: 1-d non-decreasing array of thresholds, same length as forecasts
        
    Returns:
        CRPS at point
    """

    obs_ind = np.searchsorted(thresholds, obs)  # obs is between obs_ind - 1 and obs_ind
    ans = 0
    if obs_ind == 0:
        ans += integral_above(obs, thresholds[0], fc[0], fc[0])
    for i in range(1, obs_ind):
        ans += integral_below(thresholds[i - 1], thresholds[i], fc[i - 1], fc[i])
    if (obs_ind > 0) and (obs_ind < len(fc)):
        prob_at_obs = np.interp(
            obs,
            [thresholds[obs_ind - 1], thresholds[obs_ind]],
            [fc[obs_ind - 1], fc[obs_ind]],
        )
        ans += integral_below(
            thresholds[obs_ind - 1], obs, fc[obs_ind - 1], prob_at_obs
        )
        ans += integral_above(obs, thresholds[obs_ind], prob_at_obs, fc[obs_ind])
    for i in range(obs_ind + 1, len(fc)):
        ans += integral_above(thresholds[i - 1], thresholds[i], fc[i - 1], fc[i])
    if obs_ind == len(fc):
        ans += integral_below(thresholds[-1], obs, fc[-1], fc[-1])
    res[0] = ans


def crps_realization(observations: ndarray, forecasts: ndarray):
    """Pointwise calculation of CRPS for an ensemble forecast.

    Assumes realizations correspond to quantiles 1/(n + 1), ..., n/(n + 1)
    where n is the number of realizations.
    
    Args:
        observations: n-dimensional array
        forecasts: n-dimensional array with same shape as observations
            and additional realization dimension as last dimension
    
    Returns:
        n-dimensional array with same dimensions as observations
    """

    forecasts = np.sort(forecasts, axis=-1)
    n = forecasts.shape[-1]
    probabilities = np.linspace(1 / (n + 1), n / (n + 1), n)
    return crps_at_point(observations, probabilities, forecasts)


def crps_threshold(observations: ndarray, forecasts: ndarray, thresholds: ndarray):
    """Pointwise calculation of CRPS for a thresholded probabilistic forecast.
    
    Args:
        observations: n-dimensional array
        forecasts: n-dimensional array of probability forecasts with same shape as observations
            and additional threshold dimension as last dimension. Forecasts should describe 
            a cdf, i.e. they must be in increasing order.
        thresholds: 1-dimensional monotone non-decreasing array with length forecasts.shape[-1]

    
    Returns:
        n-dimensional array with same dimensions as observations
    """

    return crps_at_point(observations, forecasts, thresholds)


def crps_exact(
    observations: DataArray,
    forecasts: DataArray,
    dim: Optional[Union[str, List[str]]] = None,
    keep_attrs: bool = False,
) -> DataArray:
    """
    Calculate exact CRPS for a probabilistic forecast expressed 
    as an ensemble or thresholded probabilities.

    Args:
        observations: DataArray broadcastable to same dimensions as forecasts
        forecasts: DataArray containing either "realization" or "threshold" dim
        dim: str or list of strings, dimensions to average over. If None, average over all dims
        keep_attrs: preserve attrs from observations

    Returns:
        DataArray with same dimensions as forecasts, except for realization or threshold dim, and
        dimensions listed in dim
    """

    if "realization" in forecasts.dims:
        crps_dim = "realization"
    elif "threshold" in forecasts.dims:
        crps_dim = "threshold"
    else:
        raise ValueError(
            "forecasts must contain 'realization' or 'threshold' dimension"
        )

    # broadcast obs to forecast
    observations, forecasts = probabilistic_broadcast(
        observations, forecasts, member_dim=crps_dim
    )

    if crps_dim == "realization":
        # doesn't work with Dask because sort is not implemented
        res = xr.apply_ufunc(
            crps_realization,
            observations,
            forecasts,
            input_core_dims=[[], ["realization"]],
            dask="forbidden",
            output_dtypes=[float],
            keep_attrs=keep_attrs,
        )
    else:
        if ("long_name" in forecasts.attrs) and (
            "above" in forecasts.attrs["long_name"]
        ):
            cdf_forecasts = 1 - forecasts
            cdf_forecasts.attrs["long_name"] = forecasts.attrs["long_name"].replace(
                "above", "below"
            )
        else:
            cdf_forecasts = forecasts
        res = xr.apply_ufunc(
            crps_threshold,
            observations,
            cdf_forecasts,
            cdf_forecasts["threshold"].values,
            input_core_dims=[[], ["threshold"], []],
            dask="allowed",
            output_dtypes=[float],
            keep_attrs=keep_attrs,
        )

    return res.mean(dim, keep_attrs=keep_attrs)
