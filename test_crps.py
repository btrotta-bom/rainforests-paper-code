import numpy as np
import pytest
import xarray as xr
from crps import (
    crps_at_point,
    crps_exact,
    crps_realization,
    crps_threshold,
    integral_above,
    integral_below,
)
from scipy.integrate import quad


@pytest.fixture
def prob_dist():
    thresholds = np.array([0, 1, 2, 3, 5])
    probabilities = np.array([0, 0.6, 0.8, 0.9, 0.95])
    return probabilities, thresholds


@pytest.fixture
def obs():
    return np.array([-1, 0, 1.6, 6])


@pytest.fixture
def ensemble_fc(obs):
    rng = np.random.default_rng(0)
    n = 10
    fc = np.empty((len(obs), n))
    for i in range(fc.shape[0]):
        fc[i] = rng.normal(obs[i], 1, n)
    return fc


@pytest.fixture
def probability_fc(prob_dist, obs):
    probabilities, thresholds = prob_dist
    probabilities = np.repeat(probabilities[np.newaxis, :], repeats=len(obs), axis=0)
    probabilities *= np.array([0.91, 0.98, 0.80, 1])[:, np.newaxis]
    return probabilities, thresholds


@pytest.mark.parametrize("obs", [-1, 0, 1.6, 4, 5, 6])
def test_crps_at_point(prob_dist, obs):
    probabilities, thresholds = prob_dist
    probability_at_obs = np.interp(obs, thresholds, probabilities)
    probabilities_extended = np.sort(
        np.concatenate([probabilities, [probability_at_obs]])
    )
    thresholds_extended = np.sort(np.concatenate([thresholds, [obs]]))

    def integrand(x, xp, fp):
        prob_at_x = np.interp(x, xp, fp)
        if x < obs:
            return prob_at_x ** 2
        else:
            return (1 - prob_at_x) ** 2

    # calculate the integral in parts, splitting on discontinuities
    expected = 0
    for i in range(1, len(thresholds_extended)):
        curr = quad(
            lambda x: integrand(x, thresholds_extended, probabilities_extended),
            thresholds_extended[i - 1],
            thresholds_extended[i],
        )[0]
        expected += curr
    np.testing.assert_allclose(expected, crps_at_point(obs, probabilities, thresholds))


def test_crps_threshold(probability_fc, obs):
    probabilities, thresholds = probability_fc
    expected = np.array(
        [crps_at_point(x, probabilities[i], thresholds) for i, x in enumerate(obs)]
    )
    np.testing.assert_allclose(expected, crps_threshold(obs, probabilities, thresholds))


def test_crps_realization(obs, ensemble_fc):
    n = ensemble_fc.shape[-1]
    probabilities = np.linspace(1 / (n + 1), n / (n + 1), n)
    ensemble_fc = np.sort(ensemble_fc, axis=-1)
    expected = np.array(
        [crps_at_point(x, probabilities, ensemble_fc[i]) for i, x in enumerate(obs)]
    )
    np.testing.assert_allclose(expected, crps_realization(obs, ensemble_fc))


def test_crps_exact_threshold(probability_fc, obs):
    probabilities_2d, thresholds = probability_fc
    probabilities = np.reshape(probabilities_2d, (2, 2, -1))
    fc_da = xr.DataArray(
        probabilities,
        dims=["x", "y", "threshold"],
        coords={
            "x": ("x", [0, 1]),
            "y": ("y", [0, 1]),
            "threshold": ("threshold", thresholds),
        },
    )
    obs_2d = obs
    obs = np.reshape(obs_2d, (2, 2))
    obs_da = xr.DataArray(
        obs, dims=["x", "y"], coords={"x": ("x", [0, 1]), "y": ("y", [0, 1])},
    )
    # test pointwise
    crps_pointwise = np.reshape(
        crps_threshold(obs_2d, probabilities_2d, thresholds), (2, 2)
    )
    res_pointwise = crps_exact(obs_da, fc_da, dim=[])
    np.testing.assert_allclose(crps_pointwise, res_pointwise)

    # test pointwise with permuted dimensions
    fc_da_perm = fc_da.transpose("threshold", "x", "y")
    np.testing.assert_allclose(crps_exact(obs_da, fc_da_perm, dim=[]), res_pointwise)

    # test aggregating along dimension
    res_mean_x = crps_exact(obs_da, fc_da_perm, dim=["x"])
    exp_mean_x = np.mean(crps_pointwise, axis=0)
    np.testing.assert_allclose(res_mean_x, exp_mean_x)
    assert res_mean_x.dims == ("y",)
