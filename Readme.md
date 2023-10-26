# Rainforests benchmarks

Code to produce hindcasts using reliability calibration and EMOS as benchmarks for rainforests.

To produce the benchmark forecasts:
* Prepare the data by running  `merge_fcst_raw_ensemble.sh` then `prepare_fcst_obs_dataframes.py`. The outputs of these scripts
are used for both calibration methods.
* For reliability calibration, run `create_reliability_tables.py`, then `apply_reliability_tables.py`.
* For EMOS, run `calculate_emos_coefficients_hclr.py`, then `apply_emos_hclr.py`.
