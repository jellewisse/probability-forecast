"""Main module."""
import sys
import logging
import pyfscache
import numpy as np
import pandas as pd
import configparser
from time import time


# User modules
from mixture_model.gaussian_mixture import GaussianMixtureModel
from bias_corrector.simple_correctors import SimpleBiasCorrector
from helpers.data_assimilation import load_data
import helpers.plotting as plot
from helpers import data_io
from helpers import metrics


# Configure cache
CACHE_DIR = '.cache'
cache = pyfscache.FSCache(CACHE_DIR)


@cache
def cached_load_data(*args):
    """Wrapper for method level caching."""
    return load_data(*args)


def _calculate_lag(forecast_hour):
    return np.ceil(forecast_hour / 24)


def _split_list(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:(i + n)]


def _model_to_group(model_names, element_name):
    """Construct column names and standard member grouping."""
    grouping = []
    ens_cols = []
    group_counter = 0
    for model_name in model_names:
        if model_name == 'eps':
            grouping += [group_counter] * 50
            ens_cols += \
                [element_name + '_EPS' + str(x).zfill(2) for x in range(1, 51)]
        else:
            grouping += [group_counter]
            ens_cols += [element_name + '_' + model_name.upper()]
        group_counter += 1
    return grouping, ens_cols


def main(element_name, model_names, station_names, issue, forecast_hours,
         forecast_hour_group_size):
    """Main ETL method."""
    logging.info("Loading data..")
    load_start_time = time()
    full_data = cached_load_data(
        element_name,
        station_names,
        issue,
        model_names
    )
    logging.info("Done loading data (%ds)." % (time() - load_start_time))
    logging.debug("data.shape: %d rows, %d columns." % full_data.shape)

    # Ensemble definition
    grouping, ens_cols = _model_to_group(model_names, element_name)
    obs_col = element_name + '_OBS'
    model_mix = GaussianMixtureModel(len(ens_cols), grouping)
    train_days = 40

    # Bias corrector definition
    model_bias = SimpleBiasCorrector(len(ens_cols), grouping)

    # TODO TdR 20-05-2016 Check what happens if multiple stations are loaded.

    # Loop over forecast hours
    forecast_hour_groups = \
        list(_split_list(forecast_hours, forecast_hour_group_size))
    for fh_count, forecast_hour_group in enumerate(forecast_hour_groups):
        fh_time = time()
        logging.info("Processing %d / %d forecast hours.." %
                     (fh_count + 1, len(forecast_hour_groups)))
        data = full_data[full_data.forecast_hour.isin(forecast_hour_group)]
        if len(data) == 0:
            logging.warn("No data for forecast hour(s) %s. Skipping." %
                         (str(forecast_hour_group)))
            continue

        # The dates to predict for
        # Take latest lag to be conservative.
        lag = _calculate_lag(forecast_hour_group[-1])
        valid_dates = data['valid_date'].unique()
        assert len(valid_dates) * len(station_names) == len(data), \
            "Each station valid date should only have a single prediction"

        # Prediction intervals
        thresholds = np.arange(-30, 30, 0.5) + 273.15
        model_mix_weights = []
        model_mix_variances = []
        model_bias_intercepts = []
        plot_valid_dates = []

        # Moving window prediction
        row_count = 0
        for index, row in data.iterrows():

            # If one of the model prediction forecasts is unavailable, skip.
            if row[ens_cols + [obs_col]].isnull().any():
                continue

            # Select data
            valid_date = row['valid_date']
            first_date = valid_date - pd.DateOffset(days=lag + train_days)
            last_date = valid_date - pd.DateOffset(days=lag)
            # TODO Selection might be expensive. Alternative is to index first.
            train_data = data[
                (data.valid_date <= last_date) & (data.valid_date > first_date)
            ].dropna(axis=0, subset=ens_cols + [obs_col])
            if len(train_data) < (train_days * 0.5):
                logging.debug(
                  "Not enough training days (%s / %d), skipping date %s." % (
                        str(len(train_data)).zfill(2), train_days,
                        str(valid_date)))
                continue
            X_train = train_data[ens_cols].as_matrix()
            y_train = train_data[obs_col].as_matrix()
            X_test = row[ens_cols].as_matrix()
            y_test = row[obs_col]

            # Train bias model
            model_bias.fit(X_train, y_train)
            X_train = model_bias.predict(X_train)

            # Train mixture model
            model_mix.fit(X_train, y_train)
            model_mix_weights.append(model_mix.weights)
            model_mix_variances.append(model_mix.get_member_variances())
            model_bias_intercepts.append(model_bias.intercept_per_model)
            plot_valid_dates.append(valid_date)

            # Predict
            X_test = model_bias.predict(X_test)
            model_mix.set_member_means(X_test)

            # Verify
            # Ensemble PDF
            full_data.loc[index, element_name + '_ENSEMBLE_PDF'] = \
                model_mix.pdf(y_test)
            # Ensemble CDF
            full_data.loc[index, element_name + '_ENSEMBLE_CDF'] = \
                model_mix.cdf(y_test)
            full_data.loc[index, element_name + '_ENSEMBLE_CDF_FREEZE'] = \
                model_mix.cdf(273.15)
            # CRPS
            threshold_cdfs = model_mix.cdf(thresholds)
            full_data.loc[index, element_name + '_CRPS'] = \
                metrics.crps(thresholds, threshold_cdfs, y_test)
            # Ensemble mean
            full_data.loc[index, element_name + '_ENSEMBLE_MEAN'] = \
                model_mix.mean()
            # Percentiles
            perc_start_time = time()
            percentiles = np.array([5, 10, 25, 75, 90, 95])
            perc_values = metrics.percentiles(
                model_mix.cdf, percentiles / 100, y_test - 15)
            for percentile, value in zip(percentiles, perc_values):
                name = element_name + '_ENSEMBLE_PERC' + str(percentile)
                full_data.loc[index, name] = value
            logging.debug("Done with %s: %.3fs" %
                          (str(valid_date), time() - perc_start_time))

            # For determining the ensemble verification rank / calibrationl
            # Rank only makes sense if the ensemble weights are uniformly
            # distributed.
            obs_in_forecasts = \
                list(model_mix.get_member_means()) + list([y_test])
            obs_in_forecasts.sort()
            full_data.loc[index, element_name + '_OBS_RANK'] = \
                obs_in_forecasts.index(y_test)
            row_count += 1

            # TODO For debugging.
            # plot.plot_distribution(model_mix, row[obs_col], valid_date)
        logging.info("Done with forecast hours %s (%.2fs)." %
                     (str(forecast_hour_group), time() - fh_time))
        for forecast_hour in forecast_hour_group:
            plot.plot_ensemble_percentiles(
                forecast_hour, percentiles, element_name, full_data)
        plot.plot_model_parameters(
            plot_valid_dates, model_mix_weights, model_mix_variances,
            model_bias_intercepts,
            forecast_hour_group, ens_cols, element_name)
    return full_data.drop(ens_cols, axis=1)


def do_verification(data, forecast_hour):
    """Method for running several plotting routines."""
    hourly_data = data[data.forecast_hour == forecast_hour]
    plot.plot_verification_rank_histogram(hourly_data)
    plot.plot_PIT_histogram(hourly_data)
    # a, b, c = plot.calculate_threshold_hits(hourly_data)
    # plot.plot_relialibilty_sharpness_diagram(a, b, c)
    mean_crps = hourly_data['2T_CRPS'].mean()
    logging.info("Mean CRPS: %f" % (mean_crps))
    deterministic_MAE = \
        abs(hourly_data['2T_ENSEMBLE_MEAN'] - hourly_data['2T_OBS']).mean()
    logging.info("Ensemble mean MAE: %f" % (deterministic_MAE))


def load_configuration(configuration_name='main'):
    """Load the program configurations from file."""
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini')
    return config_parser[configuration_name]


def log_dict(dict):
    """Write dict keys and values to log."""
    logging.info("Configuration parameters:")
    for key in dict:
        logging.info("\t%s: %s" % (key, dict[key]))
    logging.info("End of configuration parameters.")


def write_predictions(data_frame, element_name, model_names, forecast_hours):
    """Write a data_frame to file."""
    # Write predictions to file
    data_frame.sort_values(['issue_date', 'forecast_hour'],
                           ascending=True, inplace=True)
    file_path = 'output/' + element_name + '_' + '_'.join(model_names) + '_' \
        + str(forecast_hours[0]) + '_' + str(forecast_hours[-1]) + '.csv'
    data_io.write_csv(data_frame, file_path)


if __name__ == "__main__":
    """Run the probability forecast.

    Example:
        python main.py [configuration_name]
    """

    # Logger setup
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level='INFO'
    )
    logging.info("Starting program.")

    if len(sys.argv) >= 2:
        config_section_name = sys.argv[1]
    else:
        config_section_name = 'main'

    config = load_configuration(config_section_name)
    logging.info("Configuration loaded.")
    log_dict(config)

    # Forecast hour configuration
    fh_first = int(config['first_forecast_hour'])
    fh_last = int(config['last_forecast_hour'])
    fh_interval = int(config['forecast_hour_interval'])
    fh_group_size = int(config['forecast_hour_group_size'])
    forecast_hours = np.arange(fh_first, fh_last + 1, fh_interval)

    # Ensemble definition
    model_issue = config['model_issue']
    model_names = config['model_names'].split(',')
    element_name = config['element_name']

    # Station Selection
    station_names = config['station_names'].split(',')

    # Run the program
    data = main(
        element_name, model_names, station_names,
        model_issue, forecast_hours, fh_group_size)

    # Write predicitons to file
    write_predictions(data, element_name, model_names, forecast_hours)
