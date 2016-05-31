"""Main module."""
import sys
import logging
import pyfscache
import numpy as np
import configparser
from time import time


# User modules
import mixture_model
import bias_corrector
from helpers import data_io
from helpers import metrics
import helpers.plotting as plot
from helpers import data_assimilation

# Configure cache
CACHE_DIR = '.cache'
cache = pyfscache.FSCache(CACHE_DIR)


@cache
def cached_load_data(*args):
    """Wrapper for method level caching."""
    return data_assimilation.load_data(*args)


def _get_group_index(number, group_size):
    """Give group index for given number when grouping consecutive numbers.

    Examples:
        _get_group_index(5, 2) = 3
        _get_group_index(2, 3) = 1
    """
    return np.ceil(number / group_size)


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


def add_model_predictions(dataframe, model, test_observation, row_index,
                          element_name, percentiles):
    """Add model predictions to the dataframe."""
    # Local constants
    ZERO_DEGREES_KELVIN = 273.15
    THRESHOLDS = np.arange(-30, 30, 0.5) + 273.15
    # Ensemble PDF
    dataframe.loc[row_index, element_name + '_ENSEMBLE_PDF'] = \
        model.pdf(test_observation)
    # Ensemble CDF
    dataframe.loc[row_index, element_name + '_ENSEMBLE_CDF'] = \
        model.cdf(test_observation)
    # Ensemble CDF freezing
    dataframe.loc[row_index, element_name + '_ENSEMBLE_CDF_FREEZE'] = \
        model.cdf(ZERO_DEGREES_KELVIN)
    # CRPS
    threshold_cdfs = model.cdf(THRESHOLDS)
    dataframe.loc[row_index, element_name + '_CRPS'] = \
        metrics.crps(THRESHOLDS, threshold_cdfs, test_observation)
    # Ensemble mean
    dataframe.loc[row_index, element_name + '_ENSEMBLE_MEAN'] = \
        model.mean()
    # Observation rank
    dataframe.loc[row_index, element_name + '_OBS_RANK'] = \
        metrics.rank(test_observation, model.get_member_means())
    # Percentiles
    search_initialization = test_observation - 15
    perc_values = metrics.percentiles(
        model.cdf, percentiles / 100, search_initialization)
    for percentile, value in zip(percentiles, perc_values):
        name = element_name + '_ENSEMBLE_PERC' + str(percentile)
        dataframe.loc[row_index, name] = value


def main(element_name, model_names, station_names, train_days, issue,
         forecast_hours, forecast_hour_group_size):
    """Main ETL method."""
    logging.info("Loading data..")
    load_start_time = time()
    full_data = cached_load_data(
        element_name,
        station_names,
        issue,
        model_names
    )
    logging.info("Done loading data (%ds).", time() - load_start_time)
    logging.debug("data.shape: %d rows, %d columns.", full_data.shape)

    # Drop unrequested forecast hours
    data_assimilation.filter_unused_forecast_hours(full_data, forecast_hours)

    # Ensemble definition
    ensemble_grouping, ensemble_columns = \
        _model_to_group(model_names, element_name)
    observation_column = element_name + '_OBS'
    model_mix = mixture_model.GaussianMixtureModel(
        len(ensemble_columns), ensemble_grouping)

    # Bias corrector definition
    model_bias = bias_corrector.SimpleBiasCorrector(
        len(ensemble_columns), ensemble_grouping)

    # Loop over forecast hour groups
    forecast_hour_grouping = full_data.groupby(
        _get_group_index(full_data.forecast_hour, forecast_hour_group_size)
    )
    for group_id, data in forecast_hour_grouping:
        fh_time = time()
        logging.info("Processing %d / %d forecast hour groups..",
                     group_id, len(forecast_hour_grouping))
        if len(data) == 0:
            logging.warn("No data for forecast hour group %d. Skipping.",
                         group_id)
            continue

        # The dates to predict for.
        dates_in_group = data['valid_date'].unique()
        hours_in_group = data.forecast_hour.unique()
        assert len(dates_in_group) * len(station_names) == len(data), \
            "Each station valid date should only have a single prediction"

        # Variables for plotting
        model_mix_weights = []
        model_mix_variances = []
        model_bias_intercepts = []
        plot_valid_dates = []

        # Moving window prediction
        # TODO TdR 31.05.16 : Loop over prediction dates instead of rows
        for index, row in data.iterrows():
            logging.debug("Starting with row %s", str(row['valid_date']))
            # If one of the model prediction forecasts is unavailable, skip.
            if row[ensemble_columns + [observation_column]].isnull().any():
                continue

            # Select train data
            valid_date = row['valid_date']
            latest_forecast_hour = hours_in_group[-1]
            # TODO TdR 31.05.16 : select test examples as well.
            train_data = data_assimilation.select_train_data(
                data, latest_forecast_hour, valid_date, train_days,
                ensemble_columns, observation_column
            )

            # Check data quality
            train_dates = train_data['valid_date'].unique()
            if not data_assimilation.check_data_coverage(
              train_dates, train_days):
                logging.warn(
                    "Not enough training days (%s / %d), skipping date %s",
                    str(len(train_data)).zfill(2), train_days,
                    str(valid_date))
                continue

            X_train = train_data[ensemble_columns].as_matrix()
            y_train = train_data[observation_column].as_matrix()
            X_test = row[ensemble_columns].as_matrix()
            y_test = row[observation_column]

            # Train bias model
            logging.debug("Training bias model..")
            model_bias.fit(X_train, y_train)
            X_train = model_bias.predict(X_train)

            # Train mixture model
            logging.debug("Training mixture model..")
            model_mix.fit(X_train, y_train)
            model_mix_weights.append(model_mix.weights)
            model_mix_variances.append(model_mix.get_member_variances())
            model_bias_intercepts.append(model_bias.intercept_per_model)
            plot_valid_dates.append(valid_date)

            # Predict
            logging.debug("Storing predictions..")
            X_test = model_bias.predict(X_test)
            model_mix.set_member_means(X_test)

            # Add predictions to dataset
            PERCENTILES = np.array([5, 10, 25, 75, 90, 95])
            add_model_predictions(
                full_data, model_mix, y_test, index, element_name, PERCENTILES)
        logging.info("Done with forecast hour group %d (%.2fs).",
                     group_id, time() - fh_time)
        for forecast_hour in hours_in_group:
            plot.plot_ensemble_percentiles(
                forecast_hour, PERCENTILES, element_name, full_data)
        plot.plot_model_parameters(
            plot_valid_dates, model_mix_weights, model_mix_variances,
            model_bias_intercepts,
            hours_in_group, ensemble_columns, element_name)
    return full_data.drop(ensemble_columns, axis=1)


def do_verification(data, forecast_hour):
    """Method for running several plotting routines."""
    hourly_data = data[data.forecast_hour == forecast_hour]
    plot.plot_verification_rank_histogram(hourly_data)
    plot.plot_PIT_histogram(hourly_data)
    # a, b, c = plot.calculate_threshold_hits(hourly_data)
    # plot.plot_relialibilty_sharpness_diagram(a, b, c)
    mean_crps = hourly_data['2T_CRPS'].mean()
    logging.info("Mean CRPS: %f", mean_crps)
    deterministic_MAE = \
        abs(hourly_data['2T_ENSEMBLE_MEAN'] - hourly_data['2T_OBS']).mean()
    logging.info("Ensemble mean MAE: %f", deterministic_MAE)


def load_configuration(configuration_name='main'):
    """Load the program configurations from file."""
    config_parser = configparser.ConfigParser()
    config_parser.read('config.ini')
    return config_parser[configuration_name]


def log_dict(dictionary):
    """Write dict keys and values to log."""
    logging.info("Configuration parameters:")
    for key in dictionary:
        logging.info("\t%s: %s", key, dictionary[key])
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
        level='DEBUG'
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
    train_days = int(config['model_train_days'])
    model_issue = config['model_issue']
    model_names = config['model_names'].split(',')
    element_name = config['element_name']

    # Station Selection
    station_names = config['station_names'].split(',')

    # Run the program
    data = main(
        element_name, model_names, station_names, train_days,
        model_issue, forecast_hours, fh_group_size)

    # Write predicitons to file
    write_predictions(data, element_name, model_names, forecast_hours)
    logging.info("Finished program.")
