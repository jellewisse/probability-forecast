"""Main module."""

import pyfscache
import numpy as np
import pandas as pd
from time import time

# User modules
from mixture_model.gaussian_mixture import GaussianMixtureModel
from helpers.data_assimilation import load_data
import helpers.plotting as plot
from helpers import data_readers
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


def pipeline(element_name, model_names, issue, forecast_hours):
    """Main ETL method."""
    # Load data
    print("Loading data..")
    load_start_time = time()
    full_data = cached_load_data(
        element_name,
        issue,
        model_names
    )
    print("Done loading data (%ds)." % (time() - load_start_time))
    print("Data.shape: %d rows, %d columns." % full_data.shape)

    # Ensemble definition
    grouping, ens_cols = _model_to_group(model_names, element_name)
    obs_col = element_name + '_OBS'
    model = GaussianMixtureModel(len(ens_cols), grouping)
    train_days = 40

    # Loop over forecast hours
    for fh_count, forecast_hour in enumerate(forecast_hours):
        fh_time = time()
        print("Processing %d / %d forecast hours.." %
              (fh_count + 1, len(forecast_hours)))
        data = full_data[full_data.forecast_hour == forecast_hour]

        # TODO Write results to other dataframe than original data
        # The dates to predict for
        lag = _calculate_lag(forecast_hour)
        valid_dates = data['valid_date'].unique()
        assert len(valid_dates) == len(data), \
            "Each valid date should only have a single prediction"

        # Prediction intervals
        thresholds = np.arange(-30, 30, 0.5) + 273.15
        model_weights = []
        model_variances = []
        plot_valid_dates = []

        # Moving window prediction
        row_count = 0
        for index, row in data.iterrows():

            # If one of the model prediction forecasts is unavailable, skip.
            if row[ens_cols].isnull().any():
                continue

            # Select data
            valid_date = row['valid_date']
            first_date = valid_date - pd.DateOffset(days=lag + train_days)
            last_date = valid_date - pd.DateOffset(days=lag)
            # TODO Selection might be expensive. Alternative is to index first.
            train_data = data[
                (data.valid_date <= last_date) & (data.valid_date > first_date)
            ].dropna(axis=0, subset=ens_cols)
            if len(train_data) < (train_days * 0.5):
                # print(
                #   "Not enough training days (%s / %d), skipping date %s." % (
                #         str(len(train_data)).zfill(2), train_days,
                #         str(valid_date)))
                continue
            X_train = train_data[ens_cols].as_matrix()
            y_train = train_data[obs_col].as_matrix()
            X_test = row[ens_cols].as_matrix()
            y_test = row[obs_col]

            # Train
            model.fit(X_train, y_train)
            model_weights.append(model.weights)
            model_variances.append(model.get_member_variances())
            plot_valid_dates.append(valid_date)

            # Predict
            model.set_member_means(X_test)

            # Verify
            # Ensemble PDF
            full_data.loc[index, element_name + '_ENSEMBLE_PDF'] = \
                model.pdf(y_test)
            # Ensemble CDF
            full_data.loc[index, element_name + '_ENSEMBLE_CDF'] = \
                model.cdf(y_test)
            # CRPS
            threshold_cdfs = model.cdf(thresholds)
            full_data.loc[index, element_name + '_CRPS'] = \
                metrics.crps(thresholds, threshold_cdfs, y_test)
            # Ensemble mean
            full_data.loc[index, element_name + '_ENXSEMBLE_MEAN'] = \
                model.mean()
            # Percentiles
            perc_start_time = time()
            # percentiles = np.array([1, 10, 25, 50, 75, 90, 99])
            percentiles = np.arange(1, 99, 1)
            perc_values = \
                metrics.percentiles(model.cdf, percentiles / 100, y_test - 15)
            for percentile, value in zip(percentiles, perc_values):
                name = element_name + '_ENSEMBLE_PERC' + str(percentile)
                full_data.loc[index, name] = value
            print("Done with %s: %.3fs" %
                  (str(valid_date), time() - perc_start_time))

            # For determining the ensemble verification rank / calibrationl
            # Rank only makes sense if the ensemble weights are uniformly
            # distributed.
            obs_in_forecasts = list(model.get_member_means()) + list([y_test])
            obs_in_forecasts.sort()
            full_data.loc[index, element_name + '_OBS_RANK'] = \
                obs_in_forecasts.index(y_test)
            row_count += 1
            # if valid_date >= datetime(2015, 2, 14, tzinfo=timezone.utc):
            #     import pdbdata.
            #     pdb.set_trace()

            # plot.plot_distribution(model, row[obs_col], valid_date)
        print("Done (%.2fs)." % (time() - fh_time))
        # modified_weights = np.array(model_weights)[:, [0, 50, 51, 52]]
        # modified_weights[:, 0] *= 50
        # plot.plot_model_weights(
        #     plot_valid_dates, modified_weights,
        #     forecast_hour, np.array(ens_cols)[[0, 50, 51, 52]])
        # plot.plot_model_variances(
        #     plot_valid_dates, np.array(model_variances)[:, [0, 50, 51, 52]],
        #     forecast_hour, np.array(ens_cols)[[0, 50, 51, 52]])
    return full_data


def do_verification(data, forecast_hour):
    """Method for running several plotting routines."""
    hourly_data = data[data.forecast_hour == forecast_hour]
    plot.plot_verification_rank_histogram(hourly_data)
    plot.plot_PIT_histogram(hourly_data)
    plot.plot_sharpness_histogram(hourly_data)
    # a, b, c = plot.calculate_threshold_hits(hourly_data)
    # plot.plot_relialibilty_sharpness_diagram(a, b, c)
    mean_crps = hourly_data['2T_CRPS'].mean()
    print("Mean CRPS: %f" % (mean_crps))
    deterministic_MAE = \
        abs(hourly_data['2T_ENSEMBLE_MEAN'] - hourly_data['2T_OBS']).mean()
    print("Ensemble mean MAE: %f" % (deterministic_MAE))


# For testing purposes
if __name__ == "__main__":
    forecast_hours = np.arange(24, 24 + 1, 3)
    model_names = ["eps", "control", "fc", "ukmo"]
    data = pipeline("2T", model_names, "0", forecast_hours)

    # Write predictions to file
    data.sort_values(['issue_date', 'forecast_hour'],
                     ascending=True, inplace=True)
    file_path = 'output/' + '_'.join(model_names) + '_' + \
        str(forecast_hours[0]) + '_' + str(forecast_hours[-1]) + '.csv'
    data_readers.write_csv(data, file_path)
