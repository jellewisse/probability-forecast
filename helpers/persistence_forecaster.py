"""Module for persistence forecast."""


class PersistenceForecaster(object):

    def __init__(self):
        self.previous_value = None
        self.previous_date = None

    def _is_initialized(self):
        return self.previous_value is not None

    def _update(self, value, previous_date):
        self.previous_value = value
        self.previous_date = previous_date

    def _run(self, current_date):
        if not self._is_initialized():
            return_value = None
            return_date = None
        elif dates_differ_one_day(current_date, self.previous_date):
            return_value = self.previous_value
            return_date = self.previous_date
        else:
            return_value = None
            return_date = None
        return(return_value, return_date)

    def run_and_update(self, value, date):
        return_object = self._run(date)
        self._update(value, date)
        return return_object


def dates_differ_one_day(date1, date2):
    date_diff = date1 - date2
    return abs(date_diff.days) == 1


class ForecastStruct(object):

    def __init__(self, value, forecast_hour, valid_date, station_name):
        self.value = value
        self.forecast_hour = forecast_hour
        self.valid_date = valid_date
        self.station_name = station_name
