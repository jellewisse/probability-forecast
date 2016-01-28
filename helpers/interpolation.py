# interpolation.py
from math import (
    radians as deg2rad,
    sin,
    cos,
    atan2,
    sqrt
)
import numpy as np


def distance(point1, point2):
    """Computes the haversine distance between two (lat,lon) points.

    This method assumes the earth is a sphere with a constant radius.
    The error is 0.55\%, which is good enough for most uses.
    Returns distance in whole meters.
    """
    (lat1, lon1) = point1
    (lat2, lon2) = point2
    R = 6371000  # Radius of the earth in meters

    # Convert degrees to radians
    phi_1 = deg2rad(lat1)
    phi_2 = deg2rad(lat2)
    delta_phi = deg2rad(lat2 - lat1)
    delta_lambda = deg2rad(lon2 - lon1)

    # Haversine
    a = sin(delta_phi / 2) * sin(delta_phi / 2) + \
        cos(phi_1) * cos(phi_2) * \
        sin(delta_lambda / 2) * sin(delta_lambda / 2)

    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    d = R * c
    return int(round(d))


def bilinear_interpolate(points, values, query_point):
    assert len(points) == 4, "Bad number of points provided."
    assert len(points) == len(values), "Unequal number of points and values."

    # Sort points such that:
    # 0 - bottom left
    # 1 - bottom right
    # 2 - top left
    # 3 - top right

    # Assert query point lies within square.
    # TODO


def nearest_grid_point(distances):
    """Returns the index of the nearest grid point."""
    return distances.index(min(distances))


def nearest_grid_point_interpolate(forecasts,
                                   lats=None, lons=None, dists=None):
    """Nearest grid point interpolation function.

    forecasts should be a dataframe """
    assert dists is not None, "Distances should be provided."
    values = np.around(forecasts.ix[:, nearest_grid_point(dists)].values, 3)
    values.astype(np.float32, copy=False)
    return values


def interpolate(forecasts, lats, lons, dists,
                forecast_fun=nearest_grid_point_interpolate):
    """General interpolation function."""

    assert len(forecasts.columns) == 4, "Bad number of grid points provided."
    assert len(lats) == len(lons), "Latitudes and longitudes don't match."
    assert len(dists) == 4, "Bad number of distances provided."

    interpolated_forecasts = \
        forecast_fun(forecasts=forecasts, lats=lats, lons=lons, dists=dists)

    return interpolated_forecasts
