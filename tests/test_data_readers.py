# test_data_readers.py
import pytest


@pytest.fixture(
    scope="module",
    params=[
        {'model': 'fc', 'element_id': '167', 'issue': '0'},
        {'model': 'eps', 'element_id': '167', 'issue': '0'},
        {'model': 'control', 'element_id': '167', 'issue': '0'},
    ]
)
def file_parameters(request):
    return request.param


def test_read_forecast_data(file_parameters):
    """Tests whether the relevant foreacst files can be loaded."""
    # Test function relevant imports
    import helpers.data_readers as data_readers
    data_readers.read_forecast_data(**file_parameters)


def test_read_observations():
    """Tests whether the station observations can be loaded."""
    # Test function relevant imports
    import helpers.data_readers as data_readers
    obs_data = data_readers.read_observations()
    assert len(obs_data) != 0, "Observation data is empty."


def test_read_meta_data():
    """Tests whether the meta data parser function works properly."""
    # Test function relevant imports
    import helpers.data_readers as data_readers

    lats, lons, dists = data_readers.read_meta_data('fc', '167', '0')
    assert lats == [52.25, 52.25, 52.38, 52.38], "Latitudes don't match."
    assert lons == [4.88, 4.75, 4.88, 4.75], "Longitudes don't match."
    assert dists == [9.29, 7.78, 8.76, 7.15], "Distances don't match."
