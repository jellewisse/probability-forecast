"""Tests for data io."""
# test_data_readers.py
import pytest
MOCK_DATA_PATH = 'tests/mock_data/'


@pytest.fixture(
    scope="module",
    params=[
        {'model': 'fc', 'element_id': '167',
            'station_name': 'schiphol', 'issue': '0'},
        {'model': 'eps', 'element_id': '167',
            'station_name': 'schiphol', 'issue': '0'},
        {'model': 'control', 'element_id': '167',
            'station_name': 'schiphol', 'issue': '0'},
    ]
)
def file_parameters(request):
    """Py.test fixture for file parameters."""
    request.param['file_path'] = MOCK_DATA_PATH
    return request.param


def test_read_forecast_data(file_parameters):
    """Test whether the relevant foreacst files can be loaded."""
    # Test function relevant imports
    from helpers import data_io
    data_io.read_forecast_data(**file_parameters)


def test_read_meta_data():
    """Test whether the meta data parser function works properly."""
    # Test function relevant imports
    from helpers import data_io
    meta_data = data_io.read_meta_data(
            'fc', '167', 'schiphol', '0', file_path=MOCK_DATA_PATH)
    assert meta_data['latitude'] == [52.25, 52.25, 52.38, 52.38]
    assert meta_data['longitude'] == [4.88, 4.75, 4.88, 4.75]
    assert meta_data['distance'] == [9.29, 7.78, 8.76, 7.15]
