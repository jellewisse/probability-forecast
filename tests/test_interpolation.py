import pytest


@pytest.fixture(
    params=[
        # Example distance 1
        {
            'args': [(40.7486, -73.9864), (45.7486, -61.4252)],
            'return': 1157496
        },

        # Distance from Amsterdam to New york.
        {
            'args': [(52.37403, 4.88969), (40.71427, -74.00597)],
            'return': 5861938
        }
    ]
)
def mock_distances(request):
    return request.param


def test_distance(mock_distances):
    """Tests the correct implementation of the Haversine distance using
    mocked data points. Note, the distances incorporate arond 0.55\% error."""

    from helpers.interpolation import distance
    assert distance(*mock_distances['args']) == mock_distances['return']


def test_commuted_distance(mock_distances):
    """Test whether distance calculation is commutative."""
    from helpers.interpolation import distance
    assert distance(*list(reversed(mock_distances['args']))) == \
        mock_distances['return']


def test_argsort():
    from helpers.interpolation import argsort
    input_list = [1, 2, 3, 100, 5]
    output_index = [0, 1, 2, 4, 3]

    test_index = argsort(input_list)
    assert output_index == test_index


@pytest.fixture(
    params=[
        # Example rectangle 1 in terms of (lat, lon) pairs
        {
            'lats': [0, 0, 1, 1],
            'lons': [0, 1, 0, 1],
            'index': [2, 3, 0, 1]   # Sorted in order NW, NE, SW, SE
        },

        # Example rectangle 2
        {
            'lats': [-1, 1, 1, -1],
            'lons': [5, 5, -5, -5],
            'index': [2, 1, 3, 0]
        }
    ]
)
def mock_rectangle(request):
    return request.param


def test_find_left_right():
    from helpers.interpolation import find_left_right
    lons = [0, 1]
    index = [0, 1]
    test_index = find_left_right(lons, index)
    assert list(test_index) == index

    # Now revert the index. The answer should be reversed as well
    index = list(reversed(index))
    test_index = find_left_right(lons, index)
    assert list(test_index) == index


def test_grid_point_order(mock_rectangle):
    from helpers.interpolation import grid_point_order
    lats = mock_rectangle['lats']
    lons = mock_rectangle['lons']
    index = mock_rectangle['index']

    test_index = grid_point_order(lats, lons)
    assert index == list(test_index)


@pytest.fixture(
    params=[
        {
            'point': (0, 0),
            'weights': [.25, .25, .25, .25]
        }
    ]
)
def mock_request(request):
    """Mock requests for points on the grid specified in
    test_bilinear_weights"""
    return request.param


def test_bilinear_weights(mock_request):
    from helpers.interpolation import get_bilinear_weights
    lats = [-1, 1, -1, 1]
    lons = [-1, -1, 1, 1]
    test_weights = get_bilinear_weights(
        mock_request['point'][0],
        mock_request['point'][1],
        lats,
        lons
    )
    print(type(test_weights))
    print(type(mock_request['weights']))
    assert all(mock_request['weights'] == test_weights)
