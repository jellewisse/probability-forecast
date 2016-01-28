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
