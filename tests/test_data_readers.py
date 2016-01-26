# test_data_readers.py


def test_read_meta_data():

    # Test function relevant imports
    import helpers.data_readers as data_readers

    lats, lons, dists = data_readers.read_meta_data('fc', '167', '0_test')
    assert lats == [52.25, 52.25, 52.38, 52.38], "Latitudes don't match."
    assert lons == [4.88, 4.75, 4.88, 4.75], "Longitudes don't match."
    assert dists == [9.29, 7.78, 8.76, 7.15], "Distances don't match."
