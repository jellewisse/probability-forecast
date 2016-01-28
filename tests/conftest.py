import pytest


@pytest.fixture(autouse=True)
def setup_root_path(scope="session"):
    """Adds the project root to the python path so developed modules can be
    imported."""
    import sys
    import os
    myPath = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, myPath + '/../')
