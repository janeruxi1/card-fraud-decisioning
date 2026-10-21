import os

import pytest

os.environ["FRAUDRISK_SYNTHETIC"] = "1"  # tests never read the real data


@pytest.fixture(scope="session")
def synthetic():
    from fraudrisk.synthetic import make_transactions

    return make_transactions(n_customers=1_500, seed=3)
