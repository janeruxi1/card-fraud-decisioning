"""The demo renders from the committed records (skipped where streamlit or the records are missing)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
NEEDED = ["01_audit", "03_model", "05_policy", "05_policy_cells", "06_monitoring"]


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("streamlit")
    if not all((ROOT / "reports" / "metrics" / f"{n}.json").exists() for n in NEEDED):
        pytest.skip("records not present")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_runs_without_errors(app):
    assert not app.exception


def test_policy_calculator_reacts_to_capacity(app):
    before = [m.value for m in app.metric]
    app.slider[5].set_value(3.0).run()
    assert not app.exception
    assert [m.value for m in app.metric] != before
