"""The demo renders from the committed records (skipped where streamlit or the records are missing)."""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app" / "streamlit_app.py"
NEEDED = ["01_audit", "03_model", "04_anomaly", "05_policy", "05_policy_cells", "06_monitoring"]


@pytest.fixture(scope="module")
def app():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    if not all((ROOT / "reports" / "metrics" / f"{n}.json").exists() for n in NEEDED):
        pytest.skip("records not present")
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(str(APP), default_timeout=60).run()


def test_app_runs_without_errors(app):
    assert not app.exception


def test_policy_calculator_reacts_to_capacity(app):
    before = [m.value for m in app.metric]
    app.slider(key="capacity").set_value(3.0).run()
    assert not app.exception
    assert [m.value for m in app.metric] != before


def test_scenario_preset_sets_assumptions(app):
    app.selectbox(key="preset").set_value("Chargebacks cost twice as much").run()
    assert not app.exception
    assert app.slider(key="fee").value == 30.0
    assert app.slider(key="capacity").value == 1.5


def test_single_payment_verdict_changes_with_probability(app):
    app.selectbox(key="preset").set_value("Baseline (decision memo)").run()
    app.slider(key="one_p").set_value(0.0).run()
    low = [a.value for a in app.success]
    app.slider(key="one_p").set_value(0.3).run()
    assert low and low[0].startswith("**APPROVE**")
    assert [w.value for w in app.warning][0].startswith("**REVIEW**")
