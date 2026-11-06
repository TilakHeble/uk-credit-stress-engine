import pytest

from ukstress.config import BCST_PATHS_FILE, MODELS_DIR

pytest.importorskip("streamlit")
pytestmark = pytest.mark.skipif(
    not (BCST_PATHS_FILE.exists() and (MODELS_DIR / "champions.json").exists()),
    reason="data or models absent",
)


def test_dashboard_runs_and_responds_to_severity():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(MODELS_DIR.parent / "dashboards" / "app.py"), default_timeout=300).run()
    assert not at.exception
    base_total = at.metric[0].value
    at.sidebar.slider[0].set_value(0.0).run()  # hold the economy at end-2024
    assert not at.exception
    assert at.metric[0].value != base_total
