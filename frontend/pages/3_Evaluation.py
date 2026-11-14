"""Evaluation dashboard: shows results produced by scripts/run_evaluation.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st  # noqa: E402

from frontend import api_client  # noqa: E402
from frontend.components import ui  # noqa: E402
from frontend.components.metrics import render_evaluation  # noqa: E402

ui.setup_page("Evaluation", "📊", "Measured retrieval, answer and citation quality on a labelled synthetic question set.")
ok, result = ui.call_api(api_client.get_client().latest_evaluation)
if ok and result is None:
    st.info("No evaluation results exist yet. Generate them with:")
    st.code("python scripts/run_evaluation.py --live", language="powershell")
elif ok:
    render_evaluation(result)
    ok_report, report = ui.call_api(api_client.get_client().latest_report)
    if ok_report and report:
        with st.expander("Full evaluation report (reports/evaluation_report.md)"):
            st.markdown(report)
