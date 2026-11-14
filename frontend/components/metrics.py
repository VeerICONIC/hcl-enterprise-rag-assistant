"""Metric cards and the evaluation-results view. Only real, loaded numbers are ever displayed."""

from __future__ import annotations

import streamlit as st

from frontend.components import ui


def fmt(v) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def render_evaluation(result: dict) -> None:
    mode = result.get("mode", "unknown")
    if mode == "live":
        st.success(f"Live run · {result.get('timestamp', '')} · embedding `{result.get('embed_model', '?')}` · LLM `{result.get('llm_model', '?')}`")
    else:
        st.warning("**Offline dry run, not representative.** Embeddings and answers here come from test doubles, "
                   "so only the retrieval *machinery* is exercised. Run the live evaluation (see README) for "
                   "meaningful numbers.")
    st.caption(f"{result.get('n_questions', 0)} questions · corpus: {result.get('n_documents', 0)} documents · "
               f"report: `{result.get('report_path', 'reports/evaluation_report.md')}`")

    ret = result.get("retrieval") or {}
    st.subheader("Retrieval (automatically computed)")
    if ret.get("overall"):
        o = ret["overall"]
        cols = st.columns(5)
        for col, (label, key) in zip(cols, [("Recall@5", "recall@5"), ("MRR", "mrr"), ("nDCG@5", "ndcg@5"),
                                            ("Hit rate@5", "hit_rate@5"), ("Questions", "n")]):
            col.metric(label, fmt(o.get(key)))
        if ret.get("by_category"):
            st.dataframe([{"Category": k, **{m: fmt(v) for m, v in vals.items()}} for k, vals in ret["by_category"].items()],
                         hide_index=True, width="stretch")
    else:
        st.info("No retrieval metrics in this result.")

    st.subheader("Answer quality (automatic proxies)")
    ans = result.get("answers")
    if ans:
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Key-fact recall", "key_fact_recall"), ("Groundedness proxy (v2)", "groundedness"),
                                            ("Fallback accuracy", "status_accuracy"), ("Conflict surfaced", "conflict_surfaced_rate")]):
            col.metric(label, fmt(ans.get(key)))
    else:
        st.info("Answer metrics were not measured in this run (generation was not executed).")

    st.subheader("Citations")
    cit = result.get("citations")
    if cit:
        cols = st.columns(4)
        for col, (label, key) in zip(cols, [("Valid citations", "validity"), ("Citation precision", "precision"),
                                            ("Citation recall", "recall"), ("Page/section correct", "location_correct")]):
            col.metric(label, fmt(cit.get(key)))
    else:
        st.info("Citation metrics were not measured in this run.")

    st.subheader("Reliability")
    rel = result.get("reliability") or {}
    cols = st.columns(3)
    for col, (label, key) in zip(cols, [("Failed queries", "failed_queries"), ("Empty retrieval rate", "empty_retrieval_rate"),
                                        ("API error rate", "error_rate")]):
        col.metric(label, fmt(rel.get(key)))

    failures = result.get("failure_cases") or []
    counts = result.get("failure_counts")
    st.subheader(f"Flagged cases ({len(failures)} entries)")
    if counts:
        sev = counts["by_severity"]
        st.info(f"**{counts['entries']} entries on {counts['questions_flagged']} of {counts['n_questions']} questions**, "
                f"while {result['reliability']['failed_queries']} queries actually failed. "
                f"**{sev['failure']} failure** (a defect), **{sev['review']} review** (a proxy flag that needs a human look), "
                f"**{sev['note']} note** (informational). One question can raise several entries.")
    order = {"failure": 0, "review": 1, "note": 2}
    icon = {"failure": "🔴", "review": "🟡", "note": "⚪"}
    for f in sorted(failures, key=lambda f: (order.get(f.get("severity", "failure"), 0), f["id"]))[:40]:
        sev_name = f.get("severity", "failure")
        with st.expander(f"{icon.get(sev_name, '🔴')} {sev_name} · {f['id']} · {f['category']} · {f['reason']}"):
            st.write(f["question"])
            if f.get("detail"):
                st.caption(f["detail"])
    st.caption("Automatic metrics are proxies defined in the report; they are not human judgements of answer correctness.")
