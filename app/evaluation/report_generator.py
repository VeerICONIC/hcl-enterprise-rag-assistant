"""Write ``eval_results.json`` (machine-readable, shown by the dashboard) and ``evaluation_report.md``."""

from __future__ import annotations

import json
from pathlib import Path

# What changed between run 1 and the current run. The first block changed the SYSTEM; the second changed how
# results are MEASURED. Questions, labels, corpus, chunking and thresholds (MIN_RELEVANCE etc.) are unchanged.
SYSTEM_CHANGES = [
    "Tokenizer: personal pronouns (my, me, his, ...) are now stop words, so \"How do I reset my password?\" has 2 query terms instead of 3 and the lexical-rescue rule can apply (a bug, not a threshold change).",
    "Generator: evidence aliases are normalized leniently ([E1], e1, 1), and a claim list with no valid evidence no longer discards an answer whose text carries valid inline citations.",
    "Generator: a conflict only counts for an answer if it involves a passage the answer actually cites (previously any retrieved conflict forced status 'conflict').",
    "Generator/prompt: the model may not call a version latest/newer/older/previous/current/outdated unless a source says so, and a deterministic warning flags it if it does.",
]
METHODOLOGY_CHANGES = [
    "Groundedness proxy v2: words copied from the question and conflict-framing words are ignored; numbers in an answer must appear in the cited text (or its version/file metadata); a sentence is checkable if it has a citation marker or a number. The run-1 proxy is still reported as 'groundedness_v1' for the same answers.",
    "'false_conflict' now measures the conflict shown to the user (the answer's conflicts); a conflict flagged by retrieval only among passages the answer does not cite is recorded as the note 'retrieval_conflict_not_shown'. In offline mode (no answer) the retrieval-level flag is used.",
    "Groundedness proxy v2 judges word coverage only when at least 3 content words remain after exclusions (numbers are always judged); this removed false alarms on short, fully supported sentences whose remaining words were connectives.",
    "Failure cases now carry a severity: 'failure' (a defect by definition), 'review' (a proxy flag that needs a human look) or 'note' (informational, e.g. retrieval did not abstain but the answer stage correctly declined). Run-1 entries are classified with the same rules for comparison.",
]

EXAMPLE_ORDER = ["direct", "paraphrase", "conflict", "unanswerable", "ambiguous", "similar_keywords", "multi_document"]


def _f(v) -> str:
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def pick_examples(results: list[dict]) -> list[dict]:
    chosen = []
    for cat in EXAMPLE_ORDER:
        for r in results:
            if r["category"] == cat and r["retrieval"] is not None:
                chosen.append(r)
                break
    return chosen


def attach_comparison(res: dict, baseline: dict) -> None:
    """Add ``res['comparison']`` (before/after against a preserved earlier run)."""
    from app.evaluation.evaluator import SEVERITY, failure_counts

    base_cases = [{**f, "severity": f.get("severity") or SEVERITY.get(f["reason"], "failure")} for f in baseline["failure_cases"]]
    base_counts = failure_counts(base_cases, baseline["n_questions"])
    k = res["retrieval"]["k"]

    def pair(path, key):
        def get(r):
            node = r
            for p in path:
                node = (node or {}).get(p)
            return (node or {}).get(key) if isinstance(node, dict) or node is None else None
        return get(baseline), get(res)

    rows = []
    for label, path, key in [(f"Recall@{k}", ("retrieval", "overall"), f"recall@{k}"), ("MRR", ("retrieval", "overall"), "mrr"),
                             (f"nDCG@{k}", ("retrieval", "overall"), f"ndcg@{k}"),
                             ("Key-fact recall", ("answers",), "key_fact_recall"),
                             ("Status accuracy", ("answers",), "status_accuracy"),
                             ("Conflict surfaced", ("answers",), "conflict_surfaced_rate"),
                             ("Citation validity", ("citations",), "validity"), ("Citation precision", ("citations",), "precision"),
                             ("Citation recall", ("citations",), "recall")]:
        rows.append({"metric": label, "before": pair(path, key)[0], "after": pair(path, key)[1]})
    before_g = (baseline.get("answers") or {}).get("groundedness")
    rows.append({"metric": "Groundedness, run-1 proxy", "before": before_g, "after": (res.get("answers") or {}).get("groundedness_v1")})
    rows.append({"metric": "Groundedness, proxy v2 (new method)", "before": None, "after": (res.get("answers") or {}).get("groundedness")})
    rows.append({"metric": "Questions with a 'failure' entry", "before": base_counts["questions_with_failure"],
                 "after": res["failure_counts"]["questions_with_failure"]})
    rows.append({"metric": "Flagged entries (all severities)", "before": base_counts["entries"], "after": res["failure_counts"]["entries"]})
    res["comparison"] = {
        "baseline": {k2: baseline.get(k2) for k2 in ("mode", "timestamp", "llm_model", "embed_model")},
        "rows": rows, "failure_counts_before": base_counts, "failure_counts_after": res["failure_counts"],
        "system_changes": SYSTEM_CHANGES, "methodology_changes": METHODOLOGY_CHANGES,
    }


def render_markdown(res: dict) -> str:
    live = res["mode"] == "live"
    L: list[str] = ["# Evaluation Report", ""]
    if live:
        L += [f"**Mode: LIVE.** Generated {res['timestamp']} with embeddings `{res['embed_model']}` and answers from `{res['llm_model']}`.", ""]
    else:
        L += ["> **OFFLINE DRY RUN. These numbers are NOT representative.** Embeddings come from a hashed bag-of-words stand-in "
              f"(`{res['embed_model']}`) and no answers were generated. Only the retrieval machinery (BM25, rank fusion, "
              "evidence strength, conflict heuristics) was exercised. Real Titan/Nova quality has not been measured; "
              "run `python scripts/run_evaluation.py --live` once AWS access works.", "",
              f"Generated {res['timestamp']}.", ""]

    L += ["## 1. Dataset", "",
          f"* Corpus: {res['n_documents']} synthetic, fictional documents (PDF, HTML, Markdown, TXT; two deliberately conflicting handbook versions), "
          f"indexed into a temporary store as {res['n_chunks']} chunks (chunk size {res['chunking']['size']}, overlap {res['chunking']['overlap']} characters).",
          f"* Questions: {res['n_questions']}, labelled by hand: direct, paraphrase, multi-document, conflicting-version, ambiguous, unanswerable, and similar-keyword/different-meaning.",
          "* Relevance labels are text anchors (`filename` + a phrase) resolved to indexed chunk IDs at run time; ambiguous and unanswerable questions have none.",
          "* Construction assumptions: the corpus and questions were written by the project team, not drawn from real user logs; questions reuse the vocabulary of the corpus more than real users would, so scores are likely optimistic.", "",
          "## 2. Methodology", "",
          f"* Retrieval metrics (top-{res['top_k']}, binary relevance, answerable questions only): Recall@k, MRR, nDCG@k, hit rate. Computed automatically.",
          "* Answer metrics (live only) are **automatic proxies, not human judgements**: key-fact recall (expected facts appear in the answer), a lexical groundedness proxy (answer sentences covered by cited text), status accuracy (answered / conflict / needs_clarification / not_found as expected), and whether conflicts were surfaced with two sources.",
          "* Citation metrics (live only): validity (maps to a retrieved chunk, excerpt is a verbatim substring), section/page correctness, precision and recall against the labelled chunks.",
          "* Reliability: failed queries, empty-retrieval rate, error rate.", "",
          "## 3. Results", "", "### Retrieval", ""]
    ret = res["retrieval"]
    o, k = ret["overall"], ret["k"]
    if o:
        L += ["| Scope | n | Recall@%d | MRR | nDCG@%d | Hit@%d |" % (k, k, k), "|---|---|---|---|---|---|",
              f"| **Overall** | {o['n']} | {_f(o[f'recall@{k}'])} | {_f(o['mrr'])} | {_f(o[f'ndcg@{k}'])} | {_f(o[f'hit_rate@{k}'])} |"]
        for cat, m in ret["by_category"].items():
            L.append(f"| {cat} | {m['n']} | {_f(m[f'recall@{k}'])} | {_f(m['mrr'])} | {_f(m[f'ndcg@{k}'])} | {_f(m[f'hit_rate@{k}'])} |")
    cd, ab = ret["conflict_detection"], ret["abstention"]
    L += ["", f"* Conflict detection (heuristic): flagged {_f(cd['recall_on_conflict_questions'])} of {cd['n_conflict_questions']} conflicting-version questions; "
          f"false-positive rate on {cd['n_other_answerable']} other answerable questions: {_f(cd['false_positive_rate_on_other_answerable'])}.",
          f"* Retrieval-level abstention: evidence strength `none` for {_f(ab['evidence_none_rate_on_unanswerable_or_ambiguous'])} of {ab['n']} unanswerable/ambiguous questions "
          "(the answer stage can still decline when retrieval does not).", ""]
    L += ["### Answers and citations", ""]
    if res["answers"]:
        a, c = res["answers"], res["citations"] or {}
        L += ["| Metric | Value |", "|---|---|", f"| Key-fact recall | {_f(a['key_fact_recall'])} |", f"| Groundedness proxy (v2) | {_f(a['groundedness'])} |",
              f"| Groundedness, run-1 proxy (comparison only) | {_f(a.get('groundedness_v1'))} |",
              f"| Status accuracy | {_f(a['status_accuracy'])} |", f"| Conflict surfaced (2+ sources) | {_f(a['conflict_surfaced_rate'])} |",
              f"| Citations: validity | {_f(c.get('validity'))} |", f"| Citations: section/page correct | {_f(c.get('location_correct'))} |",
              f"| Citations: precision | {_f(c.get('precision'))} |", f"| Citations: recall | {_f(c.get('recall'))} |", ""]
        L += ["Status accuracy by category: " + ", ".join(f"{k} {_f(v)}" for k, v in a["status_accuracy_by_category"].items()), ""]
    else:
        L += ["_Not measured: generation was not run in this mode._", ""]
    rel = res["reliability"]
    L += ["### Reliability", "", f"Failed queries: {rel['failed_queries']} of {rel['n_questions']}; error rate {_f(rel['error_rate'])}; empty-retrieval rate {_f(rel['empty_retrieval_rate'])}.", ""]

    L += ["## 4. Evidence-backed examples", ""]
    for i, r in enumerate(pick_examples(res["questions"]), 1):
        ret_r = r["retrieval"]
        L += [f"### Example {i} ({r['category']}): {r['question']}", "",
              f"* Expected: `{r['expected_status']}`" + (f" (accepted: {', '.join(r['accepted_status'])})" if len(r["accepted_status"]) > 1 else "") + f". Reference: {r['reference_answer'] or 'n/a'}",
              f"* Retrieval: evidence `{ret_r['evidence_strength']}`, top cosine {ret_r['top_score']}, conflicts flagged: {ret_r['conflicts']['has_conflict']}"]
        for t in ret_r["top"][:3]:
            L.append(f"  * #{t['rank']} `{t['filename']}`" + (f" § {t['section']}" if t["section"] else "") + (f" p.{t['page']}" if t["page"] != -1 else "")
                     + f" (cosine {t['cosine']}{', labelled relevant' if t['relevant'] else ''}): \"{t['snippet']}\"")
        for g in ret_r["conflicts"]["groups"][:2]:
            L.append(f"  * conflict `{g['kind']}`: {g['description'][:260]}")
        if r["answer"]:
            ans = r["answer"]
            L += [f"* Answer (`{ans['status']}`, confidence {ans['confidence']}): {ans['text']}"]
            for n, c in enumerate(ans["citations"], 1):
                L.append(f"  * [{n}] `{c['source']}`" + (f" v{c['version']}" if c["version"] else "") + f": \"{c['excerpt']}\"")
        else:
            L.append("* Answer: not generated in this mode.")
        L.append("")

    L += ["## 5. Failure cases", ""]
    fc = res.get("failure_counts")
    if fc:
        sev = fc["by_severity"]
        L += [f"**{fc['entries']} flagged entries on {fc['questions_flagged']} of {fc['n_questions']} questions** "
              f"({res['reliability']['failed_queries']} queries actually failed). By severity: {sev['failure']} failure, "
              f"{sev['review']} review (proxy flags needing a human look), {sev['note']} note (informational). "
              f"{fc['questions_with_failure']} question(s) have at least one 'failure'. "
              "An entry is not a failed query: one question can raise several entries.", ""]
    if res["failure_cases"]:
        order = {"failure": 0, "review": 1, "note": 2}
        L += ["| Severity | Question | Category | Reason | Detail |", "|---|---|---|---|---|"]
        for f in sorted(res["failure_cases"], key=lambda f: (order.get(f.get("severity", "failure"), 0), f["id"])):
            L.append(f"| {f.get('severity', 'failure')} | {f['id']}: {f['question']} | {f['category']} | {f['reason']} | {f['detail'][:140]} |")
    else:
        L.append("None detected by the automatic checks.")
    L += ["", "## 6. Human review (not automatically scored)", "",
          "Automatic proxies cannot judge fluency, completeness or subtle unsupported claims. A reviewer should read the examples above and the full `eval_results.json`, then fill this in:", "",
          "| Question id | Correct? | Grounded? | Citations appropriate? | Notes |", "|---|---|---|---|---|"]
    L += [f"| {r['id']} |  |  |  |  |" for r in pick_examples(res["questions"])]
    L += ["", "## 7. Limitations", "",
          "* Small synthetic corpus and question set; team-written, so likely easier than real usage. No confidence intervals.",
          "* Groundedness and key-fact recall are lexical proxies; they can miss paraphrased unsupported claims and reward answers that simply repeat cited words.",
          "* Conflict detection is heuristic (numbers, units, dates, document families); the labelled conflicts here are exactly that kind, so its recall on this set overstates real-world performance.",
          "* Evidence-strength thresholds and `MIN_RELEVANCE` are uncalibrated for Titan; this set is a starting point for calibrating them.",
          "* Offline mode uses a stand-in embedder and no LLM.",
          "* LLM output varies slightly between runs even at temperature 0, so a single run can differ from the next on borderline questions.",
          "* The groundedness proxy only checks sentences that carry a citation or a number; an uncited qualitative sentence escapes it.", "",
          "## 8. Recommendations", "",
          "* Run the live evaluation, then tune `MIN_RELEVANCE`, the evidence margins and chunk size on the failure cases.",
          "* Add more paraphrased and longer, messier documents; add human-labelled judgements for answer correctness.",
          "* Consider an LLM-as-judge for groundedness only after validating it against human labels.", ""]
    cmp = res.get("comparison")
    if cmp:
        b = cmp["baseline"]
        L += ["", "## 9. Comparison with run 1", "",
              f"Run 1 ({b['mode']}, {b['timestamp']}, LLM `{b['llm_model']}`) is preserved unchanged in `reports/eval_results_run1_original.json` and "
              "`reports/evaluation_report_run1_original.md`. Same questions, labels, corpus, chunking and thresholds.", "",
              "| Metric | Run 1 | This run |", "|---|---|---|"]
        L += [f"| {r['metric']} | {_f(r['before'])} | {_f(r['after'])} |" for r in cmp["rows"]]
        L += ["", "**System changes between the runs:**", ""] + [f"* {c}" for c in cmp["system_changes"]]
        L += ["", "**Measurement (methodology) changes:**", ""] + [f"* {c}" for c in cmp["methodology_changes"]]
        L += ["", "Differences between runs mix system changes, measurement changes and normal LLM variation; the groundedness rows show the "
              "measurement change on its own (same answers, two proxies).", ""]
    return "\n".join(L)


def write_reports(res: dict, out_dir: Path, extra_markdown: str = "") -> tuple[Path, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path, json_path = out_dir / "evaluation_report.md", out_dir / "eval_results.json"
    res["report_path"] = str(md_path)
    res["examples"] = [r["id"] for r in pick_examples(res["questions"])]
    json_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    body = render_markdown(res) + (("\n" + extra_markdown) if extra_markdown else "")
    md_path.write_text(body, encoding="utf-8")
    return md_path, json_path
