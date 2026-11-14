"""Run the evaluation and write reports/eval_results.json + reports/evaluation_report.md.

    python scripts/run_evaluation.py              # offline dry run (no AWS; NOT representative)
    python scripts/run_evaluation.py --live       # real Titan + Nova Lite (billable, needs Bedrock access)

Live mode makes roughly one Titan call per indexed chunk, one per question, and one Nova call per question
that has usable evidence (about 30 + 28 + 25 calls). Your own Chroma collection is never read or modified:
the corpus is indexed into a temporary directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.core.bedrock import BedrockError  # noqa: E402
from app.core.config import ConfigurationError, get_settings  # noqa: E402
from app.evaluation import evaluator  # noqa: E402
from app.evaluation.dataset import DatasetError, load_manifest, load_questions  # noqa: E402
from app.evaluation.report_generator import attach_comparison, write_reports  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", action="store_true", help="use real Bedrock (Titan + Nova Lite); default is an offline dry run")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--chunk-size", type=int, default=evaluator.EVAL_CHUNK_SIZE,
                    help=f"chunk size in characters (default {evaluator.EVAL_CHUNK_SIZE}, small so the short synthetic documents "
                         "yield several chunks; the app's deployed default is 1000)")
    ap.add_argument("--chunk-overlap", type=int, default=evaluator.EVAL_CHUNK_OVERLAP)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data" / "eval")
    ap.add_argument("--baseline", type=Path, default=None, help="earlier eval_results.json to compare against (adds a before/after section)")
    ap.add_argument("--analysis", type=Path, default=None, help="Markdown file appended to the report (a human-written analysis)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: REPORTS_DIR, normally reports/)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    try:
        settings = get_settings()
        questions = load_questions(args.data_dir / "questions.jsonl")
        manifest = load_manifest(args.data_dir / "corpus.json")
    except (ConfigurationError, DatasetError, OSError) as exc:
        print(f"Setup problem: {exc}")
        return 2
    mode = "live" if args.live else "offline"
    print(f"Mode: {mode.upper()}  ({len(questions)} questions, {len(manifest)} documents)")
    if mode == "offline":
        print("Offline dry run: stand-in embedder, no generation. Results are NOT representative of Titan/Nova.")
    try:
        with evaluator.build_pipeline(mode, args.data_dir / "corpus", manifest, settings,
                                      chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap) as pipeline:
            results = evaluator.run(pipeline, questions, args.top_k)
    except (evaluator.EvaluationSetupError, DatasetError, BedrockError) as exc:
        print(f"Evaluation could not run: {exc}")
        return 1
    if args.baseline:
        import json
        attach_comparison(results, json.loads(args.baseline.read_text(encoding="utf-8")))
    extra = args.analysis.read_text(encoding="utf-8") if args.analysis else ""
    md, js = write_reports(results, args.out or Path(settings.reports_dir), extra)
    rel, o = results["reliability"], results["retrieval"]["overall"]
    print(f"Retrieval@{args.top_k}: recall {o.get(f'recall@{args.top_k}', 0):.3f}  MRR {o.get('mrr', 0):.3f}  nDCG {o.get(f'ndcg@{args.top_k}', 0):.3f}")
    fc = results["failure_counts"]
    print(f"Failed queries: {rel['failed_queries']}/{rel['n_questions']}")
    print(f"Flagged entries: {fc['entries']} on {fc['questions_flagged']} questions "
          f"(failure {fc['by_severity']['failure']}, review {fc['by_severity']['review']}, note {fc['by_severity']['note']})")
    print(f"Wrote {md}\nWrote {js}")
    return 0 if rel["failed_queries"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
