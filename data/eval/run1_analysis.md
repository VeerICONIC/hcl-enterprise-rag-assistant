## 10. Analysis of the 15 flagged entries in run 1

Run 1 reported "failure cases: 15". That number counts **entries**, not failed queries: 0 of 28 queries failed, and the 15 entries fall on 11 distinct questions (q01 raised two). The original run is preserved unchanged in `reports/eval_results_run1_original.json` / `reports/evaluation_report_run1_original.md`. Questions, labels, corpus, chunking and thresholds were not changed. Labels were checked against the evidence and no label error was found.

| Entry | Question | What happened | Type | Disposition |
|---|---|---|---|---|
| q01 `wrong_status` + `false_conflict` | sick leave days | Retrieval correctly ranked the two sick-leave chunks 1-2 and answered "10 days", but the annual-leave chunks (20 vs 25 days) were also retrieved and the generator forced status `conflict` for **any** retrieved conflict. | answer-generation defect | **Fixed**: a conflict only counts if it involves a passage the answer cites. The retrieval-level flag remains as the note `retrieval_conflict_not_shown`. |
| q07, q08, q10 `low_groundedness_proxy` | paraphrased direct questions | Answers were correct and cited the right chunk ("24 hours", "the 1st", "60 minutes"). The proxy scored 0.00 because the answer repeated words from the *question* ("notebook", "computer", "bearer") that are not in the chunk. | evaluation-metric artifact | **Metric changed** (proxy v2, see below). Answers unchanged. |
| q15, q17 `low_groundedness_proxy` | conflict questions | Correct two-sided answers; the proxy penalised the framing sentence ("differs between versions of the Employee Handbook"). | evaluation-metric artifact | **Metric changed** (proxy v2). |
| q18 `low_groundedness_proxy` | meal limit conflict | Both values correct and cited, but Nova called v2.0 "the latest version" and v1.0 "the previous version". The sources never say which is current. | **real defect** (answer wording) plus metric artifact | **Fixed**: prompt forbids latest/newer/older/previous/current/outdated unless a source says so, and a deterministic warning flags it. |
| q16 `wrong_status` | carry-over days | The retrieved evidence and conflict flag were right, but Nova's claim list failed validation and the generator withheld the whole answer as `not_found`, although the answer text carried valid `[E#]` citations. On a repro Nova produced a valid answer, so Nova's output also varies between runs. | citation-validation brittleness | **Fixed**: lenient alias normalization; a claim list with no valid evidence no longer discards an answer with valid inline citations. |
| q26 `wrong_status` | "How do I reset my password?" | The correct chunk was ranked #1 but its cosine (0.204) is below `MIN_RELEVANCE`. The "lexical rescue" rule should have applied (both query words are in the chunk), but "my" was not a stop word, so the query had 3 terms and the chunk covered only 2/3 (< 80%). | retrieval bug (tokenizer) | **Fixed**: personal pronouns are stop words. The question is now answered. |
| q11 `wrong_status` | USB memory stick | The correct chunk is ranked #1 but is a multi-topic chunk with cosine 0.269 < `MIN_RELEVANCE` (0.30), and the paraphrase shares almost no keywords, so evidence is `none` and the LLM is not called. | insufficient-evidence gate (false abstention) | **Open, not tuned.** On this set a cosine cut-off cannot separate this chunk (0.269) from unanswerable-question distractors (0.30-0.36). Needs a better signal (smaller or topic-pure chunks, or a reranker), not a lower threshold chosen to lift the score. |
| q12 `missing_key_facts` | passwords + MFA for new hires | The answer is grounded and cited but omits the 90-day rotation rule that the reference includes. | answer completeness | **Open, intermittent** (not changed): it also failed in run 2 and the deployed-defaults run but passed in the final main run, i.e. it depends on what Nova chooses to include. |
| q23, q24, q25 `no_retrieval_abstention` | unanswerable | Retrieval reported `weak` evidence (cosine 0.30-0.36) instead of `none`, but the answer stage correctly returned `not_found` with no citations. | informational | Reclassified as a **note**: behaviour is correct end to end. |

**By type (15 entries):** 6 groundedness-proxy flags (5 pure metric artifacts, 1 also a real wording defect), 4 wrong-status (2 generator defects, 1 tokenizer bug, 1 calibration limit), 3 informational abstention notes, 1 false conflict (the q01 generator defect), 1 completeness gap.

**Methodology changes (documented in section 9):** groundedness proxy v2, failure severity levels, and measuring `false_conflict` on the conflict shown to the user. The old proxy is still reported on the same answers (`groundedness_v1`), so the effect of the proxy change is visible: **v2 is more lenient than v1 by design, so the jump from 0.74 to 1.0 is partly measurement and must not be read as a pure quality gain**; the like-for-like v1 figure on the final run is 0.90.

**What is still open after the final run:** q11 (false abstention, calibration), q12 (intermittent incompleteness), residual retrieval-level conflict flags on unrelated passages (q01 note), and LLM run-to-run variance (a single 28-question run cannot show it; repeat runs would).

## 11. Sensitivity to chunk size (what the deployed default does)

The main runs index the corpus with 400-character chunks so the short synthetic documents yield several chunks. The application's deployed default is `CHUNK_SIZE=1000`. Both were evaluated live (same questions, labels, thresholds and code):

| Metric | 400 / 60 (main run) | 1000 / 150 (deployed default) |
|---|---|---|
| Chunks indexed | 23 | 22 |
| Recall@5 / MRR / nDCG@5 | 1.000 / 0.976 / 0.982 | 1.000 / 0.968 / 0.976 |
| Status accuracy | 0.964 | 0.857 |
| Key-fact recall | 0.975 | 1.000 |
| Citation validity / precision / recall | 1.000 / 1.000 / 1.000 | 1.000 / 1.000 / 1.000 |
| Questions with a `failure` entry | 2 (q11, q12) | 4 (q05, q07, q11, q26) |

Full output of the second run: `reports/deployed_defaults/`.

**Finding.** At 1000 characters the whole 5-rule IT security policy becomes **one chunk**. The correct chunk is still retrieved (Recall@5 is 1.0), but its embedding is a blend of five unrelated rules, so its cosine to any single-rule question falls below `MIN_RELEVANCE` (0.30). With no keyword rescue (a paraphrase, or one query word missing from the chunk) the evidence gate returns `none`, the LLM is never called, and the user gets "not enough evidence" for questions the documents answer (q05 "lost laptop", q07 its paraphrase, q11 USB, q26 password reset). This is a retrieval-quality effect that Recall@k cannot see, because it happens after retrieval.

**What was and was not changed.** `MIN_RELEVANCE`, the lexical-rescue rule and the chunk-size default were not changed to improve these scores. The data suggests a smaller chunk size (about 400-500 characters) suits documents made of short, unrelated rules, but this is one small synthetic corpus: whether it holds for real documents is unknown. Set `CHUNK_SIZE`/`CHUNK_OVERLAP` in `.env`, delete and re-upload documents (changing it requires re-indexing), and re-run `python scripts/run_evaluation.py --live --chunk-size N` to check on your own material.

**Weak-evidence prompt change.** Before the last fix, a correct but low-cosine chunk could be left out of the prompt entirely (only chunks with cosine >= 0.30, or the top 2, were shown). For `weak` evidence the generator now shows every returned passage; the main-run result for q26 and q16 reflects that, and the three unanswerable questions (q23-q25) were still correctly declined.

**Final-run note.** Numbers above are from the last pair of live runs (after all code changes). Across the five live runs the same questions moved between pass and fail from run to run (q12, and q26 at the deployed chunk size), because Nova's wording and its willingness to answer from a weakly-matching passage vary even at temperature 0. Treat single-question differences as noise; the stable signals are the false abstentions on the one diluted IT-policy chunk at 1000 characters (q05, q07, q11) and the consistently correct handling of conflicts, unanswerable and ambiguous questions. Earlier runs are kept in `reports/history/`.
