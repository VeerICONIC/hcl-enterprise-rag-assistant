# Evaluation Report

**Mode: LIVE.** Generated 2026-11-14T15:01:09+00:00 with embeddings `amazon.titan-embed-text-v2:0` and answers from `apac.amazon.nova-lite-v1:0`.

## 1. Dataset

* Corpus: 8 synthetic, fictional documents (PDF, HTML, Markdown, TXT; two deliberately conflicting handbook versions), indexed into a temporary store as 23 chunks (chunk size 400, overlap 60 characters).
* Questions: 28, labelled by hand: direct, paraphrase, multi-document, conflicting-version, ambiguous, unanswerable, and similar-keyword/different-meaning.
* Relevance labels are text anchors (`filename` + a phrase) resolved to indexed chunk IDs at run time; ambiguous and unanswerable questions have none.
* Construction assumptions: the corpus and questions were written by the project team, not drawn from real user logs; questions reuse the vocabulary of the corpus more than real users would, so scores are likely optimistic.

## 2. Methodology

* Retrieval metrics (top-5, binary relevance, answerable questions only): Recall@k, MRR, nDCG@k, hit rate. Computed automatically.
* Answer metrics (live only) are **automatic proxies, not human judgements**: key-fact recall (expected facts appear in the answer), a lexical groundedness proxy (answer sentences covered by cited text), status accuracy (answered / conflict / needs_clarification / not_found as expected), and whether conflicts were surfaced with two sources.
* Citation metrics (live only): validity (maps to a retrieved chunk, excerpt is a verbatim substring), section/page correctness, precision and recall against the labelled chunks.
* Reliability: failed queries, empty-retrieval rate, error rate.

## 3. Results

### Retrieval

| Scope | n | Recall@5 | MRR | nDCG@5 | Hit@5 |
|---|---|---|---|---|---|
| **Overall** | 21 | 1.000 | 0.976 | 0.982 | 1.000 |
| direct | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | 5 | 1.000 | 0.900 | 0.926 | 1.000 |
| multi_document | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| conflict | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| similar_keywords | 3 | 1.000 | 1.000 | 1.000 | 1.000 |

* Conflict detection (heuristic): flagged 1.000 of 4 conflicting-version questions; false-positive rate on 17 other answerable questions: 0.059.
* Retrieval-level abstention: evidence strength `none` for 0.571 of 7 unanswerable/ambiguous questions (the answer stage can still decline when retrieval does not).

### Answers and citations

| Metric | Value |
|---|---|
| Key-fact recall | 0.975 |
| Groundedness proxy (v2) | 1.000 |
| Groundedness, run-1 proxy (comparison only) | 0.833 |
| Status accuracy | 0.964 |
| Conflict surfaced (2+ sources) | 1.000 |
| Citations: validity | 1.000 |
| Citations: section/page correct | 1.000 |
| Citations: precision | 1.000 |
| Citations: recall | 1.000 |

Status accuracy by category: direct 1.000, paraphrase 0.800, multi_document 1.000, conflict 1.000, ambiguous 1.000, unanswerable 1.000, similar_keywords 1.000

### Reliability

Failed queries: 0 of 28; error rate 0.000; empty-retrieval rate 0.000.

## 4. Evidence-backed examples

### Example 1 (direct): How many days of sick leave do employees get?

* Expected: `answered`. Reference: 10 days per year (both handbook versions agree).
* Retrieval: evidence `strong`, top cosine 0.8345, conflicts flagged: True
  * #1 `employee_handbook_v2.md` § Leave Policy > Sick Leave (cosine 0.8345, labelled relevant): "Employees receive 10 days of sick leave per year. A medical certificate is required for absences longer than 3 days."
  * #2 `employee_handbook_v1.md` § Leave Policy > Sick Leave (cosine 0.8345, labelled relevant): "Employees receive 10 days of sick leave per year. A medical certificate is required for absences longer than 3 days."
  * #3 `employee_handbook_v2.md` § Leave Policy > Annual Leave (cosine 0.3765): "Full-time employees receive 25 days of annual leave per year. Unused annual leave may be carried over for up to 10 days into the next year."
  * conflict `version_difference`: METADATA CONFLICT: 'employee_handbook_v2.md' (version 2.0) and 'employee_handbook_v1.md' (version 1.0) appear to be different versions of the same document ('employee handbook') and both match the question. Which version is current is NOT determined here (file
  * conflict `contradiction`: POSSIBLE contradiction on 'annual employees full': '25 days' in employee_handbook_v2.md § Leave Policy > Annual Leave, version 2.0 vs '20 days' in employee_handbook_v1.md § Leave Policy > Annual Leave, version 1.0. This is a heuristic comparison of numbers/dat
* Answer (`answered`, confidence high): Employees receive 10 days of sick leave per year. [1][2]
  * [1] `employee_handbook_v2.md#Leave Policy > Sick Leave` v2.0: "Employees receive 10 days of sick leave per year."
  * [2] `employee_handbook_v1.md#Leave Policy > Sick Leave` v1.0: "Employees receive 10 days of sick leave per year."

### Example 2 (paraphrase): How long can I wait before telling IT that my notebook computer is missing?

* Expected: `answered`. Reference: Within 24 hours.
* Retrieval: evidence `strong`, top cosine 0.6683, conflicts flagged: False
  * #1 `it_security_policy.txt` (cosine 0.1357): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
  * #2 `it_security_policy.txt` (cosine 0.6683, labelled relevant): "A lost or stolen laptop must be reported to the IT help desk within 24 hours. USB storage devices are prohibited on company computers."
  * #3 `troubleshooting.md` § Router problems (cosine 0.1854): "To factory reset the router, hold the reset button for 10 seconds until the lights blink."
* Answer (`answered`, confidence high): You can wait up to 24 hours before telling IT that your notebook computer is missing [1].
  * [1] `it_security_policy.txt`: "A lost or stolen laptop must be reported to the IT help desk within 24 hours."

### Example 3 (conflict): How many days of annual leave do employees get?

* Expected: `conflict`. Reference: Handbook v1.0 says 20 days; v2.0 says 25 days. The sources disagree.
* Retrieval: evidence `strong`, top cosine 0.9002, conflicts flagged: True
  * #1 `employee_handbook_v1.md` § Leave Policy > Annual Leave (cosine 0.9002, labelled relevant): "Full-time employees receive 20 days of annual leave per year. Unused annual leave may be carried over for up to 5 days into the next year."
  * #2 `employee_handbook_v2.md` § Leave Policy > Annual Leave (cosine 0.8942, labelled relevant): "Full-time employees receive 25 days of annual leave per year. Unused annual leave may be carried over for up to 10 days into the next year."
  * #3 `employee_handbook_v2.md` § Leave Policy > Sick Leave (cosine 0.3639): "Employees receive 10 days of sick leave per year. A medical certificate is required for absences longer than 3 days."
  * conflict `version_difference`: METADATA CONFLICT: 'employee_handbook_v1.md' (version 1.0) and 'employee_handbook_v2.md' (version 2.0) appear to be different versions of the same document ('employee handbook') and both match the question. Which version is current is NOT determined here (file
  * conflict `contradiction`: POSSIBLE contradiction on 'annual employees full': '20 days' in employee_handbook_v1.md § Leave Policy > Annual Leave, version 1.0 vs '25 days' in employee_handbook_v2.md § Leave Policy > Annual Leave, version 2.0. This is a heuristic comparison of numbers/dat
* Answer (`conflict`, confidence medium): The number of days of annual leave for employees differs between versions of the Employee Handbook. [1] states that full-time employees receive 20 days of annual leave per year, while [2] states that full-time employees receive 25 days of annual leave per year.
  * [1] `employee_handbook_v1.md#Leave Policy > Annual Leave` v1.0: "Full-time employees receive 20 days of annual leave per year. Unused annual leave may be carried over for up to 5 days into the next year."
  * [2] `employee_handbook_v2.md#Leave Policy > Annual Leave` v2.0: "Full-time employees receive 25 days of annual leave per year. Unused annual leave may be carried over for up to 10 days into the next year."

### Example 4 (unanswerable): What is the company's current stock price?

* Expected: `not_found`. Reference: Not in the documents.
* Retrieval: evidence `none`, top cosine 0.1043, conflicts flagged: False
  * #1 `it_security_policy.txt` (cosine 0.0423): "A lost or stolen laptop must be reported to the IT help desk within 24 hours. USB storage devices are prohibited on company computers."
  * #2 `billing_faq.html` § Payment methods (cosine 0.1043): "You can update your payment method from the Billing page in your account settings."
  * #3 `billing_faq.html` § Refunds (cosine 0.1017): "Refunds are available within 14 days of purchase. Requests must be sent to the billing team."
* Answer (`not_found`, confidence low): I could not find enough supporting evidence in the indexed documents to answer this question. Try rephrasing it, or upload the relevant documents.

### Example 5 (ambiguous): What is the limit?

* Expected: `needs_clarification` (accepted: needs_clarification, not_found). Reference: Ambiguous: could mean API rate limit, payload limit, meal limit...
* Retrieval: evidence `none`, top cosine 0.2216, conflicts flagged: False
  * #1 `api_guide.pdf` § Rate Limits p.1 (cosine 0.2216): "The Standard plan allows 100 requests per minute. The Enterprise plan allows 1000 requests per minute."
  * #2 `product_release_notes.md` § Release 3.2 (2024-03-01) (cosine 0.1641): "Release 3.2 adds single sign-on (SSO) support. Known issue: exports fail when a report has more than 10,000 rows."
  * #3 `api_guide.pdf` § Payload and Authentication p.2 (cosine 0.1501): "The maximum request payload size is 5 MB. Bearer tokens expire after 60 minutes."
* Answer (`not_found`, confidence low): I could not find enough supporting evidence in the indexed documents to answer this question. Try rephrasing it, or upload the relevant documents.

### Example 6 (similar_keywords): How do I reset my password?

* Expected: `answered`. Reference: Use the self-service identity portal.
* Retrieval: evidence `weak`, top cosine 0.2043, conflicts flagged: False
  * #1 `it_security_policy.txt` (cosine 0.2043, labelled relevant): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
  * #2 `troubleshooting.md` § Router problems (cosine 0.2013): "To factory reset the router, hold the reset button for 10 seconds until the lights blink."
  * #3 `onboarding_guide.md` § Accounts (cosine 0.1617): "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
* Answer (`answered`, confidence low): To reset your password, use the self-service identity portal as per the IT Security Policy [1]. The help desk will never ask for your password [1].
  * [1] `it_security_policy.txt`: "The help desk will never ask for your password."

### Example 7 (multi_document): What should a new hire know about passwords and multi-factor authentication?

* Expected: `answered` (accepted: answered, conflict). Reference: Set a password that follows the IT Security Policy (12+ characters, changed every 90 days) and enrol in MFA in the first week.
* Retrieval: evidence `strong`, top cosine 0.7478, conflicts flagged: False
  * #1 `onboarding_guide.md` § Accounts (cosine 0.7478, labelled relevant): "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
  * #2 `it_security_policy.txt` (cosine 0.4054, labelled relevant): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
  * #3 `onboarding_guide.md` § First day (cosine 0.38): "New hires receive a laptop and a security badge on their first day."
* Answer (`answered`, confidence high): A new hire should know that they must set a password that is at least 12 characters long, cannot reuse the last 5 passwords, and must be changed every 90 days [1][2]. They must also enroll in multi-factor authentication for all VPN and email access [1][2].
  * [1] `onboarding_guide.md#Accounts`: "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
  * [2] `it_security_policy.txt`: "Passwords must be at least 12 characters long and must be changed every 90 days."

## 5. Failure cases

**6 flagged entries on 6 of 28 questions** (0 queries actually failed). By severity: 2 failure, 0 review (proxy flags needing a human look), 4 note (informational). 2 question(s) have at least one 'failure'. An entry is not a failed query: one question can raise several entries.

| Severity | Question | Category | Reason | Detail |
|---|---|---|---|---|
| failure | q11: Can I plug a USB memory stick into my work computer? | paraphrase | wrong_status | got not_found, accepted ['answered'] |
| failure | q12: What should a new hire know about passwords and multi-factor authentication? | multi_document | missing_key_facts | key-fact recall 0.50 |
| note | q01: How many days of sick leave do employees get? | direct | retrieval_conflict_not_shown | retrieval flagged (employee handbook; annual employees full (day)) among passages the answer does not cite; the answer shown to the user has |
| note | q23: Who is the CEO of Northwind Systems? | unanswerable | no_retrieval_abstention | evidence_strength=weak (the answer stage may still decline) |
| note | q24: How many days of paternity leave does the company offer? | unanswerable | no_retrieval_abstention | evidence_strength=weak (the answer stage may still decline) |
| note | q25: What is the Wi-Fi password for the guest network? | unanswerable | no_retrieval_abstention | evidence_strength=weak (the answer stage may still decline) |

## 6. Human review (not automatically scored)

Automatic proxies cannot judge fluency, completeness or subtle unsupported claims. A reviewer should read the examples above and the full `eval_results.json`, then fill this in:

| Question id | Correct? | Grounded? | Citations appropriate? | Notes |
|---|---|---|---|---|
| q01 |  |  |  |  |
| q07 |  |  |  |  |
| q15 |  |  |  |  |
| q22 |  |  |  |  |
| q19 |  |  |  |  |
| q26 |  |  |  |  |
| q12 |  |  |  |  |

## 7. Limitations

* Small synthetic corpus and question set; team-written, so likely easier than real usage. No confidence intervals.
* Groundedness and key-fact recall are lexical proxies; they can miss paraphrased unsupported claims and reward answers that simply repeat cited words.
* Conflict detection is heuristic (numbers, units, dates, document families); the labelled conflicts here are exactly that kind, so its recall on this set overstates real-world performance.
* Evidence-strength thresholds and `MIN_RELEVANCE` are uncalibrated for Titan; this set is a starting point for calibrating them.
* Offline mode uses a stand-in embedder and no LLM.
* LLM output varies slightly between runs even at temperature 0, so a single run can differ from the next on borderline questions.
* The groundedness proxy only checks sentences that carry a citation or a number; an uncited qualitative sentence escapes it.

## 8. Recommendations

* Run the live evaluation, then tune `MIN_RELEVANCE`, the evidence margins and chunk size on the failure cases.
* Add more paraphrased and longer, messier documents; add human-labelled judgements for answer correctness.
* Consider an LLM-as-judge for groundedness only after validating it against human labels.


## 9. Comparison with run 1

Run 1 (live, 2026-11-14T13:58:10+00:00, LLM `apac.amazon.nova-lite-v1:0`) is preserved unchanged in `reports/eval_results_run1_original.json` and `reports/evaluation_report_run1_original.md`. Same questions, labels, corpus, chunking and thresholds.

| Metric | Run 1 | This run |
|---|---|---|
| Recall@5 | 1.000 | 1.000 |
| MRR | 0.976 | 0.976 |
| nDCG@5 | 0.982 | 0.982 |
| Key-fact recall | 0.972 | 0.975 |
| Status accuracy | 0.857 | 0.964 |
| Conflict surfaced | 0.750 | 1.000 |
| Citation validity | 1.000 | 1.000 |
| Citation precision | 1.000 | 1.000 |
| Citation recall | 1.000 | 1.000 |
| Groundedness, run-1 proxy | 0.741 | 0.833 |
| Groundedness, proxy v2 (new method) | n/a | 1.000 |
| Questions with a 'failure' entry | 5 | 2 |
| Flagged entries (all severities) | 15 | 6 |

**System changes between the runs:**

* Tokenizer: personal pronouns (my, me, his, ...) are now stop words, so "How do I reset my password?" has 2 query terms instead of 3 and the lexical-rescue rule can apply (a bug, not a threshold change).
* Generator: evidence aliases are normalized leniently ([E1], e1, 1), and a claim list with no valid evidence no longer discards an answer whose text carries valid inline citations.
* Generator: a conflict only counts for an answer if it involves a passage the answer actually cites (previously any retrieved conflict forced status 'conflict').
* Generator/prompt: the model may not call a version latest/newer/older/previous/current/outdated unless a source says so, and a deterministic warning flags it if it does.

**Measurement (methodology) changes:**

* Groundedness proxy v2: words copied from the question and conflict-framing words are ignored; numbers in an answer must appear in the cited text (or its version/file metadata); a sentence is checkable if it has a citation marker or a number. The run-1 proxy is still reported as 'groundedness_v1' for the same answers.
* 'false_conflict' now measures the conflict shown to the user (the answer's conflicts); a conflict flagged by retrieval only among passages the answer does not cite is recorded as the note 'retrieval_conflict_not_shown'. In offline mode (no answer) the retrieval-level flag is used.
* Groundedness proxy v2 judges word coverage only when at least 3 content words remain after exclusions (numbers are always judged); this removed false alarms on short, fully supported sentences whose remaining words were connectives.
* Failure cases now carry a severity: 'failure' (a defect by definition), 'review' (a proxy flag that needs a human look) or 'note' (informational, e.g. retrieval did not abstain but the answer stage correctly declined). Run-1 entries are classified with the same rules for comparison.

Differences between runs mix system changes, measurement changes and normal LLM variation; the groundedness rows show the measurement change on its own (same answers, two proxies).

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
