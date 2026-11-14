# Evaluation Report

**Mode: LIVE.** Generated 2026-11-14T15:01:36+00:00 with embeddings `amazon.titan-embed-text-v2:0` and answers from `apac.amazon.nova-lite-v1:0`.

## 1. Dataset

* Corpus: 8 synthetic, fictional documents (PDF, HTML, Markdown, TXT; two deliberately conflicting handbook versions), indexed into a temporary store as 22 chunks (chunk size 1000, overlap 150 characters).
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
| **Overall** | 21 | 1.000 | 0.968 | 0.976 | 1.000 |
| direct | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| paraphrase | 5 | 1.000 | 1.000 | 1.000 | 1.000 |
| multi_document | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
| conflict | 4 | 1.000 | 1.000 | 1.000 | 1.000 |
| similar_keywords | 3 | 1.000 | 0.778 | 0.833 | 1.000 |

* Conflict detection (heuristic): flagged 1.000 of 4 conflicting-version questions; false-positive rate on 17 other answerable questions: 0.059.
* Retrieval-level abstention: evidence strength `none` for 0.571 of 7 unanswerable/ambiguous questions (the answer stage can still decline when retrieval does not).

### Answers and citations

| Metric | Value |
|---|---|
| Key-fact recall | 1.000 |
| Groundedness proxy (v2) | 1.000 |
| Groundedness, run-1 proxy (comparison only) | 0.863 |
| Status accuracy | 0.857 |
| Conflict surfaced (2+ sources) | 1.000 |
| Citations: validity | 1.000 |
| Citations: section/page correct | 1.000 |
| Citations: precision | 1.000 |
| Citations: recall | 1.000 |

Status accuracy by category: direct 0.833, paraphrase 0.600, multi_document 1.000, conflict 1.000, ambiguous 1.000, unanswerable 1.000, similar_keywords 0.667

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
* Retrieval: evidence `none`, top cosine 0.2798, conflicts flagged: False
  * #1 `it_security_policy.txt` (cosine 0.2798, labelled relevant): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
  * #2 `troubleshooting.md` § Router problems (cosine 0.1854): "To factory reset the router, hold the reset button for 10 seconds until the lights blink."
  * #3 `onboarding_guide.md` § Accounts (cosine 0.1753): "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
* Answer (`not_found`, confidence low): I could not find enough supporting evidence in the indexed documents to answer this question. Try rephrasing it, or upload the relevant documents.

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
  * #1 `it_security_policy.txt` (cosine 0.0635): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
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
* Retrieval: evidence `weak`, top cosine 0.2013, conflicts flagged: False
  * #1 `troubleshooting.md` § Router problems (cosine 0.2013): "To factory reset the router, hold the reset button for 10 seconds until the lights blink."
  * #2 `onboarding_guide.md` § Accounts (cosine 0.1617): "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
  * #3 `it_security_policy.txt` (cosine 0.1386, labelled relevant): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
* Answer (`not_found`, confidence low): The evidence does not directly state how to reset your password. mentions that if you forget your password, you should reset it yourself using the self-service identity portal, but it does not provide details on how to do this. mentions that new hires must enroll in multi-factor authentication, but it does not provide information on password reset procedures. discusses account lockouts but does not cover password resets.

### Example 7 (multi_document): What should a new hire know about passwords and multi-factor authentication?

* Expected: `answered` (accepted: answered, conflict). Reference: Set a password that follows the IT Security Policy (12+ characters, changed every 90 days) and enrol in MFA in the first week.
* Retrieval: evidence `strong`, top cosine 0.7478, conflicts flagged: False
  * #1 `onboarding_guide.md` § Accounts (cosine 0.7478, labelled relevant): "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
  * #2 `it_security_policy.txt` (cosine 0.424, labelled relevant): "Northwind IT Security Policy Passwords must be at least 12 characters long and must be changed every 90 days. Reusing the last 5 passwords is not allowed. Multi…"
  * #3 `onboarding_guide.md` § First day (cosine 0.38): "New hires receive a laptop and a security badge on their first day."
* Answer (`answered`, confidence high): A new hire should know that they must set a password that is at least 12 characters long, cannot reuse the last 5 passwords, and must be changed every 90 days [1][2]. They must also enroll in multi-factor authentication (MFA) during their first week [1]. MFA is required for all VPN and email access [2]. If they forget their password, they should reset it themselves using the self-service identity portal [2].
  * [1] `onboarding_guide.md#Accounts`: "New hires must set a password that follows the IT Security Policy and enrol in multi-factor authentication during their first week."
  * [2] `it_security_policy.txt`: "If you forget your password, reset it yourself using the self-service identity portal. The help desk will never ask for your password.

A lost or stolen laptop must be reported to the IT help desk within 24 hours."

## 5. Failure cases

**8 flagged entries on 8 of 28 questions** (0 queries actually failed). By severity: 4 failure, 0 review (proxy flags needing a human look), 4 note (informational). 4 question(s) have at least one 'failure'. An entry is not a failed query: one question can raise several entries.

| Severity | Question | Category | Reason | Detail |
|---|---|---|---|---|
| failure | q05: How soon must a lost laptop be reported? | direct | wrong_status | got not_found, accepted ['answered'] |
| failure | q07: How long can I wait before telling IT that my notebook computer is missing? | paraphrase | wrong_status | got not_found, accepted ['answered'] |
| failure | q11: Can I plug a USB memory stick into my work computer? | paraphrase | wrong_status | got not_found, accepted ['answered'] |
| failure | q26: How do I reset my password? | similar_keywords | wrong_status | got not_found, accepted ['answered'] |
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
