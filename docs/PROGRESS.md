# Project Progress

Living document. **Read at the start of every Claude Code session. Update before ending one.** Keep entries short and factual.

## Current state
- **Phase:** planning and documentation. No application code exists.
- **Last updated:** 2026-11-14
- **Awaiting:** user approval of the architecture docs, contracts and open decisions below.

## Done
- [x] Read the HCLTech problem statement and extracted FR-01 to FR-08.
- [x] Created `CLAUDE.md`, `docs/architecture.md`, `docs/api_contracts.md`, `docs/implementation_plan.md`, `docs/aws_prerequisites.md`, and this file.

## Milestone status
| Milestone | Owner | Status |
|---|---|---|
| M0 Foundations / contract freeze | M1 | not started |
| M1 Ingestion & indexing | M2 + M1 | not started |
| M2 Retrieval | M3 | not started |
| M3 Generation | M4 | not started |
| M4 Integration & UI | M1 / M4 | not started |
| M5 Evaluation & tuning | M4 | not started |
| M6 Submission polish | all | not started |

## Decisions made
- Stack and flows as listed in `CLAUDE.md` (finalized by the user).
- Contracts in `api_contracts.md` are drafts, pending approval.

## Open decisions / assumptions to confirm
1. The embeddings and vectorstore wrappers are **assigned to Member 1** (assumption, since the brief's four areas do not name them).
2. Embedding dimension 1024.
3. Reranker approach (BM25 + vector RRF proposed) and conflict detection approach (metadata + heuristics proposed).
4. Source of the document corpus, and the version-extraction convention.
5. AWS account, region and credential mechanism (`docs/aws_prerequisites.md`).
6. How much of the pipeline uses LangChain abstractions vs. plain boto3 and chromadb (proposal: LangChain for splitters and prompt templates only where they do not obscure the contracts).

## Contract change log
(none yet)

## Next steps
1. User approves or amends the docs.
2. Start M0: scaffold, config, schemas, AWS smoke test.

## Session notes
- 2026-11-14: The workspace initially contained only `docs/EnterpriseDocQ&Aassist_AIEngineer_Usecase2.pdf`. No files were overwritten. The PDF was read with `pdftotext`, because `pypdf` and the Read tool's PDF renderer were unavailable on this machine.
