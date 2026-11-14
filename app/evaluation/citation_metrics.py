"""Citation checks against the labelled relevant chunks and the real retrieved chunks.

* validity           - citation maps to a retrieved chunk, its metadata matches, and the excerpt is a verbatim
                       substring of that chunk (``generation.citation_builder.is_valid_citation``).
* location correct   - the citation's section and page equal the stored chunk's section and page.
* precision / recall - citations pointing at a labelled-relevant chunk, and questions where at least one
                       labelled-relevant chunk is cited.
"""

from __future__ import annotations

from typing import Mapping

from app.generation.citation_builder import is_valid_citation
from app.schemas import Answer, Chunk


def check(answer: Answer, retrieved: Mapping[str, Chunk], relevant: set[str]) -> dict:
    valid = location = on_target = 0
    for c in answer.citations:
        chunk = retrieved.get(c.chunk_id)
        if chunk is None:
            continue
        valid += is_valid_citation(c, chunk)
        location += (c.section == chunk.metadata.section and c.page == chunk.metadata.page)
        on_target += c.chunk_id in relevant
    return {"n": len(answer.citations), "valid": valid, "location_correct": location, "on_target": on_target,
            "hit": bool({c.chunk_id for c in answer.citations} & relevant) if relevant else None}
