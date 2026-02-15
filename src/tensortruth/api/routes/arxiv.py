"""ArXiv metadata lookup endpoint."""

import asyncio
import logging

import arxiv
from fastapi import APIRouter, HTTPException

from tensortruth.api.schemas import ArxivLookupResponse
from tensortruth.utils.validation import validate_arxiv_id

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/arxiv/{arxiv_id:path}", response_model=ArxivLookupResponse)
async def lookup_arxiv(arxiv_id: str) -> ArxivLookupResponse:
    """Look up arXiv paper metadata by ID.

    Accepts new-format IDs (2301.12345) and old-format (hep-th/9901001).
    """
    normalized = validate_arxiv_id(arxiv_id)
    if normalized is None:
        raise HTTPException(status_code=400, detail="Invalid arXiv ID")

    def _fetch():
        search = arxiv.Search(id_list=[normalized])
        try:
            return next(search.results())
        except StopIteration:
            return None

    try:
        loop = asyncio.get_event_loop()
        paper = await asyncio.wait_for(
            loop.run_in_executor(None, _fetch),
            timeout=15,
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="arXiv API timeout")
    except Exception as e:
        logger.error(f"arXiv lookup failed: {e}")
        raise HTTPException(status_code=502, detail="arXiv API error")

    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    return ArxivLookupResponse(
        arxiv_id=normalized,
        title=paper.title,
        authors=[a.name for a in paper.authors],
        published=paper.published.strftime("%Y-%m-%d"),
        categories=list(paper.categories),
        abstract=paper.summary.strip(),
        pdf_url=paper.pdf_url,
    )
