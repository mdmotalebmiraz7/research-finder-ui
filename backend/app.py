
import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import fitz
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from groq import Groq
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util

# You will implement this function in crag.py.
from crag import crag_answer


# -------------------- Configuration --------------------

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

YEAR_FROM = 2020
PAPERS_PER_TOPIC = 100
TITLE_TOP_K = 20
FINAL_TOP_K = 10

app = FastAPI(
    title="Research Finder API",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------- In-memory storage --------------------

result = None
selected_paper = None
current_pdf_bytes = None

logs: list[dict] = []
log_lock = threading.Lock()


def clear_current_pdf():
    """Remove the active PDF from RAM."""
    global current_pdf_bytes, selected_paper

    current_pdf_bytes = None
    selected_paper = None


def add_log(message: str, level: str = "INFO"):
    entry = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
    }

    with log_lock:
        logs.append(entry)

        if len(logs) > 500:
            del logs[:-500]

    print(f"[{level}] {message}")


# -------------------- Load ranking model --------------------

add_log("Loading semantic ranking model...")

semantic_model = SentenceTransformer(EMBEDDING_MODEL)

add_log("Semantic ranking model loaded.")


# -------------------- Request schemas --------------------

class ResearchRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=2000)
    api_key: str = Field(min_length=10)


class RankRequest(BaseModel):
    rank: int = Field(ge=1, le=10)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    api_key: str = Field(min_length=10)


# -------------------- LangGraph schemas --------------------

class State(TypedDict):
    idea: str
    summary: str
    titles: list[str]
    search_results: Annotated[list[list[dict]], list.__add__]
    papers: list[dict]
    final_results: list[dict]
    api_key: str


class WorkerState(TypedDict):
    title: str
    summary: str
    api_key: str


class ResearchPlan(BaseModel):
    summary: str
    titles: list[str] = Field(min_length=10, max_length=10)


# -------------------- OpenAlex search --------------------

def reconstruct_abstract(data):
    if not data:
        return ""

    words = []

    for word, positions in data.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(word for position, word in words)


def search_papers(
    query: str,
    search_limit: int = 100,
    year_from: int = 2020,
) -> list[dict]:

    if not query.strip():
        return []

    add_log(f"Searching OpenAlex for: {query}")

    papers = []
    cursor = "*"

    while len(papers) < search_limit:
        params = {
            "search": query,
            "filter": f"from_publication_date:{year_from}-01-01",
            "per-page": 100,
            "cursor": cursor,
        }

        try:
            response = requests.get(
                "https://api.openalex.org/works",
                params=params,
                timeout=30,
            )

            response.raise_for_status()
            data = response.json()

        except requests.RequestException as exc:
            add_log(f"OpenAlex search failed: {exc}", "ERROR")
            raise

        for item in data.get("results", []):
            title = item.get("title", "")

            if not title:
                continue

            best_oa = item.get("best_oa_location") or {}
            primary = item.get("primary_location") or {}

            pdf_url = (
                best_oa.get("pdf_url")
                or primary.get("pdf_url")
            )

            if not pdf_url:
                continue

            doi = item.get("doi", "")

            landing_url = (
                doi
                or best_oa.get("landing_page_url")
                or primary.get("landing_page_url")
                or item.get("id", "")
            )

            papers.append({
                "title": title,
                "abstract": reconstruct_abstract(
                    item.get("abstract_inverted_index")
                ),
                "year": item.get("publication_year"),
                "citations": item.get("cited_by_count", 0),
                "doi": doi,
                "url": landing_url,
                "link": landing_url,
                "openalex_url": item.get("id", ""),
                "pdf_url": pdf_url,
            })

            if len(papers) >= search_limit:
                break

        add_log(f"Collected {len(papers)} papers")

        cursor = data.get("meta", {}).get("next_cursor")

        if not cursor:
            break

    add_log(f"Found {len(papers)} papers for: {query}")

    return papers


# -------------------- Semantic ranking --------------------

async def semantic_title_rank(
    papers: list[dict],
    query: str,
    top_k: int = TITLE_TOP_K,
) -> list[dict]:

    if not papers:
        return []

    def rank():
        query_vector = semantic_model.encode(
            query,
            convert_to_tensor=True,
        )

        titles = [
            paper.get("title", "")
            for paper in papers
        ]

        title_vectors = semantic_model.encode(
            titles,
            convert_to_tensor=True,
        )

        scores = util.cos_sim(
            query_vector,
            title_vectors,
        )[0]

        indices = scores.argsort(descending=True)[:top_k]

        ranked = []

        for idx in indices:
            i = int(idx)

            item = papers[i].copy()
            item["Semantic"] = float(scores[i])

            ranked.append(item)

        return ranked

    return await asyncio.to_thread(rank)


async def semantic_abstract_rank(
    papers: list[dict],
    summary: str,
    top_k: int = FINAL_TOP_K,
) -> list[dict]:

    if not papers:
        return []

    def rank():
        query_vector = semantic_model.encode(
            summary,
            convert_to_tensor=True,
        )

        abstracts = [
            paper.get("abstract") or paper.get("title", "")
            for paper in papers
        ]

        abstract_vectors = semantic_model.encode(
            abstracts,
            convert_to_tensor=True,
        )

        scores = util.cos_sim(
            query_vector,
            abstract_vectors,
        )[0]

        indices = scores.argsort(descending=True)[:top_k]

        ranked = []

        for rank_num, idx in enumerate(indices, start=1):
            i = int(idx)
            paper = papers[i]

            url = paper.get("url") or paper.get("link") or ""

            ranked.append({
                "Rank": rank_num,
                "title": paper.get("title", ""),
                "link": url,
                "url": url,
                "pdf_url": paper.get("pdf_url") or "",
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "doi": paper.get("doi", ""),
                "Semantic": float(scores[i]),
            })

        return ranked

    return await asyncio.to_thread(rank)


# -------------------- LangGraph research workflow --------------------

def generate_research_plan(
    idea: str,
    api_key: str,
) -> ResearchPlan:

    client = Groq(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "titles": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["summary", "titles"],
        "additionalProperties": False,
    }

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Create a concise "
                    "research summary and exactly 10 distinct academic "
                    "search topics. Return the requested structured data."
                ),
            },
            {
                "role": "user",
                "content": f"Research idea:\n{idea}",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "research_plan",
                "strict": True,
                "schema": schema,
            },
        },
        temperature=0.2,
        max_completion_tokens=2500,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The model returned an empty response.")

    plan = ResearchPlan.model_validate_json(content)

    if len(plan.titles) != 10:
        raise ValueError("The model must return exactly 10 topics.")

    return plan


def generate(state: State):
    plan = generate_research_plan(
        state["idea"],
        state["api_key"],
    )

    return {
        "summary": plan.summary,
        "titles": plan.titles,
    }


def dispatch_search(state: State):
    return [
        Send(
            "process_title",
            {
                "title": title,
                "summary": state["summary"],
                "api_key": state["api_key"],
            },
        )
        for title in state["titles"]
    ]


async def process_title(state: WorkerState):
    papers = await asyncio.to_thread(
        search_papers,
        state["title"],
        PAPERS_PER_TOPIC,
        YEAR_FROM,
    )

    ranked = await semantic_title_rank(
        papers,
        state["title"],
        TITLE_TOP_K,
    )

    return {"search_results": [ranked]}


def collect_unique_papers(state: State):
    unique = {}

    for topic_results in state["search_results"]:
        for paper in topic_results:
            url = paper.get("url") or paper.get("link") or ""

            if url and url not in unique:
                unique[url] = paper.copy()

    papers = list(unique.values())

    add_log(f"Collected {len(papers)} unique papers.")

    return {"papers": papers}


async def process_abstract(state: State):
    final_results = await semantic_abstract_rank(
        state["papers"],
        state["summary"],
        FINAL_TOP_K,
    )

    add_log(f"Final ranking complete: {len(final_results)} papers.")

    return {"final_results": final_results}


builder = StateGraph(State)

builder.add_node("generate", generate)
builder.add_node("process_title", process_title)
builder.add_node("collect_unique_papers", collect_unique_papers)
builder.add_node("process_abstract", process_abstract)

builder.add_edge(START, "generate")

builder.add_conditional_edges(
    "generate",
    dispatch_search,
    ["process_title"],
)

builder.add_edge("process_title", "collect_unique_papers")
builder.add_edge("collect_unique_papers", "process_abstract")
builder.add_edge("process_abstract", END)

graph = builder.compile()


# -------------------- API endpoints --------------------

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Research Finder API is running.",
    }


@app.get("/logs")
async def get_logs():
    with log_lock:
        return {"logs": list(logs)}


@app.post("/research")
async def run_research(request: ResearchRequest):
    global result

    add_log(f"Research started: {request.idea}")

    try:
        # A new research search invalidates the previous selected PDF.
        clear_current_pdf()

        result = await graph.ainvoke({
            "idea": request.idea,
            "api_key": request.api_key,
            "summary": "",
            "titles": [],
            "search_results": [],
            "papers": [],
            "final_results": [],
        })

        add_log("Research completed successfully.")

        return {
            "summary": result["summary"],
            "titles": result["titles"],
            "papers": result["final_results"],
        }

    except Exception as exc:
        add_log(f"Research failed: {exc}", "ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"Research failed: {exc}",
        ) from exc


@app.post("/paper")
async def get_paper(request: RankRequest):
    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Run a research search first.",
        )

    paper = next(
        (
            item
            for item in result["final_results"]
            if item.get("Rank") == request.rank
        ),
        None,
    )

    if paper is None:
        raise HTTPException(
            status_code=404,
            detail="Paper rank not found.",
        )

    return {
        "Rank": paper["Rank"],
        "title": paper.get("title", ""),
        "Paper_URL": paper.get("link", ""),
        "PDF_URL": paper.get("pdf_url", ""),
    }


@app.post("/paper/process")
async def process_selected_paper(request: RankRequest):
    global current_pdf_bytes, selected_paper

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Run a research search first.",
        )

    paper = next(
        (
            item
            for item in result["final_results"]
            if item.get("Rank") == request.rank
        ),
        None,
    )

    if paper is None:
        clear_current_pdf()

        raise HTTPException(
            status_code=404,
            detail="Paper rank not found.",
        )

    pdf_url = paper.get("pdf_url", "")

    if not pdf_url:
        clear_current_pdf()

        raise HTTPException(
            status_code=400,
            detail="PDF URL is not available.",
        )

    # Remove the previous PDF immediately.
    clear_current_pdf()

    add_log(f"Downloading PDF: {paper.get('title', '')}")

    try:
        response = await asyncio.to_thread(
            requests.get,
            pdf_url,
            timeout=60,
            headers={
                "User-Agent": "Mozilla/5.0 ResearchFinder/1.2",
            },
        )

        response.raise_for_status()

        pdf_bytes = response.content

        # Validate PDF and count pages. No PDF file is written to disk.
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError(
                "The URL did not return a valid PDF file."
            )

        with fitz.open(
            stream=pdf_bytes,
            filetype="pdf",
        ) as pdf:
            total_pages = len(pdf)

            if total_pages == 0:
                raise ValueError("The PDF contains no pages.")

        # Keep only the active PDF in RAM.
        current_pdf_bytes = pdf_bytes
        selected_paper = paper

        add_log(
            f"PDF loaded into RAM: {total_pages} pages."
        )

        return {
            "title": paper.get("title", ""),
            "total_pages": total_pages,
            "message": "PDF is ready in the reader.",
        }

    except Exception as exc:
        clear_current_pdf()

        add_log(f"PDF processing failed: {exc}", "ERROR")

        if isinstance(exc, requests.RequestException):
            detail = f"Failed to download PDF: {exc}"
        else:
            detail = f"Failed to process PDF: {exc}"

        raise HTTPException(
            status_code=500,
            detail=detail,
        ) from exc


@app.get("/paper/pdf")
async def get_current_pdf():
    """Serve the active PDF to the browser's built-in PDF viewer."""

    if current_pdf_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="No PDF is loaded. Select and process a paper first.",
        )

    return Response(
        content=current_pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": 'inline; filename="paper.pdf"',
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    if current_pdf_bytes is None:
        raise HTTPException(
            status_code=400,
            detail="Select and process a paper before asking questions.",
        )

    try:
        question_llm = ChatGroq(
            model=MODEL_NAME,
            api_key=request.api_key,
            temperature=0,
        )

        # Retrieval and answering are implemented by you in crag.py.
        answer = await asyncio.to_thread(
            crag_answer,
            request.question,
            current_pdf_bytes,
            question_llm,
        )

        add_log(f"Answered question: {request.question}")

        return {
            "question": request.question,
            "answer": answer,
        }

    except Exception as exc:
        add_log(f"Question answering failed: {exc}", "ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer the question: {exc}",
        ) from exc


@app.on_event("startup")
async def startup_message():
    add_log("Research Finder API is ready.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
