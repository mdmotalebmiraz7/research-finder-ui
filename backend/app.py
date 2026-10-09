"""Research Finder API.

Run locally:
    pip install -r requirements.txt
    uvicorn research_finder:app --reload --host 0.0.0.0 --port 8000

Run directly (including with `%run research_finder.py` in Colab):
    python research_finder.py

Endpoints:
    GET  /health
    GET  /logs
    POST /research         {"idea": "...", "api_key": "..."}
    POST /paper            {"rank": 1}
    POST /paper/process    {"rank": 1}
    POST /ask              {"question": "...", "api_key": "..."}
    GET  /docs
"""

import asyncio
import logging
import threading
from datetime import datetime, timezone
from typing import Annotated, TypedDict

import arxiv
import fitz
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util


# -------------------- Configuration --------------------

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
YEAR_FROM = 2020
PAPERS_PER_TOPIC = 100
TITLE_TOP_K = 20
FINAL_TOP_K = 10

app = FastAPI(title="Research Finder API", version="1.0.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your frontend origin in production.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory prototype storage. Results are lost when the server restarts.
result = None
selected_paper = None
chunks: list[Document] = []
vector_store = None
retriever = None
logs: list[dict] = []
log_lock = threading.Lock()


def add_log(message: str, level: str = "INFO") -> None:
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


# -------------------- Load models once --------------------

add_log("Loading semantic ranking model...")
semantic_model = SentenceTransformer(EMBEDDING_MODEL)
add_log("Semantic ranking model loaded.")

# Embeddings are used when a PDF is processed. This model may download weights
# the first time the API starts, depending on the runtime's model cache.
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# -------------------- Request and graph schemas --------------------

class ResearchRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=2000)
    api_key: str = Field(min_length=10)


class RankRequest(BaseModel):
    rank: int = Field(ge=1, le=10)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    api_key: str = Field(min_length=10)


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


# -------------------- Paper search and ranking --------------------

import requests


def reconstruct_abstract(data):
    if not data:
        return ""

    words = []

    for word, positions in data.items():
        for position in positions:
            words.append((position, word))

    words.sort()

    return " ".join(
        word for position, word in words
    )


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
            add_log(
                f"OpenAlex search failed for '{query}': {exc}",
                "ERROR",
            )
            raise

        for item in data["results"]:
            title = item.get("title", "")

            if not title:
                continue

            best_oa = item.get("best_oa_location") or {}
            primary = item.get("primary_location") or {}

            pdf_url = (
                best_oa.get("pdf_url")
                or primary.get("pdf_url")
            )

            # PDF link na thakle skip
            if not pdf_url:
                continue

            doi = item.get("doi", "")

            papers.append({
                "title": title,
                "abstract": reconstruct_abstract(
                    item.get("abstract_inverted_index")
                ),
                "year": item.get("publication_year"),
                "citations": item.get("cited_by_count", 0),
                "doi": doi,
                "url": (
                    doi
                    or best_oa.get("landing_page_url")
                    or primary.get("landing_page_url")
                    or item.get("id", "")
                ),
                "link": (
                    doi
                    or best_oa.get("landing_page_url")
                    or primary.get("landing_page_url")
                    or item.get("id", "")
                ),
                "openalex_url": item.get("id", ""),
                "pdf_url": pdf_url,
            })

            if len(papers) >= search_limit:
                break

        add_log(f"Collected: {len(papers)} papers")

        cursor = data["meta"].get("next_cursor")

        if not cursor:
            break

    add_log(f"Found {len(papers)} papers for topic: {query}")
    return papers

async def semantic_title_rank(
    papers: list[dict], query: str, top_k: int = TITLE_TOP_K
) -> list[dict]:
    if not papers:
        return []

    def rank() -> list[dict]:
        query_vector = semantic_model.encode(query, convert_to_tensor=True)
        titles = [paper.get("title", "") for paper in papers]
        title_vectors = semantic_model.encode(titles, convert_to_tensor=True)
        scores = util.cos_sim(query_vector, title_vectors)[0]
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
    papers: list[dict], summary: str, top_k: int = FINAL_TOP_K
) -> list[dict]:
    if not papers:
        return []

    def rank() -> list[dict]:
        query_vector = semantic_model.encode(summary, convert_to_tensor=True)
        abstracts = [paper.get("abstract") or paper.get("title", "") for paper in papers]
        abstract_vectors = semantic_model.encode(abstracts, convert_to_tensor=True)
        scores = util.cos_sim(query_vector, abstract_vectors)[0]
        indices = scores.argsort(descending=True)[:top_k]

        ranked = []
        for rank_num, idx in enumerate(indices, start=1):
            i = int(idx)
            paper = papers[i]
            url = paper.get("url") or paper.get("link") or ""
            ranked.append(
                {
                    "Rank": rank_num,
                    "title": paper.get("title", ""),
                    "link": url,
                    "url": url,
                    "pdf_url": paper.get("pdf_url") or url.replace("/abs/", "/pdf/"),
                    "abstract": paper.get("abstract", ""),
                    "year": paper.get("year"),
                    "doi": paper.get("doi", ""),
                    "Semantic": float(scores[i]),
                }
            )
        return ranked

    return await asyncio.to_thread(rank)


# -------------------- LangGraph research workflow --------------------
from groq import Groq
def generate_research_plan(idea: str, api_key: str) -> ResearchPlan:
    client = Groq(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "titles": {
                "type": "array",
                "items": {"type": "string"}
            }
        },
        "required": ["summary", "titles"],
        "additionalProperties": False
    }

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. "
                    "Create a concise research summary and exactly "
                    "10 distinct academic search topics. "
                    "Return only the requested structured data."
                )
            },
            {
                "role": "user",
                "content": f"Research idea:\n{idea}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "research_plan",
                "strict": True,
                "schema": schema
            }
        },
        temperature=0.2,
        max_completion_tokens=2500
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The model returned an empty response.")

    plan = ResearchPlan.model_validate_json(content)

    if len(plan.titles) != 10:
        raise ValueError(
            f"Expected 10 research topics, got {len(plan.titles)}"
        )

    return plan
def generate(state):
    idea = state["idea"]
    api_key = state["api_key"]

    plan = generate_research_plan(idea, api_key)

    return {
        "summary": plan.summary,
        "titles": plan.titles,
    }
def dispatch_search(state: State):
    return [
        Send(
            "process_title",
            {"title": title, "summary": state["summary"], "api_key": state["api_key"]},
        )
        for title in state["titles"]
    ]

async def process_title(state: WorkerState) -> dict:
    title = state["title"]
    papers = await asyncio.to_thread(search_papers, title, PAPERS_PER_TOPIC, YEAR_FROM)
    ranked = await semantic_title_rank(papers, title, TITLE_TOP_K)
    return {"search_results": [ranked]}


def collect_unique_papers(state: State) -> dict:
    unique = {}
    for topic_results in state["search_results"]:
        for paper in topic_results:
            url = paper.get("url") or paper.get("link") or ""
            if url and url not in unique:
                unique[url] = paper.copy()
    papers = list(unique.values())
    add_log(f"Collected {len(papers)} unique papers from topic searches.")
    return {"papers": papers}


async def process_abstract(state: State) -> dict:
    final_results = await semantic_abstract_rank(
        state["papers"], state["summary"], FINAL_TOP_K
    )
    add_log(f"Final ranking complete: {len(final_results)} papers.")
    return {"final_results": final_results}


builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("process_title", process_title)
builder.add_node("collect_unique_papers", collect_unique_papers)
builder.add_node("process_abstract", process_abstract)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", dispatch_search, ["process_title"])
builder.add_edge("process_title", "collect_unique_papers")
builder.add_edge("collect_unique_papers", "process_abstract")
builder.add_edge("process_abstract", END)
graph = builder.compile()


# -------------------- PDF processing and paper Q&A --------------------

def extract_pdf_chunks(pdf_bytes: bytes) -> tuple[int, list[Document]]:
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        page_docs = [
            Document(page_content=page.get_text(), metadata={"page": i + 1})
            for i, page in enumerate(pdf)
            if page.get_text().strip()
        ]
    finally:
        pdf.close()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return len(page_docs), splitter.split_documents(page_docs)


def retrieve_text(question: str) -> str:
    if retriever is None:
        return ""
    docs = retriever.invoke(question)
    return "\n\n".join(doc.page_content for doc in docs if doc.page_content.strip())


qa_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a precise question-answering assistant. Use only the provided "
            "paper context. Do not invent facts. If the context does not contain the "
            "answer, say: 'The provided context does not contain this information.' "
            "If only part is answered, state what is missing. Be concise and answer "
            "in the same language as the question.",
        ),
        ("human", "Paper context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"),
    ]
)


def answer_question(question: str, question_llm) -> str:
    context = retrieve_text(question)
    if not context.strip():
        return "The provided context does not contain this information."
    chain = qa_prompt | question_llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# -------------------- API endpoints --------------------

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Research Finder API is running."}


@app.get("/logs")
async def get_logs():
    with log_lock:
        return {"logs": list(logs)}


@app.post("/research")
async def run_research(request: ResearchRequest):
    global result, selected_paper, chunks, vector_store, retriever
    add_log(f"Research started: {request.idea}")
    try:
        result = await graph.ainvoke(
            {
                "idea": request.idea,
                "api_key": request.api_key,
                "summary": "",
                "titles": [],
                "search_results": [],
                "papers": [],
                "final_results": [],
            }
        )
        selected_paper = None
        chunks = []
        vector_store = None
        retriever = None
        add_log("Research completed successfully.")
        return {
            "summary": result["summary"],
            "titles": result["titles"],
            "papers": result["final_results"],
        }
    except Exception as exc:
        add_log(f"Research failed: {exc}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Research failed: {exc}") from exc


@app.post("/paper")
async def get_paper(request: RankRequest):
    if result is None:
        raise HTTPException(status_code=400, detail="Run a research search first.")
    paper = next(
        (paper for paper in result["final_results"] if paper.get("Rank") == request.rank),
        None,
    )
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper rank not found.")
    return {
        "Rank": paper["Rank"],
        "title": paper.get("title", ""),
        "Paper_URL": paper.get("link", ""),
        "PDF_URL": paper.get("pdf_url", ""),
    }


@app.post("/paper/process")
async def process_selected_paper(request: RankRequest):
    global selected_paper, chunks, vector_store, retriever
    if result is None:
        raise HTTPException(status_code=400, detail="Run a research search first.")

    paper = next(
        (paper for paper in result["final_results"] if paper.get("Rank") == request.rank),
        None,
    )
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper rank not found.")

    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        raise HTTPException(status_code=400, detail="PDF URL is not available.")

    add_log(f"Downloading PDF for rank {request.rank}: {paper.get('title', '')}")
    try:
        response = await asyncio.to_thread(
            requests.get,
            pdf_url,
            timeout=60,
            headers={"User-Agent": "ResearchFinder/1.0"},
        )
        response.raise_for_status()
        total_pages, new_chunks = await asyncio.to_thread(extract_pdf_chunks, response.content)
        if not new_chunks:
            raise ValueError("No selectable text found in the PDF. It may be a scanned document.")

        new_vector_store = await asyncio.to_thread(FAISS.from_documents, new_chunks, embeddings)
        chunks = new_chunks
        vector_store = new_vector_store
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        selected_paper = paper
        add_log(f"PDF processed: {total_pages} pages, {len(chunks)} chunks.")
        return {
            "title": paper.get("title", ""),
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "message": "Paper processed successfully.",
        }
    except Exception as exc:
        add_log(f"PDF processing failed: {exc}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Failed to download or process the PDF: {exc}") from exc


@app.post("/ask")
async def ask_question_endpoint(request: QuestionRequest):
    if not chunks or retriever is None:
        raise HTTPException(status_code=400, detail="Process a paper PDF before asking questions.")
    try:
        question_llm = ChatGroq(model=MODEL_NAME, api_key=request.api_key, temperature=0)
        answer = await asyncio.to_thread(answer_question, request.question, question_llm)
        add_log(f"Answered question: {request.question}")
        return {"question": request.question, "answer": answer}
    except Exception as exc:
        add_log(f"Question answering failed: {exc}", "ERROR")
        raise HTTPException(status_code=500, detail=f"Failed to answer the question: {exc}") from exc


@app.on_event("startup")
async def startup_message():
    add_log("Research Finder API is ready.")



if __name__ == "__main__":
    import asyncio
    import threading
    import uvicorn

    config = uvicorn.Config(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )

    server = uvicorn.Server(config)

    def start_server():
        asyncio.run(server.serve())

    thread = threading.Thread(
        target=start_server,
        daemon=True,
    )
    thread.start()

    print("FastAPI server is starting on port 8000...")
