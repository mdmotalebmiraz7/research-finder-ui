
"""
Research Finder API with CRAG.

Run:
    pip install -r requirements.txt
    uvicorn research_finder:app --reload --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health
    GET  /logs
    POST /research
    POST /paper
    POST /paper/process
    GET  /paper/pdf
    POST /ask
    GET  /docs
"""

import asyncio
import logging
import os
import re
import subprocess
import threading

from datetime import datetime, timezone
from typing import Annotated, TypedDict

import arxiv
import fitz
import gdown
import requests
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from pydantic import BaseModel, Field
from groq import Groq

from transformers import T5Tokenizer, T5ForSequenceClassification

from sentence_transformers import SentenceTransformer, util

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import START, END, StateGraph
from langgraph.types import Send


# =========================================================
# 1. CONFIGURATION
# =========================================================

MODEL_NAME = "openai/gpt-oss-20b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

YEAR_FROM = 2020
PAPERS_PER_TOPIC = 100
TITLE_TOP_K = 20
FINAL_TOP_K = 10

UPPER_TH = 0.7
LOWER_TH = 0.3
SENTENCE_THRESHOLD = 0.59

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# On Colab, use Google Drive.
# On a local computer, use a persistent local directory.
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

if IS_COLAB:
    BASE_DIR = "/content/drive/MyDrive/CRAG"
else:
    BASE_DIR = os.path.abspath("./CRAG_models")

CRAG_DIR = os.path.join(BASE_DIR, "CRAG")
EVALUATOR_DIR = os.path.join(BASE_DIR, "evaluator")

CRAG_REPOSITORY_URL = (
    "https://github.com/HuskyInSalt/CRAG.git"
)

EVALUATOR_DOWNLOAD_URL = (
    "https://drive.google.com/drive/folders/"
    "1CRFGsyNguXJwKSvFvJm_82GOOlkWSkW7?usp=drive_link"
)


# =========================================================
# 2. FASTAPI
# =========================================================

app = FastAPI(
    title="Research Finder API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# In-memory application state.
# Restarting the API clears this data.

result = None
selected_paper = None

chunks: list[Document] = []
vector_store = None
retriever = None

current_pdf_bytes: bytes | None = None

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


# =========================================================
# 3. DOWNLOAD AND LOAD CRAG MODEL AUTOMATICALLY
# =========================================================

def find_model_directory(base_dir: str) -> str | None:
    """Find the folder containing Hugging Face config.json."""

    if not os.path.isdir(base_dir):
        return None

    for root, _, files in os.walk(base_dir):
        if "config.json" in files:
            return root

    return None


def setup_crag_model():
    """
    Prepare the CRAG repository and evaluator.

    Existing files are reused. The evaluator is loaded once
    when this Python module starts.
    """

    if IS_COLAB:
        from google.colab import drive

        if not os.path.isdir("/content/drive/MyDrive"):
            drive.mount("/content/drive")

    os.makedirs(BASE_DIR, exist_ok=True)

    # Download repository only if missing.
    if not os.path.isdir(CRAG_DIR):
        add_log("Downloading CRAG repository...")

        subprocess.run(
            [
                "git",
                "clone",
                CRAG_REPOSITORY_URL,
                CRAG_DIR,
            ],
            check=True,
        )
    else:
        add_log("CRAG repository already exists.")

    # Check for an actual model checkpoint, not just a folder.
    model_dir = find_model_directory(EVALUATOR_DIR)

    if model_dir is None:
        add_log("Downloading CRAG evaluator...")

        os.makedirs(EVALUATOR_DIR, exist_ok=True)

        downloaded = gdown.download_folder(
            EVALUATOR_DOWNLOAD_URL,
            output=EVALUATOR_DIR,
            quiet=False,
        )

        model_dir = find_model_directory(EVALUATOR_DIR)

        if model_dir is None:
            raise RuntimeError(
                "CRAG evaluator download did not produce a "
                "config.json. Check the Google Drive folder "
                "permissions and the downloaded folder structure. "
                f"gdown result: {downloaded}"
            )
    else:
        add_log("CRAG evaluator already exists.")

    add_log(f"Loading CRAG model from: {model_dir}")

    tokenizer = T5Tokenizer.from_pretrained(model_dir)

    model = T5ForSequenceClassification.from_pretrained(
        model_dir
    )

    model.to(DEVICE)
    model.eval()

    add_log(f"CRAG model loaded on {DEVICE}.")

    return tokenizer, model


# Download if required and load once at startup/import.
tokenizer, crag_model = setup_crag_model()


# =========================================================
# 4. LOAD RESEARCH RANKING MODEL
# =========================================================

add_log("Loading semantic ranking model...")

semantic_model = SentenceTransformer(EMBEDDING_MODEL)

add_log("Semantic ranking model loaded.")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)


# =========================================================
# 5. API SCHEMAS
# =========================================================

class ResearchRequest(BaseModel):
    idea: str = Field(min_length=3, max_length=2000)
    api_key: str = Field(min_length=10)


class RankRequest(BaseModel):
    rank: int = Field(ge=1, le=10)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    api_key: str = Field(min_length=10)


class ResearchPlan(BaseModel):
    summary: str
    titles: list[str] = Field(min_length=10, max_length=10)


class ResearchState(TypedDict):
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


# =========================================================
# 6. RESEARCH PAPER SEARCH
# =========================================================

def reconstruct_abstract(data) -> str:
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
    search_limit: int = PAPERS_PER_TOPIC,
    year_from: int = YEAR_FROM,
) -> list[dict]:

    if not query.strip():
        return []

    add_log(f"Searching OpenAlex: {query}")

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
                f"OpenAlex search failed: {exc}",
                "ERROR",
            )
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

            paper_url = (
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
                "url": paper_url,
                "link": paper_url,
                "openalex_url": item.get("id", ""),
                "pdf_url": pdf_url,
            })

            if len(papers) >= search_limit:
                break

        cursor = data.get("meta", {}).get("next_cursor")

        if not cursor:
            break

    add_log(f"Found {len(papers)} papers for: {query}")

    return papers


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

        indices = scores.argsort(
            descending=True
        )[:top_k]

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

        indices = scores.argsort(
            descending=True
        )[:top_k]

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
                "pdf_url": paper.get("pdf_url", ""),
                "abstract": paper.get("abstract", ""),
                "year": paper.get("year"),
                "doi": paper.get("doi", ""),
                "Semantic": float(scores[i]),
            })

        return ranked

    return await asyncio.to_thread(rank)


# =========================================================
# 7. RESEARCH PLANNING WITH GROQ
# =========================================================

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
        raise ValueError(
            f"Expected 10 research topics, got {len(plan.titles)}"
        )

    return plan


def generate_research_node(state: ResearchState) -> dict:
    plan = generate_research_plan(
        state["idea"],
        state["api_key"],
    )

    return {
        "summary": plan.summary,
        "titles": plan.titles,
    }


def dispatch_search(state: ResearchState):
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


async def process_title(state: WorkerState) -> dict:
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


def collect_unique_papers(state: ResearchState) -> dict:
    unique = {}

    for topic_results in state["search_results"]:
        for paper in topic_results:
            url = paper.get("url") or paper.get("link") or ""

            if url and url not in unique:
                unique[url] = paper.copy()

    papers = list(unique.values())

    add_log(f"Collected {len(papers)} unique papers.")

    return {"papers": papers}


async def process_abstract(state: ResearchState) -> dict:
    final_results = await semantic_abstract_rank(
        state["papers"],
        state["summary"],
        FINAL_TOP_K,
    )

    add_log(
        f"Final research ranking complete: {len(final_results)} papers."
    )

    return {"final_results": final_results}


research_builder = StateGraph(ResearchState)

research_builder.add_node(
    "generate",
    generate_research_node,
)

research_builder.add_node(
    "process_title",
    process_title,
)

research_builder.add_node(
    "collect_unique_papers",
    collect_unique_papers,
)

research_builder.add_node(
    "process_abstract",
    process_abstract,
)

research_builder.add_edge(START, "generate")

research_builder.add_conditional_edges(
    "generate",
    dispatch_search,
    ["process_title"],
)

research_builder.add_edge(
    "process_title",
    "collect_unique_papers",
)

research_builder.add_edge(
    "collect_unique_papers",
    "process_abstract",
)

research_builder.add_edge(
    "process_abstract",
    END,
)

research_graph = research_builder.compile()


# =========================================================
# 8. PDF EXTRACTION AND RETRIEVAL
# =========================================================

def extract_pdf_chunks(
    pdf_bytes: bytes,
) -> tuple[int, list[Document]]:

    pdf = fitz.open(
        stream=pdf_bytes,
        filetype="pdf",
    )

    try:
        total_pages = len(pdf)

        page_docs = []

        for index, page in enumerate(pdf):
            page_text = page.get_text().strip()

            if page_text:
                page_docs.append(
                    Document(
                        page_content=page_text,
                        metadata={"page": index + 1},
                    )
                )
    finally:
        pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    return (
        total_pages,
        splitter.split_documents(page_docs),
    )


def retrieve_documents(question: str) -> list[Document]:
    if retriever is None:
        return []

    return retriever.invoke(question)


# =========================================================
# 9. CRAG EVALUATOR
# =========================================================

def crag_evaluate(
    question: str,
    document: str,
) -> float:

    text = f"{question} [SEP] {document}"

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = crag_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )

    logits = outputs.logits.squeeze()

    if logits.numel() != 1:
        raise ValueError(
            "CRAG evaluator returned multiple logits. "
            "Check whether the checkpoint expects a different "
            "classification or relevance-scoring method."
        )

    return float(logits.item())


def evaluate_documents(
    question: str,
    docs: list[Document],
) -> tuple[str, list[Document], list[float]]:

    scores = []
    good_docs = []

    for doc in docs:
        score = crag_evaluate(
            question,
            doc.page_content,
        )

        scores.append(score)

        if score > LOWER_TH:
            good_docs.append(doc)

    if scores and any(score > UPPER_TH for score in scores):
        verdict = "CORRECT"

    elif not scores or all(score < LOWER_TH for score in scores):
        verdict = "INCORRECT"
        good_docs = []

    else:
        verdict = "AMBIGUOUS"

    return verdict, good_docs, scores


# =========================================================
# 10. ALWAYS REWRITE THE QUESTION FIRST
# =========================================================

class WebQuery(BaseModel):
    query: str


rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You rewrite detailed questions into concise search queries.

Rules:
- Always rewrite the question.
- Preserve its central intent and important technical terms.
- For a long multi-part question, focus on the main information need.
- Aim for 6–14 words when possible.
- Do not answer the question.
- Return the query using the required schema.
""",
    ),
    ("human", "Question: {question}"),
])

# The actual LLM is created per request, using that request's API key.
def create_rewrite_chain(api_key: str):
    rewrite_llm = ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0,
    )

    return rewrite_prompt | rewrite_llm.with_structured_output(WebQuery)


def rewrite_query(question: str, api_key: str) -> str:
    chain = create_rewrite_chain(api_key)

    output = chain.invoke({
        "question": question,
    })

    query = output.query.strip()

    return query or question


# =========================================================
# 11. DUCKDUCKGO WEB SEARCH — NO TAVILY KEY
# =========================================================

duckduckgo_search = DuckDuckGoSearchResults(
    num_results=5,
    output_format="list",
)


def search_web(query: str) -> list[Document]:
    try:
        results = duckduckgo_search.invoke(query)
    except Exception as exc:
        add_log(f"DuckDuckGo search failed: {exc}", "ERROR")
        return []

    web_docs = []

    if isinstance(results, str):
        if results.strip():
            web_docs.append(
                Document(
                    page_content=results,
                    metadata={"source": "DuckDuckGo"},
                )
            )

        return web_docs

    for item in results or []:
        if not isinstance(item, dict):
            continue

        title = item.get("title", "")
        url = item.get("link") or item.get("url", "")
        snippet = (
            item.get("snippet")
            or item.get("body")
            or item.get("content")
            or ""
        )

        if not (title or url or snippet):
            continue

        content = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{snippet}"
        )

        web_docs.append(
            Document(
                page_content=content,
                metadata={
                    "title": title,
                    "url": url,
                    "source": "DuckDuckGo",
                },
            )
        )

    return web_docs


# =========================================================
# 12. SENTENCE-LEVEL RELEVANCE FILTER
# =========================================================

def get_relevant_text(
    question: str,
    text: str,
) -> list[str]:

    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text,
    )

    relevant_sentences = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        try:
            score = crag_evaluate(question, sentence)
        except Exception as exc:
            add_log(
                f"Sentence relevance evaluation failed: {exc}",
                "ERROR",
            )
            continue

        if score >= SENTENCE_THRESHOLD:
            relevant_sentences.append(sentence)

    return relevant_sentences


# =========================================================
# 13. CRAG QUESTION-ANSWERING GRAPH
# =========================================================

class QAState(TypedDict, total=False):
    question: str
    api_key: str

    web_query: str

    docs: list[Document]
    good_docs: list[Document]
    web_docs: list[Document]

    scores: list[float]
    verdict: str
    reason: str

    strips: list[str]
    refined_context: str
    answer: str


def qa_rewrite_node(state: QAState) -> dict:
    query = rewrite_query(
        state["question"],
        state["api_key"],
    )

    add_log(f"Rewritten question: {query}")

    return {"web_query": query}


def qa_retrieve_node(state: QAState) -> dict:
    docs = retrieve_documents(
        state["question"]
    )

    return {"docs": docs}


def qa_evaluate_node(state: QAState) -> dict:
    verdict, good_docs, scores = evaluate_documents(
        state["question"],
        state.get("docs", []),
    )

    add_log(
        f"CRAG verdict: {verdict}; scores={scores}"
    )

    return {
        "verdict": verdict,
        "good_docs": good_docs,
        "scores": scores,
    }


def qa_route_after_eval(state: QAState) -> str:
    if state["verdict"] == "CORRECT":
        return "refine"

    return "web_search"


def qa_web_search_node(state: QAState) -> dict:
    query = (
        state.get("web_query")
        or state["question"]
    )

    web_docs = search_web(query)

    return {"web_docs": web_docs}


def qa_refine_node(state: QAState) -> dict:
    verdict = state["verdict"]

    local_docs = state.get("good_docs", [])
    web_docs = state.get("web_docs", [])

    if verdict == "CORRECT":
        docs_to_use = local_docs

    elif verdict == "INCORRECT":
        docs_to_use = web_docs

    else:
        docs_to_use = local_docs + web_docs

    context = "\n\n".join(
        doc.page_content
        for doc in docs_to_use
        if doc.page_content.strip()
    )

    relevant_sentences = get_relevant_text(
        state["question"],
        context,
    )

    refined_context = " ".join(relevant_sentences)

    # Avoid returning an empty context just because the sentence
    # evaluator rejected every sentence. Keep source evidence as
    # a fallback; the answer prompt still enforces evidence-based answers.
    if not refined_context.strip() and context.strip():
        refined_context = context

    return {
        "strips": relevant_sentences,
        "refined_context": refined_context,
    }


qa_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a careful academic research assistant.

Answer the user's ORIGINAL question using only the supplied context.
Use clear explanations and preserve relevant technical details.
If the context does not support an answer, say so honestly.
If only part of the question can be answered, explain what is missing.
Do not invent facts or claim that web sources were independently verified.
Answer in the same language as the user's question.
""",
    ),
    (
        "human",
        "Original question:\n{question}\n\n"
        "Evidence context:\n{context}",
    ),
])


def qa_generate_node(state: QAState) -> dict:
    context = state.get("refined_context", "").strip()

    if not context:
        return {
            "answer": (
                "I couldn't find enough relevant information "
                "in the selected paper or web search results "
                "to answer this question reliably."
            )
        }

    question_llm = ChatGroq(
        model=MODEL_NAME,
        api_key=state["api_key"],
        temperature=0,
    )

    chain = (
        qa_answer_prompt
        | question_llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        # The full original question is preserved.
        "question": state["question"],
        "context": context,
    })

    return {"answer": answer}


qa_builder = StateGraph(QAState)

qa_builder.add_node(
    "rewrite_query",
    qa_rewrite_node,
)

qa_builder.add_node(
    "retrieve",
    qa_retrieve_node,
)

qa_builder.add_node(
    "evaluate",
    qa_evaluate_node,
)

qa_builder.add_node(
    "web_search",
    qa_web_search_node,
)

qa_builder.add_node(
    "refine",
    qa_refine_node,
)

qa_builder.add_node(
    "generate",
    qa_generate_node,
)

# Rewrite happens first for every question.
qa_builder.add_edge(START, "rewrite_query")
qa_builder.add_edge("rewrite_query", "retrieve")
qa_builder.add_edge("retrieve", "evaluate")

qa_builder.add_conditional_edges(
    "evaluate",
    qa_route_after_eval,
    {
        "refine": "refine",
        "web_search": "web_search",
    },
)

qa_builder.add_edge("web_search", "refine")
qa_builder.add_edge("refine", "generate")
qa_builder.add_edge("generate", END)

qa_graph = qa_builder.compile()


# =========================================================
# 14. API ENDPOINTS
# =========================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "Research Finder API is running.",
        "device": str(DEVICE),
    }


@app.get("/logs")
async def get_logs():
    with log_lock:
        return {"logs": list(logs)}


@app.post("/research")
async def run_research(request: ResearchRequest):
    global result
    global selected_paper
    global chunks
    global vector_store
    global retriever
    global current_pdf_bytes

    add_log(f"Research started: {request.idea}")

    try:
        new_result = await research_graph.ainvoke({
            "idea": request.idea,
            "api_key": request.api_key,
            "summary": "",
            "titles": [],
            "search_results": [],
            "papers": [],
            "final_results": [],
        })

        result = new_result

        # A new research session clears the selected PDF.
        selected_paper = None
        chunks = []
        vector_store = None
        retriever = None
        current_pdf_bytes = None

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
            paper
            for paper in result["final_results"]
            if paper.get("Rank") == request.rank
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
    global selected_paper
    global chunks
    global vector_store
    global retriever
    global current_pdf_bytes

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="Run a research search first.",
        )

    paper = next(
        (
            paper
            for paper in result["final_results"]
            if paper.get("Rank") == request.rank
        ),
        None,
    )

    if paper is None:
        raise HTTPException(
            status_code=404,
            detail="Paper rank not found.",
        )

    pdf_url = paper.get("pdf_url", "")

    if not pdf_url:
        raise HTTPException(
            status_code=400,
            detail="PDF URL is not available.",
        )

    add_log(
        f"Downloading PDF for rank {request.rank}: "
        f"{paper.get('title', '')}"
    )

    try:
        response = await asyncio.to_thread(
            requests.get,
            pdf_url,
            timeout=60,
            headers={
                "User-Agent": "ResearchFinder/1.0",
            },
        )

        response.raise_for_status()

        pdf_bytes = response.content

        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError(
                "The returned URL did not provide a valid PDF."
            )

        total_pages, new_chunks = await asyncio.to_thread(
            extract_pdf_chunks,
            pdf_bytes,
        )

        if not new_chunks:
            raise ValueError(
                "No selectable text found. "
                "This PDF may be scanned or image-only."
            )

        new_vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            new_chunks,
            embeddings,
        )

        # Replace the previous PDF only after processing succeeds.
        current_pdf_bytes = pdf_bytes
        chunks = new_chunks
        vector_store = new_vector_store

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5},
        )

        selected_paper = paper

        add_log(
            f"PDF processed: {total_pages} pages, "
            f"{len(chunks)} chunks. PDF stored in RAM."
        )

        return {
            "title": paper.get("title", ""),
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "message": "Paper processed successfully.",
        }

    except Exception as exc:
        add_log(
            f"PDF processing failed: {exc}",
            "ERROR",
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to download or process PDF: {exc}",
        ) from exc


@app.get("/paper/pdf")
async def get_current_pdf():
    if current_pdf_bytes is None:
        raise HTTPException(
            status_code=404,
            detail="No PDF has been processed.",
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
    if not chunks or retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Process a paper PDF before asking questions.",
        )

    try:
        # The compiled CRAG graph rewrites the question first,
        # evaluates local documents, searches the web if needed,
        # filters context and generates the final answer.
        output = await asyncio.to_thread(
            qa_graph.invoke,
            {
                "question": request.question,
                "api_key": request.api_key,
                "docs": [],
                "good_docs": [],
                "web_docs": [],
                "scores": [],
                "verdict": "",
                "web_query": "",
                "strips": [],
                "refined_context": "",
                "answer": "",
            },
        )

        add_log(
            f"Answered question using CRAG: {request.question}"
        )

        return {
            "question": request.question,
            "rewritten_query": output.get("web_query", ""),
            "verdict": output.get("verdict", ""),
            "answer": output.get("answer", ""),
        }

    except Exception as exc:
        add_log(
            f"Question answering failed: {exc}",
            "ERROR",
        )

        raise HTTPException(
            status_code=500,
            detail=f"Failed to answer the question: {exc}",
        ) from exc


@app.on_event("startup")
async def startup_message():
    add_log("Research Finder API is ready.")


# =========================================================
# 15. RUN DIRECTLY
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
