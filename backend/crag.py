
import asyncio
import os
import re
import threading
from datetime import datetime, timezone
from typing import TypedDict

import fitz
import requests
import torch

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END


# ==================================================
# CONFIGURATION
# ==================================================

MODEL_NAME = "openai/gpt-oss-20b"

RANKING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Configure this to a verified, compatible evaluator checkpoint.
CRAG_MODEL_ID = os.getenv(
    "CRAG_MODEL_ID",
    "Mindie/CRAG-Evaluator",
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOWER_TH = 0.30
UPPER_TH = 0.70
SENTENCE_THRESHOLD = 0.59

app = FastAPI(title="Research Finder API", version="2.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# All selected PDF data stays in memory.
selected_paper = None
current_pdf_bytes = None
chunks = []
vector_store = None
retriever = None

logs = []
log_lock = threading.Lock()


def add_log(message: str, level: str = "INFO"):
    item = {
        "time": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
    }

    with log_lock:
        logs.append(item)
        del logs[:-500]

    print(f"[{level}] {message}")


# ==================================================
# LOAD MODELS DIRECTLY FROM MODEL REPOSITORIES
# ==================================================

add_log(f"Loading CRAG evaluator from {CRAG_MODEL_ID}")

crag_tokenizer = AutoTokenizer.from_pretrained(CRAG_MODEL_ID)

crag_model = AutoModelForSequenceClassification.from_pretrained(
    CRAG_MODEL_ID
).to(DEVICE)

crag_model.eval()

add_log(f"CRAG evaluator loaded on {DEVICE}")

semantic_model = SentenceTransformer(RANKING_MODEL)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

add_log("Ranking and embedding models loaded")


# ==================================================
# REQUEST SCHEMAS
# ==================================================

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    api_key: str = Field(min_length=10)


class PaperProcessRequest(BaseModel):
    pdf_url: str = Field(min_length=5)


# ==================================================
# PDF PROCESSING
# ==================================================

def extract_pdf_chunks(pdf_bytes: bytes):
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

    try:
        page_documents = []

        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()

            if text:
                page_documents.append(
                    Document(
                        page_content=text,
                        metadata={"page": page_number},
                    )
                )

        page_count = len(pdf)

    finally:
        pdf.close()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )

    document_chunks = splitter.split_documents(page_documents)

    return page_count, document_chunks


# ==================================================
# CRAG EVALUATOR
# ==================================================

def crag_evaluate(question: str, document: str) -> float:
    """
    Assumes this checkpoint returns exactly one relevance logit.
    A different output shape requires a checkpoint-specific adapter.
    """

    text = f"Question: {question}\nDocument: {document}"

    inputs = crag_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = crag_model(**inputs)

    if output.logits.numel() != 1:
        raise ValueError(
            "CRAG evaluator must return one logit per question/document "
            "pair. Check that CRAG_MODEL_ID points to a compatible model."
        )

    logit = output.logits.reshape(-1)[0]
    score = float(torch.sigmoid(logit).item())

    return score


def evaluate_documents(question: str, docs: list[Document]):
    scores = []
    good_docs = []

    for doc in docs:
        score = crag_evaluate(question, doc.page_content)
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


# ==================================================
# RETRIEVAL
# ==================================================

def retrieve_documents(question: str):
    if retriever is None:
        return []

    return retriever.invoke(question)


def get_relevant_sentences(question: str, text: str):
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    relevant = []

    for sentence in sentences:
        sentence = sentence.strip()

        if not sentence:
            continue

        score = crag_evaluate(question, sentence)

        if score >= SENTENCE_THRESHOLD:
            relevant.append(sentence)

    return relevant


# ==================================================
# QUERY REWRITE AND WEB SEARCH
# ==================================================

rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Rewrite the user's question into a concise web search query. "
        "Preserve technical terms. Return only the search query."
    ),
    ("human", "{question}"),
])


def rewrite_query(question: str, api_key: str):
    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0,
    )

    chain = rewrite_prompt | llm | StrOutputParser()

    return chain.invoke({"question": question}).strip()


web_search_tool = DuckDuckGoSearchResults(
    num_results=5,
    output_format="list",
)


def search_web(query: str):
    try:
        results = web_search_tool.invoke(query)
    except Exception as exc:
        add_log(f"Web search failed: {exc}", "ERROR")
        return []

    if isinstance(results, str):
        return [
            Document(page_content=results)
        ] if results.strip() else []

    documents = []

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

        content = (
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content: {snippet}"
        ).strip()

        if content:
            documents.append(
                Document(
                    page_content=content,
                    metadata={"title": title, "url": url},
                )
            )

    return documents


# ==================================================
# LANGGRAPH QA WORKFLOW
# ==================================================

class QAState(TypedDict, total=False):
    question: str
    api_key: str
    web_query: str
    docs: list[Document]
    good_docs: list[Document]
    web_docs: list[Document]
    scores: list[float]
    verdict: str
    context: str
    answer: str


def rewrite_node(state: QAState):
    query = rewrite_query(
        state["question"],
        state["api_key"],
    )

    add_log(f"Search query: {query}")

    return {"web_query": query}


def retrieve_node(state: QAState):
    docs = retrieve_documents(state["question"])

    return {"docs": docs}


def evaluate_node(state: QAState):
    verdict, good_docs, scores = evaluate_documents(
        state["question"],
        state.get("docs", []),
    )

    add_log(f"CRAG verdict: {verdict}; scores={scores}")

    return {
        "verdict": verdict,
        "good_docs": good_docs,
        "scores": scores,
    }


def route_after_evaluation(state: QAState):
    if state.get("verdict") == "CORRECT":
        return "refine"

    return "web_search"


def web_search_node(state: QAState):
    query = state.get("web_query") or state["question"]
    return {"web_docs": search_web(query)}


def refine_node(state: QAState):
    verdict = state.get("verdict", "INCORRECT")

    if verdict == "CORRECT":
        source_docs = state.get("good_docs", [])

    elif verdict == "INCORRECT":
        source_docs = state.get("web_docs", [])

    else:
        source_docs = (
            state.get("good_docs", [])
            + state.get("web_docs", [])
        )

    combined_text = "\n\n".join(
        doc.page_content
        for doc in source_docs
        if doc.page_content.strip()
    )

    relevant_sentences = get_relevant_sentences(
        state["question"],
        combined_text,
    )

    context = " ".join(relevant_sentences)

    # Do not discard the evidence entirely if sentence filtering
    # returns no sentences.
    if not context.strip():
        context = combined_text

    return {"context": context}


answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a careful research assistant. Answer the original "
        "question using the provided evidence. Do not invent facts. "
        "If the evidence is insufficient, clearly say so."
    ),
    (
        "human",
        "Question:\n{question}\n\nEvidence:\n{context}"
    ),
])


def generate_answer_node(state: QAState):
    context = state.get("context", "").strip()

    if not context:
        return {
            "answer": (
                "I could not find enough relevant evidence in the selected "
                "paper or web search results to answer reliably."
            )
        }

    llm = ChatGroq(
        model=MODEL_NAME,
        api_key=state["api_key"],
        temperature=0,
    )

    chain = answer_prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "question": state["question"],
        "context": context,
    })

    return {"answer": answer}


qa_builder = StateGraph(QAState)

qa_builder.add_node("rewrite", rewrite_node)
qa_builder.add_node("retrieve", retrieve_node)
qa_builder.add_node("evaluate", evaluate_node)
qa_builder.add_node("web_search", web_search_node)
qa_builder.add_node("refine", refine_node)
qa_builder.add_node("generate", generate_answer_node)

qa_builder.add_edge(START, "rewrite")
qa_builder.add_edge("rewrite", "retrieve")
qa_builder.add_edge("retrieve", "evaluate")

qa_builder.add_conditional_edges(
    "evaluate",
    route_after_evaluation,
    {
        "refine": "refine",
        "web_search": "web_search",
    },
)

qa_builder.add_edge("web_search", "refine")
qa_builder.add_edge("refine", "generate")
qa_builder.add_edge("generate", END)

qa_graph = qa_builder.compile()


# ==================================================
# API ENDPOINTS
# ==================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "crag_model": CRAG_MODEL_ID,
    }


@app.get("/logs")
async def get_logs():
    with log_lock:
        return {"logs": list(logs)}


@app.post("/paper/process")
async def process_paper(request: PaperProcessRequest):
    global current_pdf_bytes, selected_paper
    global chunks, vector_store, retriever

    try:
        response = await asyncio.to_thread(
            requests.get,
            request.pdf_url,
            timeout=60,
            headers={"User-Agent": "ResearchFinder/2.1"},
        )

        response.raise_for_status()
        pdf_bytes = response.content

        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError("The URL did not return a valid PDF.")

        page_count, new_chunks = await asyncio.to_thread(
            extract_pdf_chunks,
            pdf_bytes,
        )

        if not new_chunks:
            raise ValueError(
                "No selectable text found in the PDF. "
                "It may be scanned or image-only."
            )

        new_vector_store = await asyncio.to_thread(
            FAISS.from_documents,
            new_chunks,
            embeddings,
        )

        # Replace old PDF only after new PDF processing succeeds.
        current_pdf_bytes = pdf_bytes
        chunks = new_chunks
        vector_store = new_vector_store
        retriever = vector_store.as_retriever(
            search_kwargs={"k": 5}
        )
        selected_paper = request.pdf_url

        add_log(
            f"PDF processed: {page_count} pages, "
            f"{len(chunks)} chunks. PDF is stored in RAM."
        )

        return {
            "message": "PDF processed successfully.",
            "total_pages": page_count,
            "total_chunks": len(chunks),
        }

    except Exception as exc:
        add_log(f"PDF processing failed: {exc}", "ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {exc}",
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
async def ask_question(request: QuestionRequest):
    if not chunks or retriever is None:
        raise HTTPException(
            status_code=400,
            detail="Process a paper before asking a question.",
        )

    try:
        output = await asyncio.to_thread(
            qa_graph.invoke,
            {
                "question": request.question,
                "api_key": request.api_key,
            },
        )

        return {
            "question": request.question,
            "rewritten_query": output.get("web_query", ""),
            "verdict": output.get("verdict", ""),
            "answer": output.get("answer", ""),
        }

    except Exception as exc:
        add_log(f"Question answering failed: {exc}", "ERROR")

        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {exc}",
        ) from exc


@app.post("/research")
async def reset_research():
    global selected_paper, current_pdf_bytes
    global chunks, vector_store, retriever

    selected_paper = None
    current_pdf_bytes = None
    chunks = []
    vector_store = None
    retriever = None

    add_log("Current paper and in-memory PDF data cleared.")

    return {"message": "Current paper cleared."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
    )
