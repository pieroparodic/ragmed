import asyncio
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from google.genai import types as genai_types

from rag_pubmed_v2 import SemanticReranker, answer_with_pubmed_rag_v2

# On Windows, uvicorn requires SelectorEventLoop instead of ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Load .env before reading any environment variables
load_dotenv()

# ──────────────────────────────────────────────────────────────
# GEMINI CONFIG
# ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# Shared state — holds the reranker and (optionally) Gemini client
state: dict = {}


# ──────────────────────────────────────────────────────────────
# GEMINI HELPERS  (synchronous — called via ThreadPoolExecutor)
# ──────────────────────────────────────────────────────────────
def _score_article_sync(client, question: str, title: str, abstract: str) -> dict:
    """Ask Gemini to score one article's relevance (1–5) and explain why."""
    prompt = (
        "You are a medical research expert evaluating whether a PubMed article "
        "is relevant to a clinical question.\n\n"
        f"Clinical question: {question}\n\n"
        f"Article title: {title}\n\n"
        f"Article abstract: {abstract[:1200]}\n\n"
        "Rate the relevance on a scale of 1 to 5:\n"
        "1 = Not relevant at all\n"
        "2 = Slightly relevant\n"
        "3 = Moderately relevant (related topic but indirect answer)\n"
        "4 = Highly relevant (directly addresses the question)\n"
        "5 = Perfectly relevant (exactly what was asked, strong evidence)\n\n"
        'Respond ONLY with valid JSON: {"score": <integer 1-5>, "reason": "<one sentence>"}'
    )
    safety_settings = [
        genai_types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE",
        ),
    ]
    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                safety_settings=safety_settings,
            ),
        )
        text = response.text.strip()
        match = re.search(r"\{.*?\}", text, re.DOTALL)
        if match:
            data = json.loads(match.group())
            score = max(1, min(5, int(data.get("score", 3))))
            return {"score": score, "reason": data.get("reason", "")}
    except Exception as e:
        print(f"[Gemini scoring error] {e}")
    return {"score": 3, "reason": "Could not evaluate."}


def _generate_answer_sync(client, question: str, articles: List[dict]) -> str:
    """Ask Gemini to synthesize a concise answer from the top retrieved articles."""
    context_parts = []
    for i, a in enumerate(articles, 1):
        context_parts.append(
            f"[{i}] {a.get('title', '')}\n"
            f"Journal: {a.get('journal', '')} ({a.get('year', '')})\n"
            f"Abstract: {a.get('abstract', '')[:900]}"
        )
    context = "\n\n".join(context_parts)

    prompt = (
        "You are a clinical research assistant. Based on the PubMed articles below, "
        "write a concise, evidence-based answer to the clinical question. "
        "Cite articles inline using their bracket numbers [1], [2], etc.\n\n"
        f"Clinical question: {question}\n\n"
        f"Retrieved articles:\n{context}\n\n"
        "Instructions:\n"
        "- Write 3–5 sentences\n"
        "- Cite sources inline with [1], [2], [3]\n"
        "- Be factual and based only on the provided abstracts\n"
        "- Do not speculate beyond what the evidence shows\n"
        "- Use plain clinical language\n\n"
        "Answer:"
    )

    # Relax safety settings — medical content often triggers Gemini's safety filters
    safety_settings = [
        genai_types.SafetySetting(
            category="HARM_CATEGORY_DANGEROUS_CONTENT",
            threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HARASSMENT",
            threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_HATE_SPEECH",
            threshold="BLOCK_NONE",
        ),
        genai_types.SafetySetting(
            category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
            threshold="BLOCK_NONE",
        ),
    ]

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                safety_settings=safety_settings,
            ),
        )

        # Try response.text first, then dig into candidates
        text = ""
        try:
            text = response.text.strip() if response.text else ""
        except Exception:
            pass

        if not text and response.candidates:
            try:
                parts = response.candidates[0].content.parts
                text = " ".join(p.text for p in parts if hasattr(p, "text")).strip()
            except Exception as inner:
                print(f"[Gemini generation] Could not extract from candidates: {inner}")

        if not text:
            # Log the full response for debugging
            print(f"[Gemini generation] Empty response.")
            if response.candidates:
                for i, c in enumerate(response.candidates):
                    print(f"  Candidate {i}: finish_reason={c.finish_reason}, safety_ratings={c.safety_ratings}")
            else:
                print(f"  No candidates returned. prompt_feedback={getattr(response, 'prompt_feedback', 'N/A')}")

        return text
    except Exception as e:
        print(f"[Gemini generation error] {type(e).__name__}: {e}")
    return ""


# ──────────────────────────────────────────────────────────────
# APP LIFESPAN
# ──────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading embedding model...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        state["reranker"] = await loop.run_in_executor(pool, SemanticReranker)
    print("Embedding model ready.")

    if GEMINI_API_KEY:
        from google import genai as google_genai
        state["gemini"] = google_genai.Client(api_key=GEMINI_API_KEY)
        print("Gemini client ready.")
    else:
        print("Warning: GEMINI_API_KEY not set — AI scoring and generation disabled.")

    print("Server is available.")
    yield
    state.clear()


app = FastAPI(title="PubMed RAG v2", version="3.0.0", lifespan=lifespan)


# ──────────────────────────────────────────────────────────────
# REQUEST MODEL
# ──────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    question: str
    domain: str = ""          # empty = no domain filter (all of PubMed)
    retmax: int = 50          # increased from 20 for better reranking pool
    top_k: int = 5
    filter_pub_types: bool = True   # restrict to RCTs, meta-analyses, systematic reviews
    generate: bool = True     # run Gemini scoring + answer generation


# ──────────────────────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_ready": "reranker" in state,
        "gemini_ready": "gemini" in state,
    }


@app.post("/query")
def query(req: QueryRequest):
    if "reranker" not in state:
        raise HTTPException(
            status_code=503,
            detail="Model is still loading. Please retry in a few seconds.",
        )

    # ── Step 1: Retrieve & re-rank ──────────────────────────────
    rag_output = answer_with_pubmed_rag_v2(
        user_question=req.question,
        domain=req.domain,
        pubmed_retmax=req.retmax,
        final_k=req.top_k,
        filter_pub_types=req.filter_pub_types,
        reranker=state["reranker"],
    )

    # ── Step 2: Gemini scoring + generation (if enabled) ────────
    if req.generate and "gemini" in state and rag_output["results"]:
        client = state["gemini"]

        # Score all articles in parallel
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [
                pool.submit(
                    _score_article_sync,
                    client,
                    req.question,
                    a.get("title", ""),
                    a.get("abstract", ""),
                )
                for a in rag_output["results"]
            ]
            scores = [f.result() for f in futures]

        for article, score_data in zip(rag_output["results"], scores):
            article["relevance_score"] = score_data["score"]
            article["relevance_reason"] = score_data["reason"]

        # Delay to avoid Gemini rate limits after parallel scoring
        time.sleep(4)

        # Generate synthesized answer
        print("[Gemini] Starting answer generation...")
        answer = _generate_answer_sync(
            client, req.question, rag_output["results"]
        )
        print(f"[Gemini] Generation result: {'OK' if answer else 'EMPTY'} ({len(answer)} chars)")
        rag_output["generated_answer"] = answer if answer else None
        rag_output["generation_model"] = GEMINI_MODEL
    else:
        rag_output["generated_answer"] = None
        rag_output["generation_model"] = None

    return rag_output
