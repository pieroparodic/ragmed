import asyncio
import sys
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_pubmed_v2 import SemanticReranker, answer_with_pubmed_rag_v2

# On Windows, uvicorn requires SelectorEventLoop instead of ProactorEventLoop
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Shared state — holds the reranker instance once loaded
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the embedding model once at startup (runs in a thread to avoid blocking the event loop)
    print("Loading embedding model...")
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        state["reranker"] = await loop.run_in_executor(pool, SemanticReranker)
    print("Model ready. Server is available.")
    yield
    state.clear()


app = FastAPI(title="PubMed RAG v2", version="2.0.0", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str
    domain: str = ""        # empty string = no domain filter (all of PubMed)
    retmax: int = 20
    top_k: int = 5


@app.get("/", response_class=HTMLResponse)
def root():
    with open("index.html", encoding="utf-8") as f:
        return f.read()


@app.get("/health")
def health():
    ready = "reranker" in state
    return {"status": "ok", "model_ready": ready}


@app.post("/query")
def query(req: QueryRequest):
    if "reranker" not in state:
        raise HTTPException(status_code=503, detail="Model is still loading. Please retry in a few seconds.")
    return answer_with_pubmed_rag_v2(
        user_question=req.question,
        domain=req.domain,
        pubmed_retmax=req.retmax,
        final_k=req.top_k,
        reranker=state["reranker"],
    )
