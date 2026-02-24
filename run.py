"""
Start the PubMed RAG v2 server.
Usage: python run.py
"""
import sys
import asyncio

# Windows requires SelectorEventLoop for uvicorn to bind correctly
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
