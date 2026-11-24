# FreightRates RAG Assistant - Flask backend API
import io
import json
import logging
import os
import threading
import time

import pandas as pd
from flask import Flask, jsonify, request

import rag_core as rc

# ---------- configuration (environment variables override the defaults) ----------
INDEX_DIR = os.getenv("INDEX_DIR", "rag_index")
with open(os.path.join(INDEX_DIR, "rag_config.json")) as f:
    CONFIG = json.load(f)          # tuned settings + model names saved by the notebook

EMBED_MODEL = os.getenv("EMBED_MODEL", CONFIG["embed_model"])
LLM_REPO = os.getenv("LLM_REPO", CONFIG["llm_repo"])
LLM_FILE = os.getenv("LLM_FILE", CONFIG["llm_file"])
N_CTX = int(os.getenv("N_CTX", "4096"))
N_THREADS = int(os.getenv("N_THREADS", "0")) or None      # 0 = let llama.cpp decide
MAX_QUESTION_CHARS = 500

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rag_api")

if EMBED_MODEL != CONFIG["embed_model"]:
    # Query and document vectors must come from the same model, or retrieval silently breaks
    raise RuntimeError(f"EMBED_MODEL={EMBED_MODEL} but the index was built with {CONFIG['embed_model']}; rebuild the index.")

# ---------- app + models ----------
rag_api = Flask("FreightRates RAG Assistant")
rag_api.json.sort_keys = False
index = rc.RagIndex.load(INDEX_DIR)
_, embed_query = rc.load_embedder(EMBED_MODEL, device="cpu")
chat = rc.load_llm(LLM_REPO, LLM_FILE, n_ctx=N_CTX, n_gpu_layers=0, n_threads=N_THREADS)
llm_lock = threading.Lock()        # llama.cpp handles one generation at a time
logger.info("Loaded %d documents; embedder %s; LLM %s/%s", len(index.docs), EMBED_MODEL, LLM_REPO, LLM_FILE)


def _int(value, default, lo, hi, name):
    if value is None:
        return default
    try:
        v = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be an integer")
    if not lo <= v <= hi:
        raise ValueError(f"'{name}' must be between {lo} and {hi}")
    return v


def _question(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("'question' is required and must be a non-empty string")
    q = value.strip()
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"'question' is limited to {MAX_QUESTION_CHARS} characters")
    return q


def ask(question: str, k: int, max_tokens: int, include_context: bool = False) -> dict:
    t0 = time.time()
    with llm_lock:
        result = rc.answer(question, index, embed_query, chat, k=k,
                           temperature=CONFIG["temperature"], max_tokens=max_tokens)
    out = {"question": question, "answer": result["answer"], "sources": result["sources"],
           "understood": result["understood"], "seconds": round(time.time() - t0, 2)}
    if include_context:
        out["context"] = result["context"]
    return out


@rag_api.get("/")
def home():
    """Health check / welcome route."""
    return jsonify({"message": "Welcome to the FreightRates RAG Assistant! POST /v1/ask, /v1/askbatch or /v1/search.",
                    "documents": len(index.docs), "embed_model": EMBED_MODEL, "llm": f"{LLM_REPO}/{LLM_FILE}",
                    "k": CONFIG["k"], "temperature": CONFIG["temperature"], "data_through": CONFIG.get("data_through"),
                    "data_source": CONFIG.get("data_source")})


@rag_api.post("/v1/ask")
def ask_one():
    """Online: {"question": "...", "k": optional, "max_tokens": optional, "include_context": optional}."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        q = _question(body.get("question"))
        k = _int(body.get("k"), CONFIG["k"], 1, 8, "k")
        mt = _int(body.get("max_tokens"), CONFIG["max_tokens"], 32, 800, "max_tokens")
        return jsonify(ask(q, k, mt, bool(body.get("include_context", False))))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.exception("Answer failed")
        return jsonify({"error": f"Answer failed: {e}"}), 500


@rag_api.post("/v1/askbatch")
def ask_batch():
    """Batch: CSV file (form field 'file') with a 'question' column -> {row_index: {answer, sources}}."""
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file uploaded under the key 'file'"}), 400
        batch = pd.read_csv(io.BytesIO(request.files["file"].read()))
        if "question" not in batch.columns:
            return jsonify({"error": "The CSV needs a 'question' column"}), 400
        if len(batch) > 50:
            return jsonify({"error": "Batches are limited to 50 questions"}), 400
        results = {}
        for i, q in enumerate(batch["question"]):
            try:
                r = ask(_question(q), CONFIG["k"], CONFIG["max_tokens"])
                results[str(i)] = {"answer": r["answer"], "sources": [s["id"] for s in r["sources"]]}
            except ValueError as e:
                results[str(i)] = {"error": str(e)}
        return jsonify(results)
    except Exception as e:
        logger.exception("Batch failed")
        return jsonify({"error": f"Batch failed: {e}"}), 500


@rag_api.post("/v1/search")
def search():
    """Retrieval only (no LLM, fast): {"question": "...", "k": optional} -> documents with scores and text."""
    try:
        body = request.get_json(force=True, silent=True) or {}
        q = _question(body.get("question"))
        k = _int(body.get("k"), CONFIG["k"], 1, 20, "k")
        hits, info = index.search(q, embed_query(q), k=k)
        return jsonify({"question": q, "understood": info.to_dict(), "results": hits})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    rag_api.run(host="0.0.0.0", port=7870, debug=False)
