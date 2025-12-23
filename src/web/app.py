import os
import uuid
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from langchain_community.retrievers import BM25Retriever

from src.pipelines.agentic import (
    resolve_doc,
    ALL_DOCS,
    get_llm,
    chat_wrap,
    write_memory,
    read_memory,
    delete_memory,
    ask_entrypoint,
    switch
)

from scripts.ingest import ingest_documents
from scripts.delete import delete_docs_from_chroma

# ──────────── Flask Setup ────────────
app = Flask(__name__, static_folder="static", static_url_path="/static")
UPLOAD_FOLDER      = "data"
ALLOWED_EXTENSIONS = {".pdf", ".docx"}  # keep as-is
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ─────────── Helpers (app-side only; no agent changes) ───────────
def _ensure_vs_persist():
    """Ensure agent.vs has a .persist() method (back-compat across langchain/chromadb versions)."""
    import src.pipelines.agentic as agent
    try:
        vs = agent.vs
    except Exception:
        return
    if hasattr(vs, "persist") and callable(getattr(vs, "persist")):
        return
    client = getattr(vs, "_client", None)
    if client and hasattr(client, "persist") and callable(client.persist):
        setattr(vs, "persist", client.persist)
    else:
        setattr(vs, "persist", lambda: None)


def _rebuild_retrievers_from_store():
    """Rebuild in-memory retrievers after deletes."""
    import src.pipelines.agentic as agent
    try:
        # Use the public Chroma API instead of the private _collection
        data = agent.vs.get(include=["documents", "metadatas"])
        docs  = data.get("documents") or []
        metas = data.get("metadatas") or []

        # Rebuild all_docs, filtering out blanks defensively
        all_docs = []
        for d, m in zip(docs, metas):
            if (d or "").strip():
                all_docs.append(agent.Document(page_content=d, metadata=m or {}))

        agent.all_docs = all_docs

        # Rebuild BM25 retriever with smaller k to avoid huge prompts
        if agent.all_docs:
            agent.bm25_retriever = BM25Retriever.from_documents(agent.all_docs, k=10)
        else:
            class _EmptyBM25:
                def invoke(self, *_a, **_kw):
                    return []
            agent.bm25_retriever = _EmptyBM25()

        # Rebuild vectorstore retriever with smaller k and same chunk filter
        agent.chunk_retriever = agent.vs.as_retriever(
            search_kwargs={"k": 10, "filter": {"chunk_type": "chunk"}}
        )

        print(f"[REBUILD] retrievers refreshed | chunks={len(agent.all_docs)}")
    except Exception as e:
        print(f"[REBUILD] warning: {e}")



# Apply once at startup to avoid AttributeError in agent ingest()
_ensure_vs_persist()


# ─── Serve your top-level Docs/ folder at /Docs/<filename> ───
@app.route("/Docs/<path:filename>")
def serve_docs(filename):
    docs_dir = os.path.join(app.root_path, "Docs")
    ext = Path(filename).suffix.lower()
    mime_types = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc":  "application/msword"
    }
    mime = mime_types.get(ext)
    return send_from_directory(docs_dir, filename, as_attachment=False, mimetype=mime)


# ─── Simple in-memory session tracking ───
chat_sessions = {"default": []}


# ─── HTML pages ───
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/chatbot.html")
def chatbot():
    return render_template("chatbot.html")

@app.route("/files.html")
def files():
    return render_template("files.html")

@app.route("/about")
def about():
    return render_template("about.html")


# ─── File upload & management ───
@app.route("/upload", methods=["POST"])
def upload_files():
    if "files" not in request.files:
        return jsonify({"status": "error", "message": "No files part"}), 400

    uploaded    = request.files.getlist("files")
    saved_paths = []
    for file in uploaded:
        if not file or not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)
            saved_paths.append(Path(filepath))

    if not saved_paths:
     return jsonify({"status": "ok", "added_chunks": 0})

    # Ingest into Chroma
    _ensure_vs_persist()
    added = ingest_documents(saved_paths)

    try:
        import src.pipelines.agentic as agent
        agent.vs.persist()
    except Exception as e:
        # Log full stack trace for debugging
        app.logger.exception("Chroma persist failed after ingestion")
        return jsonify({
            "status": "error",
            "message": "Persist failed after ingestion. Please try again."
        }), 500

    # Success path
    return jsonify({
        "status": "success",
        "added_chunks": added
    })


@app.route("/delete", methods=["POST"])
def delete_files():
    filenames = request.json.get("filenames", [])
    if not filenames:
        return jsonify({"status": "ok", "files": [], "deleted": 0})

    deleted = []
    paths   = []
    for fname in filenames:
        p = Path(os.path.join(UPLOAD_FOLDER, fname))
        if p.exists():
            try:
                os.remove(p)
                deleted.append(fname)
                paths.append(p)
            except Exception as e:
                print(f"[DELETE] file remove error {fname}: {e}")

    if paths:
        print(f"[DELETE] removing {len(paths)} from vector store")
        _ensure_vs_persist()
        delete_docs_from_chroma(paths)
        try:
            import src.pipelines.agentic as agent
            agent.vs.persist()
        except Exception as e:
            print(f"[DELETE] persist warning: {e}")
            # pass
        _rebuild_retrievers_from_store()

    return jsonify({"status": "deleted", "files": deleted, "deleted_count": len(deleted)})


@app.route("/files", methods=["GET"])
def list_files():
    out = []
    for doc in Path(UPLOAD_FOLDER).glob("*"):
        if doc.suffix.lower() in ALLOWED_EXTENSIONS:
            out.append({
                "name": doc.name,
                "type": doc.suffix[1:].upper(),
                "timestamp": doc.stat().st_mtime
            })
    return jsonify(out)


# ─── Chat endpoint ───
@app.route("/chat", methods=["POST"])
def chat():
    data        = request.json
    user_msg    = data.get("message", "").strip()
    chat_id     = data.get("chat_id", "default")
    web_search  = bool(data.get("web_search", False))

    if not user_msg:
        return jsonify({"response": "Please enter a message."})

    # toggle web search
    switch(web_search=1 if web_search else 0)

    # run the agent
    result  = ask_entrypoint(user_msg, chat_id=chat_id)
    print("Result: ", result)
    payload = result.get("final", result) or {}

    # pull structured fields directly (with graceful fallbacks)
    # final contract: answer + (sources, requirements, external_refs, subtopics, next_questions)
    sections_list = payload.get("response_sections") or []

    # quick index by title to recover if fields are only in response_sections
    idx = { (sec.get("title","") or "").strip().lower(): sec for sec in sections_list }

    def items_from(title):
        sec = idx.get(title.lower())
        return [i for i in (sec.get("items") or []) if str(i).strip()] if sec else []

    def text_from(title):
        sec = idx.get(title.lower())
        if not sec: return None
        if isinstance(sec.get("content"), str) and sec["content"].strip():
            return sec["content"].strip()
        return None

    answer          = payload.get("answer") or text_from("Answer") or ""
    sources         = payload.get("sources") or payload.get("used_sources") or items_from("Sources")
    requirements    = payload.get("requirements") or items_from("Requirements")
    external_refs   = payload.get("external_refs") or items_from("External references")
    subtopics       = payload.get("subtopics") or items_from("Subtopics")
    next_questions  = payload.get("next_questions") or items_from("Next questions")

    # minimal UX when nothing found
    if not any([sources, requirements, external_refs, subtopics, next_questions]) and not answer:
        answer = "Content related to your query was not found in the uploaded documents."
    raw_doc_names = payload.get("used_sources") or []
    # normalize sources to simple strings for the UI (file — § … — p. …)
    def norm_src(s):
        if isinstance(s, dict):
            doc = s.get("doc_name") or s.get("document") or s.get("name") or s.get("source") or ""
            sec = s.get("section") or ""
            pg  = s.get("page") or s.get("page_label") or s.get("page_index")
            bits = [doc]
            if sec: bits.append(f"§ {sec}")
            if pg is not None: bits.append(f"p. {pg}")
            return " — ".join([b for b in bits if str(b).strip()])
        return str(s)

    sources = [norm_src(s) for s in (sources or []) if str(s).strip()]

    # write memory (so /chat/history returns all fields for the UI)
    write_memory("user", user_msg, chat_id=chat_id)
    write_memory(
        "assistant",
        answer,
        doc_ids=sources,          # ← actual doc identifiers
        chat_id=chat_id,
        extra_meta={
            "sources":        sources,  # ← pretty strings for UI
            "requirements":   requirements or [],
            "external_refs":  external_refs or [],
            "subtopics":      subtopics or [],
            "next_questions": next_questions or []
        }
    )

    # return clean payload to the caller (UI may ignore it and reload history; still useful)
    return jsonify({
        "answer":         answer,
        "sources":        sources,
        "requirements":   requirements,
        "external_refs":  external_refs,
        "subtopics":      subtopics,
        "next_questions": next_questions
    })


# ─── Chat session management ───
@app.route("/chat/start", methods=["POST"])
def start_chat():
    session_id = str(uuid.uuid4())
    existing   = chat_sessions.setdefault("default", [])
    name       = f"Chat {len(existing)+1}"
    existing.append({"id": session_id, "name": name})
    return jsonify({"session_id": session_id, "session_name": name})

@app.route("/chat/sessions", methods=["POST"])
def get_sessions():
    return jsonify(chat_sessions.get("default", []))

@app.route("/chat/history", methods=["POST"])
def chat_history():
    chat_id = request.json.get("chat_id")
    if not chat_id:
        return jsonify([])
    history = read_memory(chat_id=chat_id, with_text=True)
    return jsonify(history)

@app.route("/chat/rename", methods=["POST"])
def rename_chat():
    cid      = request.json.get("chat_id")
    new_name = request.json.get("new_name")
    for sess in chat_sessions.get("default", []):
        if sess["id"] == cid:
            sess["name"] = new_name
            break
    return jsonify({"status": "renamed", "chat_id": cid, "new_name": new_name})

@app.route("/chat/delete", methods=["POST"])
def delete_chat():
    cid = request.json.get("chat_id")
    chat_sessions["default"] = [s for s in chat_sessions["default"] if s["id"] != cid]
    delete_memory(chat_id=cid)
    return jsonify({"status": "deleted", "chat_id": cid})


if __name__ == "__main__":
    app.run(debug=True)
