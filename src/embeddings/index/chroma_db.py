from pathlib import Path
from typing import List, Dict, Tuple, Any

import numpy as np
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from chromadb import Settings

from src.chunking.chunker import (
    _collect_doc_text_pieces,
    parse_docx_to_chunks,
    parse_pdf_to_chunks,
    to_documents_and_ids,
)
from src.data_ingest.preprocess import build_doc_summary_chunk_gpt, now_utc_iso
from src.configs.helper import STORE_DIR, DATA_DIR, DEVICE, dbg

np.float_ = np.float64  # compatibility for some numpy users

client_settings = Settings(
    is_persistent=True,
    persist_directory=str(STORE_DIR),
    anonymized_telemetry=False,  # quiets noisy telemetry warnings
)

def setup_vectorstores(
    store_dir: str = STORE_DIR,
    data_dir: Path = DATA_DIR,
    device: str = DEVICE,
    client_settings: Settings = client_settings,
) -> Tuple[
    Chroma,             # vs
    Chroma,             # memory_vs
    Any,                # doc_retriever
    List[Document],     # all_docs
    Any,                # bm25_retriever
    Any,                # chunk_retriever
]:
    """
    • Build/load three Chroma collections: docs_and_headings, chat_memory, and doc_index
    • Ingest PDFs/DOCX in data_dir once if docs_and_headings is empty
    • Return handles and retrievers:
        vs, memory_vs, doc_retriever, all_docs, bm25_retriever, chunk_retriever
    """
    # ── Globals ────────────────────────────────────────────────────────────────
    global vs, memory_vs, all_docs, bm25_retriever, chunk_retriever, doc_retriever, _EMB_INSTANCE, doc_vs
    #                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    #                          NEW: expose doc_vs so delete.py can access it

    # ── 0) ensure dirs exist ───────────────────────────────────────────────────
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    store_dir = Path(store_dir)
    print(f"Building store dir: {store_dir}")
    store_dir.mkdir(parents=True, exist_ok=True)
    print("Store dir made")

    # ── 1) embeddings (cached singleton) ──────────────────────────────────────
    try:
        emb = _EMB_INSTANCE
    except NameError:
        _EMB_INSTANCE = HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        emb = _EMB_INSTANCE

    # ── 2) open/create collections (persist in store_dir) ─────────────────────
    client_settings.persist_directory = str(store_dir)
    vs = Chroma(
        collection_name="docs_and_headings",
        embedding_function=emb,
        client_settings=client_settings,
    )
    memory_vs = Chroma(
        collection_name="chat_memory",
        embedding_function=emb,
        client_settings=client_settings,
    )

    # ── 3) first-time ingestion into docs_and_headings ────────────────────────
    try:
        cur_count = int(vs._collection.count())
    except Exception:
        cur_count = 0

    if cur_count == 0:
        print("⏳ Building Chroma from data_dir …")
        new_chunks: List[dict] = []

        for fp in sorted(data_dir.iterdir()):
            if fp.suffix.lower() not in (".pdf", ".docx"):
                dbg(f"[INGEST] skipping unsupported {fp.name} (ext={fp.suffix.lower()})")
                continue

            print(f"  • file: {fp}")
            try:
                if fp.suffix.lower() == ".pdf":
                    parsed = parse_pdf_to_chunks(str(fp))
                else:
                    parsed = parse_docx_to_chunks(str(fp))
            except Exception as e:
                dbg(f"[INGEST] failed on {fp.name}: {e}")
                continue

            if not parsed:
                dbg(f"[INGEST] no chunks parsed from {fp.name}")
                continue

            new_chunks.extend(parsed)

        # Group chunks per (doc_name, file_hash) to build summaries
        by_doc: Dict[Tuple[str, str], List[dict]] = {}
        for ch in new_chunks:
            key = (ch.get("doc_name", ""), ch.get("file_hash", ""))
            by_doc.setdefault(key, []).append(ch)

        doc_sum_chunks: List[dict] = []
        for (doc_name, file_hash), items in by_doc.items():
            ingested_at = items[0].get("ingested_at") or now_utc_iso()
            pieces = _collect_doc_text_pieces(items, doc_name)
            sum_chunk = build_doc_summary_chunk_gpt(doc_name, file_hash, ingested_at, pieces)
            if sum_chunk:
                doc_sum_chunks.append(sum_chunk)

        if doc_sum_chunks:
            new_chunks.extend(doc_sum_chunks)

        # Convert to LangChain Documents and filter blanks defensively
        docs, ids = to_documents_and_ids(new_chunks)
        keep_docs, keep_ids = [], []
        for d, _id in zip(docs, ids):
            if (d.page_content or "").strip():
                keep_docs.append(d)
                keep_ids.append(_id)
        docs, ids = keep_docs, keep_ids

        if docs:
            # ---- batched upsert to avoid "Batch size exceeds maximum" errors ----
            def _chunked(seq, size):
                for i in range(0, len(seq), size):
                    yield seq[i:i + size]

            import os
            BATCH = int(os.getenv("CHROMA_BATCH", "64"))  # safe under typical limits
            total = len(docs)
            print(f"✅ Upserting {total} chunks to Chroma in batches of {BATCH}")
            sent = 0
            for d_chunk, id_chunk in zip(_chunked(docs, BATCH), _chunked(ids, BATCH)):
                if len(d_chunk) != len(id_chunk):
                    raise ValueError("Batch length mismatch")
                vs.add_documents(d_chunk, ids=id_chunk)
                sent += len(d_chunk)
            vs.persist()
            print(f"✅ Indexed {total} chunks into Chroma")
        else:
            print("⚠️  No usable PDF/DOCX content found — collection stays empty")
    else:
        print(f"✓ Loaded existing Chroma with {cur_count} docs")

    # ── 4) rebuild all_docs list from store (filter blanks) ────────────────────
    data = vs._collection.get(include=["documents", "metadatas"])
    all_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip((data.get("documents") or []), (data.get("metadatas") or []))
        if (d or "").strip()
    ]

    # Build a 1-per-document proxy text (first ~1KB) for doc-level index
    doc_texts: Dict[str, str] = {}
    for d in all_docs:
        src = d.metadata.get("source", "")
        if src and src not in doc_texts:
            txt = (d.page_content or "").strip()
            if txt:
                doc_texts[src] = txt[:1024]

    doc_docs = [
        Document(page_content=text, metadata={"source": src})
        for src, text in doc_texts.items()
        if (text or "").strip()
    ]

    # Create doc-level index safely even if empty
    if doc_docs:
        doc_vs = Chroma.from_documents(
            documents=doc_docs,
            embedding=emb,
            collection_name="doc_index",
            client_settings=client_settings,
        )
    else:
        doc_vs = Chroma(
            collection_name="doc_index",
            embedding_function=emb,
            client_settings=client_settings,
        )

    # make a retriever for top-3 docs
    doc_retriever = doc_vs.as_retriever(search_kwargs={"k": 3})

    # ── 5) retrievers for chunks ───────────────────────────────────────────────

    if all_docs:
        # Reduced k to 10 to avoid huge prompts / timeouts
        bm25_retriever = BM25Retriever.from_documents(all_docs, k=10)
    else:
        class _EmptyBM25:
            def invoke(self, *_a, **_kw): return []
        bm25_retriever = _EmptyBM25()

    # Reduced k to 10 for Chroma retriever as well
    chunk_retriever = vs.as_retriever(search_kwargs={"k": 10})


    print("✓ Retriever setup complete:")
    print("   • BM25Retriever   →", type(bm25_retriever))
    print("   • chunk_retriever →", type(chunk_retriever))

    # ────────────────────────── Refresh after deletes ─────────────────────────────
    def refresh_indices_from_chroma() -> None:
        """
        Rebuild in-memory state (all_docs, bm25_retriever, doc_vs, doc_retriever)
        from the current contents of the Chroma collections.

        Call this after deleting documents from vs/doc_vs so that subsequent queries
        don't see stale documents.
        """
        global all_docs, bm25_retriever, doc_vs, doc_retriever, _EMB_INSTANCE

        # 1) Rebuild all_docs from the main chunk collection
        data = vs._collection.get(include=["documents", "metadatas"])
        docs_raw = data.get("documents") or []
        metas_raw = data.get("metadatas") or []

        all_docs = [
            Document(page_content=d, metadata=m)
            for d, m in zip(docs_raw, metas_raw)
            if (d or "").strip()
        ]

        # 2) Rebuild the doc-level index (doc_vs) from remaining docs
        from langchain_huggingface import HuggingFaceEmbeddings
        emb = _EMB_INSTANCE  # already created in setup_vectorstores

        doc_texts: Dict[str, str] = {}
        for d in all_docs:
            src = d.metadata.get("source", "")
            if src and src not in doc_texts:
                txt = (d.page_content or "").strip()
                if txt:
                    doc_texts[src] = txt[:1024]

        doc_docs = [
            Document(page_content=text, metadata={"source": src})
            for src, text in doc_texts.items()
            if (text or "").strip()
        ]

        from chromadb import Settings as _Settings
        # reuse client_settings: same persist directory, etc.
        if doc_docs:
            doc_vs_new = Chroma.from_documents(
                documents=doc_docs,
                embedding=emb,
                collection_name="doc_index",
                client_settings=client_settings,
            )
        else:
            doc_vs_new = Chroma(
                collection_name="doc_index",
                embedding_function=emb,
                client_settings=client_settings,
            )

        doc_vs = doc_vs_new
        doc_retriever = doc_vs.as_retriever(search_kwargs={"k": 3})

        # 3) Rebuild BM25 retriever from all_docs
        if all_docs:
            # Keep BM25 in sync with the global k=10 setting
            bm25_retriever = BM25Retriever.from_documents(all_docs, k=10)
        else:
            class _EmptyBM25:
                def invoke(self, *_a, **_kw):
                    return []
            bm25_retriever = _EmptyBM25()


    return vs, memory_vs, doc_retriever, all_docs, bm25_retriever, chunk_retriever
