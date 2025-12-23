agentic_features_dict = {
    # Retrieval & reasoning
    "reranker":         {"desc": "MiniLM re-rank retrieved chunks",   "enabled": 1},
    "web_search":       {"desc": "DuckDuckGo fallback search",        "enabled": 0},
    "follow_up":        {"desc": "Clarifying-question agent",         "enabled": 1},

    # Domain-specific
    "clause_locator":   {"desc": "Locate exact RegDoc sections",      "enabled": 1},
    "req_extractor":    {"desc": "Pull SHALL / MUST requirement lines",  "enabled": 1},
    "external_refs":    {"desc": "List IAEA / CSA / ISO refs in answer", "enabled": 1},

    # Trust & provenance
    "citations":        {"desc": "Attach paragraph-level citations",  "enabled": 1},
    "topic_suggestion": {"desc": "Suggest related subtopics & follow‑up questions",  "enabled": 1},

    # Ops
    "query_gap_logger": {"desc": "Log low-confidence queries",        "enabled": 0},
    "memory":           {"desc": "Memory for chats",                  "enabled": 1}
}