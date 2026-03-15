# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, The Exgentic organization and its contributors.

import json

from transformers import AutoTokenizer


class BCPSearchToolHandler:
    def __init__(
        self,
        searcher,
        snippet_max_tokens: int | None = None,
        k: int = 5,
        include_get_document: bool = True,
        full_doc_max_tokens: int | None = None,
    ):
        self.searcher = searcher
        self.snippet_max_tokens = snippet_max_tokens
        self.k = k
        self.include_get_document = include_get_document

        self.tokenizer = None
        self.full_doc_max_tokens = None
        if snippet_max_tokens and snippet_max_tokens > 0:
            self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
            self.full_doc_max_tokens = full_doc_max_tokens

    def execute_tool(self, tool_name: str, arguments: dict):
        if tool_name == "search":
            return self._search(arguments["query"])
        if tool_name == "get_document":
            return self._get_document(arguments["docid"])
        raise ValueError(f"Unknown tool: {tool_name}")

    def _search(self, query: str):
        candidates = self.searcher.search(query, self.k)

        if self.snippet_max_tokens and self.snippet_max_tokens > 0 and self.tokenizer:
            for cand in candidates:
                text = cand["text"]
                cand["snippet"] = self._truncate_text(text, self.snippet_max_tokens)
        else:
            for cand in candidates:
                cand["snippet"] = cand["text"]

        results = []
        for cand in candidates:
            if cand.get("score") is None:
                results.append({"docid": cand["docid"], "snippet": cand["snippet"]})
            else:
                results.append(
                    {
                        "docid": cand["docid"],
                        "score": cand["score"],
                        "snippet": cand["snippet"],
                    }
                )

        return json.dumps(results, indent=2)

    def _get_document(self, docid: str):
        try:
            result = self.searcher.get_document(docid)
        except Exception:
            result = None
        if result is None:
            return json.dumps({"error": f"Document {docid} not found"})

        text = result.get("text")
        result["text"] = self._truncate_text(text, max_len=self.full_doc_max_tokens)
        return json.dumps(result, indent=2)

    def _truncate_text(self, text: str, max_len: int) -> str:
        if not self.tokenizer:
            raise RuntimeError("Tokenizer not initialized")
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        if max_len and len(tokens) > max_len:
            truncated_tokens = tokens[:max_len]
            return self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        return text
