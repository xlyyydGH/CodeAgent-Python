from __future__ import annotations

import math
import re
import time
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


class MemoryCategory(StrEnum):
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    TEAM = "team"

    @classmethod
    def from_tag(cls, value: str | None) -> "MemoryCategory":
        if not value:
            return cls.SEMANTIC
        lowered = value.lower()
        for item in cls:
            if item.value == lowered:
                return item
        return cls.SEMANTIC


STOP_WORDS = {
    "的",
    "了",
    "在",
    "是",
    "我",
    "有",
    "和",
    "就",
    "不",
    "the",
    "is",
    "at",
    "which",
    "on",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "to",
    "for",
    "of",
    "with",
    "this",
    "that",
}


@dataclass(slots=True)
class MemoryDocument:
    id: str
    title: str
    content: str
    category: MemoryCategory
    source: str = "USER"
    createdAt: str | None = None
    updatedAt: str | None = None
    keywords: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["category"] = self.category.value
        return data


@dataclass(slots=True)
class SearchResult:
    memory: MemoryDocument
    score: float
    bm25Score: float
    rerankScore: float | None = None
    matchedTokens: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.memory.to_dict(),
            "score": self.score,
            "bm25Score": self.bm25Score,
            "rerankScore": self.rerankScore,
            "matchedTokens": self.matchedTokens or [],
        }


class MemorySearchEngine:
    K1 = 1.2
    B = 0.75
    TITLE_BOOST = 2.0
    WORD_PATTERN = re.compile(r"[a-zA-Z0-9_]+")

    def tokenize(self, text: str | None) -> list[str]:
        if not text:
            return []
        lowered = text.lower()
        tokens: list[str] = []
        for match in self.WORD_PATTERN.finditer(lowered):
            word = match.group(0)
            if len(word) > 1 and word not in STOP_WORDS:
                tokens.append(word)
        cjk_buffer = []
        for char in lowered:
            if "\u4e00" <= char <= "\u9fff":
                cjk_buffer.append(char)
            else:
                self._flush_cjk(cjk_buffer, tokens)
        self._flush_cjk(cjk_buffer, tokens)
        return tokens

    def search(self, entries: list[MemoryDocument], query: str, top_k: int = 5) -> list[SearchResult]:
        query_tokens = self.tokenize(query)
        if not entries or not query_tokens:
            return []
        doc_tokens = [self.tokenize(item.content + "\n" + item.keywords) for item in entries]
        title_tokens = [self.tokenize(item.title) for item in entries]
        avg_dl = sum(len(tokens) for tokens in doc_tokens) / max(1, len(doc_tokens))
        idf = self._idf(query_tokens, doc_tokens)
        results: list[SearchResult] = []
        for index, memory in enumerate(entries):
            body_score = self._bm25(query_tokens, doc_tokens[index], idf, avg_dl)
            title_score = self._bm25(query_tokens, title_tokens[index], idf, max(1.0, avg_dl / 5.0))
            score = body_score + title_score * self.TITLE_BOOST
            matched = sorted(set(query_tokens) & set(doc_tokens[index] + title_tokens[index]))
            if score > 0 or matched:
                results.append(SearchResult(memory, score, score, matchedTokens=matched))
        return sorted(results, key=lambda item: item.score, reverse=True)[: max(1, top_k)]

    def _flush_cjk(self, buffer: list[str], tokens: list[str]) -> None:
        if not buffer:
            return
        text = "".join(buffer)
        for char in text:
            if char not in STOP_WORDS:
                tokens.append(char)
        for index in range(len(text) - 1):
            tokens.append(text[index : index + 2])
        buffer.clear()

    def _idf(self, query_tokens: list[str], docs: list[list[str]]) -> dict[str, float]:
        total = len(docs)
        values = {}
        for token in query_tokens:
            df = sum(1 for doc in docs if token in doc)
            values[token] = math.log((total - df + 0.5) / (df + 0.5) + 1.0)
        return values

    def _bm25(self, query_tokens: list[str], doc_tokens: list[str], idf: dict[str, float], avg_dl: float) -> float:
        if not doc_tokens:
            return 0.0
        tf: dict[str, int] = {}
        for token in doc_tokens:
            tf[token] = tf.get(token, 0) + 1
        score = 0.0
        doc_len = len(doc_tokens)
        for token in query_tokens:
            freq = tf.get(token, 0)
            if not freq:
                continue
            numerator = freq * (self.K1 + 1)
            denominator = freq + self.K1 * (1 - self.B + self.B * (doc_len / max(1.0, avg_dl)))
            score += idf.get(token, 0.0) * (numerator / denominator)
        return score


class MemoryRerankService:
    def rerank(self, query: str, candidates: list[SearchResult], top_k: int) -> list[SearchResult]:
        query_tokens = set(MemorySearchEngine().tokenize(query))
        reranked: list[SearchResult] = []
        for result in candidates:
            memory = result.memory
            text = f"{memory.title}\n{memory.content}\n{memory.keywords}".lower()
            semantic_boost = 0.0
            if memory.category == MemoryCategory.PROCEDURAL and any(token in text for token in ("流程", "步骤", "workflow", "deploy", "build")):
                semantic_boost += 0.15
            if memory.category == MemoryCategory.TEAM:
                semantic_boost += 0.1
            if memory.updatedAt:
                semantic_boost += 0.03
            overlap = len(query_tokens & set(MemorySearchEngine().tokenize(text))) / max(1, len(query_tokens))
            rerank_score = min(1.0, overlap * 0.7 + semantic_boost)
            reranked.append(SearchResult(memory, result.bm25Score + rerank_score, result.bm25Score, rerank_score, result.matchedTokens))
        return sorted(reranked, key=lambda item: item.score, reverse=True)[: max(1, top_k)]


class MemdirService:
    MAX_PROMPT_LINES = 200
    MAX_PROMPT_BYTES = 25_000

    def __init__(self, root: Path, state: dict[str, Any], search_engine: MemorySearchEngine | None = None, reranker: MemoryRerankService | None = None) -> None:
        self.root = root
        self.state = state
        self.search_engine = search_engine or MemorySearchEngine()
        self.reranker = reranker or MemoryRerankService()

    def entries(self) -> list[MemoryDocument]:
        return [self._normalize(item) for item in self.state.setdefault("memories", [])]

    def categories(self) -> list[dict[str, str]]:
        return [{"name": item.name, "tag": item.value} for item in MemoryCategory]

    def search(self, query: str, top_k: int = 5, category: str | None = None, rerank: bool = True) -> list[SearchResult]:
        entries = self.entries()
        if category:
            parsed = MemoryCategory.from_tag(category)
            entries = [item for item in entries if item.category == parsed]
        candidates = self.search_engine.search(entries, query, max(top_k, 20 if rerank else top_k))
        if rerank and len(candidates) > top_k:
            return self.reranker.rerank(query, candidates, top_k)
        return candidates[: max(1, top_k)]

    def search_by_category(self, category: str, max_count: int = 10) -> list[MemoryDocument]:
        parsed = MemoryCategory.from_tag(category)
        entries = [item for item in self.entries() if item.category == parsed]
        return sorted(entries, key=lambda item: item.updatedAt or item.createdAt or "", reverse=True)[: max(1, max_count)]

    def build_prompt(self, project_root: Path | None = None) -> str:
        parts: list[str] = []
        personal = self.read_for_prompt()
        if personal:
            parts.append("## Personal Memory\n" + personal)
        team = self.load_team_memories(project_root or self.root)
        if team:
            team_text = "\n\n".join(f"### {item.title}\n{item.content}" for item in team)
            parts.append("## Team Memory\n" + team_text)
        if not parts:
            return ""
        return "\n\n".join(parts) + "\n\n---\nMemory categories: episodic, semantic, procedural, team\n"

    def read_for_prompt(self) -> str:
        content = "\n\n".join(f"### {item.title}\n{item.content}" for item in self.entries() if item.category != MemoryCategory.TEAM)
        lines = content.splitlines()
        if len(lines) > self.MAX_PROMPT_LINES:
            content = "\n".join(lines[: self.MAX_PROMPT_LINES]) + f"\n<!-- truncated: exceeded {self.MAX_PROMPT_LINES} lines -->"
        encoded = content.encode("utf-8")
        if len(encoded) > self.MAX_PROMPT_BYTES:
            content = encoded[: self.MAX_PROMPT_BYTES].decode("utf-8", errors="ignore") + f"\n<!-- truncated: exceeded {self.MAX_PROMPT_BYTES} bytes -->"
        return content

    def load_team_memories(self, project_root: Path) -> list[MemoryDocument]:
        team_dir = project_root / ".zhikun" / "team-memories"
        if not team_dir.is_dir():
            return []
        memories: list[MemoryDocument] = []
        for path in sorted(team_dir.glob("*.md")):
            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            memories.append(
                MemoryDocument(
                    id=f"team:{path.name}",
                    title=path.stem,
                    content=content,
                    category=MemoryCategory.TEAM,
                    source="TEAM",
                    createdAt=None,
                    updatedAt=None,
                )
            )
        return memories

    def tool(self, action: str, content: str = "", title: str = "", category: str = "semantic", query: str = "", limit: int = 5) -> dict[str, Any]:
        if action == "read":
            return {"content": self.read_for_prompt(), "entries": [item.to_dict() for item in self.entries()]}
        if action == "search":
            return {"results": [item.to_dict() for item in self.search(query or content, limit, rerank=True)]}
        if action == "write":
            if not content.strip():
                return {"error": "Content is required for write action."}
            now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            entry = {
                "id": f"memory-{int(time.time() * 1000)}",
                "title": title or content[:60],
                "content": content,
                "category": MemoryCategory.from_tag(category).value,
                "source": "TOOL",
                "createdAt": now,
                "updatedAt": now,
            }
            self.state.setdefault("memories", []).append(entry)
            return {"success": True, "memory": entry}
        if action == "delete":
            before = len(self.state.setdefault("memories", []))
            needle = content.lower()
            self.state["memories"] = [item for item in self.state["memories"] if needle not in str(item.get("content") or "").lower()]
            return {"success": len(self.state["memories"]) < before, "deleted": before - len(self.state["memories"])}
        return {"error": f"Unknown action: {action}"}

    def _normalize(self, item: dict[str, Any]) -> MemoryDocument:
        content = str(item.get("content") or "")
        return MemoryDocument(
            id=str(item.get("id") or f"memory-{abs(hash(content))}"),
            title=str(item.get("title") or item.get("summary") or content[:60] or "Memory"),
            content=content,
            category=MemoryCategory.from_tag(str(item.get("category") or "semantic")),
            source=str(item.get("source") or "USER"),
            createdAt=item.get("createdAt") or item.get("created_at"),
            updatedAt=item.get("updatedAt") or item.get("updated_at"),
            keywords=str(item.get("keywords") or ""),
        )
