from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MIGRATIONS: list[tuple[str, str]] = [
    (
        "V001_init_global_schema",
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            model TEXT,
            working_dir TEXT,
            message_count INTEGER DEFAULT 0,
            total_cost_usd REAL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            seq_num INTEGER NOT NULL,
            role TEXT NOT NULL,
            content_json TEXT NOT NULL,
            stop_reason TEXT,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            created_at TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            UNIQUE(session_id, seq_num),
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """,
    ),
    (
        "V002_project_context_and_memory",
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            title TEXT,
            category TEXT,
            content TEXT,
            source TEXT,
            created_at TEXT,
            raw_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS activities (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            type TEXT,
            title TEXT,
            created_at TEXT,
            raw_json TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
        """,
    ),
    (
        "V003_file_snapshots",
        """
        CREATE TABLE IF NOT EXISTS file_snapshots (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            message_id TEXT,
            path TEXT,
            operation TEXT,
            created_at TEXT,
            raw_json TEXT NOT NULL
        );
        """,
    ),
    (
        "V004_anomaly_cost_evidence",
        """
        CREATE TABLE IF NOT EXISTS cost_events (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            model TEXT,
            cost_usd REAL DEFAULT 0,
            created_at TEXT,
            raw_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS anomaly_events (
            id TEXT PRIMARY KEY,
            type TEXT,
            severity TEXT,
            status TEXT,
            created_at TEXT,
            raw_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS evidence_bundles (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            created_at TEXT,
            raw_json TEXT NOT NULL
        );
        """,
    ),
]


@dataclass(slots=True)
class DatabaseStats:
    path: str
    migrations: int
    sessions: int
    messages: int
    memories: int
    activities: int
    journalMode: str
    foreignKeys: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "migrations": self.migrations,
            "sessions": self.sessions,
            "messages": self.messages,
            "memories": self.memories,
            "activities": self.activities,
            "journalMode": self.journalMode,
            "foreignKeys": self.foreignKeys,
        }


class SQLiteStateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def migrate(self) -> list[str]:
        applied: list[str] = []
        with self._lock, self.connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            existing = {row["version"] for row in conn.execute("SELECT version FROM schema_migrations")}
            for version, sql in MIGRATIONS:
                if version in existing:
                    continue
                conn.executescript(sql)
                conn.execute("INSERT INTO schema_migrations(version) VALUES (?)", (version,))
                applied.append(version)
            conn.commit()
        return applied

    def migration_status(self) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            existing = {row["version"]: row["applied_at"] for row in conn.execute("SELECT version, applied_at FROM schema_migrations")}
        return [{"version": version, "applied": version in existing, "appliedAt": existing.get(version)} for version, _ in MIGRATIONS]

    def sync_state(self, state: dict[str, Any]) -> None:
        self.migrate()
        with self._lock, self.connect() as conn:
            with conn:
                self._sync_sessions(conn, state.get("sessions", {}))
                self._sync_memories(conn, state.get("memories", []))
                self._sync_activities(conn, state.get("activities", {}))
                self._sync_file_snapshots(conn, state.get("fileSnapshots", []))
                self._sync_cost_events(conn, state.get("costEvents", []))
                self._sync_anomalies(conn, state.get("anomalies", []))
                self._sync_evidence(conn, state.get("evidence", {}))

    def _sync_sessions(self, conn: sqlite3.Connection, sessions: dict[str, Any]) -> None:
        ids = set(sessions.keys())
        if ids:
            placeholders = ",".join("?" for _ in ids)
            conn.execute(f"DELETE FROM sessions WHERE id NOT IN ({placeholders})", tuple(ids))
        else:
            conn.execute("DELETE FROM sessions")
        for session_id, session in sessions.items():
            messages = session.get("messages") or []
            usage = session.get("usage") or {}
            conn.execute(
                """
                INSERT INTO sessions(id, title, model, working_dir, message_count, total_cost_usd, created_at, updated_at, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title=excluded.title,
                    model=excluded.model,
                    working_dir=excluded.working_dir,
                    message_count=excluded.message_count,
                    total_cost_usd=excluded.total_cost_usd,
                    created_at=excluded.created_at,
                    updated_at=excluded.updated_at,
                    raw_json=excluded.raw_json
                """,
                (
                    session_id,
                    session.get("title"),
                    session.get("model"),
                    session.get("workingDir") or session.get("workingDirectory"),
                    len(messages),
                    float(usage.get("totalCostUsd") or session.get("totalCostUsd") or 0),
                    session.get("createdAt") or session.get("created_at") or "",
                    session.get("updatedAt") or session.get("updated_at") or "",
                    json.dumps(session, ensure_ascii=False),
                ),
            )
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            for seq, message in enumerate(messages, start=1):
                message_id = str(message.get("id") or f"{session_id}-{seq}")
                usage_obj = message.get("usage") or {}
                conn.execute(
                    """
                    INSERT INTO messages(id, session_id, seq_num, role, content_json, stop_reason, input_tokens, output_tokens, created_at, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message_id,
                        session_id,
                        seq,
                        message.get("type") or message.get("role") or "unknown",
                        json.dumps(message.get("content") or "", ensure_ascii=False),
                        message.get("stopReason"),
                        int(usage_obj.get("inputTokens") or 0),
                        int(usage_obj.get("outputTokens") or 0),
                        message.get("timestamp") or message.get("createdAt") or "",
                        json.dumps(message, ensure_ascii=False),
                    ),
                )

    def _sync_memories(self, conn: sqlite3.Connection, memories: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM memories")
        for index, memory in enumerate(memories):
            memory_id = str(memory.get("id") or f"memory-{index}")
            conn.execute(
                "INSERT INTO memories(id, title, category, content, source, created_at, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    memory_id,
                    memory.get("title"),
                    memory.get("category"),
                    memory.get("content"),
                    memory.get("source"),
                    memory.get("createdAt"),
                    json.dumps(memory, ensure_ascii=False),
                ),
            )

    def _sync_activities(self, conn: sqlite3.Connection, activities: dict[str, list[dict[str, Any]]]) -> None:
        conn.execute("DELETE FROM activities")
        existing_sessions = {row["id"] for row in conn.execute("SELECT id FROM sessions")}
        for session_id, items in activities.items():
            if session_id not in existing_sessions:
                continue
            for index, activity in enumerate(items or []):
                activity_id = str(activity.get("id") or f"{session_id}-activity-{index}")
                conn.execute(
                    "INSERT OR IGNORE INTO activities(id, session_id, type, title, created_at, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        activity_id,
                        session_id,
                        activity.get("type"),
                        activity.get("title") or activity.get("message"),
                        activity.get("createdAt") or activity.get("timestamp"),
                        json.dumps(activity, ensure_ascii=False),
                    ),
                )

    def _sync_file_snapshots(self, conn: sqlite3.Connection, snapshots: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM file_snapshots")
        for index, snapshot in enumerate(snapshots):
            snapshot_id = str(snapshot.get("id") or f"snapshot-{index}")
            conn.execute(
                "INSERT INTO file_snapshots(id, session_id, message_id, path, operation, created_at, raw_json) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    snapshot.get("sessionId") or "",
                    snapshot.get("messageId"),
                    snapshot.get("path") or snapshot.get("filePath"),
                    snapshot.get("operation"),
                    snapshot.get("createdAt"),
                    json.dumps(snapshot, ensure_ascii=False),
                ),
            )

    def _sync_cost_events(self, conn: sqlite3.Connection, events: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM cost_events")
        for index, event in enumerate(events):
            event_id = str(event.get("id") or f"cost-{index}")
            conn.execute(
                "INSERT INTO cost_events(id, session_id, model, cost_usd, created_at, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    event.get("sessionId"),
                    event.get("model"),
                    float(event.get("costUsd") or event.get("cost") or 0),
                    event.get("createdAt"),
                    json.dumps(event, ensure_ascii=False),
                ),
            )

    def _sync_anomalies(self, conn: sqlite3.Connection, anomalies: list[dict[str, Any]]) -> None:
        conn.execute("DELETE FROM anomaly_events")
        for index, anomaly in enumerate(anomalies):
            anomaly_id = str(anomaly.get("id") or f"anomaly-{index}")
            conn.execute(
                "INSERT INTO anomaly_events(id, type, severity, status, created_at, raw_json) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    anomaly_id,
                    anomaly.get("type"),
                    anomaly.get("severity"),
                    anomaly.get("status"),
                    anomaly.get("createdAt"),
                    json.dumps(anomaly, ensure_ascii=False),
                ),
            )

    def _sync_evidence(self, conn: sqlite3.Connection, evidence: dict[str, dict[str, Any]]) -> None:
        conn.execute("DELETE FROM evidence_bundles")
        for bundle_id, bundle in evidence.items():
            conn.execute(
                "INSERT INTO evidence_bundles(id, session_id, created_at, raw_json) VALUES (?, ?, ?, ?)",
                (bundle_id, bundle.get("sessionId"), bundle.get("createdAt"), json.dumps(bundle, ensure_ascii=False)),
            )

    def stats(self) -> DatabaseStats:
        self.migrate()
        with self._lock, self.connect() as conn:
            journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            foreign_keys = bool(conn.execute("PRAGMA foreign_keys").fetchone()[0])
            counts = {
                table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in ("schema_migrations", "sessions", "messages", "memories", "activities")
            }
        return DatabaseStats(
            path=str(self.db_path),
            migrations=counts["schema_migrations"],
            sessions=counts["sessions"],
            messages=counts["messages"],
            memories=counts["memories"],
            activities=counts["activities"],
            journalMode=str(journal_mode),
            foreignKeys=foreign_keys,
        )

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock, self.connect() as conn:
            row = conn.execute("SELECT raw_json FROM sessions WHERE id=?", (session_id,)).fetchone()
        return json.loads(row["raw_json"]) if row else None

    def list_messages(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock, self.connect() as conn:
            rows = conn.execute("SELECT raw_json FROM messages WHERE session_id=? ORDER BY seq_num", (session_id,)).fetchall()
        return [json.loads(row["raw_json"]) for row in rows]

    def delete_messages_after(self, session_id: str, seq_num: int) -> int:
        with self._lock, self.connect() as conn:
            with conn:
                cursor = conn.execute("DELETE FROM messages WHERE session_id=? AND seq_num > ?", (session_id, seq_num))
                return int(cursor.rowcount)
