import json
import os
import sqlite3
import time
from typing import Any


class UsageStore:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "")
        self.sqlite_path = os.getenv("SQLITE_PATH", "outputs/usage.db")
        self.use_postgres = self.database_url.startswith("postgres")
        self.init()

    def init(self):
        if self.use_postgres:
            import psycopg

            last_error: Exception | None = None
            for _ in range(20):
                try:
                    with psycopg.connect(self.database_url) as conn:
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS usage_events (
                                request_id TEXT PRIMARY KEY,
                                route TEXT NOT NULL,
                                task_type TEXT NOT NULL,
                                prompt_tokens INTEGER NOT NULL,
                                completion_tokens INTEGER NOT NULL,
                                total_tokens INTEGER NOT NULL,
                                estimated_cost_usd DOUBLE PRECISION NOT NULL,
                                payload JSONB NOT NULL,
                                created_at TIMESTAMPTZ DEFAULT now()
                            )
                            """
                        )
                    return
                except Exception as exc:
                    last_error = exc
                    time.sleep(1)
            raise RuntimeError(f"database initialization failed: {last_error}")
        else:
            parent = os.path.dirname(self.sqlite_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS usage_events (
                        request_id TEXT PRIMARY KEY,
                        route TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        prompt_tokens INTEGER NOT NULL,
                        completion_tokens INTEGER NOT NULL,
                        total_tokens INTEGER NOT NULL,
                        estimated_cost_usd REAL NOT NULL,
                        payload TEXT NOT NULL,
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )

    def insert(self, row: dict[str, Any]):
        payload = json.dumps(row["payload"], ensure_ascii=False)
        values = (
            row["request_id"],
            row["route"],
            row["task_type"],
            row["prompt_tokens"],
            row["completion_tokens"],
            row["total_tokens"],
            row["estimated_cost_usd"],
            payload,
        )
        if self.use_postgres:
            import psycopg

            with psycopg.connect(self.database_url) as conn:
                conn.execute(
                    """
                    INSERT INTO usage_events (
                        request_id, route, task_type, prompt_tokens, completion_tokens,
                        total_tokens, estimated_cost_usd, payload
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    ON CONFLICT (request_id) DO NOTHING
                    """,
                    values,
                )
        else:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO usage_events (
                        request_id, route, task_type, prompt_tokens, completion_tokens,
                        total_tokens, estimated_cost_usd, payload
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

    def stats(self, limit: int = 20) -> dict[str, Any]:
        if self.use_postgres:
            return self._postgres_stats(limit)
        return self._sqlite_stats(limit)

    def _sqlite_stats(self, limit: int) -> dict[str, Any]:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            totals = conn.execute(
                "SELECT COUNT(*) c, COALESCE(SUM(total_tokens), 0) t, COALESCE(SUM(estimated_cost_usd), 0) cost FROM usage_events"
            ).fetchone()
            routes = conn.execute("SELECT route, COUNT(*) c FROM usage_events GROUP BY route").fetchall()
            tasks = conn.execute("SELECT task_type, COUNT(*) c FROM usage_events GROUP BY task_type").fetchall()
            recent_rows = conn.execute(
                "SELECT payload FROM usage_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "totalRequests": int(totals["c"]),
            "totalTokens": int(totals["t"]),
            "estimatedCostUsd": float(totals["cost"]),
            "routeCounts": {row["route"]: int(row["c"]) for row in routes},
            "taskCounts": {row["task_type"]: int(row["c"]) for row in tasks},
            "recent": [json.loads(row["payload"]) for row in recent_rows],
        }

    def _postgres_stats(self, limit: int) -> dict[str, Any]:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            totals = conn.execute(
                "SELECT COUNT(*) c, COALESCE(SUM(total_tokens), 0) t, COALESCE(SUM(estimated_cost_usd), 0) cost FROM usage_events"
            ).fetchone()
            routes = conn.execute("SELECT route, COUNT(*) c FROM usage_events GROUP BY route").fetchall()
            tasks = conn.execute("SELECT task_type, COUNT(*) c FROM usage_events GROUP BY task_type").fetchall()
            recent_rows = conn.execute(
                "SELECT payload FROM usage_events ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return {
            "totalRequests": int(totals[0]),
            "totalTokens": int(totals[1]),
            "estimatedCostUsd": float(totals[2]),
            "routeCounts": {row[0]: int(row[1]) for row in routes},
            "taskCounts": {row[0]: int(row[1]) for row in tasks},
            "recent": [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in recent_rows],
        }
