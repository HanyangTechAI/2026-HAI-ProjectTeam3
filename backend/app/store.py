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
                        conn.execute(
                            """
                            CREATE TABLE IF NOT EXISTS feedback_events (
                                id BIGSERIAL PRIMARY KEY,
                                request_id TEXT NOT NULL REFERENCES usage_events(request_id),
                                reviewer_id TEXT NOT NULL DEFAULT 'anonymous',
                                rating INTEGER NOT NULL,
                                quality_score DOUBLE PRECISION,
                                reward DOUBLE PRECISION NOT NULL,
                                comment TEXT NOT NULL DEFAULT '',
                                created_at TIMESTAMPTZ DEFAULT now()
                            )
                            """
                        )
                        conn.execute(
                            "ALTER TABLE feedback_events ADD COLUMN IF NOT EXISTS reviewer_id TEXT NOT NULL DEFAULT 'anonymous'"
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
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feedback_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        request_id TEXT NOT NULL,
                        reviewer_id TEXT NOT NULL DEFAULT 'anonymous',
                        rating INTEGER NOT NULL,
                        quality_score REAL,
                        reward REAL NOT NULL,
                        comment TEXT NOT NULL DEFAULT '',
                        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY(request_id) REFERENCES usage_events(request_id)
                    )
                    """
                )
                self._sqlite_add_column_if_missing(
                    conn,
                    table="feedback_events",
                    column="reviewer_id",
                    definition="TEXT NOT NULL DEFAULT 'anonymous'",
                )

    def _sqlite_add_column_if_missing(self, conn: sqlite3.Connection, table: str, column: str, definition: str):
        columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

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

    def insert_feedback(self, row: dict[str, Any]):
        values = (
            row["request_id"],
            row.get("reviewer_id", "anonymous"),
            row["rating"],
            row.get("quality_score"),
            row["reward"],
            row.get("comment", ""),
        )
        if self.use_postgres:
            import psycopg

            with psycopg.connect(self.database_url) as conn:
                conn.execute(
                    """
                    INSERT INTO feedback_events (
                        request_id, reviewer_id, rating, quality_score, reward, comment
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    values,
                )
        else:
            with sqlite3.connect(self.sqlite_path) as conn:
                conn.execute(
                    """
                    INSERT INTO feedback_events (
                        request_id, reviewer_id, rating, quality_score, reward, comment
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    values,
                )

    def feedback_training_rows(self, limit: int | None = None) -> list[dict[str, Any]]:
        if self.use_postgres:
            return self._postgres_feedback_training_rows(limit)
        return self._sqlite_feedback_training_rows(limit)

    def request_exists(self, request_id: str) -> bool:
        if self.use_postgres:
            import psycopg

            with psycopg.connect(self.database_url) as conn:
                row = conn.execute("SELECT 1 FROM usage_events WHERE request_id = %s", (request_id,)).fetchone()
            return row is not None
        with sqlite3.connect(self.sqlite_path) as conn:
            row = conn.execute("SELECT 1 FROM usage_events WHERE request_id = ?", (request_id,)).fetchone()
        return row is not None

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
            feedback_total = conn.execute("SELECT COUNT(*) c FROM feedback_events").fetchone()
            feedback_counts = conn.execute("SELECT rating, COUNT(*) c FROM feedback_events GROUP BY rating").fetchall()
            reviewer_counts = conn.execute("SELECT reviewer_id, COUNT(*) c FROM feedback_events GROUP BY reviewer_id").fetchall()
            recent_rows = conn.execute(
                "SELECT payload FROM usage_events ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return {
            "totalRequests": int(totals["c"]),
            "totalFeedback": int(feedback_total["c"]),
            "totalTokens": int(totals["t"]),
            "estimatedCostUsd": float(totals["cost"]),
            "routeCounts": {row["route"]: int(row["c"]) for row in routes},
            "taskCounts": {row["task_type"]: int(row["c"]) for row in tasks},
            "feedbackCounts": {str(row["rating"]): int(row["c"]) for row in feedback_counts},
            "reviewerCounts": {row["reviewer_id"]: int(row["c"]) for row in reviewer_counts},
            "recent": [json.loads(row["payload"]) for row in recent_rows],
        }

    def _sqlite_feedback_training_rows(self, limit: int | None) -> list[dict[str, Any]]:
        query = """
            SELECT u.payload, f.reviewer_id, f.rating, f.quality_score, f.reward, f.comment, f.created_at
            FROM feedback_events f
            JOIN usage_events u ON u.request_id = f.request_id
            ORDER BY f.created_at DESC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (limit,)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "payload": json.loads(row["payload"]),
                "reviewer_id": row["reviewer_id"],
                "rating": int(row["rating"]),
                "quality_score": None if row["quality_score"] is None else float(row["quality_score"]),
                "reward": float(row["reward"]),
                "comment": row["comment"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _postgres_stats(self, limit: int) -> dict[str, Any]:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            totals = conn.execute(
                "SELECT COUNT(*) c, COALESCE(SUM(total_tokens), 0) t, COALESCE(SUM(estimated_cost_usd), 0) cost FROM usage_events"
            ).fetchone()
            routes = conn.execute("SELECT route, COUNT(*) c FROM usage_events GROUP BY route").fetchall()
            tasks = conn.execute("SELECT task_type, COUNT(*) c FROM usage_events GROUP BY task_type").fetchall()
            feedback_total = conn.execute("SELECT COUNT(*) c FROM feedback_events").fetchone()
            feedback_counts = conn.execute("SELECT rating, COUNT(*) c FROM feedback_events GROUP BY rating").fetchall()
            reviewer_counts = conn.execute("SELECT reviewer_id, COUNT(*) c FROM feedback_events GROUP BY reviewer_id").fetchall()
            recent_rows = conn.execute(
                "SELECT payload FROM usage_events ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return {
            "totalRequests": int(totals[0]),
            "totalFeedback": int(feedback_total[0]),
            "totalTokens": int(totals[1]),
            "estimatedCostUsd": float(totals[2]),
            "routeCounts": {row[0]: int(row[1]) for row in routes},
            "taskCounts": {row[0]: int(row[1]) for row in tasks},
            "feedbackCounts": {str(row[0]): int(row[1]) for row in feedback_counts},
            "reviewerCounts": {row[0]: int(row[1]) for row in reviewer_counts},
            "recent": [row[0] if isinstance(row[0], dict) else json.loads(row[0]) for row in recent_rows],
        }

    def _postgres_feedback_training_rows(self, limit: int | None) -> list[dict[str, Any]]:
        import psycopg

        query = """
            SELECT u.payload, f.reviewer_id, f.rating, f.quality_score, f.reward, f.comment, f.created_at
            FROM feedback_events f
            JOIN usage_events u ON u.request_id = f.request_id
            ORDER BY f.created_at DESC
        """
        params: tuple[Any, ...] = ()
        if limit is not None:
            query += " LIMIT %s"
            params = (limit,)
        with psycopg.connect(self.database_url) as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "payload": row[0] if isinstance(row[0], dict) else json.loads(row[0]),
                "reviewer_id": row[1],
                "rating": int(row[2]),
                "quality_score": None if row[3] is None else float(row[3]),
                "reward": float(row[4]),
                "comment": row[5],
                "created_at": str(row[6]),
            }
            for row in rows
        ]
