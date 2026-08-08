"""实验登记（Experiment Registry）：把每次实验的规格、运行、指标持久化到 SQLite。

对应 AgentX 论文的 Experiment KB / 结构化知识资产。Registry 让研究过程可追踪、
可检索、可形成实验 DAG（通过 ``parent_id``），是 Brainstorm Agent 未来的记忆层。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from ticknet.research.spec import ExperimentResult, ExperimentSpec


class ExperimentRegistry:
    """基于 SQLite 的实验登记簿。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.database_path))
        self._connection.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                parent_id TEXT,
                status TEXT,
                stage TEXT,
                hypothesis TEXT,
                experiment_type TEXT,
                git_sha TEXT,
                dataset_fingerprint TEXT,
                spec_json TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                seed INTEGER,
                status TEXT,
                duration_seconds REAL,
                result_path TEXT
            );

            CREATE TABLE IF NOT EXISTS metrics (
                experiment_id TEXT,
                seed INTEGER,
                metric TEXT,
                value REAL
            );

            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT,
                review_type TEXT,
                decision TEXT,
                payload_json TEXT
            );
            """
        )
        self._connection.commit()

    def record_experiment(
        self,
        experiment_id: str,
        result: ExperimentResult,
        spec: ExperimentSpec,
    ) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO experiments
                (experiment_id, parent_id, status, stage, hypothesis,
                 experiment_type, git_sha, dataset_fingerprint, spec_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment_id,
                result.spec.parent_id,
                result.status,
                spec.stage,
                spec.hypothesis,
                spec.experiment_type,
                result.git_sha,
                result.dataset_fingerprint,
                json.dumps(spec.to_dict(), ensure_ascii=False),
                result.created_at,
            ),
        )
        self._connection.commit()

    def record_run(
        self,
        experiment_id: str,
        seed: int,
        status: str,
        duration_seconds: float | None,
        result_path: str | None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO runs (experiment_id, seed, status, duration_seconds, result_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (experiment_id, seed, status, duration_seconds, result_path),
        )
        self._connection.commit()

    def record_metrics(
        self,
        experiment_id: str,
        seed: int,
        metrics: dict[str, Any],
    ) -> None:
        for metric, value in metrics.items():
            if isinstance(value, (int, float)) and value == value:
                self._connection.execute(
                    "INSERT INTO metrics (experiment_id, seed, metric, value) VALUES (?, ?, ?, ?)",
                    (experiment_id, seed, metric, float(value)),
                )
        self._connection.commit()

    def record_review(
        self,
        experiment_id: str,
        review_type: str,
        decision: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO reviews (experiment_id, review_type, decision, payload_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                experiment_id,
                review_type,
                decision,
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
        self._connection.commit()

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def list_experiments(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def average_metrics(
        self,
        experiment_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """按实验和指标返回平均值，供对比命令使用。"""
        query = "SELECT experiment_id, metric, AVG(value) as mean_value FROM metrics "
        parameters: list[str] = []
        if experiment_ids:
            placeholders = ",".join("?" for _ in experiment_ids)
            query += f"WHERE experiment_id IN ({placeholders}) "
            parameters.extend(experiment_ids)
        query += "GROUP BY experiment_id, metric"
        rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        self._connection.close()
