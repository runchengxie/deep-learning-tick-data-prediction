"""带唯一性、递归指标和 artifact 指纹的 SQLite 实验登记簿。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ticknet.research.evaluation import flatten_numeric_metrics
from ticknet.research.spec import ExperimentResult, ExperimentSpec


class RegistryConflict(RuntimeError):
    """实验、run、metric、review 或 artifact 与已有记录冲突。"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ExperimentRegistry:
    """基于 SQLite 的实验登记簿。"""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.database_path))
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def _init_schema(self) -> None:
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                parent_id TEXT REFERENCES experiments(experiment_id),
                status TEXT NOT NULL,
                stage TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                experiment_type TEXT NOT NULL,
                executor TEXT,
                git_sha TEXT,
                dataset_fingerprint TEXT,
                spec_json TEXT NOT NULL,
                artifact_dir TEXT,
                evaluation_decision TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                seed INTEGER NOT NULL,
                status TEXT NOT NULL,
                duration_seconds REAL,
                result_path TEXT,
                exit_code INTEGER,
                stdout_path TEXT,
                stderr_path TEXT,
                UNIQUE(experiment_id, seed)
            );

            CREATE TABLE IF NOT EXISTS metrics (
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                seed INTEGER NOT NULL,
                metric TEXT NOT NULL,
                value REAL NOT NULL,
                UNIQUE(experiment_id, seed, metric)
            );

            CREATE TABLE IF NOT EXISTS reviews (
                review_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                review_type TEXT NOT NULL,
                decision TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(experiment_id, review_type)
            );

            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                seed INTEGER NOT NULL,
                name TEXT NOT NULL,
                path TEXT NOT NULL,
                sha256 TEXT NOT NULL,
                size_bytes INTEGER NOT NULL,
                UNIQUE(experiment_id, seed, name),
                UNIQUE(experiment_id, seed, path)
            );
            """
        )
        self._migrate_legacy_columns()
        try:
            self._connection.executescript(
                """
                CREATE UNIQUE INDEX IF NOT EXISTS uq_runs_experiment_seed
                    ON runs(experiment_id, seed);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_metrics_experiment_seed_metric
                    ON metrics(experiment_id, seed, metric);
                CREATE UNIQUE INDEX IF NOT EXISTS uq_reviews_experiment_type
                    ON reviews(experiment_id, review_type);
                """
            )
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise RegistryConflict("旧 Registry 含重复记录，不能安全升级到 v2") from error
        self._connection.commit()

    def _migrate_legacy_columns(self) -> None:
        additions = {
            "experiments": {
                "executor": "TEXT",
                "artifact_dir": "TEXT",
                "evaluation_decision": "TEXT",
                "error": "TEXT",
                "updated_at": "TEXT",
            },
            "runs": {
                "exit_code": "INTEGER",
                "stdout_path": "TEXT",
                "stderr_path": "TEXT",
            },
        }
        for table, columns in additions.items():
            present = {
                str(row["name"])
                for row in self._connection.execute(f"PRAGMA table_info({table})").fetchall()
            }
            for name, sql_type in columns.items():
                if name not in present:
                    self._connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")

    def has_experiment(self, experiment_id: str) -> bool:
        row = self._connection.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return row is not None

    def _require_experiment(self, experiment_id: str) -> None:
        if not self.has_experiment(experiment_id):
            raise RegistryConflict(f"实验尚未登记: {experiment_id}")

    def create_experiment(
        self,
        experiment_id: str,
        spec: ExperimentSpec,
        *,
        status: str,
        git_sha: str,
        artifact_dir: str,
    ) -> None:
        if spec.parent_id is not None and not self.has_experiment(spec.parent_id):
            raise RegistryConflict(f"parent_id 对应实验不存在: {spec.parent_id}")
        now = _utc_now()
        try:
            self._connection.execute(
                """
                INSERT INTO experiments
                    (experiment_id, parent_id, status, stage, hypothesis,
                     experiment_type, executor, git_sha, dataset_fingerprint,
                     spec_json, artifact_dir, evaluation_decision, error,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, NULL, NULL, ?, ?)
                """,
                (
                    experiment_id,
                    spec.parent_id,
                    status,
                    spec.stage,
                    spec.hypothesis,
                    spec.experiment_type,
                    spec.executor,
                    git_sha,
                    json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True),
                    artifact_dir,
                    now,
                    now,
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise RegistryConflict(f"实验 ID 或 parent_id 冲突: {experiment_id}") from error

    def update_experiment(
        self,
        experiment_id: str,
        *,
        status: str,
        dataset_fingerprint: str | None = None,
        evaluation_decision: str | None = None,
        error: str | None = None,
    ) -> None:
        self._require_experiment(experiment_id)
        self._connection.execute(
            """
            UPDATE experiments
            SET status = ?, dataset_fingerprint = COALESCE(?, dataset_fingerprint),
                evaluation_decision = COALESCE(?, evaluation_decision),
                error = ?, updated_at = ?
            WHERE experiment_id = ?
            """,
            (
                status,
                dataset_fingerprint,
                evaluation_decision,
                error,
                _utc_now(),
                experiment_id,
            ),
        )
        self._connection.commit()

    def record_experiment(
        self,
        experiment_id: str,
        result: ExperimentResult,
        spec: ExperimentSpec,
    ) -> None:
        """兼容人工登记；已有 ID 只允许更新状态，不能覆盖 spec。"""
        existing = self.get_experiment(experiment_id)
        if existing is None:
            self.create_experiment(
                experiment_id,
                spec,
                status=result.status,
                git_sha=result.git_sha,
                artifact_dir=result.artifact_dir,
            )
        else:
            stored_spec = json.loads(str(existing["spec_json"]))
            normalized_spec = json.loads(json.dumps(spec.to_dict(), ensure_ascii=False))
            if stored_spec != normalized_spec:
                raise RegistryConflict(f"实验 ID 对应不同 spec: {experiment_id}")
        self.update_experiment(
            experiment_id,
            status=result.status,
            dataset_fingerprint=result.dataset_fingerprint,
            evaluation_decision=result.evaluation_decision,
        )

    def start_run(self, experiment_id: str, seed: int) -> None:
        self._require_experiment(experiment_id)
        try:
            self._connection.execute(
                "INSERT INTO runs (experiment_id, seed, status) VALUES (?, ?, 'running')",
                (experiment_id, seed),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise RegistryConflict(f"run 已存在: {experiment_id} seed={seed}") from error

    def finish_run(
        self,
        experiment_id: str,
        seed: int,
        *,
        status: str,
        duration_seconds: float | None,
        result_path: str | None,
        exit_code: int | None,
        stdout_path: str | None,
        stderr_path: str | None,
    ) -> None:
        cursor = self._connection.execute(
            """
            UPDATE runs
            SET status = ?, duration_seconds = ?, result_path = ?, exit_code = ?,
                stdout_path = ?, stderr_path = ?
            WHERE experiment_id = ? AND seed = ? AND status = 'running'
            """,
            (
                status,
                duration_seconds,
                result_path,
                exit_code,
                stdout_path,
                stderr_path,
                experiment_id,
                seed,
            ),
        )
        if cursor.rowcount != 1:
            self._connection.rollback()
            raise RegistryConflict(f"找不到唯一 running run: {experiment_id} seed={seed}")
        self._connection.commit()

    def record_run(
        self,
        experiment_id: str,
        seed: int,
        status: str,
        duration_seconds: float | None,
        result_path: str | None,
    ) -> None:
        self.start_run(experiment_id, seed)
        self.finish_run(
            experiment_id,
            seed,
            status=status,
            duration_seconds=duration_seconds,
            result_path=result_path,
            exit_code=None,
            stdout_path=None,
            stderr_path=None,
        )

    def record_metrics(
        self,
        experiment_id: str,
        seed: int,
        metrics: dict[str, Any],
    ) -> None:
        self._require_experiment(experiment_id)
        flattened = flatten_numeric_metrics(metrics)
        try:
            self._connection.executemany(
                "INSERT INTO metrics (experiment_id, seed, metric, value) VALUES (?, ?, ?, ?)",
                [
                    (experiment_id, seed, metric, value)
                    for metric, value in sorted(flattened.items())
                ],
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise RegistryConflict(f"metric 重复: {experiment_id} seed={seed}") from error

    def record_artifact(
        self,
        experiment_id: str,
        seed: int,
        name: str,
        path: str | Path,
    ) -> dict[str, Any]:
        self._require_experiment(experiment_id)
        artifact = Path(path).expanduser().resolve()
        if not artifact.is_file():
            raise RegistryConflict(f"artifact 不存在: {artifact}")
        digest = _sha256_file(artifact)
        size = artifact.stat().st_size
        try:
            self._connection.execute(
                """
                INSERT INTO artifacts
                    (experiment_id, seed, name, path, sha256, size_bytes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (experiment_id, seed, name, str(artifact), digest, size),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise RegistryConflict(
                f"artifact 名称或路径重复: {experiment_id} seed={seed} {name}"
            ) from error
        return {"name": name, "path": str(artifact), "sha256": digest, "size_bytes": size}

    def record_review(
        self,
        experiment_id: str,
        review_type: str,
        decision: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        self._require_experiment(experiment_id)
        try:
            self._connection.execute(
                """
                INSERT INTO reviews (experiment_id, review_type, decision, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    review_type,
                    decision,
                    json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                ),
            )
            self._connection.commit()
        except sqlite3.IntegrityError as error:
            self._connection.rollback()
            raise RegistryConflict(f"review 重复: {experiment_id} {review_type}") from error

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    def get_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM runs WHERE experiment_id = ? ORDER BY seed",
            (experiment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_metrics(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM metrics WHERE experiment_id = ? ORDER BY seed, metric",
            (experiment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_reviews(self, experiment_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT * FROM reviews WHERE experiment_id = ? ORDER BY review_id",
            (experiment_id,),
        ).fetchall()
        return [dict(row) for row in rows]

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
        query = "SELECT experiment_id, metric, AVG(value) as mean_value FROM metrics "
        parameters: list[str] = []
        if experiment_ids:
            placeholders = ",".join("?" for _ in experiment_ids)
            query += f"WHERE experiment_id IN ({placeholders}) "
            parameters.extend(experiment_ids)
        query += "GROUP BY experiment_id, metric ORDER BY experiment_id, metric"
        rows = self._connection.execute(query, parameters).fetchall()
        return [dict(row) for row in rows]

    def spec_sha256(self, experiment_id: str) -> str:
        row = self.get_experiment(experiment_id)
        if row is None:
            raise RegistryConflict(f"实验尚未登记: {experiment_id}")
        return hashlib.sha256(str(row["spec_json"]).encode()).hexdigest()

    def close(self) -> None:
        self._connection.close()
