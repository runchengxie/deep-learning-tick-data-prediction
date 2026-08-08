"""一次性、内容绑定的 locked-test 人工批准与消费。"""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ticknet.research.audit import PredictionTable, audit_predictions
from ticknet.research.registry import (
    ExperimentRegistry,
    RegistryConflict,
    file_sha256,
)


class LockedTestNotApproved(RuntimeError):
    """locked test 未获得有效批准，或批准绑定已经失效。"""


class LockedTestFailed(RuntimeError):
    """批准已经消费，但 locked test 审计执行失败。"""


@dataclass(frozen=True)
class LockedTestApproval:
    """执行阶段只持有一次性 bearer token；批准元数据来自 Registry。"""

    token: str

    def __post_init__(self) -> None:
        if not self.token.strip():
            raise LockedTestNotApproved("locked-test token 不能为空")


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _token_sha256(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _resolved_file(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise LockedTestNotApproved(f"{label}不存在: {source}")
    return source


def _checkpoint_bundle(
    registry: ExperimentRegistry,
    experiment_id: str,
    artifact_name: str,
) -> list[dict[str, Any]]:
    artifacts = registry.get_artifacts(experiment_id, name=artifact_name)
    if not artifacts:
        raise LockedTestNotApproved(f"实验缺少待绑定 checkpoint artifact: {artifact_name}")
    bundle: list[dict[str, Any]] = []
    for artifact in artifacts:
        path = _resolved_file(str(artifact["path"]), label="checkpoint artifact ")
        digest = file_sha256(path)
        if digest != artifact["sha256"]:
            raise LockedTestNotApproved(f"checkpoint 与 Registry SHA-256 不一致: {path}")
        bundle.append(
            {
                "seed": int(artifact["seed"]),
                "name": str(artifact["name"]),
                "path": str(path),
                "sha256": digest,
                "size_bytes": int(artifact["size_bytes"]),
            }
        )
    return bundle


def issue_locked_test_approval(
    predictions_path: str | Path,
    *,
    registry: ExperimentRegistry,
    experiment_id: str,
    reason: str,
    approved_by: str,
    checkpoint_artifact_name: str = "best_checkpoint",
) -> dict[str, Any]:
    """为已经通过 KEEP 门槛的实验签发只显示一次的内容绑定 token。"""
    if not reason.strip():
        raise LockedTestNotApproved("locked-test 批准需要填写 reason")
    if not approved_by.strip():
        raise LockedTestNotApproved("locked-test 批准需要填写 approved_by")
    if not checkpoint_artifact_name.strip():
        raise LockedTestNotApproved("checkpoint_artifact_name 不能为空")
    experiment = registry.get_experiment(experiment_id)
    if experiment is None:
        raise LockedTestNotApproved(f"实验尚未登记: {experiment_id}")
    if experiment["status"] != "completed":
        raise LockedTestNotApproved("只有 completed 实验可以签发 locked approval")
    if experiment["stage"] != "release":
        raise LockedTestNotApproved("只有 stage=release 的实验可以签发 locked approval")
    if experiment["evaluation_decision"] != "KEEP":
        raise LockedTestNotApproved("只有 Evaluation=KEEP 的实验可以签发 locked approval")
    dataset_fingerprint = str(experiment["dataset_fingerprint"] or "")
    if not dataset_fingerprint:
        raise LockedTestNotApproved("实验缺少 dataset_fingerprint，不能冻结")

    predictions = _resolved_file(predictions_path, label="locked predictions ")
    predictions_sha256 = file_sha256(predictions)
    checkpoint_bundle = _checkpoint_bundle(
        registry,
        experiment_id,
        checkpoint_artifact_name,
    )
    checkpoint_bundle_sha256 = _canonical_sha256(checkpoint_bundle)
    spec_sha256 = registry.spec_sha256(experiment_id)
    approved_at = datetime.now(timezone.utc).isoformat()
    token = secrets.token_urlsafe(32)
    approval_id = registry.issue_locked_approval(
        experiment_id,
        token_sha256=_token_sha256(token),
        spec_sha256=spec_sha256,
        checkpoint_artifact_name=checkpoint_artifact_name,
        checkpoint_bundle=checkpoint_bundle,
        checkpoint_bundle_sha256=checkpoint_bundle_sha256,
        predictions_path=str(predictions),
        predictions_sha256=predictions_sha256,
        dataset_fingerprint=dataset_fingerprint,
        reason=reason,
        approved_by=approved_by,
        approved_at=approved_at,
    )
    return {
        "approval_id": approval_id,
        "experiment_id": experiment_id,
        "token": token,
        "approved_by": approved_by,
        "approved_at": approved_at,
        "binding": {
            "spec_sha256": spec_sha256,
            "checkpoint_bundle_sha256": checkpoint_bundle_sha256,
            "predictions_sha256": predictions_sha256,
            "dataset_fingerprint": dataset_fingerprint,
        },
    }


def _validate_approval_binding(
    predictions: Path,
    *,
    approval: LockedTestApproval,
    registry: ExperimentRegistry,
    experiment_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stored = registry.get_locked_approval(experiment_id, _token_sha256(approval.token))
    if stored is None or stored["status"] != "issued":
        raise LockedTestNotApproved("locked approval 无效、已消费或不属于该实验")
    experiment = registry.get_experiment(experiment_id)
    if experiment is None or experiment["status"] != "frozen":
        raise LockedTestNotApproved("实验未处于 frozen 状态")
    if registry.spec_sha256(experiment_id) != stored["spec_sha256"]:
        raise LockedTestNotApproved("实验 spec 已变化，locked approval 失效")
    if experiment["dataset_fingerprint"] != stored["dataset_fingerprint"]:
        raise LockedTestNotApproved("dataset_fingerprint 已变化，locked approval 失效")
    if str(predictions) != stored["predictions_path"]:
        raise LockedTestNotApproved("predictions 路径与批准绑定不一致")
    if file_sha256(predictions) != stored["predictions_sha256"]:
        raise LockedTestNotApproved("predictions SHA-256 与批准绑定不一致")

    checkpoint_bundle = _checkpoint_bundle(
        registry,
        experiment_id,
        str(stored["checkpoint_artifact_name"]),
    )
    if checkpoint_bundle != json.loads(str(stored["checkpoint_bundle_json"])):
        raise LockedTestNotApproved("checkpoint artifact 集合与批准绑定不一致")
    if _canonical_sha256(checkpoint_bundle) != stored["checkpoint_bundle_sha256"]:
        raise LockedTestNotApproved("checkpoint bundle SHA-256 与批准绑定不一致")
    return stored, checkpoint_bundle


def run_locked_test(
    predictions_path: str | Path,
    *,
    approval: LockedTestApproval,
    registry: ExperimentRegistry,
    experiment_id: str,
    min_symbols_per_day: int = 50,
    portfolio_quantile: float = 0.1,
) -> dict[str, Any]:
    """原子消费批准，并且只审计批准时绑定的预测、spec、checkpoint 与数据。"""
    predictions = _resolved_file(predictions_path, label="locked predictions ")
    stored, checkpoint_bundle = _validate_approval_binding(
        predictions,
        approval=approval,
        registry=registry,
        experiment_id=experiment_id,
    )
    try:
        consumed_at = registry.consume_locked_approval(int(stored["approval_id"]))
    except RegistryConflict as error:
        raise LockedTestNotApproved(str(error)) from error

    try:
        table = PredictionTable.from_parquet(predictions)
        report = audit_predictions(
            table,
            min_symbols_per_day=min_symbols_per_day,
            portfolio_quantile=portfolio_quantile,
        )
        if file_sha256(predictions) != stored["predictions_sha256"]:
            raise LockedTestNotApproved("predictions 在 locked test 执行期间发生变化")
    except Exception as error:
        registry.record_review(
            experiment_id,
            review_type="locked_test_result",
            decision="FAILED",
            payload={
                "approval_id": int(stored["approval_id"]),
                "consumed_at": consumed_at,
                "error": str(error),
            },
        )
        registry.update_experiment(
            experiment_id,
            status="locked_test_failed",
            error=str(error),
        )
        raise LockedTestFailed(f"locked test 执行失败且批准已消费: {error}") from error
    result = {
        "experiment_id": experiment_id,
        "approval_id": int(stored["approval_id"]),
        "mode": "locked_test",
        "approved_by": str(stored["approved_by"]),
        "approved_at": str(stored["approved_at"]),
        "consumed_at": consumed_at,
        "binding": {
            "spec_sha256": str(stored["spec_sha256"]),
            "checkpoint_bundle_sha256": str(stored["checkpoint_bundle_sha256"]),
            "checkpoint_count": len(checkpoint_bundle),
            "predictions_sha256": str(stored["predictions_sha256"]),
            "dataset_fingerprint": str(stored["dataset_fingerprint"]),
        },
        "audit": report.to_dict(),
        "predictions": str(predictions),
    }
    registry.record_review(
        experiment_id,
        review_type="locked_test_result",
        decision="RECORDED",
        payload={"result": result},
    )
    registry.update_experiment(experiment_id, status="locked_tested")
    return result
