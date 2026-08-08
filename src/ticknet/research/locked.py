"""锁定测试（Locked Test）隔离：样本外评估需要显式人工批准。

对应 AgentX 论文"Agent 永远拿不到 locked test 权限，人放行后一次性评估"。
即使 Runner/Agent 有读取 locked 数据的能力，也必须提供一个批准 token，
且批准记录写入 Registry，保证评估可审计、不可被 Agent 静默触发。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ticknet.research.audit import PredictionTable, audit_predictions
from ticknet.research.registry import ExperimentRegistry


class LockedTestNotApproved(RuntimeError):
    """锁定测试未获人工批准。"""


@dataclass(frozen=True)
class LockedTestApproval:
    """一次人工批准的锁定测试评估授权。"""

    reason: str
    approved_by: str = "human-reviewer"
    approved_at: str = ""
    token: str = ""

    def __post_init__(self) -> None:
        if self.token != "APPROVED":
            raise LockedTestNotApproved("locked-test 需要显式批准 token=APPROVED")
        if not self.reason.strip():
            raise LockedTestNotApproved("locked-test 批准需要填写 reason")


def run_locked_test(
    predictions_path: str | Path,
    *,
    approval: LockedTestApproval,
    registry: ExperimentRegistry,
    experiment_id: str,
    min_symbols_per_day: int = 50,
    portfolio_quantile: float = 0.1,
) -> dict:
    """在显式批准下评估锁定测试集，并把批准与结果写入 Registry。"""
    table = PredictionTable.from_parquet(predictions_path)
    report = audit_predictions(
        table,
        min_symbols_per_day=min_symbols_per_day,
        portfolio_quantile=portfolio_quantile,
    )
    approved_at = approval.approved_at or datetime.now(timezone.utc).isoformat()
    registry.record_review(
        experiment_id,
        review_type="locked_test_approval",
        decision="APPROVED",
        payload={
            "reason": approval.reason,
            "approved_by": approval.approved_by,
            "approved_at": approved_at,
            "predictions": str(predictions_path),
        },
    )
    result = {
        "experiment_id": experiment_id,
        "mode": "locked_test",
        "approved_by": approval.approved_by,
        "approved_at": approved_at,
        "audit": report.to_dict(),
        "predictions": str(predictions_path),
    }
    registry.record_review(
        experiment_id,
        review_type="locked_test_result",
        decision="RECORDED",
        payload={"result": result},
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
