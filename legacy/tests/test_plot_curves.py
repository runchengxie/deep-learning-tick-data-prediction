"""训练曲线脚本测试。"""

from __future__ import annotations

import json

from legacy.scripts.plot_curves import load_histories


def test_load_histories_reads_current_filename_pattern(tmp_path):
    history = [
        {
            "epoch": 1,
            "train_loss": 1.0,
            "val_accuracy": 0.5,
            "val_macro_f1": 0.4,
        }
    ]
    path = tmp_path / "train_history.setup2.k10.json"
    path.write_text(json.dumps(history), encoding="utf-8")
    assert load_histories(str(tmp_path)) == [("setup2.k10", history)]
