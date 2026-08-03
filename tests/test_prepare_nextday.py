"""次日预测数据准备脚本测试。"""

import json

import numpy as np

from scripts.prepare_nextday import main


def test_prepare_nextday_writes_labels_and_event_shards(tmp_path):
    daily_bars = tmp_path / "daily-bars.csv"
    daily_bars.write_text(
        "symbol,trading_date,open,close\n"
        "A,2024-01-02,100,100\n"
        "A,2024-01-03,100,99\n"
        "B,2024-01-02,100,100\n"
        "B,2024-01-03,100,100\n"
        "C,2024-01-02,100,100\n"
        "C,2024-01-03,100,101\n",
        encoding="utf-8",
    )
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("2024-01-02\n2024-01-03\n", encoding="utf-8")
    event_lines = []
    for index, symbol in enumerate(("A", "B", "C")):
        feature_path = tmp_path / f"{symbol}.npy"
        np.save(feature_path, np.full((5, 40), index, dtype=np.float32))
        event_lines.append(
            json.dumps(
                {
                    "symbol": symbol,
                    "trading_date": "2024-01-02",
                    "features_path": feature_path.name,
                    "last_event_timestamp": "2024-01-02T14:54:59",
                    "signal_timestamp": "2024-01-02T14:55:00",
                }
            )
        )
    events = tmp_path / "events.jsonl"
    events.write_text("\n".join(event_lines), encoding="utf-8")

    output = tmp_path / "output"
    main(
        [
            "--daily-bars",
            str(daily_bars),
            "--calendar",
            str(calendar),
            "--events-manifest",
            str(events),
            "--output-dir",
            str(output),
            "--min-cross-section",
            "3",
            "--chunks-per-sample",
            "1",
            "--chunk-size",
            "4",
        ]
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["samples"]) == 3
    assert {row["label"] for row in manifest["samples"]} == {0, 1, 2}
    assert {row["label_date"] for row in manifest["samples"]} == {"2024-01-03"}
    assert np.load(output / "shards/part-00000.npy").shape == (3, 1, 4, 40)
