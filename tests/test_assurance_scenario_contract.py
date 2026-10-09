import hashlib
import importlib
import json

import numpy as np
import pytest


def fixture(tmp_path):
    data = b"prediction,return\n1,-0.015625\n-1,-0.015625\n1,0.015625\n-1,0.015625\n"
    (tmp_path / "signals.csv").write_bytes(data)
    payload = {
        "schema_version": "quant.assurance-scenario.v1",
        "name": "null_signal",
        "seed": 0,
        "truth": {"covariance": 0},
        "inventory": [
            {
                "name": "signals",
                "path": "signals.csv",
                "sha256": hashlib.sha256(data).hexdigest(),
                "rows": 4,
                "dtypes": {"prediction": "int64", "return": "float64"},
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def test_independent_reader_preserves_declared_null_truth(tmp_path):
    path, _ = fixture(tmp_path)
    module = importlib.import_module("ticknet.research.scenario_contract")
    frames, truth = module.load_scenario(path)
    assert (
        np.cov(frames["signals"]["prediction"], frames["signals"]["return"])[0, 1]
        == truth["covariance"]
    )


@pytest.mark.parametrize("mutation", ["hash", "path", "schema", "duplicate"])
def test_malformed_inputs_fail_before_consumer(tmp_path, mutation):
    path, payload = fixture(tmp_path)
    if mutation == "hash":
        (tmp_path / "signals.csv").write_bytes(b"changed")
    elif mutation == "path":
        payload["inventory"][0]["path"] = "../signals.csv"
    elif mutation == "schema":
        payload["schema_version"] = "unknown"
    else:
        payload["inventory"].append(payload["inventory"][0])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"scenario|inventory"):
        importlib.import_module("ticknet.research.scenario_contract").load_scenario(path)
