"""固定 best checkpoint 的 2024 validation 多周期评估 CLI。"""

from __future__ import annotations

import argparse
from pathlib import Path

from ticknet.nextday.horizon_evaluation import evaluate_validation_horizons
from ticknet.nextday.train import load_config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="固定既有 best checkpoint，只评估 2024 validation 的多周期 Rank IC",
    )
    parser.add_argument("--config", type=Path, required=True, help="原始 H=1 训练 YAML")
    parser.add_argument("--sidecar", type=Path, required=True, help="多周期标签 sidecar manifest")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--inference-batch-size", type=int, default=128)
    parser.add_argument("--source-revision")
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--verify-data-checksums",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser


def _training_config_arguments(arguments: argparse.Namespace) -> list[str]:
    values = ["--config", str(arguments.config)]
    optional_values = (
        ("manifest-path", arguments.manifest_path),
        ("checkpoint-dir", arguments.checkpoint_dir),
        ("device", arguments.device),
        ("num-workers", arguments.num_workers),
    )
    for name, value in optional_values:
        if value is not None:
            values.extend((f"--{name}", str(value)))
    if arguments.verify_data_checksums is not None:
        values.append(
            "--verify-data-checksums"
            if arguments.verify_data_checksums
            else "--no-verify-data-checksums"
        )
    return values


def main(argv: list[str] | None = None) -> None:
    arguments = _parser().parse_args(argv)
    config = load_config(_training_config_arguments(arguments))
    if config.evaluate_test:
        raise ValueError("多周期 validation CLI 必须保持 evaluate_test=False")
    evaluate_validation_horizons(
        config,
        arguments.sidecar,
        seeds=arguments.seeds,
        horizons=arguments.horizons,
        output_dir=arguments.output_dir,
        inference_batch_size=arguments.inference_batch_size,
        source_revision=arguments.source_revision,
    )


if __name__ == "__main__":
    main()
