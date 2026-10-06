"""把官方 FI-2010 文本文件转换成训练使用的 ``float32`` NPY。

官方文件按 ``149 × N`` 保存，每一列是一个样本：

* 第 0 至 39 行是 DeepLOB 使用的十档订单簿价格和数量
* 第 40 至 143 行是本项目保留但不送入模型的手工特征
* 第 144 至 148 行是 ``k=10/20/30/50/100`` 的标签

转换结果按 ``N × 149`` 保存。脚本同时生成带有源文件边界的
``<输出文件名>_meta.json``，训练时需要一起使用。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

RAW_FEATURE_COLUMNS = 144
MODEL_FEATURE_COLUMNS = 40
LABEL_COLUMNS = 5
TOTAL_ROWS = RAW_FEATURE_COLUMNS + LABEL_COLUMNS
HORIZONS = (10, 20, 30, 50, 100)

NORMALISATION_TOKENS = {
    "z-score": ("Zscore", "ZScore", 1),
    "min-max": ("MinMax", "MinMax", 2),
    "decimal": ("DecPre", "DecPre", 3),
}


def find_files(
    base_dir: str,
    norm: str,
    auction: str,
    folds: Sequence[int],
) -> list[str]:
    """按官方目录和文件名查找 Training 与 Testing 文件。"""
    directory_token, filename_token, index = NORMALISATION_TOKENS[norm]
    norm_root = f"{index}.{auction}_{directory_token}"
    found: list[str] = []
    for split in ("Training", "Testing"):
        prefix = "Train" if split == "Training" else "Test"
        directory = Path(base_dir) / auction / norm_root / f"{auction}_{directory_token}_{split}"
        for fold in folds:
            filename = f"{prefix}_Dst_{auction}_{filename_token}_CF_{fold}.txt"
            path = directory / filename
            if path.is_file():
                found.append(str(path))
            else:
                print(f"缺少文件：{path}")
    return found


def _read_txt(path: str) -> np.ndarray:
    """读取一个官方文本文件并转置为 ``N × 149``。"""
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[0] != TOTAL_ROWS:
        raise ValueError(f"{path} 应有 {TOTAL_ROWS} 行，实际形状为 {raw.shape}")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{path} 包含 NaN 或无穷值")
    labels = raw[RAW_FEATURE_COLUMNS:]
    unexpected = set(np.unique(labels).tolist()) - {1.0, 2.0, 3.0}
    if unexpected:
        raise ValueError(f"{path} 含无效标签：{sorted(unexpected)}")
    return np.ascontiguousarray(raw.T)


def _sample_count(path: str) -> int:
    """从文件第一行取得样本数，避免为确定输出形状而加载整个文件。"""
    with open(path, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                return len(line.split())
    raise ValueError(f"{path} 是空文件")


def _metadata_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_meta.json")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="转换官方 FI-2010 数据")
    parser.add_argument(
        "--base-dir",
        required=True,
        help="包含 NoAuction 和 Auction 目录的 BenchmarkDatasets 路径",
    )
    parser.add_argument(
        "--norm",
        choices=list(NORMALISATION_TOKENS),
        default="z-score",
        help="归一化版本",
    )
    parser.add_argument(
        "--auction",
        choices=["NoAuction", "Auction"],
        default="NoAuction",
        help="是否包含竞价时段，论文使用 NoAuction",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=list(range(1, 10)),
        help="需要转换的 CF 编号",
    )
    parser.add_argument("--out", help="输出 .npy 路径")
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="只检查第一个匹配文件",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    folds = sorted(set(args.folds))
    if not folds or any(fold not in range(1, 10) for fold in folds):
        raise SystemExit("--folds 只能包含 1 至 9")

    files = find_files(args.base_dir, args.norm, args.auction, folds)
    if not files:
        raise SystemExit("没有找到 FI-2010 文本文件，请检查 --base-dir 等参数")
    if args.inspect_only:
        sample = _read_txt(files[0])
        print(f"文件：{files[0]}")
        print(f"转置后形状：{sample.shape}")
        print("前 40 列用于模型，原始特征共 144 列")
        print(f"标签列：{dict(zip(HORIZONS, range(144, 149), strict=True))}")
        print(f"标签取值：{np.unique(sample[:, 144:149]).tolist()}")
        return

    expected_files = len(folds) * 2
    if len(files) != expected_files:
        raise SystemExit(f"完整转换需要 {expected_files} 个文件，当前找到 {len(files)} 个")
    if not args.out:
        raise SystemExit("完整转换需要 --out")

    output = Path(args.out).expanduser().resolve()
    if output.suffix != ".npy":
        raise SystemExit("--out 应使用 .npy 扩展名")
    output.parent.mkdir(parents=True, exist_ok=True)
    sample_counts = [_sample_count(path) for path in files]
    total_samples = sum(sample_counts)
    temporary = output.with_suffix(".tmp.npy")
    metadata_segments: list[dict[str, int | str]] = []
    offset = 0
    committed = False
    try:
        destination = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(total_samples, TOTAL_ROWS),
        )
        for path, expected_samples in zip(files, sample_counts, strict=True):
            array = _read_txt(path)
            if array.shape[0] != expected_samples:
                raise ValueError(
                    f"{path} 首行记录 {expected_samples} 个样本，完整读取后得到 {array.shape[0]} 个"
                )
            end = offset + expected_samples
            destination[offset:end] = array
            match = re.search(r"CF_(\d+)\.txt$", Path(path).name)
            if match is None:
                raise ValueError(f"无法从文件名解析 CF 编号：{path}")
            role = "test" if Path(path).name.startswith("Test_") else "train"
            metadata_segments.append(
                {
                    "cf": int(match.group(1)),
                    "role": role,
                    "start": offset,
                    "end": end,
                    "source": Path(path).name,
                }
            )
            offset = end
            print(f"已写入：{Path(path).name}，累计 {offset:,} 行")
        destination.flush()
        del destination
        os.replace(temporary, output)
        committed = True
    finally:
        if not committed and temporary.exists():
            temporary.unlink()

    metadata = {
        "schema_version": 1,
        "rows": total_samples,
        "columns": TOTAL_ROWS,
        "raw_feature_columns": RAW_FEATURE_COLUMNS,
        "model_feature_columns": MODEL_FEATURE_COLUMNS,
        "horizons": list(HORIZONS),
        "auction": args.auction,
        "normalisation": args.norm,
        "folds": folds,
        "segments": metadata_segments,
    }
    metadata_path = _metadata_path(output)
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    print(f"已保存数据：{output}，形状为 ({total_samples:,}, {TOTAL_ROWS})")
    print(f"已保存元数据：{metadata_path}")


if __name__ == "__main__":
    main()
