"""数据集测试，用合成的 149 列 .npy 模拟官方 FI-2010 布局，无需真实数据。"""

import numpy as np

from src.dataset import K_TO_LABEL_COLUMN, NUM_FEATURES, WINDOW_SIZE, FI2010WindowDataset


def _fake_fi2010(n: int = 500, seed: int = 1):
    rng = np.random.default_rng(seed)
    fake = np.zeros((n, 149), dtype=np.float32)
    fake[:, :NUM_FEATURES] = rng.standard_normal((n, NUM_FEATURES)).astype(np.float32)
    # 标签列 144/145 用 1/2/3 编码（FI-2010 约定），后面会归一化成 0/1/2
    fake[:, 144] = rng.integers(1, 4, n).astype(np.float32)
    fake[:, 145] = rng.integers(1, 4, n).astype(np.float32)
    return fake


def test_window_shape_and_label_range(tmp_path):
    path = tmp_path / "fake.npy"
    np.save(path, _fake_fi2010())

    for k in (10, 20):
        ds = FI2010WindowDataset(str(path), k=k, window_size=WINDOW_SIZE, split="train")
        x, y = ds[0]
        assert x.shape == (1, WINDOW_SIZE, NUM_FEATURES)
        assert y in (0, 1, 2)


def test_k_selects_different_label_column(tmp_path):
    path = tmp_path / "fake.npy"
    np.save(path, _fake_fi2010())

    ds10 = FI2010WindowDataset(str(path), k=10, window_size=WINDOW_SIZE, split="train")
    ds20 = FI2010WindowDataset(str(path), k=20, window_size=WINDOW_SIZE, split="train")
    x10, y10 = ds10[0]
    x20, y20 = ds20[0]
    # 两个 k 选的是不同标签列，且取出的标签都在归一化后的 {0,1,2} 内
    assert ds10.label_col == K_TO_LABEL_COLUMN[10]
    assert ds20.label_col == K_TO_LABEL_COLUMN[20]
    assert y10 in (0, 1, 2)
    assert y20 in (0, 1, 2)


def test_train_split_length(tmp_path):
    path = tmp_path / "fake.npy"
    np.save(path, _fake_fi2010(n=500))
    ds = FI2010WindowDataset(str(path), k=10, window_size=WINDOW_SIZE, split="train")
    tr_rows = int(500 * 0.7)
    assert len(ds) == tr_rows - WINDOW_SIZE + 1
