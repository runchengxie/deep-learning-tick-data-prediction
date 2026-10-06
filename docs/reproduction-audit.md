# DeepLOB Reproduction Audit

The implementation audited here is maintained in `src/ticknet/fi2010/` as a separate paper-reproduction research track. Its results do not participate in A-share next-day prediction evidence.

## Conclusion

The current code has been updated from an early lightweight implementation to match the paper architecture and FI-2010's official split protocol. Synthetic-data tests cover the model, label columns, training parameters, and data selection. The repository does not contain results from a real training run, so the experiment can run but its numerical reproduction status remains unverified.

## Model architecture

| Component | Paper setting | Current implementation |
|---|---|---|
| Input | Most recent 100 states, each with 40 raw LOB features | `B × 1 × 100 × 40` |
| Convolution channels | 16 | 16 |
| Spatial convolutions | `1 × 2`, `1 × 2`, `1 × 10` | Implemented |
| Temporal convolutions | Two `4 × 1` convolutions per block | Implemented |
| Activation and normalization | Leaky ReLU and Batch Normalization | Implemented |
| Inception | Three branches with 32 channels each | 96 channels after concatenation |
| Temporal module | 64-unit LSTM | Implemented |
| Output | Three-class softmax | Returns logits; cross-entropy computes softmax internally |
| Parameter count | About 60k | Automated test constrains it to 55k–70k |

Returning logits is the more stable PyTorch training interface. `CrossEntropyLoss` internally applies `log_softmax` and negative log likelihood, giving the same optimization objective as the paper.

## FI-2010 features and labels

Each official FI-2010 text file has 149 rows. DeepLOB uses only the first 40 rows, containing order-book prices and volumes. The 104 handcrafted features remain in the converted NPY files, but the dataset does not include them in model windows.

The five label columns are:

| `k` | Column index |
|---:|---:|
| 10 | 144 |
| 20 | 145 |
| 30 | 146 |
| 50 | 147 |
| 100 | 148 |

Original labels 1, 2, and 3 are mapped to 0, 1, and 2. The dataset rejects non-integer labels, unknown labels, an incorrect number of columns, and arrays that are not `float32`.

## Experiment protocol

### Setup 1

`CF_1` through `CF_9` are anchored forward splits defined by the FI-2010 publisher. The implementation trains separately for each fold:

```text
Train_CF_1 -> Test_CF_1
Train_CF_2 -> Test_CF_2
...
Train_CF_9 -> Test_CF_9
```

Each Training file contains the history for that fold. Combining the other eight Training files would duplicate samples and could place test-period observations in training data. The implementation has removed that earlier path.

### Setup 2

The implementation uses:

```text
Training: Train_CF_7
Testing:  Test_CF_7 + Test_CF_8 + Test_CF_9
```

The model trains once, and the three Testing files form the test set. Sliding windows are created separately for each file and never cross file boundaries.

## Training process

| Setting | Current value |
|---|---|
| Optimizer | Adam |
| Learning rate | `0.01` |
| Epsilon | `1.0` |
| Batch size | `32` |
| Early stopping | Stop when validation accuracy does not improve for 20 epochs |
| Validation fraction | Last 20% of the Training file |
| Random seed | `0` by default; configurable |
| Main metrics | Accuracy, macro F1, weighted F1, and per-class precision and recall |

Training and validation windows do not overlap on the original row axis. Checkpoints store both the latest and best states. Training recovery loads the latest state and stops if the protocol, prediction horizon, or training parameters conflict.

## Remaining work

- Run Setup 1 and Setup 2 for all five prediction horizons on official FI-2010 data.
- Summarize per-fold metrics, means, and standard deviations.
- Compare results with Tables I and II of the paper.
- Run multiple random seeds and report variability.
- Add a small real-data validation sample that covers conversion through DataLoader.
- Reproduce the paper's LSE experiment, transfer experiments, trading simulation, and LIME analysis.

## Known limitations

The official FI-2010 matrices do not provide stock or date boundaries inside files. Metadata can record Training and Testing file boundaries. The project can prevent windows from crossing source files, but it cannot verify whether a file contains internal stock or date boundaries that should split windows. Reports must retain this limitation.

GPU kernels, PyTorch versions, and hardware may cause small numerical differences. The project sets common random seeds and CuDNN deterministic options. Every run records Python, NumPy, PyTorch, CUDA, device name, and elapsed time in result JSON. Reports should still include variability across repeated runs.

## Sources

- `docs/references/1808.03668v6.pdf` in this repository
- `docs/references/deeplob-paper-notes.md`
- [Authors' public PyTorch implementation](https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books)
- [FI-2010 dataset page](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649)
