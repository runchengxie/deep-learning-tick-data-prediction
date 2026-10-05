# Event-Stream Label-Scale and Supervision-Position Experiments

`EVT-LABEL-SCALE-001` tested whether daily-return target scale limited the 100M event-stream Transformer's ability to learn H5 cross-sectional signals. Seed 0 completed on the recent and adjacent folds, followed by `EVT-SUPERVISION-POSITION-001`. Final decision: `KEEP_ALL`.

## Label-scale question and method

The gradient audit found that the daily-task gradient at each fold's best checkpoint was only about 2% and 4% of the median gradient from the three generative tasks. This experiment changed only the H5 daily target in training. For each training day with labels, values were winsorized at median ± 5 raw MAD and standardized using the winsorized cross-sectional mean and population standard deviation. Event windows, sample order, model capacity, task weights, validation, OOS, and H3 monitoring targets remained unchanged. Samples without an H5 label at split boundaries remained masked by `day_valid=0`.

## Data and run identity

| Fold | Training | Validation | OOS | Data fingerprint | Z-label fingerprint |
|---|---|---|---|---|---|
| Recent | 2025-08 to 2025-10 | 2025-11 | 2025-12 | `5a7d9216c7b4a8f680ef8a22ca760b482b6ccd38f6a8df587bd7deb44f445314` | `7f8223c0581e08115b18c19e36756e8431e2e2b395231e5a047860eb5ae53832` |
| `fold-54-oos-202511` | 2025-07 to 2025-09 | 2025-10 | 2025-11 | `596daa34cfe2a44ad94f884db95d9ce164fd6aff38e05fad61ce1869cc8e9403` | `faefc4f65fe97bdebd27be3dd77033976beab8fd94ef45ce9f5f80015f149e60` |

Both folds used `capacity100m` (100,604,180 parameters) and seed 0. Training identity used source revision `1b4c0f163d1f0aab2c930468f20eecaafe8b60f3`. The adjacent-fold Colab scheduler used revision `35f90d722e6dacd98cd9d0608d6fa3c3c7737b3e`, while the checkpoint continued to validate against the original experiment revision. Locked 2026 did not enter training or evaluation.

## Label-scale results

| Fold | Training target | Best epoch | Validation Rank IC | OOS Rank IC | Validation extreme-group spread | OOS extreme-group spread |
|---|---|---:|---:|---:|---:|---:|
| Recent | Raw H5 return | 4 | 0.04345 | 0.05879 | -0.38744% | 0.34105% |
| Recent | Daily cross-sectional z label | 4 | 0.11747 | 0.07446 | 1.27275% | 0.08885% |
| `fold-54-oos-202511` | Raw H5 return | 11 | 0.08735 | 0.03305 | 1.78031% | -0.51240% |
| `fold-54-oos-202511` | Daily cross-sectional z label | 8 | 0.13534 | 0.07755 | 2.68055% | 0.48367% |

Validation and OOS Rank IC improved on both folds. Both extreme-group spreads improved on the adjacent fold; on the recent fold, validation changed from negative to positive, but OOS spread declined from 0.34105% to 0.08885%.

H3 was a monitoring target, not a checkpoint-selection metric:

| Fold | Training target | Validation H3 Rank IC | OOS H3 Rank IC |
|---|---|---:|---:|
| Recent | Raw H5 return | 0.04095 | 0.05101 |
| Recent | Daily cross-sectional z label | 0.09230 | 0.05928 |
| `fold-54-oos-202511` | Raw H5 return | 0.04840 | 0.04231 |
| `fold-54-oos-202511` | Daily cross-sectional z label | 0.09066 | 0.07326 |

The preregistered gate required validation, OOS, and extreme-group spread to improve on both folds. Since recent-fold OOS spread declined, seeds 1 and 2 were not run. Z labels improved H5 Rank IC in all four evaluation segments and H3 Rank IC in all four segments, making them the shared target for the next training-mechanism experiment.

## Supervision-position contract

`EVT-SUPERVISION-POSITION-001` keeps the z-label results as the `all` control and adds `last` and `tail_weighted`:

| Mode | Daily loss positions |
|---|---|
| `all` | Every valid position; current implementation and baseline |
| `last` | The final valid position for each sample |
| `tail_weighted` | Every valid position, weighted linearly from sequence start to end |

For a valid sequence of length L, `tail_weighted` uses position weight `(t + 1) / L`, with t starting at 0. Padding receives zero weight, and the loss is normalized by the total valid weight in the batch. Generative task losses, daily-task weight, optimizer, data, and selection metric remain fixed. `day_supervision_mode` and the tail-weight version are part of checkpoint identity; each mode uses its own output directory.

The trainer accepts `--day-supervision-mode all|last|tail_weighted`. The Colab runner allows new modes only for label-scale workflows and records the mode in the run summary. Checkpoint names, local directories, and Drive directories for `last` and `tail_weighted` are isolated. First-round runs keep OOS closed:

```bash
python scripts/run_colab_nextday.py \
  --workflow eventstream-recent-label-scale-train \
  --day-supervision-mode last \
  --session ticknet-supervision-recent-last-seed0 \
  --gpu A100 --seeds 0 --no-evaluate-test --keep-on-failure \
  --local-output-dir artifacts/eventstream-supervision-position/recent-last-seed0
```

For the adjacent fold, use `eventstream-rolling-label-scale-train` and add `--eventstream-fold-id fold-54-oos-202511`. Change the mode to `tail_weighted` to run the linear-tail control.

Do not retrain the existing `all` z-label baseline. Run `last` and `tail_weighted` with seed 0 on both folds, keeping OOS closed via `--no-evaluate-test`. A candidate must meet all validation gates:

1. Rank IC exceeds the same-fold `all` baseline on both folds, with mean gain at least 0.005.
2. Extreme-group spread is no lower than the `all` baseline on either fold and remains positive.
3. If both candidates pass, choose the higher mean Rank IC gain. If their gains differ by less than 0.002, choose `tail_weighted`.

Only the selected candidate may access OOS on both folds. Add seeds 1 and 2 only if its OOS Rank IC and extreme-group spread both exceed the same-fold `all` baseline. If neither mode passes, inspect daily-task weight and a cost-aware ranking objective.

## Supervision-position results

`last` and `tail_weighted` used seed 0, daily cross-sectional z labels, and recent-fold data. Model capacity, generative tasks, daily-task weight, optimizer, selection metric, and evaluation samples were fixed. Both runs used revision `41290ff056fb318d37ce44ba89bcbf31453c07f3` and the same data and z-label fingerprints as the `all` baseline.

| Mode | Best epoch | Validation Rank IC | Change vs `all` | Validation extreme-group spread | Change vs `all` |
|---|---:|---:|---:|---:|---:|
| `all` | 4 | 0.11747 | Baseline | 1.27275% | Baseline |
| `last` | 7 | 0.07802 | -0.03945 | 0.92692% | -0.34583 pp |
| `tail_weighted` | 7 | 0.11289 | -0.00458 | 1.10797% | -0.16478 pp |

`last` substantially reduced both Rank IC and extreme-group spread. `tail_weighted` stayed closer to `all` but still scored lower on both primary measures. H3 monitoring Rank IC was 0.06055 and 0.08667, respectively, below `all` at 0.09230.

Both candidates failed the necessary preregistered requirement to exceed `all` on the recent fold. Adjacent-fold results could not change that decision, so the experiment stopped early without running the adjacent fold or opening any OOS. The stop used validation only and did not change the selection rule.

Final decision: `KEEP_ALL`. Continue applying daily-label loss at every valid position. Do not add supervision-position seeds 1 and 2 or start `probe150m`. Next, review the daily task's weight in the combined loss and then decide whether to implement a ranking objective closer to stock selection and trading costs.

## Artifacts

Local formal results:

- `artifacts/eventstream-label-scale/recent-seed0/training`
- `artifacts/eventstream-label-scale/fold54-seed0/training`
- `artifacts/eventstream-supervision-position/recent-last-seed0`
- `artifacts/eventstream-supervision-position/recent-tail-weighted-seed0`

Drive uses the corresponding `eventstream-top400-h5-capacity100m-*-label-z/training`, `eventstream-top400-h5-capacity100m-recent-label-z-day-last/training`, and `eventstream-top400-h5-capacity100m-recent-label-z-day-tail-weighted/training` paths. Each contains best/last checkpoints, training history, results, preflight report, and Colab run summary.

`last` best, last, and result JSON SHA-256 values are `6148eb9d4b8d83134c625e7af0570069f733e852ffd5bbc33dddcb7aecf26b5b`, `eb14826d9080cff3c466334735fc584adb921af17faa66c93a0b382fe7050f7e`, and `28d1420ae275ee28c02d255afa03fbfd5bb6495d2710363bb34adea7622a0c04`. `tail_weighted` values are `3f6a4a5956631d85f74aae43ea7a3b13a61015bcca8e86306ca8879418a0a819`, `b74b141603667680afe8717e874b5a2fd0f08f3025eb6e4cf0e37028b250b2e2`, and `b463d172fc470abf63eb3dd4fedd6c41c3f01d219925e3786069fe9ffb5e4fdd`. Each Drive directory has six formal files matching local names and sizes.
