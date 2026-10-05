# Historical systemd Workflow Record

> This page describes TickNet workflows that existed in the user's systemd configuration before cleanup on 2026-09-02. Their units have been deleted. This is a maintenance and migration record, not a recommendation to restore those jobs.

## Scope

These jobs were managed by the user's systemd instance, with unit files under `~/.config/systemd/user/`. They referenced an older project checkout under `CODE_ROOT` and a former `research-workspace` checkout. Large data and run artifacts lived under `DATA_ROOT`. Within the project checkout, `artifacts/`, `data/`, `results/`, `logs/`, `checkpoints*/`, and `.venv` were local symlinks to the external data area.

## Historical unit inventory

### Data preparation, packing, and audits

| Unit | Former purpose |
|---|---|
| `ticknet-eventstream-202101-top400.service` | Prepare and pack January 2021 Top-400 event-stream data |
| `ticknet-eventstream-202102-202105-top400.service` | Prepare and pack February through May 2021 Top-400 event-stream data |
| `ticknet-eventstream-202508-top400.service` | Pack August 2025 Top-400 event-stream data |
| `ticknet-eventstream-202509-202512-top400.service` | Pack September through December 2025 Top-400 event-stream data |
| `ticknet-eventstream-202101-audit.service` | Check and audit January 2021 event-stream data |
| `ticknet-eventstream-202101-audit.path` | Watch audit-related paths and trigger the audit service |
| `ticknet-eventstream-top400-preflight.service` | Run preflight checks for Top-400 event-stream jobs |

### Benchmarks, scans, and orchestration

| Unit | Former purpose |
|---|---|
| `ticknet-eventstream-h5-a100-benchmark.service` | A100 benchmark for H5 event-stream training |
| `ticknet-eventstream-h5-recent-a100-benchmark.service` | A100 benchmark on the recent dataset |
| `ticknet-eventstream-h5-recent-a100-sweep.service` | Batch-size sweep on the recent dataset using A100 |
| `ticknet-eventstream-h5-recent-chain.service` | Wait for upstream jobs, then start downstream jobs in sequence |

### Uploads

| Unit | Former purpose |
|---|---|
| `ticknet-eventstream-h5-benchmark-upload.service` | Upload benchmark artifacts to remote storage |
| `ticknet-eventstream-h5-recent-upload.service` | Upload recent dataset packs, labels, and manifests |

## Cleanup record

Before cleanup:

- No `ticknet-*` user service was active.
- Every service was `static` and none was enabled.
- `ticknet-eventstream-202101-audit.path` was disabled.
- No process associated with a `.pid` file in the old project checkout was running.

The listed `ticknet-*.service` and `ticknet-*.path` units and this project's leftover `.pid` files were then removed. The following command reloaded the unit state:

```bash
systemctl --user daemon-reload
```

Logs and experiment artifacts were not removed during systemd cleanup. They remain in the external data area.

## Future convention

- Do not register experiment or data-processing jobs as persistent systemd user services by default.
- Before restoring automation, specify inputs, outputs, retry behavior, resource use, and shutdown behavior, then design a separate scheduler configuration.
- Run jobs from a stable project release path. Keep data, checkpoints, logs, and artifacts outside the repository.
- When cleaning up old jobs, check `~/.config/systemd/user/`, the user's crontab, PID files, and old path references in project documentation.
