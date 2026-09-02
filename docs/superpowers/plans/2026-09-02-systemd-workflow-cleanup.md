# Systemd Workflow Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Document the historical ticknet systemd workflows and remove their obsolete user-level service configuration.

**Architecture:** Add a self-contained operations note under the deep-learning project's `docs/` directory. Verify all matching user units are inactive, remove the unit files and stale PID files, then verify no matching configuration remains while preserving the project's current data and code layout.

**Tech Stack:** Markdown, systemd user units, shell verification, Git.

**Spec:** User request in the conversation: document the existing systemd workflows in the deep-learning project's `docs/` folder and clean up the systemd configuration.

## Global Constraints

- Do not modify tracked source code or Git history.
- Do not delete the deep-learning data stored under `/home/richard/data/deep-learning-tick-data-prediction/`.
- Do not remove unrelated `production/` release worktrees.
- Only remove the `ticknet-*` user units and their obsolete local PID files after confirming they are inactive.

---

### Task 1: Document the historical workflows

**Files:**
- Create: `docs/operations/systemd-workflows.md`

- [x] **Step 1: Record the unit inventory and purpose**

Document the 12 `ticknet-*.service` units and the `ticknet-eventstream-202101-audit.path` watcher, grouped into eventstream packing/audit, benchmark/sweep, upload, and chained orchestration. State that the units referenced the former checkout path and that the current inspection found no active instances.

- [x] **Step 2: Record the cleanup and replacement guidance**

Document that the workflows are historical and that future recurring jobs should use an explicitly reviewed scheduler or manual invocation. Include the current canonical code path and external data path.

### Task 2: Remove obsolete systemd configuration

**Files:**
- Delete: `~/.config/systemd/user/ticknet-eventstream-202101-audit.path`
- Delete: `~/.config/systemd/user/ticknet-eventstream-202101-audit.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-202101-top400.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-202102-202105-top400.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-202508-top400.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-202509-202512-top400.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-h5-a100-benchmark.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-h5-benchmark-upload.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-h5-recent-a100-benchmark.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-h5-recent-a100-sweep.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-h5-recent-chain.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-h5-recent-upload.service`
- Delete: `~/.config/systemd/user/ticknet-eventstream-top400-preflight.service`
- Delete: stale PID files `prepare-nextday-raw-200-preflight.pid`, `prepare-nextday-raw-200.pid`, `upload-nextday-raw-200.pid`, and `upload-nextday-raw-200.watcher.pid` under `/home/richard/data/deep-learning-tick-data-prediction/logs/`

- [x] **Step 1: Verify units are inactive and not enabled**

Run `systemctl --user list-units --all 'ticknet-*'` and `systemctl --user list-unit-files 'ticknet-*'`. The pre-cleanup result contained no active units. All 12 services were `static` and the path unit was `disabled`.

- [x] **Step 2: Remove the matching unit files and stale PID files**

Remove only the 12 service files, the single path file, and the four listed PID files. Each recorded PID had no live process at cleanup time.

- [x] **Step 3: Reload the user manager**

Run `systemctl --user daemon-reload` so the removed unit definitions are no longer visible to the user manager. This was executed after deletion.

### Task 3: Verify the result

**Files:**
- Verify: `docs/operations/systemd-workflows.md`

- [x] **Step 1: Confirm no matching files or active units remain**

Run `find ~/.config/systemd/user -maxdepth 1 -name 'ticknet-*'`, `systemctl --user list-units --all 'ticknet-*'`, and `systemctl --user list-unit-files 'ticknet-*'`. All three checks returned no matching entries after reload.

- [x] **Step 2: Confirm Git and path integrity**

Run `git status --short --branch` in the deep-learning submodule and the parent workspace, verify the canonical checkout and external data links resolve, and confirm no source files were changed. The canonical checkout resolves to the workspace submodule, external artifacts resolve to `/home/richard/data/deep-learning-tick-data-prediction/artifacts`, and the only tracked changes are these documentation files.
