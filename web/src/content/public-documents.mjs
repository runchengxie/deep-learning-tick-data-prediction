// Explicit allowlist copied from the former public documentation navigation.
// Internal agent plans, private experiment artifacts, and source PDFs are excluded.
export const publicDocuments = [
  'docs/index.md',
  'docs/documentation-index.md',
  'docs/project-status.md',
  'docs/architecture/data-boundary.md',
  'docs/architecture/repository-boundaries.md',
  'docs/model-catalog.md',
  'docs/nextday/cross-sectional-prediction.md',
  'docs/nextday/eventstream.md',
  'docs/nextday/h5-rolling-eventstream-roadmap.md',
  'docs/nextday/hardware-constraints-and-experiment-roadmap.md',
  'docs/nextday/multi-horizon-data-expansion-roadmap.md',
  'docs/nextday/nextday-100m-raw1000-benchmark.md',
  'docs/nextday/raw-200-end-to-end-pipeline.md',
  'docs/nextday/raw-data-expansion-roadmap.md',
  'docs/research/topk-agentx-research-roadmap.md',
  'docs/research/experiment-log.md',
  'docs/research/eventstream-gradient-audit.md',
  'docs/research/eventstream-label-scale.md',
  'docs/research/eventstream-signal-trading-diagnostics.md',
  'docs/research/external-l2-research-comparison.md',
  'docs/research/historical-data-eligibility-2026-08-27.md',
  'docs/research/m3-eventstream-representation.md',
  'docs/research/opening-coverage-inventory-2026-08-27.md',
  'docs/research/resource-strategy-and-pilot-gates.md',
  'docs/research/shanghai-opening-contract-audit-2026-08-27.md',
  'docs/research/topk-agentx-m0-research-contract.md',
  'docs/research/topk-agentx-m1-portfolio-evaluator.md',
  'docs/research/topk-agentx-m2a-deterministic-loop.md',
  'docs/research/topk-agentx-m2b-locked-approval.md',
  'docs/research/topk-agentx-m2c-executors-comparison.md',
  'docs/research/topk-agentx-m2d-registry-context.md',
  'docs/research/topk-agentx-m3-topk-diagnostics.md',
  'docs/dev/development-guide.md',
  'docs/dev/colab-cli-automation.md',
  'docs/dev/historical-colab-snapshots.md',
  'docs/operations/development-guide.md',
  'docs/operations/systemd-workflows.md',
  'docs/reproduction-audit.md',
  'docs/reports/multi-horizon-decision-2026-08-10/source-inspection.md',
  'docs/references/README.md',
  'docs/references/agentx-paper-notes.md',
  'docs/references/debang-minute-gru-notes.md',
  'docs/references/deeplob-paper-notes.md',
];

export function documentSlug(source) {
  if (source === 'docs/index.md') return 'documentation';
  return source.slice('docs/'.length).replace(/\.md$/i, '');
}

export function documentRoute(source) {
  return `/${documentSlug(source)}/`;
}
