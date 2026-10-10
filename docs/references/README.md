# Papers and Reading Notes

This directory preserves papers used by the project and reading notes derived from them. Some sources relate to completed or discontinued research stages. For the current implementation and experiment status, use [Project status](../project-status.md).

## DeepLOB

- Title: DeepLOB: Deep Convolutional Neural Networks for Limit Order Books
- Authors: Zihao Zhang, Stefan Zohren, and Stephen Roberts
- Affiliation: Department of Engineering Science and Oxford-Man Institute of Quantitative Finance, University of Oxford
- Paper: [arXiv:1808.03668](https://arxiv.org/abs/1808.03668)
- Notes: [deeplob-paper-notes.md](deeplob-paper-notes.md)

The paper combines CNN, Inception, and LSTM layers to extract spatial and temporal features from raw limit-order-book prices and volumes. Its experiments cover FI-2010, London Stock Exchange data, cross-stock generalization, a simple trading simulation, and LIME explanations.

The implementation-to-paper comparison is in the [reproduction audit](../reproduction-audit.md). The reading notes explain the paper and do not replace the current implementation guide.

## AgentX

- Title: AgentX: Towards Agent-Driven Self-Iteration of Industrial Recommender Systems
- Authors: AgentX Team, Kuaishou
- Paper: [arXiv:2606.26859](https://arxiv.org/abs/2606.26859)
- Notes: [agentx-paper-notes.md](agentx-paper-notes.md)

The paper proposes a production multi-agent loop for recommender-system development. A Brainstorm Agent creates evidence-backed proposals, a Developing Agent turns proposals into production code, an Evaluation Agent runs guardrail-based A/B decisions and records negative findings, and SGPO uses execution traces to improve the agents. The paper reports that a three-week deployment produced 10 launchable results from 374 ideas, with substantial throughput and online-metric improvements.

This project draws on the closed-loop research idea for automated quantitative research. The implementation discussion is at the end of `agentx-paper-notes.md`.

## Debang Securities

- Source title: <!-- preserved-source:start -->基于分钟数据的 GRU 模型在选股策略中的应用初探<!-- preserved-source:end -->
- Series: Debang Securities quantitative research, Machine Learning Series No. 6
- Source document: cited in the reading notes; the source PDF is not included in this public repository
- Notes: [debang-minute-gru-notes.md](debang-minute-gru-notes.md)

The report studies a GRU that uses minute-level price and volume sequences for cross-sectional stock selection. It is related to this project's minute-data work in `nextday`, including `minute_baseline`, `minute_tcn`, and `minute_gru`: first test whether minute features contain next-day information, then compare deep sequence models with tree baselines out of sample and account for net returns after costs.

The PDF is not redistributed through this public repository. The notes summarize the report in our own words and distinguish its findings from this project's evidence.

## Files

| File | Description |
|---|---|
| `1808.03668v6.pdf` | DeepLOB arXiv v6 paper |
| `deeplob-paper-notes.md` | Notes organized by the DeepLOB paper's sections |
| `2606.26859v2.pdf` | AgentX arXiv v2 paper |
| `agentx-paper-notes.md` | Notes organized by the AgentX paper's sections |
| Source PDF | Not included because redistribution rights have not been established |
| `debang-minute-gru-notes.md` | English reading notes for the minute-data GRU report |
| `README.md` | This index |
