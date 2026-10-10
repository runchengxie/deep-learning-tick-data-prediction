const text = (en, zh) => ({ en, zh });

export const studyReports = {
  'event-stream-ranking': {
    sources: [
      { path: 'docs/research/topk-agentx-research-roadmap.md', label: text('Research roadmap', '研究路线图') },
      { path: 'docs/project-status.md', label: text('Project status and current decision', '项目状态与当前结论') },
      { path: 'docs/research/experiment-log.md', label: text('Dated experiment results', '分日期实验记录') },
      { path: 'docs/research/eventstream-label-scale.md', label: text('Label-scale study', '标签尺度研究') },
    ],
    sections: [
      {
        title: text('Question and model', '研究问题与模型'),
        paragraphs: [
          text(
            'Can a model learn a useful next-day stock ranking from the order, trade, and order-book events recorded during a trading day? The event-stream model reads these records in event order and uses a causal Transformer that attends only to current and earlier sequence positions. Information availability also depends on message publication times, ordering, and feature construction.',
            '模型能否从交易日内的委托、成交和盘口快照事件中学到有用的次日选股排序信号？事件流模型按发生顺序读取这些记录，并使用因果 Transformer，使注意力只读取当前及之前的序列位置。数据是否当时可用，还要核对发布时间、排序规则和特征计算。',
          ),
          text(
            'The 100M-parameter model receives windows of 512 merged events and 80 features per event. It learns several event-level prediction tasks alongside an H5 daily return signal. The shared event representation can also be exported and tested with downstream rankers.',
            '1 亿参数模型每次读取 512 个合并后的事件，每个事件包含 80 项特征。模型同时学习多个事件级预测任务和 H5 收益信号（本实验定义的五交易日目标，起止时点见来源记录）。学到的事件表示也可以导出，交给下游排序模型评估。',
          ),
        ],
        pipeline: {
          title: text('From event records to a daily score', '从事件记录到每日评分'),
          steps: [
            text('Orders, trades, and book snapshots', '委托、成交与盘口快照'),
            text('Merge in event-time order', '按事件时间合并'),
            text('Causal Transformer', '因果 Transformer'),
            text('H5 stock ranking', 'H5 股票排序'),
            text('Portfolio and cost evaluation', '组合与成本评估'),
          ],
        },
      },
      {
        title: text('Time split and main result', '时间切分与主要结果'),
        paragraphs: [
          text(
            'The three-seed run trained on August–October 2025, selected checkpoints on November validation data, and evaluated once on December 2025. Each seed used 120,000 fixed training windows; the validation and out-of-sample sets contained 5,807 and 6,966 stock-days.',
            '三随机种子实验使用 2025 年 8–10 月训练，在 11 月验证集上选择检查点，并在 2025 年 12 月进行样本外评估。每个种子使用 120,000 个固定训练窗口；验证集和样本外集分别包含 5,807 和 6,966 个股票日。',
          ),
          text(
            'All three December Rank IC values were positive. The mean was 0.04300, compared with a validation mean of 0.07259. The lower out-of-sample mean and the variation across seeds are reasons to treat this as an early signal, not a stable performance estimate.',
            '三个种子的 12 月 Rank IC 均为正，均值为 0.04300；验证集均值为 0.07259。样本外均值更低，种子间也有差异，因此这只能视为初步信号，不能当作稳定表现的估计。',
          ),
        ],
        chart: {
          kind: 'grouped', min: 0, max: 0.1,
          title: text('Rank IC by seed and period', '各随机种子在不同区间的 Rank IC'),
          description: text('The out-of-sample period is December 2025. Every seed is positive, but the test covers one month.', '样本外区间为 2025 年 12 月。三个种子结果均为正，但测试只覆盖一个月。'),
          axisMin: text('0', '0'), axisMax: text('0.10', '0.10'),
          categories: [
            { label: text('Seed 0', '种子 0'), values: [{ value: 0.04345, display: '0.04345' }, { value: 0.05879, display: '0.05879' }] },
            { label: text('Seed 1', '种子 1'), values: [{ value: 0.09403, display: '0.09403' }, { value: 0.03730, display: '0.03730' }] },
            { label: text('Seed 2', '种子 2'), values: [{ value: 0.08029, display: '0.08029' }, { value: 0.03291, display: '0.03291' }] },
          ],
          series: [text('Validation · Nov 2025', '验证集 · 2025 年 11 月'), text('Out of sample · Dec 2025', '样本外 · 2025 年 12 月')],
        },
      },
      {
        title: text('A stronger ranking score did not pass the trading check', '排序指标较好，但交易检验仍未通过'),
        paragraphs: [
          text(
            'A separate joint-model evaluation combined the event representation with minute features. Its December OOS Rank IC averaged 0.06398 across three seeds, while daily net active return after costs averaged −9.67 bp. All three seeds had negative net active return.',
            '另一项联合模型评估将事件表示与分钟特征结合。三个种子的 12 月样本外 Rank IC 均值为 0.06398，但计入成本后的日均净超额收益均值为 −9.67 个基点，三个种子的净超额收益都为负。',
          ),
          text(
            'The chart reports active return relative to the evaluation benchmark, not absolute portfolio return. These results show why Rank IC and portfolio performance need separate checks.',
            '图中的超额收益是相对该评估基准的表现，不是组合的绝对收益。这说明 Rank IC 与组合表现需要分别检验。',
          ),
        ],
        chart: {
          kind: 'diverging', min: -16, max: 0,
          title: text('Joint-model daily net active return by seed', '联合模型各随机种子的日均净超额收益'),
          description: text('December 2025 OOS; basis points per day after costs. Negative values mean the portfolio underperformed its benchmark.', '2025 年 12 月样本外结果；计入成本后的日均基点数。负值表示组合表现低于评估基准。'),
          axisMin: text('−16 bp', '−16 个基点'), axisMax: text('0 bp', '0 个基点'),
          categories: [
            { label: text('Seed 0', '种子 0'), values: [{ value: -9.26, display: '−9.26 bp' }] },
            { label: text('Seed 1', '种子 1'), values: [{ value: -14.40, display: '−14.40 bp' }] },
            { label: text('Seed 2', '种子 2'), values: [{ value: -5.34, display: '−5.34 bp' }] },
          ],
          series: [text('Net active return', '净超额收益')],
        },
        callout: {
          title: text('Interpretation', '如何解读'),
          body: text(
            'The event-stream model produced a repeatable positive ranking metric in this month. The joint-model portfolio still fell short of its benchmark after costs. More time periods and improvements to the trading objective are needed before claiming durable trading value.',
            '事件流模型在这个月呈现出重复的正向排序指标，但联合模型组合计入成本后仍未跑赢基准。要判断交易价值能否持续，还需要更多时间区间的结果，并改进交易目标。',
          ),
        },
      },
      {
        title: text('What remains uncertain', '仍待回答的问题'),
        paragraphs: [
          text(
            'The three-seed result covers one recent out-of-sample month. A separate adjacent-fold run has one seed and one test month; its H5 Rank IC was positive, but its OOS extreme-group return spread was negative. Neither result establishes persistence across market regimes.',
            '三随机种子结果只覆盖一个近期样本外月份。另一相邻时间区间目前只有一个种子和一个测试月；该实验的 H5 Rank IC 为正，但样本外极端分组收益差为负。两项结果都不能证明信号能跨市场阶段持续。',
          ),
          text(
            'Daily-task weighting is the next planned model ablation. Capacity expansion to the 150M model remains paused until ranking improvements also show up in cost-adjusted portfolio results across more than one window.',
            '下一步计划测试日级任务权重。150M 参数模型扩容实验仍暂停，直到排序指标的改善也能在多个时间区间的成本后组合结果中体现出来。',
          ),
        ],
      },
    ],
  },
  'raw-order-book-capacity': {
    sources: [
      { path: 'docs/research/experiment-log.md', label: text('Four-setting capacity and window experiment', '四种模型规模与输入窗口实验') },
      { path: 'docs/nextday/cross-sectional-prediction.md', label: text('Prediction inputs, labels, and evaluation', '预测输入、标签与评估方法') },
      { path: 'docs/project-status.md', label: text('Current project status', '当前项目状态') },
    ],
    sections: [
      {
        title: text('What goes into the model', '模型读取哪些数据'),
        paragraphs: [
          text(
            'For each stock and trading day, the model reads the final 200 or 1,000 valid ten-level order-book snapshots before the signal time. The selected 200-snapshot candidate splits its input into two 100-snapshot chunks. A shared DeepLOB encoder extracts local patterns, and a GRU combines the chunks into a next-day open-to-close excess-return score.',
            '模型按股票和交易日取样，在信号时点前读取最后 200 或 1,000 条有效的十档盘口快照。当前保留的 200 条快照候选方案将输入分成两个各含 100 条快照的片段，由共享 DeepLOB 编码器提取局部模式，再由 GRU 汇总为次日开盘至收盘的超额收益评分。',
          ),
          text(
            'The controlled comparison used a Top-100 stock universe and the same next-day excess-return target in all four settings. Training dates were 2021–2023, validation was 2024, and 2025 remained locked.',
            '受控对比在四种配置中都使用 Top-100 股票池和相同的次日超额收益目标。训练区间为 2021–2023 年，验证区间为 2024 年，2025 年测试集保持锁定。',
          ),
        ],
        pipeline: {
          title: text('How a raw-book sample becomes a prediction', '原始盘口样本如何变成预测'),
          steps: [
            text('Last 200 book snapshots', '最后 200 条盘口快照'),
            text('Two 100-snapshot chunks', '两个 100 条快照片段'),
            text('Shared DeepLOB encoder', '共享 DeepLOB 编码器'),
            text('GRU sequence summary', 'GRU 汇总序列'),
            text('Next-day return score', '次日收益评分'),
          ],
        },
      },
      {
        title: text('The controlled comparison', '受控对比'),
        paragraphs: [
          text(
            'The experiment varied model capacity and input length while holding the universe, dates, target, batch size, learning rate, and checkpoint rule fixed. Each of the four settings used three random seeds and the same 241 validation days.',
            '实验只改变模型规模和输入长度，股票池、日期、目标、批次大小、学习率和检查点选择规则保持一致。四种配置各使用三个随机种子，并在相同的 241 个验证日上评估。',
          ),
          text(
            'Bars show the three-seed mean validation Rank IC. Whiskers show one sample standard deviation across seeds. This spread describes seed-to-seed variation; it is not a confidence interval.',
            '柱表示三个随机种子的验证集 Rank IC 均值，误差线表示种子间一个样本标准差。这个范围描述种子间差异，不是置信区间。',
          ),
        ],
        chart: {
          kind: 'grouped', min: 0, max: 0.045,
          title: text('Validation Rank IC · mean ± sample SD', '验证集 Rank IC · 均值 ± 样本标准差'),
          description: text('Each cell uses three seeds. The test period was not opened.', '每种配置使用三个随机种子；测试集尚未开启。'),
          axisMin: text('0', '0'), axisMax: text('0.045', '0.045'),
          categories: [
            { label: text('200-snapshot window', '200 条快照窗口'), values: [{ value: 0.03748, error: 0.00096, display: '0.03748 ± 0.00096' }, { value: 0.02740, error: 0.00412, display: '0.02740 ± 0.00412' }] },
            { label: text('1,000-snapshot window', '1,000 条快照窗口'), values: [{ value: 0.03530, error: 0.00241, display: '0.03530 ± 0.00241' }, { value: 0.03152, error: 0.00287, display: '0.03152 ± 0.00287' }] },
          ],
          series: [text('1M parameters', '1M 参数'), text('100M parameters', '100M 参数')],
        },
      },
      {
        title: text('What the comparison supports', '这组对比说明了什么'),
        paragraphs: [
          text(
            '`1M / raw-200` had the highest mean Rank IC and the smallest seed variation. Increasing capacity to 100M lowered the mean at both window lengths. Extending the window from 200 to 1,000 snapshots had only a small average effect, with the direction depending on model size.',
            '`1M / raw-200` 的 Rank IC 均值最高，种子间差异也最小。模型规模增至 100M 后，两种输入长度下的均值都下降。窗口从 200 延长至 1,000 条快照的平均影响较小，而且效果方向随模型规模而变。',
          ),
          text(
            'The observed capacity effect was −0.00693 on average, while the window effect was +0.00097. The interaction was +0.00631. With three seeds and validation-only selection, these are descriptive comparisons—not significance or out-of-sample results.',
            '观察到的模型规模主效应均值为 −0.00693，窗口主效应为 +0.00097，交互效应为 +0.00631。由于每种配置只有三个种子，且选择依据是验证集，这些数值只描述本次对比，不代表统计显著或样本外结果。',
          ),
        ],
        chart: {
          kind: 'diverging', min: -0.012, max: 0.012,
          title: text('Average effects in the 2 × 2 comparison', '2 × 2 对比中的平均效应'),
          description: text('Rank IC change; zero marks no average change. The capacity effect is 100M minus 1M; the window effect is raw-1000 minus raw-200.', 'Rank IC 变化；零表示平均无变化。模型规模效应为 100M 减 1M，窗口效应为 raw-1000 减 raw-200。'),
          axisMin: text('−0.012', '−0.012'), axisMax: text('+0.012', '+0.012'),
          categories: [
            { label: text('Capacity effect', '模型规模效应'), values: [{ value: -0.00693, display: '−0.00693' }] },
            { label: text('Window effect', '输入窗口效应'), values: [{ value: 0.00097, display: '+0.00097' }] },
            { label: text('Interaction', '交互效应'), values: [{ value: 0.00631, display: '+0.00631' }] },
          ],
          series: [text('Validation Rank IC', '验证集 Rank IC')],
        },
        callout: {
          title: text('Current status', '当前状态'),
          body: text(
            'Keep `1M / raw-200` as the sole candidate. Its 2025 locked test has not been opened, so the comparison does not establish generalization or trading performance.',
            '当前只保留 `1M / raw-200` 作为候选方案。2025 年锁定测试集尚未开启，因此这组对比还不能证明泛化能力或交易表现。',
          ),
        },
      },
    ],
  },
  'minute-baselines': {
    sources: [
      { path: 'docs/research/topk-agentx-m3-topk-diagnostics.md', label: text('Formal Top-K cost diagnostics', '正式 Top-K 成本诊断') },
      { path: 'docs/research/experiment-log.md', label: text('Dated experiment results', '分日期实验记录') },
      { path: 'docs/project-status.md', label: text('Current project status', '当前项目状态') },
      { path: 'docs/nextday/cross-sectional-prediction.md', label: text('Feature and target definitions', '特征与目标定义') },
    ],
    sections: [
      {
        title: text('A lower-cost model path', '成本较低的模型路线'),
        paragraphs: [
          text(
            'Minute models compress intraday market activity into aggregate features rather than processing every order-book event. The formal HGB model used 120 aggregate features built from order, trade, and snapshot data, then ranked a daily universe of 400 stocks.',
            '分钟模型将日内市场活动压缩为聚合特征，而不是逐条处理盘口事件。正式 HGB 模型使用由委托、成交和快照数据生成的 120 项聚合特征，并在每日 400 只股票的股票池中进行排序。',
          ),
          text(
            'The formal dataset covers July 2021 through December 2025. It contains 436,800 candidate stock-days; 436,256 have complete features (99.88%). The final six-month period includes 124 evaluation days and 49,600 candidates.',
            '正式数据集覆盖 2021 年 7 月至 2025 年 12 月，共有 436,800 个候选股票日，其中 436,256 个特征完整（99.88%）。最后六个月包含 124 个评估日和 49,600 个候选样本。',
          ),
        ],
        pipeline: {
          title: text('From minute features to cost checks', '从分钟特征到成本检验'),
          steps: [
            text('Published L2 minute data', '已发布的 L2 分钟数据'),
            text('120 aggregate features', '120 项聚合特征'),
            text('HGB daily ranking', 'HGB 每日排序'),
            text('Top-K portfolio rules', 'Top-K 组合规则'),
            text('Costs and benchmark comparison', '成本与基准对比'),
          ],
        },
      },
      {
        title: text('Ranking held up; the cost gate did not', '排序指标为正，但成本门槛未通过'),
        paragraphs: [
          text(
            'The 2025 H2 formal run had positive monthly Rank IC in all six months and a mean daily Rank IC of 0.06994. The separate annual chart above shows earlier rolling OOS baselines; its 2025 bar is not the same calculation as this formal H2 evaluation.',
            '2025 年下半年的正式评估中，六个月的月度 Rank IC 均为正，日均 Rank IC 为 0.06994。上方年度图展示的是较早的滚动样本外基线；其中 2025 年的柱与本次 H2 正式评估不是同一项计算。',
          ),
          text(
            'The cost matrix tested four portfolio sizes (K=25/50/75/100) and four exit buffers (0/10/25/50) at a 10 bp one-way cost plus 5 bp sell stamp duty. All 16 combinations had negative net active return versus the equal-weight Top-400 benchmark.',
            '成本矩阵测试了四种持仓数量（K=25/50/75/100）和四种卖出缓冲（0/10/25/50），成本设为单边 10 个基点，卖出另计 5 个基点印花税。16 种配置的净超额收益都低于 Top-400 等权基准。',
          ),
        ],
        matrix: {
          title: text('Cost-gate outcome at the 10 bp decision cost', '单边 10 个基点决策成本下的检验结果'),
          description: text('Every tested K/buffer pair had negative net active return after costs. Buffer values are exit-buffer ranks.', '所有 K/缓冲配置计入成本后的净超额收益均为负。缓冲值表示退出排序缓冲。'),
          rowHeader: text('Portfolio size K', '组合持仓数 K'),
          columnHeader: text('Exit buffer', '退出缓冲'),
          ks: [25, 50, 75, 100],
          buffers: [0, 10, 25, 50],
          cellLabel: text('Below benchmark', '低于基准'),
        },
      },
      {
        title: text('The best absolute return still lagged the benchmark', '绝对收益最高的配置仍跑输基准'),
        paragraphs: [
          text(
            'K=100 with a 50-rank exit buffer had the best absolute net return in the tested matrix: 12.15 bp per day, with a net Sharpe ratio of 1.41 and 41.87% one-way daily turnover. Its net active return was still −4.75 bp per day, and only one of the six months was positive.',
            '在测试矩阵中，K=100、退出缓冲为 50 的配置取得最高绝对净收益：日均 12.15 个基点，净 Sharpe 为 1.41，日均单边换手率为 41.87%。但其日均净超额收益仍为 −4.75 个基点，六个月中只有一个月为正。',
          ),
          text(
            "Its active-return break-even cost was about 4.33 bp one way, below the 10 bp decision cost. Positive absolute return partly reflected the broad market's rise and is not standalone evidence of model alpha.",
            '该配置的超额收益单边盈亏平衡成本约为 4.33 个基点，低于 10 个基点的决策成本。绝对收益为正部分来自同期大盘上涨，不能单独作为模型 alpha 的证据。',
          ),
        ],
        metrics: [
          { value: '12.15 bp', label: text('Daily absolute net return · best setting', '日均绝对净收益 · 最佳配置') },
          { value: '−4.75 bp', label: text('Daily net active return vs Top-400', '日均净超额收益 · 相对 Top-400') },
          { value: '4.33 bp', label: text('Active break-even cost, one way', '超额收益单边盈亏平衡成本') },
        ],
        callout: {
          title: text('Decision', '结论'),
          body: text(
            'The formal result is `NO_TRADEABLE_REGION` at the target cost. Minute HGB remains useful as a low-cost ranking baseline, but these experiments did not identify a portfolio setting that passed the pre-set net-return checks.',
            '正式结论为目标成本下 `NO_TRADEABLE_REGION`。分钟 HGB 仍可作为低成本排序基线，但本次实验没有找到通过预设净收益检验的组合配置。',
          ),
        },
      },
    ],
  },
};
