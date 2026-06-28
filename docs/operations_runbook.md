# 选股策略运维手册（持续跟踪用）

> 目标：找出未来 2 个交易日累计涨幅 ≥ 3% 概率最高的 10 只股票。
> 本手册说明模型如何保存、每天/每周该做什么、以及如何一键运行。

## 一、模型是否保存？已保存

每周训练后，生产模型保存在：

- `model_outputs/production_model.joblib` —— LightGBM 模型本体
- `model_outputs/production_model_meta.json` —— 元数据（训练截止日、因子列、横截面rank标志等）

每日预测直接**加载这个模型打分（约 6 秒）**，不再重训。重训只在每周（或每月）做一次。

## 二、运行节奏

| 频率 | 命令 | 做什么 | 耗时 |
|------|------|--------|------|
| **每个交易日盘后** | `run_strategy.py --mode daily` | 刷新池→下载最新→因子→**加载模型预测**→出信号 | ~10 分钟（主要在下载） |
| **每周末（或每月）** | `run_strategy.py --mode weekly` | 池→下载→因子→标签→**重训并保存模型**→信号→回测复盘 | ~15 分钟 |

为什么这样分：
- 行情每天变，所以**每天都要重新下载数据、重算因子、重新预测**。
- 但模型不必天天变。每周重训一次，既跟上最新数据，又让信号可复盘、不漂移。
- 模型从未训练过时，daily 模式会报错提示先跑一次 weekly。

## 三、一键运行

```bash
# 首次 / 每周：训练并保存模型 + 回测
.venv/Scripts/python.exe run_strategy.py --mode weekly

# 每个交易日盘后：加载模型，生成当日 Top10 信号
.venv/Scripts/python.exe run_strategy.py --mode daily

# 调试时复用已有数据（跳过下载）
.venv/Scripts/python.exe run_strategy.py --mode daily --skip-download

# 如想强制排除下跌趋势股（会小幅降低胜率）
.venv/Scripts/python.exe run_strategy.py --mode daily --trend-filter
```

## 四、每天看哪个文件

- **最终交易信号**：`final_signals/final_signals.csv` 和 `final_signals.xlsx`
  （含：代码、名称、上涨概率、市值、成交额、生成时间）
- 全部候选打分：`model_outputs/latest_predictions_all.csv`
- 运行日志：`model_outputs/run_strategy.log`

## 五、每周复盘看什么

- 回测报告：`backtest_outputs/backtest_report.csv`
- 净值曲线：`backtest_outputs/equity_curve.csv`（策略 vs 沪深300）
- 滚动 AUC：`model_outputs/rolling_auc_metrics.csv`（监控模型是否退化）
- 逐笔交易：`backtest_outputs/trades.csv`

关注点：滚动 AUC 是否持续 > 0.55、每日 Top10 命中率是否稳定在 0.33 附近、净值是否还跑赢基准。
若 AUC 跌破 0.52 或连续多周跑输基准，说明市场风格变了，需重新审视因子。

## 六、定时自动化（可选）

Windows 任务计划程序，每个交易日 15:30 跑 daily、每周六 09:00 跑 weekly：

```powershell
# 每日（周一到周五 15:30）
schtasks /create /tn "StockDaily" /tr "E:\private_work\A-kid\.venv\Scripts\python.exe E:\private_work\A-kid\run_strategy.py --mode daily" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:30

# 每周（周六 09:00）
schtasks /create /tn "StockWeekly" /tr "E:\private_work\A-kid\.venv\Scripts\python.exe E:\private_work\A-kid\run_strategy.py --mode weekly" /sc weekly /d SAT /st 09:00
```

## 七、用真实涨跌做反馈与监控（关键）

每天 daily 流程会自动运行 `track_performance.py`，把当日 Top10 登记到台账
`tracking/signal_ledger.csv`，并在每只票满两个交易日后，用真实日线回填实际涨跌。

### 一个必须避开的坑：不要拿这 10 只单独重训模型

被模型选中的 10 只是**高度选择偏置**的小样本，用它们单独重训会过拟合、让模型越来越极端、反而变差。
这是量化里的经典错误。真实结果对模型的反馈，**靠每周 weekly 在全市场几百只票的真实标签上重训**完成——
那才是无偏的学习信号。这 10 只的真实结果，正确用途是**监控、诊断、触发决策**：

1. **监控衰减**：台账算出"近20日滚动命中率"。若持续低于 0.25（回测基准约0.33），
   `track_performance.py` 会打出 ⚠ 预警 → 提示你提前 weekly 重训、并复盘因子。
2. **概率校准**：检查"模型说0.80的票，真实命中率是否真的更高"。看 `tracking/calibration.csv`。
   若预测概率和真实命中率严重对不上，说明模型需要重新校准/重训。
3. **败因诊断**：看 `tracking/performance_by_day.csv` 哪些日子集体失手，是否对应特定市场环境，
   据此判断要不要调整因子。

### 跟踪相关文件

- `tracking/signal_ledger.csv` —— 主台账（每条信号的预测概率 + 真实涨跌 + 是否命中/盈利）
- `tracking/performance_summary.csv` —— 整体与滚动命中率、胜率、平均净收益
- `tracking/performance_by_day.csv` —— 按信号日的逐日表现
- `tracking/calibration.csv` —— 预测概率分桶 vs 真实命中率（校准曲线）

### 口径说明

- **标签口径**（命中判定）：信号日收盘 → 第2日后收盘，涨幅 ≥ 3% 记命中。与训练标签一致。
- **交易口径**（实际盈亏）：次日开盘买入 → 第3日收盘卖出，扣双边费率后净收益。与 T+1 回测一致。

单独手动跑跟踪报告（不改台账）：
```bash
.venv/Scripts/python.exe track_performance.py --report-only
```

## 八、注意事项

1. **节假日**：A股休市日 daily 跑出来的会是上一交易日数据，不影响（重复信号不会重复登记），但不要据此交易。
2. **下载方式与耗时**：
   - **daily 增量下载**（`--incremental`）：每只票只拉最近约15个交易日的小窗口，与已有文件重叠日期比对收盘价——一致就只追加新增交易日；若对不上（说明发生除权除息、前复权价整段被缩放）则自动回退该股全量重下，保证历史一致性。
   - **weekly 全量下载**（`--overwrite`）：每周完整重下一次作为权威基线。
   - 耗时大头是防封禁的 0.8 秒/只延时（约600只 ≈ 8 分钟），增量主要省网络传输/解析并降低被封风险，总时间略短于全量。
3. **信号是 T+1 口径**：当日盘后生成信号 → 次日开盘买入 → 第三日收盘卖出（与回测一致）。
4. **概率不是收益保证**：Top10 命中率约 33%、单笔胜率约 50%，是统计优势而非确定性，需控制仓位与风险。
