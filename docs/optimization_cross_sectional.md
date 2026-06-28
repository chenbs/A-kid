# 选股优化记录（第二轮）：横截面 rank 选股，最大化胜率

> 本文档记录 2026-06-28 第二轮优化。用户目标已澄清为：**不限定追涨/起涨，只要保证胜率，
> 输出未来2日涨≥3%概率最高的10只**。本轮从"模型选股能力"本身入手。

## 一、关键诊断：模型的预测力大量来自"时间序列"而非"横截面选股"

对最新模型做因子重要性 + 横截面 IC 分析，发现两个结构性问题：

### 1. 市场因子 hs300_return_20d 占据虚高的重要性

| 指标 | 数值 |
|------|------|
| LightGBM 分裂次数 | 2445（第二名的 10 倍） |
| 每日横截面 IC | ≈ NaN（同一天所有股票同值） |

`hs300_return_20d` 对"同一天选哪只票"毫无区分力（同日同值），但它在"全样本池化"
时能区分牛熊时段，从而抬高了池化 AUC。**模型把大量容量花在判断大盘时段，而非横截面选股。**

### 2. 横截面上最强的因子全是"波动率类"

每日横截面 IC 排名（top-N 选股真正相关的指标）：

```
atr_14_pct          0.161
volatility_20d      0.151
intraday_amplitude  0.139
distance_to_20d_low 0.117
```

模型横截面上主要在"挑波动大的票" → 高波动的下跌股容易被选入。这解释了第一轮观察到的
下跌票入选现象。

## 二、优化方案：横截面 rank 标准化

把每个因子在**同一交易日的所有股票内**转成百分位排名（`groupby(日期).rank(pct=True)`），
再喂给 LightGBM。效果：

- 让模型学"今天谁相对最强"（横截面选股），而非"现在是不是好时段"（时间序列）。
- 自动中性化 hs300 这类同日同值的市场因子（rank 后全为 0.5，无信号）。
- 直接对应"每日 Top10"的选股目标。

同时新增一个因子 `pos_60d_range`（收盘在 60 日高低区间的位置），横截面 IC 0.073。

## 三、回测验证（滚动 precision@10，即每日Top10实际命中率）

| 方案 | 均值 precision@10 |
|------|------------------|
| 原始（全特征，含hs300主导） | 0.3470 |
| **横截面 rank + pos_60d_range** | **0.3514** |
| 横截面 rank + 趋势过滤 | 0.3370 |

两点结论：
1. **横截面 rank 提升胜率**（0.347 → 0.351），方向正确。
2. **趋势过滤在新模型上反而降低胜率**（0.351 → 0.337）。因为横截面模型已具备选股能力，
   趋势过滤会砍掉一些会涨的超跌反弹票。

## 四、与第一轮（趋势过滤）的关系

第一轮的趋势过滤是在"波动率驱动的旧模型"上验证的，当时能纠偏（0.329→0.342）。
但在本轮的横截面 rank 模型上，趋势过滤变成净负贡献。

由于用户最新目标是**纯粹最大化胜率**，因此：
- **横截面 rank：默认开启**（`rolling_lightgbm_train.py`，`--no-cross-sectional-rank` 可关）。
- **趋势过滤：默认关闭**（`signal_filter.py`，`--trend-filter` 可显式开启，用于强制排除下跌股）。

## 五、代码改动

### `factor_engineering.py`
- 新增因子 `pos_60d_range = (close - 60日最低) / (60日最高 - 60日最低)`。

### `rolling_lightgbm_train.py`
- 新增 `TrainConfig.cross_sectional_rank: bool = True`。
- 新增 `cross_sectional_rank_transform()`：按交易日对因子做横截面百分位排名。
- `load_labeled_samples` / `load_latest_features` 应用该变换；`load_latest_features` 预留
  `__raw` 趋势列，供第6步可选过滤还原真实值。
- 命令行新增 `--no-cross-sectional-rank`。

### `signal_filter.py`
- `trend_filter` 默认改为 `False`，命令行由 `--no-trend-filter` 改为 `--trend-filter`（显式开启）。

## 六、运行方式

```bash
# 因子列变化，需从第3步重生成
.venv/Scripts/python.exe factor_engineering.py      # 第3步（含新因子）
.venv/Scripts/python.exe label_engineering.py       # 第4步
.venv/Scripts/python.exe rolling_lightgbm_train.py  # 第5步（默认横截面rank）
.venv/Scripts/python.exe signal_filter.py           # 第6步（默认不过滤，纯概率Top10）

# 如想强制排除下跌趋势股（会小幅降低胜率）
.venv/Scripts/python.exe signal_filter.py --trend-filter
```

## 七、仍存在的本质权衡

即便横截面建模，最强的因子仍是波动率类——这是"未来2日涨≥3%"标签的数学必然
（高波动票更容易触及 ±3%）。所以胜率天花板（约 35%）主要由标签定义决定。
若要进一步提升，可探索的方向：
- 改标签为"相对市场超额收益排名前N"（横截面标签），让目标本身横截面化。
- 用 LambdaRank 等排序目标直接优化每日排序。
这些属于更大的重构，需另行评估收益。
