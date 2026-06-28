#!/usr/bin/env python3
"""预测-真实结果台账与模型监控。

每天把当日 Top10 信号登记到台账；待两个交易日后，用真实日线回填实际涨跌，
计算滚动命中率、胜率、概率校准、相对基准超额，并在模型疑似衰减时给出预警。

重要：本工具只做"监控/诊断/触发决策"，绝不拿这 10 只单独去重训模型——
被选中的 10 只是高度选择偏置的小样本，单独训练会过拟合、让模型越来越差。
真实结果对模型的反馈，靠每周 `run_strategy.py --mode weekly` 在全市场样本上重训完成。

口径与 t1_backtest.py 一致：信号日 D 出信号 → D+1 开盘买入 → D+2 收盘卖出。
同时也记录"标签口径"（D 收盘 → D+2 收盘）便于和训练标签对照。

用法：
  .venv/Scripts/python.exe track_performance.py            # 登记今日信号 + 回填已到期的旧信号 + 出报告
  .venv/Scripts/python.exe track_performance.py --report-only   # 只看报告，不改台账
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

# 与 t1_backtest.py 保持一致的费率
BUY_FEE_RATE = 0.0003
SELL_FEE_RATE = 0.0013

LEDGER_COLUMNS = [
    "信号日", "股票代码", "名称", "上涨概率",
    "信号日收盘", "买入日", "买入开盘", "卖出日", "卖出收盘",
    "标签口径收益", "交易口径净收益", "命中(标签>=3%)", "盈利(交易>0)", "状态",
]


@dataclass(frozen=True)
class TrackConfig:
    signals_path: Path = Path("final_signals/final_signals.csv")
    stock_data_dir: Path = Path("stock_data")
    hs300_path: Path = Path("stock_data/HS300.csv")
    ledger_path: Path = Path("tracking/signal_ledger.csv")
    output_dir: Path = Path("tracking")
    threshold: float = 0.03
    decay_window: int = 20       # 用最近多少个已结算"信号日"算滚动指标
    decay_hit_floor: float = 0.25  # 滚动命中率低于此值给衰减预警（回测基准约0.33）


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "track_performance.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测-真实结果台账与模型监控。")
    parser.add_argument("--signals", type=Path, default=TrackConfig.signals_path)
    parser.add_argument("--stock-data-dir", type=Path, default=TrackConfig.stock_data_dir)
    parser.add_argument("--hs300", type=Path, default=TrackConfig.hs300_path)
    parser.add_argument("--ledger", type=Path, default=TrackConfig.ledger_path)
    parser.add_argument("--output-dir", type=Path, default=TrackConfig.output_dir)
    parser.add_argument("--report-only", action="store_true", help="只输出报告，不登记/回填台账。")
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_code(value) -> str:
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits.zfill(6)[-6:] if digits else ""


def load_price(stock_data_dir: Path, code: str) -> pd.DataFrame | None:
    path = stock_data_dir / f"{code}.csv"
    if not path.exists():
        return None
    df = read_csv(path)
    if not {"日期", "开盘", "收盘"}.issubset(df.columns):
        return None
    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    for column in ["开盘", "收盘"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["日期", "开盘", "收盘"]).sort_values("日期").reset_index(drop=True)


def net_trade_return(buy_open: float, sell_close: float) -> float:
    if buy_open <= 0 or sell_close <= 0:
        return float("nan")
    gross = sell_close / buy_open
    return gross * (1.0 - SELL_FEE_RATE) - (1.0 + BUY_FEE_RATE)


def load_ledger(ledger_path: Path) -> pd.DataFrame:
    if ledger_path.exists() and ledger_path.stat().st_size > 0:
        df = read_csv(ledger_path)
        for column in LEDGER_COLUMNS:
            if column not in df.columns:
                df[column] = np.nan
        df["股票代码"] = df["股票代码"].map(normalize_code)
        return df[LEDGER_COLUMNS]
    return pd.DataFrame(columns=LEDGER_COLUMNS)


def register_today(ledger: pd.DataFrame, config: TrackConfig) -> pd.DataFrame:
    if not config.signals_path.exists():
        LOGGER.warning("未找到信号文件 %s，跳过登记。", config.signals_path)
        return ledger
    signals = read_csv(config.signals_path)
    if signals.empty or "日期" not in signals.columns:
        LOGGER.warning("信号文件为空或缺列，跳过登记。")
        return ledger
    signals = signals.copy()
    signals["股票代码"] = signals["股票代码"].map(normalize_code)
    new_rows = []
    for _, row in signals.iterrows():
        signal_date = str(row["日期"])
        code = row["股票代码"]
        exists = ((ledger["信号日"] == signal_date) & (ledger["股票代码"] == code)).any()
        if exists:
            continue
        new_rows.append({
            "信号日": signal_date,
            "股票代码": code,
            "名称": row.get("名称", ""),
            "上涨概率": row.get("上涨概率", np.nan),
            "信号日收盘": np.nan, "买入日": np.nan, "买入开盘": np.nan,
            "卖出日": np.nan, "卖出收盘": np.nan,
            "标签口径收益": np.nan, "交易口径净收益": np.nan,
            "命中(标签>=3%)": np.nan, "盈利(交易>0)": np.nan, "状态": "待结算",
        })
    if new_rows:
        ledger = pd.concat([ledger, pd.DataFrame(new_rows)], ignore_index=True)
        LOGGER.info("登记新信号 %d 条（信号日 %s）。", len(new_rows), signals["日期"].iloc[0])
    else:
        LOGGER.info("今日信号已在台账中，无需重复登记。")
    return ledger


def settle_pending(ledger: pd.DataFrame, config: TrackConfig) -> pd.DataFrame:
    # 新建台账时这些列可能被推断为 float，写入日期字符串会报 dtype 错，先统一为 object。
    for column in ["买入日", "卖出日", "名称", "状态"]:
        ledger[column] = ledger[column].astype(object)
    pending = ledger[ledger["状态"] != "已结算"]
    settled_count = 0
    price_cache: dict[str, pd.DataFrame | None] = {}
    for idx in pending.index:
        code = ledger.at[idx, "股票代码"]
        signal_date = pd.to_datetime(ledger.at[idx, "信号日"], errors="coerce")
        if pd.isna(signal_date):
            continue
        if code not in price_cache:
            price_cache[code] = load_price(config.stock_data_dir, code)
        prices = price_cache[code]
        if prices is None:
            continue
        on_signal = prices[prices["日期"] == signal_date]
        future = prices[prices["日期"] > signal_date].head(2)
        if len(future) < 2:
            continue  # 还没满两个交易日，下次再结算
        buy_row, sell_row = future.iloc[0], future.iloc[1]
        signal_close = float(on_signal.iloc[0]["收盘"]) if not on_signal.empty else np.nan
        buy_open = float(buy_row["开盘"])
        sell_close = float(sell_row["收盘"])
        label_ret = (sell_close / signal_close - 1.0) if signal_close and signal_close > 0 else np.nan
        trade_ret = net_trade_return(buy_open, sell_close)
        ledger.at[idx, "信号日收盘"] = signal_close
        ledger.at[idx, "买入日"] = buy_row["日期"].strftime("%Y-%m-%d")
        ledger.at[idx, "买入开盘"] = buy_open
        ledger.at[idx, "卖出日"] = sell_row["日期"].strftime("%Y-%m-%d")
        ledger.at[idx, "卖出收盘"] = sell_close
        ledger.at[idx, "标签口径收益"] = label_ret
        ledger.at[idx, "交易口径净收益"] = trade_ret
        ledger.at[idx, "命中(标签>=3%)"] = int(label_ret >= config.threshold) if pd.notna(label_ret) else np.nan
        ledger.at[idx, "盈利(交易>0)"] = int(trade_ret > 0) if pd.notna(trade_ret) else np.nan
        ledger.at[idx, "状态"] = "已结算"
        settled_count += 1
    if settled_count:
        LOGGER.info("回填结算 %d 条历史信号。", settled_count)
    return ledger


def build_report(ledger: pd.DataFrame, config: TrackConfig) -> None:
    settled = ledger[ledger["状态"] == "已结算"].copy()
    if settled.empty:
        LOGGER.info("尚无已结算信号（信号需满两个交易日后才能评估），暂无报告。")
        return
    settled["命中(标签>=3%)"] = pd.to_numeric(settled["命中(标签>=3%)"], errors="coerce")
    settled["盈利(交易>0)"] = pd.to_numeric(settled["盈利(交易>0)"], errors="coerce")
    settled["交易口径净收益"] = pd.to_numeric(settled["交易口径净收益"], errors="coerce")
    settled["上涨概率"] = pd.to_numeric(settled["上涨概率"], errors="coerce")

    overall_hit = settled["命中(标签>=3%)"].mean()
    overall_win = settled["盈利(交易>0)"].mean()
    overall_ret = settled["交易口径净收益"].mean()

    # 每个信号日的指标
    by_day = settled.groupby("信号日").agg(
        命中率=("命中(标签>=3%)", "mean"),
        胜率=("盈利(交易>0)", "mean"),
        平均净收益=("交易口径净收益", "mean"),
        只数=("股票代码", "count"),
    ).reset_index()
    by_day.to_csv(config.output_dir / "performance_by_day.csv", index=False, encoding="utf-8-sig")

    # 滚动衰减监控
    recent_days = by_day.tail(config.decay_window)
    rolling_hit = recent_days["命中率"].mean() if not recent_days.empty else float("nan")
    rolling_win = recent_days["胜率"].mean() if not recent_days.empty else float("nan")

    # 概率校准：预测概率分桶 vs 真实命中率
    calib = pd.DataFrame()
    if settled["上涨概率"].notna().sum() >= 20:
        settled["概率桶"] = pd.qcut(settled["上涨概率"], min(5, settled["上涨概率"].nunique()), duplicates="drop")
        calib = settled.groupby("概率桶", observed=True).agg(
            平均预测概率=("上涨概率", "mean"),
            真实命中率=("命中(标签>=3%)", "mean"),
            样本数=("股票代码", "count"),
        ).reset_index()
        calib.to_csv(config.output_dir / "calibration.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([{
        "已结算信号数": int(len(settled)),
        "已结算信号日数": int(settled["信号日"].nunique()),
        "整体命中率(标签>=3%)": round(overall_hit, 4),
        "整体胜率(交易>0)": round(overall_win, 4),
        "整体平均净收益": round(overall_ret, 5),
        f"近{config.decay_window}日滚动命中率": round(rolling_hit, 4),
        f"近{config.decay_window}日滚动胜率": round(rolling_win, 4),
    }])
    summary.to_csv(config.output_dir / "performance_summary.csv", index=False, encoding="utf-8-sig")

    LOGGER.info("===== 实盘跟踪报告 =====")
    for key, value in summary.iloc[0].items():
        LOGGER.info("  %s: %s", key, value)
    if not calib.empty:
        LOGGER.info("  概率校准（预测概率 vs 真实命中率）:")
        for _, r in calib.iterrows():
            LOGGER.info("    预测%.3f -> 真实命中%.3f (n=%d)", r["平均预测概率"], r["真实命中率"], int(r["样本数"]))

    # 衰减预警
    if pd.notna(rolling_hit) and len(recent_days) >= max(5, config.decay_window // 2):
        if rolling_hit < config.decay_hit_floor:
            LOGGER.warning(
                "⚠ 模型衰减预警：近%d日滚动命中率 %.3f 低于阈值 %.2f（回测基准约0.33）。"
                "建议立即 run_strategy.py --mode weekly 重训，并复盘 rolling_auc_metrics.csv 与因子有效性。",
                config.decay_window, rolling_hit, config.decay_hit_floor,
            )
        else:
            LOGGER.info("模型状态正常：近%d日滚动命中率 %.3f ≥ 阈值 %.2f。", config.decay_window, rolling_hit, config.decay_hit_floor)


def main() -> int:
    args = parse_args()
    config = TrackConfig(
        signals_path=args.signals,
        stock_data_dir=args.stock_data_dir,
        hs300_path=args.hs300,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
    )
    setup_logging(config.output_dir)
    config.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        ledger = load_ledger(config.ledger_path)
        if not args.report_only:
            ledger = register_today(ledger, config)
            ledger = settle_pending(ledger, config)
            ledger.to_csv(config.ledger_path, index=False, encoding="utf-8-sig")
            LOGGER.info("台账已更新: %s (共 %d 条)", config.ledger_path, len(ledger))
        build_report(ledger, config)
    except Exception as exc:
        LOGGER.exception("跟踪失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
