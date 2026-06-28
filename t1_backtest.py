#!/usr/bin/env python3
"""Step 7: T+1 style backtest from rolling prediction signals."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class BacktestConfig:
    predictions_path: Path = Path("model_outputs/rolling_predictions.csv")
    stock_data_dir: Path = Path("stock_data")
    hs300_path: Path = Path("stock_data/HS300.csv")
    output_dir: Path = Path("backtest_outputs")
    initial_capital: float = 1_000_000.0
    max_positions_bull: int = 10
    max_positions_bear: int = 3
    buy_fee_rate: float = 0.0003
    sell_fee_rate: float = 0.0013
    min_fee: float = 5.0
    min_turnover: float = 500_000_000.0


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "t1_backtest.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第7步：T+1 回测模拟。")
    parser.add_argument("--predictions", type=Path, default=BacktestConfig.predictions_path)
    parser.add_argument("--stock-data-dir", type=Path, default=BacktestConfig.stock_data_dir)
    parser.add_argument("--hs300", type=Path, default=BacktestConfig.hs300_path)
    parser.add_argument("--output-dir", type=Path, default=BacktestConfig.output_dir)
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_price_data(stock_data_dir: Path) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for path in sorted(stock_data_dir.glob("*.csv")):
        if path.stem.upper() == "HS300" or path.stat().st_size == 0:
            continue
        df = read_csv(path)
        required = {"日期", "开盘", "收盘", "成交量", "成交额"}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        for column in ["开盘", "收盘", "成交量", "成交额"]:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        df = df.dropna(subset=["日期", "开盘", "收盘"]).sort_values("日期").reset_index(drop=True)
        df["prev_close"] = df["收盘"].shift(1)
        df["prev_volume"] = df["成交量"].shift(1)
        data[path.stem.zfill(6)] = df
    return data


def load_market_regime(hs300_path: Path) -> pd.DataFrame:
    hs300 = read_csv(hs300_path)
    hs300["日期"] = pd.to_datetime(hs300["日期"], errors="coerce")
    for column in ["开盘", "收盘"]:
        hs300[column] = pd.to_numeric(hs300[column], errors="coerce")
    hs300 = hs300.dropna(subset=["日期", "开盘", "收盘"]).sort_values("日期").reset_index(drop=True)
    hs300["ma20"] = hs300["收盘"].rolling(20, min_periods=20).mean()
    hs300["ma60"] = hs300["收盘"].rolling(60, min_periods=60).mean()
    hs300["ma60_prev20"] = hs300["ma60"].shift(20)
    hs300["max_positions"] = np.where(hs300["收盘"] < hs300["ma20"], 3, 10)
    hs300["empty_market"] = (hs300["收盘"] < hs300["ma20"]) & (hs300["ma60"] < hs300["ma60_prev20"])
    return hs300


def code_limit_pct(code: str, date_value: pd.Timestamp) -> float:
    if code.startswith(("300", "301")) and date_value >= pd.Timestamp("2020-08-24"):
        return 0.20
    return 0.10


def buy_blocked_by_limit_up(row: pd.Series, code: str) -> bool:
    if pd.isna(row.get("prev_close")) or row["prev_close"] <= 0:
        return False
    limit_price = row["prev_close"] * (1.0 + code_limit_pct(code, row["日期"]))
    open_at_limit = row["开盘"] >= limit_price * 0.995
    tiny_volume = pd.notna(row.get("prev_volume")) and row["成交量"] <= row["prev_volume"] * 0.2
    return bool(open_at_limit and tiny_volume)


def net_trade_return(buy_price: float, sell_price: float, capital: float, config: BacktestConfig) -> float:
    if buy_price <= 0 or sell_price <= 0 or capital <= 0:
        return float("nan")
    shares = capital / buy_price
    buy_fee = max(capital * config.buy_fee_rate, config.min_fee)
    gross_sell = shares * sell_price
    sell_fee = max(gross_sell * config.sell_fee_rate, config.min_fee)
    return (gross_sell - sell_fee - capital - buy_fee) / capital


def select_signal_day(
    day_predictions: pd.DataFrame,
    signal_date: pd.Timestamp,
    price_data: dict[str, pd.DataFrame],
    market: pd.DataFrame,
    config: BacktestConfig,
) -> pd.DataFrame:
    market_row = market[market["日期"] == signal_date]
    if market_row.empty:
        return day_predictions.head(0)
    market_row = market_row.iloc[0]
    if bool(market_row["empty_market"]):
        return day_predictions.head(0)
    max_positions = int(market_row["max_positions"])

    rows = []
    for row in day_predictions.sort_values("上涨概率", ascending=False).itertuples(index=False):
        code = str(row.股票代码).zfill(6)
        if code.startswith(("688", "8")):
            continue
        prices = price_data.get(code)
        if prices is None:
            continue
        price_row = prices[prices["日期"] == signal_date]
        if price_row.empty:
            continue
        if float(price_row.iloc[0].get("成交额", 0.0) or 0.0) < config.min_turnover:
            continue
        rows.append(row._asdict())
        if len(rows) >= max_positions:
            break
    return pd.DataFrame(rows)


def run_backtest(config: BacktestConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = read_csv(config.predictions_path)
    predictions["日期"] = pd.to_datetime(predictions["日期"], errors="coerce")
    predictions["股票代码"] = predictions["股票代码"].astype(str).str.zfill(6)
    predictions["上涨概率"] = pd.to_numeric(predictions["上涨概率"], errors="coerce")
    predictions = predictions.dropna(subset=["日期", "上涨概率"]).sort_values(["日期", "上涨概率"], ascending=[True, False])
    price_data = load_price_data(config.stock_data_dir)
    market = load_market_regime(config.hs300_path)

    trades = []
    portfolio_returns = []
    for signal_date, day_predictions in predictions.groupby("日期", sort=True):
        selected = select_signal_day(day_predictions, signal_date, price_data, market, config)
        if selected.empty:
            portfolio_returns.append({"日期": signal_date, "strategy_return": 0.0, "benchmark_return": 0.0, "持仓数": 0})
            continue
        capital_per_trade = config.initial_capital / len(selected)
        day_trade_returns = []
        benchmark_return = 0.0
        hs_rows = market[market["日期"] > signal_date].head(2)
        if len(hs_rows) == 2 and hs_rows.iloc[0]["开盘"] > 0:
            benchmark_return = hs_rows.iloc[1]["收盘"] / hs_rows.iloc[0]["开盘"] - 1.0

        for selected_row in selected.itertuples(index=False):
            code = str(selected_row.股票代码).zfill(6)
            prices = price_data[code]
            future_rows = prices[prices["日期"] > signal_date].head(2)
            if len(future_rows) < 2:
                continue
            buy_row = future_rows.iloc[0]
            sell_row = future_rows.iloc[1]
            if buy_blocked_by_limit_up(buy_row, code):
                trades.append(
                    {
                        "信号日": signal_date,
                        "股票代码": code,
                        "上涨概率": selected_row.上涨概率,
                        "买入日": buy_row["日期"],
                        "卖出日": sell_row["日期"],
                        "成交": False,
                        "收益率": 0.0,
                    }
                )
                continue
            trade_return = net_trade_return(buy_row["开盘"], sell_row["收盘"], capital_per_trade, config)
            if not math.isnan(trade_return):
                day_trade_returns.append(trade_return)
                trades.append(
                    {
                        "信号日": signal_date,
                        "股票代码": code,
                        "上涨概率": selected_row.上涨概率,
                        "买入日": buy_row["日期"],
                        "卖出日": sell_row["日期"],
                        "成交": True,
                        "收益率": trade_return,
                    }
                )
        day_return = float(np.mean(day_trade_returns)) if day_trade_returns else 0.0
        portfolio_returns.append(
            {
                "日期": signal_date,
                "strategy_return": day_return,
                "benchmark_return": benchmark_return,
                "持仓数": len(day_trade_returns),
            }
        )

    curve = pd.DataFrame(portfolio_returns).sort_values("日期")
    curve["strategy_equity"] = (1.0 + curve["strategy_return"]).cumprod()
    curve["benchmark_equity"] = (1.0 + curve["benchmark_return"]).cumprod()
    trades_df = pd.DataFrame(trades)
    report = build_report(curve, trades_df)
    return curve, trades_df, report


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min()) if not drawdown.empty else 0.0


def build_report(curve: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if curve.empty:
        raise ValueError("回测曲线为空。")
    periods = max(len(curve), 1)
    final_equity = float(curve["strategy_equity"].iloc[-1])
    annual_return = final_equity ** (252.0 / periods) - 1.0
    returns = curve["strategy_return"]
    sharpe = 0.0 if returns.std(ddof=0) == 0 else float(returns.mean() / returns.std(ddof=0) * np.sqrt(252))
    executed = trades[trades["成交"]] if not trades.empty else trades
    wins = executed[executed["收益率"] > 0] if not executed.empty else executed
    losses = executed[executed["收益率"] < 0] if not executed.empty else executed
    win_rate = float(len(wins) / len(executed)) if len(executed) else 0.0
    profit_loss_ratio = (
        float(wins["收益率"].mean() / abs(losses["收益率"].mean()))
        if len(wins) and len(losses) and losses["收益率"].mean() != 0
        else float("nan")
    )
    report = pd.DataFrame(
        [
            {
                "策略累计净值": final_equity,
                "沪深300累计净值": float(curve["benchmark_equity"].iloc[-1]),
                "年化收益率": annual_return,
                "最大回撤": max_drawdown(curve["strategy_equity"]),
                "夏普比率": sharpe,
                "胜率": win_rate,
                "盈亏比": profit_loss_ratio,
                "交易次数": int(len(executed)),
            }
        ]
    )
    return report


def main() -> int:
    args = parse_args()
    config = BacktestConfig(
        predictions_path=args.predictions,
        stock_data_dir=args.stock_data_dir,
        hs300_path=args.hs300,
        output_dir=args.output_dir,
    )
    setup_logging(config.output_dir)
    try:
        curve, trades, report = run_backtest(config)
        for frame in (curve, trades):
            for column in frame.columns:
                if pd.api.types.is_datetime64_any_dtype(frame[column]):
                    frame[column] = frame[column].dt.strftime("%Y-%m-%d")
        curve.to_csv(config.output_dir / "equity_curve.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(config.output_dir / "trades.csv", index=False, encoding="utf-8-sig")
        report.to_csv(config.output_dir / "backtest_report.csv", index=False, encoding="utf-8-sig")
        LOGGER.info("回测完成: %s", report.to_dict("records")[0])
    except Exception as exc:
        LOGGER.exception("第7步失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
