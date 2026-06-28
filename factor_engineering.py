#!/usr/bin/env python3
"""Step 3: vectorized factor engineering for A-share daily bars."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

RAW_COLUMNS = ("日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额")
FACTOR_COLUMNS = (
    "return_1d",
    "return_2d",
    "return_3d",
    "close_ma5_pos",
    "close_ma10_pos",
    "close_ma20_pos",
    "close_ma60_pos",
    "ma_arrangement",
    "high_20d_breakout",
    "macd_dif",
    "macd_dea",
    "macd_hist",
    "macd_dif_positive",
    "macd_dea_positive",
    "volume_ratio_5",
    "volume_ratio_10",
    "volume_ratio_20",
    "amount_ratio_20",
    "turnover_amount_1d_change",
    "price_volume_corr_10",
    "volume_shrink_candle",
    "return_5d",
    "return_10d",
    "return_20d",
    "up_days_ratio_5",
    "rsi_14",
    "atr_14",
    "atr_14_pct",
    "volatility_20d",
    "distance_to_20d_high",
    "distance_to_20d_low",
    "pos_60d_range",
    "open_to_close_return",
    "high_to_close_return",
    "low_to_close_return",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "intraday_range_body_ratio",
    "intraday_amplitude",
    "hs300_return_20d",
    "relative_hs300_return_20d",
)


@dataclass(frozen=True)
class FactorConfig:
    stock_data_dir: Path = Path("stock_data")
    output_dir: Path = Path("features_data")
    hs300_path: Path = Path("stock_data/HS300.csv")
    min_rows: int = 200
    overwrite: bool = True
    limit: int | None = None


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "factor_engineering.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第3步：为每只股票生成向量化因子。")
    parser.add_argument("--stock-data-dir", type=Path, default=FactorConfig.stock_data_dir)
    parser.add_argument("--output-dir", type=Path, default=FactorConfig.output_dir)
    parser.add_argument("--hs300-path", type=Path, default=FactorConfig.hs300_path)
    parser.add_argument("--min-rows", type=int, default=FactorConfig.min_rows)
    parser.add_argument("--limit", type=int, default=None, help="仅处理前 N 只股票，便于测试。")
    parser.add_argument("--no-overwrite", action="store_true", help="跳过已存在的因子文件。")
    return parser.parse_args()


def load_daily_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"股票代码": str})
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{path} 缺少必要列: {', '.join(missing)}")

    df = df.copy()
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).sort_values("日期").drop_duplicates("日期", keep="last")
    for column in RAW_COLUMNS:
        if column != "日期":
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["开盘", "收盘", "最高", "最低", "成交量"])
    if "成交额" not in df.columns:
        df["成交额"] = np.nan
    return df.reset_index(drop=True)


def load_hs300_factor(hs300_path: Path) -> pd.DataFrame:
    hs300 = load_daily_csv(hs300_path)
    hs300["hs300_return_20d"] = safe_divide(hs300["收盘"], hs300["收盘"].shift(20)) - 1.0
    return hs300[["日期", "hs300_return_20d"]]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def compute_factors(daily: pd.DataFrame, hs300_factor: pd.DataFrame) -> pd.DataFrame:
    """Return original OHLCV columns plus all factors without future leakage."""
    df = daily.copy().sort_values("日期").reset_index(drop=True)
    close = df["收盘"]
    open_ = df["开盘"]
    high = df["最高"]
    low = df["最低"]
    volume = df["成交量"]
    amount = df["成交额"]

    ma5 = close.rolling(5, min_periods=1).mean()
    ma10 = close.rolling(10, min_periods=1).mean()
    ma20 = close.rolling(20, min_periods=1).mean()
    ma60 = close.rolling(60, min_periods=1).mean()
    df["return_1d"] = safe_divide(close, close.shift(1)) - 1.0
    df["return_2d"] = safe_divide(close, close.shift(2)) - 1.0
    df["return_3d"] = safe_divide(close, close.shift(3)) - 1.0
    df["close_ma5_pos"] = safe_divide(close - ma5, ma5)
    df["close_ma10_pos"] = safe_divide(close - ma10, ma10)
    df["close_ma20_pos"] = safe_divide(close - ma20, ma20)
    df["close_ma60_pos"] = safe_divide(close - ma60, ma60)

    bull = (ma5 > ma10) & (ma10 > ma20)
    bear = (ma5 < ma10) & (ma10 < ma20)
    df["ma_arrangement"] = np.select([bull, bear], [1, -1], default=0)
    previous_20d_high = high.rolling(20, min_periods=20).max().shift(1)
    df["high_20d_breakout"] = (high >= previous_20d_high).astype(int)

    ema12 = close.ewm(span=12, adjust=False, min_periods=1).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=1).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=1).mean()
    df["macd_dif"] = dif
    df["macd_dea"] = dea
    df["macd_hist"] = 2.0 * (dif - dea)
    df["macd_dif_positive"] = (dif > 0).astype(int)
    df["macd_dea_positive"] = (dea > 0).astype(int)

    df["volume_ratio_5"] = safe_divide(volume, volume.rolling(5, min_periods=1).mean())
    df["volume_ratio_10"] = safe_divide(volume, volume.rolling(10, min_periods=1).mean())
    df["volume_ratio_20"] = safe_divide(volume, volume.rolling(20, min_periods=1).mean())
    df["amount_ratio_20"] = safe_divide(amount, amount.rolling(20, min_periods=1).mean())
    df["turnover_amount_1d_change"] = safe_divide(amount, amount.shift(1)) - 1.0
    df["price_volume_corr_10"] = close.pct_change().rolling(10, min_periods=5).corr(volume.pct_change())
    shrink = volume < volume.shift(1) * 0.8
    up_candle = close >= open_
    down_candle = close < open_
    df["volume_shrink_candle"] = np.select([shrink & up_candle, shrink & down_candle], [1, -1], default=0)

    df["return_5d"] = safe_divide(close, close.shift(5)) - 1.0
    df["return_10d"] = safe_divide(close, close.shift(10)) - 1.0
    df["return_20d"] = safe_divide(close, close.shift(20)) - 1.0
    df["up_days_ratio_5"] = (close > close.shift(1)).astype(float).rolling(5, min_periods=1).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14, min_periods=14).mean()
    avg_loss = loss.rolling(14, min_periods=14).mean()
    rs = safe_divide(avg_gain, avg_loss)
    df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))
    df.loc[(avg_loss == 0) & (avg_gain > 0), "rsi_14"] = 100.0
    df.loc[(avg_loss == 0) & (avg_gain == 0), "rsi_14"] = 50.0

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = true_range.rolling(14, min_periods=1).mean()
    df["atr_14_pct"] = safe_divide(df["atr_14"], close)
    df["volatility_20d"] = close.pct_change().rolling(20, min_periods=5).std()
    rolling_20d_high = high.rolling(20, min_periods=1).max()
    rolling_20d_low = low.rolling(20, min_periods=1).min()
    df["distance_to_20d_high"] = safe_divide(close, rolling_20d_high) - 1.0
    df["distance_to_20d_low"] = safe_divide(close, rolling_20d_low) - 1.0
    rolling_60d_high = high.rolling(60, min_periods=20).max()
    rolling_60d_low = low.rolling(60, min_periods=20).min()
    df["pos_60d_range"] = safe_divide(close - rolling_60d_low, rolling_60d_high - rolling_60d_low)
    body = (open_ - close).abs()
    candle_top = pd.concat([open_, close], axis=1).max(axis=1)
    candle_bottom = pd.concat([open_, close], axis=1).min(axis=1)
    df["open_to_close_return"] = safe_divide(close, open_) - 1.0
    df["high_to_close_return"] = safe_divide(close, high) - 1.0
    df["low_to_close_return"] = safe_divide(close, low) - 1.0
    df["upper_shadow_ratio"] = safe_divide(high - candle_top, close)
    df["lower_shadow_ratio"] = safe_divide(candle_bottom - low, close)
    df["intraday_range_body_ratio"] = safe_divide(high - low, body)
    df["intraday_amplitude"] = safe_divide(high - low, close)

    df = df.merge(hs300_factor, on="日期", how="left", suffixes=("", "_market"))
    if "hs300_return_20d_market" in df.columns:
        df["hs300_return_20d"] = df["hs300_return_20d_market"].combine_first(df["hs300_return_20d"])
        df = df.drop(columns=["hs300_return_20d_market"])
    df["relative_hs300_return_20d"] = df["return_20d"] - df["hs300_return_20d"]

    output = df[list(RAW_COLUMNS) + list(FACTOR_COLUMNS)].copy()
    numeric_columns = [column for column in output.columns if column != "日期"]
    output[numeric_columns] = output[numeric_columns].replace([np.inf, -np.inf], np.nan)
    output[list(FACTOR_COLUMNS)] = output[list(FACTOR_COLUMNS)].fillna(0.0)
    output["日期"] = output["日期"].dt.strftime("%Y-%m-%d")
    return output


def stock_files(stock_data_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(path for path in stock_data_dir.glob("*.csv") if path.stem.upper() != "HS300")
    if limit is not None:
        return files[: max(0, limit)]
    return files


def generate_factor_files(config: FactorConfig) -> dict[str, int]:
    setup_logging(config.output_dir)
    hs300_factor = load_hs300_factor(config.hs300_path)
    files = stock_files(config.stock_data_dir, config.limit)
    LOGGER.info("第3步开始: 待处理股票文件=%d", len(files))

    success = 0
    skipped = 0
    failed = 0
    for index, path in enumerate(files, start=1):
        output_path = config.output_dir / path.name
        progress = f"[{index}/{len(files)}] {path.stem}"
        if output_path.exists() and not config.overwrite:
            skipped += 1
            LOGGER.info("%s 已存在，跳过", progress)
            continue
        try:
            daily = load_daily_csv(path)
            if len(daily) < config.min_rows:
                raise ValueError(f"交易日数量不足: {len(daily)} < {config.min_rows}")
            factors = compute_factors(daily, hs300_factor)
            factors.to_csv(output_path, index=False, encoding="utf-8-sig")
            success += 1
            LOGGER.info("%s 因子保存完成: rows=%d", progress, len(factors))
        except Exception as exc:
            failed += 1
            LOGGER.exception("%s 因子生成失败: %s", progress, exc)

    LOGGER.info("第3步完成: 成功=%d, 跳过=%d, 失败=%d, 输出目录=%s", success, skipped, failed, config.output_dir.resolve())
    return {"success": success, "skipped": skipped, "failed": failed}


def main() -> int:
    args = parse_args()
    config = FactorConfig(
        stock_data_dir=args.stock_data_dir,
        output_dir=args.output_dir,
        hs300_path=args.hs300_path,
        min_rows=args.min_rows,
        overwrite=not args.no_overwrite,
        limit=args.limit,
    )
    result = generate_factor_files(config)
    return 0 if result["success"] + result["skipped"] > 0 and result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
