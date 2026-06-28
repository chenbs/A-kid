#!/usr/bin/env python3
"""A-share tradable stock pool initial hard filter.

Fetches real-time A-share quotes from AKShare/Eastmoney, normalizes useful
fields, applies conservative liquidity and board filters, and writes the
qualified pool to a local CSV file.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import akshare as ak
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class FilterConfig:
    min_total_market_cap: float = 20_000_000_000
    min_turnover: float = 500_000_000
    request_delay_seconds: float = 1.0
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    output_path: Path = Path("qualified_stock_pool.csv")


COLUMN_ALIASES = {
    "代码": ("代码", "证券代码"),
    "名称": ("名称", "证券简称"),
    "最新价": ("最新价", "最新", "收盘"),
    "成交量": ("成交量", "成交量(手)"),
    "成交额": ("成交额", "成交额(元)"),
    "总市值": ("总市值", "总市值-元"),
    "市盈率": ("市盈率-动态", "市盈率", "动态市盈率"),
    "市净率": ("市净率",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 A 股可交易股票池初始筛选结果。")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=FilterConfig.output_path,
        help="输出 CSV 路径，默认 qualified_stock_pool.csv",
    )
    parser.add_argument(
        "--min-market-cap",
        type=float,
        default=FilterConfig.min_total_market_cap,
        help="最小总市值，单位元，默认 200 亿。",
    )
    parser.add_argument(
        "--min-turnover",
        type=float,
        default=FilterConfig.min_turnover,
        help="最小当日成交额，单位元，默认 5 亿。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=FilterConfig.request_delay_seconds,
        help="请求前延时秒数，默认 1 秒。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=FilterConfig.max_retries,
        help="接口失败后的最大尝试次数，默认 3 次。",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=FilterConfig.retry_backoff_seconds,
        help="重试退避基准秒数，默认 2 秒。",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def pick_column(df: pd.DataFrame, aliases: Iterable[str]) -> str | None:
    for column in aliases:
        if column in df.columns:
            return column
    return None


def require_columns(df: pd.DataFrame, required_fields: Iterable[str]) -> None:
    missing = []
    for field in required_fields:
        if pick_column(df, COLUMN_ALIASES[field]) is None:
            missing.append(f"{field}({', '.join(COLUMN_ALIASES[field])})")
    if missing:
        raise ValueError(f"AKShare 返回数据缺少必要字段: {'; '.join(missing)}")


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def fetch_a_share_spot(
    delay_seconds: float = 1.0,
    max_retries: int = 3,
    retry_backoff_seconds: float = 2.0,
) -> pd.DataFrame:
    """Fetch real-time A-share spot data with delay, retries, and error wrapping."""
    if delay_seconds > 0:
        time.sleep(delay_seconds)

    attempts = max(1, max_retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            data = ak.stock_zh_a_spot_em()
            break
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                raise RuntimeError(f"获取 AKShare A 股实时行情失败: {exc}") from exc
            sleep_seconds = retry_backoff_seconds * attempt
            LOGGER.warning(
                "第 %d/%d 次获取失败，%.1f 秒后重试: %s",
                attempt,
                attempts,
                sleep_seconds,
                exc,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    else:
        raise RuntimeError(f"获取 AKShare A 股实时行情失败: {last_error}")

    if data is None or data.empty:
        raise RuntimeError("AKShare A 股实时行情返回为空。")

    return data


def normalize_stock_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize AKShare columns while preserving spare valuation fields."""
    require_columns(raw, ("代码", "名称", "最新价", "成交量", "总市值"))

    normalized = raw.copy()
    rename_map = {}
    for target, aliases in COLUMN_ALIASES.items():
        source = target if target in normalized.columns else pick_column(normalized, aliases)
        if source and source != target:
            rename_map[source] = target
    normalized = normalized.rename(columns=rename_map)

    normalized["代码"] = normalized["代码"].astype(str).str.zfill(6)
    normalized["名称"] = normalized["名称"].astype(str).str.strip()

    for column in ("最新价", "成交量", "成交额", "总市值", "市盈率", "市净率"):
        if column in normalized.columns:
            normalized[column] = to_numeric(normalized[column])

    if "成交额" not in normalized.columns:
        normalized["成交额"] = pd.NA

    estimated_turnover = normalized["成交量"] * normalized["最新价"] * 100
    normalized["成交额"] = normalized["成交额"].fillna(estimated_turnover)

    preferred_columns = [
        "代码",
        "名称",
        "总市值",
        "最新价",
        "成交量",
        "成交额",
        "市盈率",
        "市净率",
    ]
    remaining_columns = [
        column for column in normalized.columns if column not in preferred_columns
    ]
    output_columns = [
        column for column in preferred_columns if column in normalized.columns
    ] + remaining_columns

    return normalized[output_columns]


def apply_hard_filters(df: pd.DataFrame, config: FilterConfig) -> pd.DataFrame:
    """Apply strict initial eligibility filters."""
    code = df["代码"].astype(str)
    name = df["名称"].astype(str)

    mask = (
        (df["总市值"] >= config.min_total_market_cap)
        & (df["成交额"] > config.min_turnover)
        & ~code.str.startswith("688")
        & ~(code.str.len().eq(6) & code.str.startswith("8"))
        & ~name.str.contains("ST", case=False, na=False)
    )

    filtered = df.loc[mask].copy()
    return filtered.sort_values(["总市值", "成交额"], ascending=[False, False])


def build_stock_pool(config: FilterConfig) -> pd.DataFrame:
    raw = fetch_a_share_spot(
        config.request_delay_seconds,
        config.max_retries,
        config.retry_backoff_seconds,
    )
    LOGGER.info("过滤前股票数量: %d", len(raw))

    normalized = normalize_stock_data(raw)
    qualified = apply_hard_filters(normalized, config)
    LOGGER.info("过滤后股票数量: %d", len(qualified))

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    qualified.to_csv(config.output_path, index=False, encoding="utf-8-sig")
    LOGGER.info("已保存股票池: %s", config.output_path.resolve())
    return qualified


def main() -> int:
    setup_logging()
    args = parse_args()
    config = FilterConfig(
        min_total_market_cap=args.min_market_cap,
        min_turnover=args.min_turnover,
        request_delay_seconds=args.delay,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff,
        output_path=args.output,
    )

    try:
        build_stock_pool(config)
    except Exception as exc:
        LOGGER.exception("构建股票池失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
