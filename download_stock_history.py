#!/usr/bin/env python3
"""Download daily A-share history for a qualified stock pool.

Reads stock codes from qualified_stock_pool.csv, downloads forward-adjusted
daily bars with AKShare, stores one CSV per stock, and downloads HS300 as the
market benchmark.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable, TypeVar

import akshare as ak
import pandas as pd


LOGGER = logging.getLogger(__name__)
T = TypeVar("T")

REQUIRED_STOCK_COLUMNS = ("日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额")
STOCK_COLUMN_ALIASES = {
    "日期": ("日期", "date"),
    "开盘": ("开盘", "open"),
    "收盘": ("收盘", "close"),
    "最高": ("最高", "high"),
    "最低": ("最低", "low"),
    "成交量": ("成交量", "volume"),
    "成交额": ("成交额", "amount"),
}
INDEX_COLUMN_ALIASES = {
    "日期": ("日期", "date"),
    "开盘": ("开盘", "open"),
    "收盘": ("收盘", "close"),
    "最高": ("最高", "high"),
    "最低": ("最低", "low"),
    "成交量": ("成交量", "volume"),
    "成交额": ("成交额", "amount"),
}


@dataclass(frozen=True)
class DownloadConfig:
    pool_path: Path = Path("qualified_stock_pool.csv")
    output_dir: Path = Path("stock_data")
    start_date: str = "20200101"
    end_date: str = date.today().strftime("%Y%m%d")
    min_trading_days: int = 200
    request_delay_seconds: float = 0.8
    max_retries: int = 3
    retry_backoff_seconds: float = 2.0
    timeout_seconds: float = 15.0
    hs300_symbol: str = "sh000300"
    overwrite: bool = False
    limit: int | None = None
    # 增量更新：只拉最近 overlap_days 的小窗口，与已有文件重叠日期比对收盘价；
    # 一致则只追加新增交易日，不一致(除权导致前复权整段缩放)则回退全量重下该股。
    incremental: bool = False
    overlap_days: int = 15


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载股票池内个股和沪深300日线数据。")
    parser.add_argument(
        "--pool",
        type=Path,
        default=DownloadConfig.pool_path,
        help="股票池 CSV 路径，默认 qualified_stock_pool.csv。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DownloadConfig.output_dir,
        help="日线 CSV 输出目录，默认 stock_data。",
    )
    parser.add_argument(
        "--start-date",
        default=DownloadConfig.start_date,
        help="开始日期，支持 YYYYMMDD 或 YYYY-MM-DD，默认 20200101。",
    )
    parser.add_argument(
        "--end-date",
        default=DownloadConfig.end_date,
        help="结束日期，支持 YYYYMMDD 或 YYYY-MM-DD，默认今天。",
    )
    parser.add_argument(
        "--min-trading-days",
        type=int,
        default=DownloadConfig.min_trading_days,
        help="最少交易日数量，低于该值自动跳过，默认 200。",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DownloadConfig.request_delay_seconds,
        help="每次请求前延时秒数，默认 0.8 秒。",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DownloadConfig.max_retries,
        help="单个标的最大尝试次数，默认 3。",
    )
    parser.add_argument(
        "--retry-backoff",
        type=float,
        default=DownloadConfig.retry_backoff_seconds,
        help="重试退避基准秒数，默认 2 秒。",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DownloadConfig.timeout_seconds,
        help="AKShare 股票日线接口 timeout 参数，默认 15 秒。",
    )
    parser.add_argument(
        "--hs300-symbol",
        default=DownloadConfig.hs300_symbol,
        help="沪深300指数代码，stock_zh_index_daily 通常使用 sh000300。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已存在且有效的 CSV；默认跳过以支持断点续跑。",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量更新：只拉最近小窗口追加到已有文件；遇除权自动回退全量重下该股。",
    )
    parser.add_argument(
        "--overlap-days",
        type=int,
        default=DownloadConfig.overlap_days,
        help="增量模式回看的交易日窗口（含与已有数据的重叠校验），默认 15。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="仅处理前 N 只股票，便于测试；默认处理全部。",
    )
    return parser.parse_args()


def normalize_akshare_date(value: str) -> str:
    parsed = datetime.strptime(value.replace("-", ""), "%Y%m%d")
    return parsed.strftime("%Y%m%d")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "download_stock_history.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def retry_call(
    func: Callable[[], T],
    *,
    label: str,
    max_retries: int,
    retry_backoff_seconds: float,
) -> T:
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_seconds = retry_backoff_seconds * attempt
            LOGGER.warning(
                "%s 第 %d/%d 次失败，%.1f 秒后重试: %s",
                label,
                attempt,
                attempts,
                sleep_seconds,
                exc,
            )
            if sleep_seconds > 0:
                time.sleep(sleep_seconds)
    raise RuntimeError(f"{label} 未能完成")


def find_column(df: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    for column in aliases:
        if column in df.columns:
            return column
    return None


def normalize_columns(
    df: pd.DataFrame,
    aliases_map: dict[str, tuple[str, ...]],
    required_columns: tuple[str, ...],
) -> pd.DataFrame:
    rename_map = {}
    for target, aliases in aliases_map.items():
        source = target if target in df.columns else find_column(df, aliases)
        if source and source != target:
            rename_map[source] = target

    result = df.rename(columns=rename_map).copy()
    missing = [column for column in required_columns if column not in result.columns]
    if missing:
        raise ValueError(f"缺少必要字段: {', '.join(missing)}")

    result["日期"] = pd.to_datetime(result["日期"], errors="coerce")
    result = result.dropna(subset=["日期"])
    result = result.sort_values("日期").drop_duplicates(subset=["日期"], keep="last")
    result["日期"] = result["日期"].dt.strftime("%Y-%m-%d")

    numeric_columns = [column for column in required_columns if column != "日期"]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    preferred_columns = list(required_columns)
    remaining_columns = [column for column in result.columns if column not in preferred_columns]
    return result[preferred_columns + remaining_columns]


def filter_by_date(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    dates = pd.to_datetime(df["日期"], errors="coerce")
    start = pd.to_datetime(start_date, format="%Y%m%d")
    end = pd.to_datetime(end_date, format="%Y%m%d")
    return df.loc[(dates >= start) & (dates <= end)].copy()


def load_stock_codes(pool_path: Path, limit: int | None = None) -> list[str]:
    if not pool_path.exists():
        raise FileNotFoundError(f"股票池文件不存在: {pool_path}")

    pool = pd.read_csv(pool_path, dtype={"代码": str, "股票代码": str})
    code_column = "代码" if "代码" in pool.columns else "股票代码"
    if code_column not in pool.columns:
        raise ValueError("股票池 CSV 中未找到 '代码' 或 '股票代码' 列。")

    codes = (
        pool[code_column]
        .dropna()
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    if limit is not None:
        codes = codes[: max(0, limit)]
    if not codes:
        raise ValueError("股票池中没有可用股票代码。")
    return codes


def existing_file_is_valid(path: Path, min_trading_days: int) -> bool:
    if not path.exists():
        return False
    try:
        existing = pd.read_csv(path, usecols=["日期"])
    except Exception as exc:
        LOGGER.warning("已有文件无法读取，将重新下载: %s, %s", path, exc)
        return False
    return len(existing) >= min_trading_days


def fetch_stock_range(symbol: str, start_date: str, end_date: str, config: DownloadConfig) -> pd.DataFrame:
    """拉取单只股票 [start_date, end_date] 的前复权日线（不做最少交易日校验）。"""
    if config.request_delay_seconds > 0:
        time.sleep(config.request_delay_seconds)

    raw = retry_call(
        lambda: ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
            timeout=config.timeout_seconds,
        ),
        label=f"股票 {symbol}",
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )
    if raw is None or raw.empty:
        raise ValueError("返回数据为空")
    normalized = normalize_columns(raw, STOCK_COLUMN_ALIASES, REQUIRED_STOCK_COLUMNS)
    return filter_by_date(normalized, start_date, end_date)


def download_stock(symbol: str, config: DownloadConfig) -> pd.DataFrame:
    normalized = fetch_stock_range(symbol, config.start_date, config.end_date, config)
    if len(normalized) < config.min_trading_days:
        raise ValueError(f"交易日数量不足: {len(normalized)} < {config.min_trading_days}")
    return normalized


def update_stock_incremental(symbol: str, output_path: Path, config: DownloadConfig) -> tuple[pd.DataFrame, str]:
    """增量更新单只股票。返回 (最终DataFrame, 模式)；模式 ∈ {appended, no_new, full_refetch}。

    策略：拉最近 overlap_days 的小窗口，与已有文件重叠日期比对收盘价。
    - 一致 → 只把新增交易日追加到已有数据（快）。
    - 不一致（除权导致前复权整段缩放）→ 全量重下，保证历史一致。
    """
    existing = pd.read_csv(output_path, dtype={"日期": str})
    existing["日期"] = pd.to_datetime(existing["日期"], errors="coerce")
    existing = existing.dropna(subset=["日期"]).sort_values("日期").drop_duplicates("日期", keep="last")
    last_date = existing["日期"].max()

    # 回看窗口：用日历天数覆盖 overlap_days 个交易日（约 1.6 倍 + 缓冲）。
    lookback_start = (last_date - pd.Timedelta(days=config.overlap_days * 2 + 10)).strftime("%Y%m%d")
    window = fetch_stock_range(symbol, lookback_start, config.end_date, config)
    window["日期"] = pd.to_datetime(window["日期"], errors="coerce")
    window = window.dropna(subset=["日期"]).sort_values("日期")

    # 重叠日期收盘价比对（前复权一致性校验）。
    overlap = window[window["日期"] <= last_date]
    merged_overlap = overlap.merge(existing[["日期", "收盘"]], on="日期", suffixes=("_new", "_old"))
    mismatch = False
    if not merged_overlap.empty:
        rel_diff = (merged_overlap["收盘_new"] - merged_overlap["收盘_old"]).abs() / merged_overlap["收盘_old"].replace(0, pd.NA)
        mismatch = bool((rel_diff > 0.001).any())  # 容忍 0.1% 浮点误差
    else:
        mismatch = True  # 无重叠（停牌过久或窗口太短），稳妥起见全量重下

    if mismatch:
        full = download_stock(symbol, config)
        return full, "full_refetch"

    new_rows = window[window["日期"] > last_date]
    if new_rows.empty:
        existing["日期"] = existing["日期"].dt.strftime("%Y-%m-%d")
        return existing[list(REQUIRED_STOCK_COLUMNS)], "no_new"

    combined = pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.sort_values("日期").drop_duplicates("日期", keep="last")
    combined["日期"] = combined["日期"].dt.strftime("%Y-%m-%d")
    return combined[list(REQUIRED_STOCK_COLUMNS)], "appended"


def download_hs300(config: DownloadConfig) -> pd.DataFrame:
    if config.request_delay_seconds > 0:
        time.sleep(config.request_delay_seconds)

    raw = retry_call(
        lambda: ak.stock_zh_index_daily(symbol=config.hs300_symbol),
        label=f"沪深300 {config.hs300_symbol}",
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )
    if raw is None or raw.empty:
        raise ValueError("沪深300返回数据为空")

    normalized = normalize_columns(
        raw,
        INDEX_COLUMN_ALIASES,
        ("日期", "开盘", "收盘", "最高", "最低", "成交量"),
    )
    normalized = filter_by_date(normalized, config.start_date, config.end_date)
    if "成交额" not in normalized.columns:
        normalized["成交额"] = pd.NA
    if len(normalized) < config.min_trading_days:
        raise ValueError(f"沪深300交易日数量不足: {len(normalized)} < {config.min_trading_days}")
    return normalized[["日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额"]]


def save_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def run(config: DownloadConfig) -> int:
    setup_logging(config.output_dir)
    LOGGER.info(
        "开始下载日线数据: pool=%s, output_dir=%s, range=%s-%s",
        config.pool_path,
        config.output_dir,
        config.start_date,
        config.end_date,
    )

    hs300_path = config.output_dir / "HS300.csv"
    if not config.overwrite and not config.incremental and existing_file_is_valid(hs300_path, config.min_trading_days):
        LOGGER.info("沪深300已存在，跳过: %s", hs300_path)
    else:
        try:
            hs300 = download_hs300(config)
            save_csv(hs300, hs300_path)
            LOGGER.info("沪深300保存完成: %s, rows=%d", hs300_path, len(hs300))
        except Exception as exc:
            LOGGER.exception("沪深300下载失败: %s", exc)

    codes = load_stock_codes(config.pool_path, config.limit)
    LOGGER.info("待处理股票数量: %d", len(codes))

    success_count = 0
    skipped_count = 0
    failed_count = 0
    appended_count = 0
    refetch_count = 0
    for index, symbol in enumerate(codes, start=1):
        output_path = config.output_dir / f"{symbol}.csv"
        progress = f"[{index}/{len(codes)}] {symbol}"

        # 增量模式：已有有效文件则只拉小窗口追加；无文件则首次全量下载。
        if config.incremental and existing_file_is_valid(output_path, config.min_trading_days):
            try:
                df, mode = update_stock_incremental(symbol, output_path, config)
                save_csv(df, output_path)
                success_count += 1
                if mode == "appended":
                    appended_count += 1
                    LOGGER.info("%s 增量追加完成: rows=%d", progress, len(df))
                elif mode == "full_refetch":
                    refetch_count += 1
                    LOGGER.info("%s 检测到除权，已全量重下: rows=%d", progress, len(df))
                else:
                    LOGGER.info("%s 无新增交易日", progress)
                continue
            except Exception as exc:
                failed_count += 1
                LOGGER.exception("%s 增量更新失败: %s", progress, exc)
                continue

        if not config.overwrite and not config.incremental and existing_file_is_valid(output_path, config.min_trading_days):
            skipped_count += 1
            LOGGER.info("%s 已存在且有效，跳过", progress)
            continue

        try:
            df = download_stock(symbol, config)
            save_csv(df, output_path)
            success_count += 1
            LOGGER.info("%s 保存完成: rows=%d", progress, len(df))
        except Exception as exc:
            failed_count += 1
            LOGGER.exception("%s 跳过，原因: %s", progress, exc)

    LOGGER.info(
        "完成: 成功=%d (增量追加=%d, 除权重下=%d), 已跳过=%d, 失败/不足=%d, 输出目录=%s",
        success_count,
        appended_count,
        refetch_count,
        skipped_count,
        failed_count,
        config.output_dir.resolve(),
    )
    return 0 if success_count + skipped_count > 0 else 1


def main() -> int:
    args = parse_args()
    config = DownloadConfig(
        pool_path=args.pool,
        output_dir=args.output_dir,
        start_date=normalize_akshare_date(args.start_date),
        end_date=normalize_akshare_date(args.end_date),
        min_trading_days=args.min_trading_days,
        request_delay_seconds=args.delay,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff,
        timeout_seconds=args.timeout,
        hs300_symbol=args.hs300_symbol,
        overwrite=args.overwrite,
        limit=args.limit,
        incremental=args.incremental,
        overlap_days=args.overlap_days,
    )
    return run(config)


if __name__ == "__main__":
    raise SystemExit(main())
