#!/usr/bin/env python3
"""Step 4: create no-leakage forward return labels."""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LabelConfig:
    feature_dir: Path = Path("features_data")
    output_dir: Path = Path("labeled_data")
    horizon_days: int = 2
    threshold: float = 0.03
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
            logging.FileHandler(output_dir / "label_engineering.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第4步：生成未来2个交易日涨幅标签。")
    parser.add_argument("--feature-dir", type=Path, default=LabelConfig.feature_dir)
    parser.add_argument("--output-dir", type=Path, default=LabelConfig.output_dir)
    parser.add_argument("--horizon-days", type=int, default=LabelConfig.horizon_days)
    parser.add_argument("--threshold", type=float, default=LabelConfig.threshold)
    parser.add_argument("--min-rows", type=int, default=LabelConfig.min_rows)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-overwrite", action="store_true")
    return parser.parse_args()


def add_forward_return_target(
    df: pd.DataFrame,
    *,
    horizon_days: int = 2,
    threshold: float = 0.03,
) -> pd.DataFrame:
    if "日期" not in df.columns or "收盘" not in df.columns:
        raise ValueError("输入数据必须包含 '日期' 和 '收盘' 列。")

    result = df.copy()
    result["日期"] = pd.to_datetime(result["日期"], errors="coerce")
    result["收盘"] = pd.to_numeric(result["收盘"], errors="coerce")
    result = result.dropna(subset=["日期", "收盘"]).sort_values("日期").reset_index(drop=True)
    future_return = result["收盘"].shift(-horizon_days) / result["收盘"] - 1.0
    result["target"] = np.where(future_return >= threshold, 1, 0).astype(float)
    result.loc[future_return.isna(), "target"] = np.nan
    result = result.dropna(subset=["target"]).copy()
    result["target"] = result["target"].astype(int)
    result["日期"] = result["日期"].dt.strftime("%Y-%m-%d")
    return result


def feature_files(feature_dir: Path, limit: int | None) -> list[Path]:
    files = sorted(feature_dir.glob("*.csv"))
    if limit is not None:
        return files[: max(0, limit)]
    return files


def generate_labeled_files(config: LabelConfig) -> dict[str, int]:
    setup_logging(config.output_dir)
    files = feature_files(config.feature_dir, config.limit)
    LOGGER.info("第4步开始: 待处理因子文件=%d", len(files))

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
            df = pd.read_csv(path)
            labeled = add_forward_return_target(
                df,
                horizon_days=config.horizon_days,
                threshold=config.threshold,
            )
            if len(labeled) < config.min_rows:
                raise ValueError(f"加标签后样本不足: {len(labeled)} < {config.min_rows}")
            labeled.to_csv(output_path, index=False, encoding="utf-8-sig")
            positive_rate = labeled["target"].mean()
            success += 1
            LOGGER.info("%s 标签保存完成: rows=%d, positive_rate=%.4f", progress, len(labeled), positive_rate)
        except Exception as exc:
            failed += 1
            LOGGER.exception("%s 标签生成失败: %s", progress, exc)

    LOGGER.info("第4步完成: 成功=%d, 跳过=%d, 失败=%d, 输出目录=%s", success, skipped, failed, config.output_dir.resolve())
    return {"success": success, "skipped": skipped, "failed": failed}


def main() -> int:
    args = parse_args()
    config = LabelConfig(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        horizon_days=args.horizon_days,
        threshold=args.threshold,
        min_rows=args.min_rows,
        overwrite=not args.no_overwrite,
        limit=args.limit,
    )
    result = generate_labeled_files(config)
    return 0 if result["success"] + result["skipped"] > 0 and result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
