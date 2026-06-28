#!/usr/bin/env python3
"""每日预测：加载已保存的生产模型，对最新一日因子做预测（无需重训）。

配合 run_strategy.py --mode daily 使用。模型由 rolling_lightgbm_train.py（每周）
训练并保存到 model_outputs/production_model.joblib。本脚本只做推理，秒级完成。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from factor_engineering import FACTOR_COLUMNS
from rolling_lightgbm_train import (
    MODEL_FILENAME,
    MODEL_META_FILENAME,
    TREND_OUTPUT_COLUMNS,
    cross_sectional_rank_transform,
    load_latest_features,
    stock_code_from_path,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DailyPredictConfig:
    feature_dir: Path = Path("features_data")
    model_dir: Path = Path("model_outputs")
    output_dir: Path = Path("model_outputs")
    latest_top_n: int = 20


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "predict_daily.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="每日预测：加载保存的模型对最新因子打分。")
    parser.add_argument("--feature-dir", type=Path, default=DailyPredictConfig.feature_dir)
    parser.add_argument("--model-dir", type=Path, default=DailyPredictConfig.model_dir)
    parser.add_argument("--output-dir", type=Path, default=DailyPredictConfig.output_dir)
    parser.add_argument("--latest-top-n", type=int, default=DailyPredictConfig.latest_top_n)
    return parser.parse_args()


def load_model_and_meta(model_dir: Path):
    model_path = model_dir / MODEL_FILENAME
    meta_path = model_dir / MODEL_META_FILENAME
    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到生产模型 {model_path}。请先运行 rolling_lightgbm_train.py（或 run_strategy.py --mode weekly）训练并保存模型。"
        )
    model = joblib.load(model_path)
    meta = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as handle:
            meta = json.load(handle)
    return model, meta


def build_predictions(config: DailyPredictConfig) -> pd.DataFrame:
    model, meta = load_model_and_meta(config.model_dir)
    cross_sectional_rank = bool(meta.get("cross_sectional_rank", True))
    LOGGER.info(
        "加载模型成功: 训练截止=%s, 横截面rank=%s",
        meta.get("train_end", "未知"),
        cross_sectional_rank,
    )

    latest = load_latest_features(config.feature_dir, cross_sectional_rank)
    latest["上涨概率"] = model.predict_proba(latest[list(FACTOR_COLUMNS)])[:, 1]
    latest["日期"] = latest["日期"].dt.strftime("%Y-%m-%d")
    latest["股票代码"] = latest["股票代码"].astype(str).str.zfill(6)
    for column in TREND_OUTPUT_COLUMNS:
        raw_column = f"{column}__raw"
        if raw_column in latest.columns:
            latest[column] = latest[raw_column]
    output_columns = ["日期", "股票代码", "上涨概率", *TREND_OUTPUT_COLUMNS]
    return latest[output_columns].sort_values("上涨概率", ascending=False).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    config = DailyPredictConfig(
        feature_dir=args.feature_dir,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        latest_top_n=args.latest_top_n,
    )
    setup_logging(config.output_dir)
    try:
        predictions = build_predictions(config)
        predictions.to_csv(config.output_dir / "latest_predictions_all.csv", index=False, encoding="utf-8-sig")
        predictions.head(config.latest_top_n).to_csv(
            config.output_dir / "latest_top20_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        LOGGER.info(
            "每日预测完成: date=%s, 候选=%d, top=%d",
            predictions["日期"].max(),
            len(predictions),
            config.latest_top_n,
        )
    except Exception as exc:
        LOGGER.exception("每日预测失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
