#!/usr/bin/env python3
"""Step 5: aggregate labeled samples and run rolling LightGBM training."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from factor_engineering import FACTOR_COLUMNS


LOGGER = logging.getLogger(__name__)
RAW_COLUMNS = ("日期", "开盘", "收盘", "最高", "最低", "成交量", "成交额")


@dataclass(frozen=True)
class TrainConfig:
    labeled_dir: Path = Path("labeled_data")
    feature_dir: Path = Path("features_data")
    output_dir: Path = Path("model_outputs")
    train_window_days: int = 600
    retrain_frequency_days: int = 120
    predict_horizon_days: int = 20
    n_splits: int = 3
    early_stopping_rounds: int = 30
    latest_top_n: int = 20
    random_state: int = 42
    n_estimators: int = 300
    min_final_estimators: int = 80
    learning_rate: float = 0.04
    num_leaves: int = 31
    min_child_samples: int = 80
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    scale_pos_weight: float | str | None = "auto"
    # 横截面 rank 标准化：把每个因子在"同一交易日所有股票"内转成百分位排名。
    # 这让模型学"今天谁相对最强"（横截面选股），而非"现在是不是好时段"（时间序列），
    # 直接提升每日 Top10 的命中率，并自动中性化 hs300 这类同日同值的市场因子。
    cross_sectional_rank: bool = True


def cross_sectional_rank_transform(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """将因子列按交易日做横截面百分位排名（同日内 rank(pct=True)）。"""
    ranked = data.copy()
    ranked[columns] = (
        data.groupby("日期")[columns].rank(pct=True).fillna(0.5)
    )
    return ranked


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "rolling_lightgbm_train.log", encoding="utf-8"),
        ],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="第5步：样本汇总与 LightGBM 滚动训练。")
    parser.add_argument("--labeled-dir", type=Path, default=TrainConfig.labeled_dir)
    parser.add_argument("--feature-dir", type=Path, default=TrainConfig.feature_dir)
    parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
    parser.add_argument("--train-window-days", type=int, default=TrainConfig.train_window_days)
    parser.add_argument("--retrain-frequency-days", type=int, default=TrainConfig.retrain_frequency_days)
    parser.add_argument("--predict-horizon-days", type=int, default=TrainConfig.predict_horizon_days)
    parser.add_argument("--n-splits", type=int, default=TrainConfig.n_splits)
    parser.add_argument("--latest-top-n", type=int, default=TrainConfig.latest_top_n)
    parser.add_argument("--n-estimators", type=int, default=TrainConfig.n_estimators)
    parser.add_argument("--min-final-estimators", type=int, default=TrainConfig.min_final_estimators)
    parser.add_argument(
        "--scale-pos-weight",
        default=TrainConfig.scale_pos_weight,
        help="正样本权重；默认 auto，即按每个训练窗口的负/正样本比例设置。",
    )
    parser.add_argument(
        "--no-cross-sectional-rank",
        action="store_true",
        help="关闭横截面 rank 标准化（默认开启，提升每日 Top10 命中率）。",
    )
    return parser.parse_args()


def stock_code_from_path(path: Path) -> str:
    return path.stem.zfill(6)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_labeled_samples(labeled_dir: Path, cross_sectional_rank: bool = True) -> pd.DataFrame:
    frames = []
    files = sorted(labeled_dir.glob("*.csv"))
    for path in files:
        if path.stat().st_size == 0:
            raise ValueError(f"标签文件为空: {path}")
        df = read_csv(path)
        missing = [column for column in ("日期", "target", *FACTOR_COLUMNS) if column not in df.columns]
        if missing:
            raise ValueError(f"{path} 缺少列: {', '.join(missing)}")
        df = df.copy()
        df["股票代码"] = stock_code_from_path(path)
        frames.append(df)
    if not frames:
        raise ValueError(f"标签目录没有 CSV: {labeled_dir}")
    data = pd.concat(frames, ignore_index=True)
    data["日期"] = pd.to_datetime(data["日期"], errors="coerce")
    data = data.dropna(subset=["日期", "target"]).sort_values(["日期", "股票代码"]).reset_index(drop=True)
    for column in FACTOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[list(FACTOR_COLUMNS)] = data[list(FACTOR_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    data["target"] = data["target"].astype(int)
    if cross_sectional_rank:
        data = cross_sectional_rank_transform(data, list(FACTOR_COLUMNS))
    return data


def load_latest_features(feature_dir: Path, cross_sectional_rank: bool = True) -> pd.DataFrame:
    rows = []
    for path in sorted(feature_dir.glob("*.csv")):
        if path.stat().st_size == 0:
            continue
        df = read_csv(path)
        missing = [column for column in ("日期", *FACTOR_COLUMNS) if column not in df.columns]
        if missing:
            raise ValueError(f"{path} 缺少列: {', '.join(missing)}")
        row = df.tail(1).copy()
        row["股票代码"] = stock_code_from_path(path)
        rows.append(row)
    if not rows:
        raise ValueError(f"因子目录没有可预测样本: {feature_dir}")
    latest = pd.concat(rows, ignore_index=True)
    latest["日期"] = pd.to_datetime(latest["日期"], errors="coerce")
    for column in FACTOR_COLUMNS:
        latest[column] = pd.to_numeric(latest[column], errors="coerce")
    latest[list(FACTOR_COLUMNS)] = latest[list(FACTOR_COLUMNS)].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 保留趋势列原始值（横截面 rank 后会被覆盖），供第6步可选趋势过滤使用。
    for column in TREND_OUTPUT_COLUMNS:
        latest[f"{column}__raw"] = latest[column]
    if cross_sectional_rank:
        latest = cross_sectional_rank_transform(latest, list(FACTOR_COLUMNS))
    return latest.sort_values(["日期", "股票代码"]).reset_index(drop=True)


def make_model(config: TrainConfig) -> lgb.LGBMClassifier:
    params = {
        "objective": "binary",
        "n_estimators": config.n_estimators,
        "learning_rate": config.learning_rate,
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample_bytree,
        "random_state": config.random_state,
        "n_jobs": -1,
        "verbose": -1,
    }
    if isinstance(config.scale_pos_weight, (int, float)):
        params["scale_pos_weight"] = config.scale_pos_weight
    return lgb.LGBMClassifier(**params)


def apply_auto_class_weight(model: lgb.LGBMClassifier, y_train: pd.Series, config: TrainConfig) -> None:
    if config.scale_pos_weight != "auto":
        return
    positive = int((y_train == 1).sum())
    negative = int((y_train == 0).sum())
    if positive > 0 and negative > 0:
        model.set_params(scale_pos_weight=negative / positive)


def fit_with_validation(
    model: lgb.LGBMClassifier,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    config: TrainConfig,
) -> lgb.LGBMClassifier:
    x_train = train_df[list(FACTOR_COLUMNS)]
    y_train = train_df["target"]
    apply_auto_class_weight(model, y_train, config)
    if valid_df is None or valid_df["target"].nunique() < 2:
        model.fit(x_train, y_train)
        return model
    model.fit(
        x_train,
        y_train,
        eval_set=[(valid_df[list(FACTOR_COLUMNS)], valid_df["target"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(config.early_stopping_rounds, verbose=False)],
    )
    return model


def auc_or_nan(y_true: pd.Series, score: np.ndarray) -> float:
    if y_true.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def cross_validate_by_dates(train_df: pd.DataFrame, train_dates: np.ndarray, config: TrainConfig) -> float:
    if len(train_dates) <= config.n_splits:
        return float("nan")
    splitter = TimeSeriesSplit(n_splits=min(config.n_splits, len(train_dates) - 1))
    aucs = []
    for train_index, valid_index in splitter.split(train_dates):
        fold_train_dates = set(train_dates[train_index])
        fold_valid_dates = set(train_dates[valid_index])
        fold_train = train_df[train_df["日期"].isin(fold_train_dates)]
        fold_valid = train_df[train_df["日期"].isin(fold_valid_dates)]
        if fold_train["target"].nunique() < 2:
            continue
        model = make_model(config)
        fit_with_validation(model, fold_train, fold_valid, config)
        score = model.predict_proba(fold_valid[list(FACTOR_COLUMNS)])[:, 1]
        aucs.append(auc_or_nan(fold_valid["target"], score))
    aucs = [value for value in aucs if not np.isnan(value)]
    return float(np.mean(aucs)) if aucs else float("nan")


def train_final_model(train_df: pd.DataFrame, train_dates: np.ndarray, config: TrainConfig) -> lgb.LGBMClassifier:
    split_at = max(1, int(len(train_dates) * 0.8))
    if split_at >= len(train_dates):
        fit_df = train_df
        valid_df = None
    else:
        fit_dates = set(train_dates[:split_at])
        valid_dates = set(train_dates[split_at:])
        fit_df = train_df[train_df["日期"].isin(fit_dates)]
        valid_df = train_df[train_df["日期"].isin(valid_dates)]
    model = make_model(config)
    fit_with_validation(model, fit_df, valid_df, config)
    best_iteration = getattr(model, "best_iteration_", None)
    if valid_df is None or not best_iteration or best_iteration <= 0:
        return model

    # Early stopping selects complexity on the tail validation slice; the final
    # production model must still learn from the full rolling window.
    final_config = TrainConfig(
        labeled_dir=config.labeled_dir,
        feature_dir=config.feature_dir,
        output_dir=config.output_dir,
        train_window_days=config.train_window_days,
        retrain_frequency_days=config.retrain_frequency_days,
        predict_horizon_days=config.predict_horizon_days,
        n_splits=config.n_splits,
        early_stopping_rounds=config.early_stopping_rounds,
        latest_top_n=config.latest_top_n,
        random_state=config.random_state,
        n_estimators=best_iteration if best_iteration >= config.min_final_estimators else config.n_estimators,
        min_final_estimators=config.min_final_estimators,
        learning_rate=config.learning_rate,
        num_leaves=config.num_leaves,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        scale_pos_weight=config.scale_pos_weight,
    )
    final_model = make_model(final_config)
    apply_auto_class_weight(final_model, train_df["target"], final_config)
    final_model.fit(train_df[list(FACTOR_COLUMNS)], train_df["target"])
    return final_model


def run_rolling_training(data: pd.DataFrame, config: TrainConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = np.array(sorted(data["日期"].dropna().unique()))
    if len(dates) <= config.train_window_days:
        raise ValueError(f"交易日不足: {len(dates)} <= {config.train_window_days}")

    predictions = []
    metrics = []
    for start in range(config.train_window_days, len(dates), config.retrain_frequency_days):
        train_dates = dates[start - config.train_window_days : start]
        predict_dates = dates[start : min(start + config.predict_horizon_days, len(dates))]
        if len(predict_dates) == 0:
            continue

        train_df = data[data["日期"].isin(set(train_dates))]
        predict_df = data[data["日期"].isin(set(predict_dates))].copy()
        if train_df["target"].nunique() < 2 or predict_df.empty:
            LOGGER.warning("跳过窗口 start=%s: 训练标签单一或预测样本为空", pd.Timestamp(dates[start]).date())
            continue

        cv_auc = cross_validate_by_dates(train_df, train_dates, config)
        model = train_final_model(train_df, train_dates, config)
        score = model.predict_proba(predict_df[list(FACTOR_COLUMNS)])[:, 1]
        predict_auc = auc_or_nan(predict_df["target"], score)

        predict_df["上涨概率"] = score
        predict_df["训练结束日"] = pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d")
        predict_df["预测窗口开始"] = pd.Timestamp(predict_dates[0]).strftime("%Y-%m-%d")
        predict_df["预测窗口结束"] = pd.Timestamp(predict_dates[-1]).strftime("%Y-%m-%d")
        predictions.append(predict_df[["日期", "股票代码", "上涨概率", "target", "训练结束日", "预测窗口开始", "预测窗口结束"]])
        metrics.append(
            {
                "训练开始日": pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
                "训练结束日": pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"),
                "预测开始日": pd.Timestamp(predict_dates[0]).strftime("%Y-%m-%d"),
                "预测结束日": pd.Timestamp(predict_dates[-1]).strftime("%Y-%m-%d"),
                "cv_auc": cv_auc,
                "predict_auc": predict_auc,
                "train_rows": len(train_df),
                "predict_rows": len(predict_df),
            }
        )
        LOGGER.info(
            "窗口 %s -> %s, CV AUC=%.4f, 预测区间 AUC=%.4f, rows=%d",
            pd.Timestamp(train_dates[0]).date(),
            pd.Timestamp(train_dates[-1]).date(),
            cv_auc,
            predict_auc,
            len(predict_df),
        )

    if not predictions:
        raise RuntimeError("没有生成任何滚动预测。")
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics)


# 趋势列随预测一起输出，供第6步做"健康上升趋势"硬过滤，剔除下跌趋势股票。
TREND_OUTPUT_COLUMNS = ("close_ma20_pos", "close_ma60_pos", "ma_arrangement", "return_20d", "return_5d")

MODEL_FILENAME = "production_model.joblib"
MODEL_META_FILENAME = "production_model_meta.json"


def save_production_model(
    model: lgb.LGBMClassifier,
    config: TrainConfig,
    train_dates: np.ndarray,
    train_rows: int,
) -> None:
    """保存生产模型与元数据，供 predict_daily.py 每日加载预测（无需重训）。"""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.output_dir / MODEL_FILENAME)
    meta = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_start": pd.Timestamp(train_dates[0]).strftime("%Y-%m-%d"),
        "train_end": pd.Timestamp(train_dates[-1]).strftime("%Y-%m-%d"),
        "train_window_days": int(config.train_window_days),
        "train_rows": int(train_rows),
        "cross_sectional_rank": bool(config.cross_sectional_rank),
        "factor_columns": list(FACTOR_COLUMNS),
        "n_estimators_effective": int(getattr(model, "n_estimators", 0)),
    }
    with open(config.output_dir / MODEL_META_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    LOGGER.info("生产模型已保存: %s (训练截止=%s)", MODEL_FILENAME, meta["train_end"])


def predict_latest(data: pd.DataFrame, latest_features: pd.DataFrame, config: TrainConfig) -> pd.DataFrame:
    labeled_dates = np.array(sorted(data["日期"].dropna().unique()))
    train_dates = labeled_dates[-config.train_window_days :]
    train_df = data[data["日期"].isin(set(train_dates))]
    model = train_final_model(train_df, train_dates, config)
    save_production_model(model, config, train_dates, len(train_df))
    latest = latest_features.copy()
    latest["上涨概率"] = model.predict_proba(latest[list(FACTOR_COLUMNS)])[:, 1]
    latest["日期"] = latest["日期"].dt.strftime("%Y-%m-%d")
    latest["股票代码"] = latest["股票代码"].astype(str).str.zfill(6)
    # 趋势列输出原始值（若横截面 rank 已覆盖原列，则用预留的 __raw 列还原），供第6步可选过滤。
    for column in TREND_OUTPUT_COLUMNS:
        raw_column = f"{column}__raw"
        if raw_column in latest.columns:
            latest[column] = latest[raw_column]
    output_columns = ["日期", "股票代码", "上涨概率", *TREND_OUTPUT_COLUMNS]
    return latest[output_columns].sort_values("上涨概率", ascending=False).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    config = TrainConfig(
        labeled_dir=args.labeled_dir,
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        train_window_days=args.train_window_days,
        retrain_frequency_days=args.retrain_frequency_days,
        predict_horizon_days=args.predict_horizon_days,
        n_splits=args.n_splits,
        latest_top_n=args.latest_top_n,
        n_estimators=args.n_estimators,
        min_final_estimators=args.min_final_estimators,
        scale_pos_weight=(
            "auto"
            if args.scale_pos_weight == "auto"
            else None
            if str(args.scale_pos_weight).lower() in {"none", "false", "0"}
            else float(args.scale_pos_weight)
        ),
        cross_sectional_rank=not args.no_cross_sectional_rank,
    )
    setup_logging(config.output_dir)
    try:
        data = load_labeled_samples(config.labeled_dir, config.cross_sectional_rank)
        LOGGER.info("合并样本完成: rows=%d, stocks=%d", len(data), data["股票代码"].nunique())
        combined = data.copy()
        combined["日期"] = combined["日期"].dt.strftime("%Y-%m-%d")
        combined.to_csv(config.output_dir / "combined_labeled_data.csv", index=False, encoding="utf-8-sig")

        rolling_predictions, metrics = run_rolling_training(data, config)
        rolling_predictions["日期"] = pd.to_datetime(rolling_predictions["日期"]).dt.strftime("%Y-%m-%d")
        rolling_predictions["股票代码"] = rolling_predictions["股票代码"].astype(str).str.zfill(6)
        rolling_predictions.to_csv(config.output_dir / "rolling_predictions.csv", index=False, encoding="utf-8-sig")
        metrics.to_csv(config.output_dir / "rolling_auc_metrics.csv", index=False, encoding="utf-8-sig")

        latest_features = load_latest_features(config.feature_dir, config.cross_sectional_rank)
        latest_predictions = predict_latest(data, latest_features, config)
        latest_predictions.to_csv(config.output_dir / "latest_predictions_all.csv", index=False, encoding="utf-8-sig")
        latest_predictions.head(config.latest_top_n).to_csv(
            config.output_dir / "latest_top20_predictions.csv",
            index=False,
            encoding="utf-8-sig",
        )
        LOGGER.info("最新预测完成: date=%s, top=%d", latest_predictions["日期"].max(), config.latest_top_n)
    except Exception as exc:
        LOGGER.exception("第5步失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
