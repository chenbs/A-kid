#!/usr/bin/env python3
"""选股策略日常运维编排器。

两种模式：

  --mode daily   每日盘后运行：刷新股票池 → 下载最新日线 → 计算因子 →
                 加载已保存模型预测 → 生成最终信号。不重训，秒级出模型预测。

  --mode weekly  每周（或每月）运行：刷新股票池 → 下载日线 → 因子 → 标签 →
                 滚动训练并保存生产模型 → 生成信号 → T+1 回测复盘。

依赖：所有子步骤复用现有脚本，保证逻辑单一来源。
用法示例：
  .venv/Scripts/python.exe run_strategy.py --mode daily
  .venv/Scripts/python.exe run_strategy.py --mode weekly
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

LOGGER = logging.getLogger(__name__)

PYTHON = sys.executable
ROOT = Path(__file__).resolve().parent


def setup_logging() -> None:
    log_dir = ROOT / "model_outputs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_dir / "run_strategy.log", encoding="utf-8"),
        ],
    )


def run_step(name: str, args: list[str]) -> None:
    """运行一个子步骤，失败即抛异常中断整条流水线。"""
    LOGGER.info("▶ 开始: %s  ->  %s", name, " ".join(args))
    started = time.time()
    result = subprocess.run([PYTHON, *args], cwd=ROOT)
    elapsed = time.time() - started
    if result.returncode != 0:
        raise RuntimeError(f"步骤失败: {name} (exit={result.returncode})")
    LOGGER.info("✔ 完成: %s  耗时 %.1f 秒", name, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="选股策略日/周运维编排器。")
    parser.add_argument("--mode", choices=["daily", "weekly"], required=True, help="daily=每日预测；weekly=重训+回测。")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载步骤（用已有 stock_data 调试）。")
    parser.add_argument("--skip-backtest", action="store_true", help="weekly 模式下跳过回测。")
    parser.add_argument("--trend-filter", action="store_true", help="信号阶段开启趋势过滤（默认关闭）。")
    return parser.parse_args()


def daily_pipeline(args: argparse.Namespace) -> None:
    """每日：刷新池 → 下载最新 → 因子 → 加载模型预测 → 信号。"""
    if not (ROOT / "model_outputs" / "production_model.joblib").exists():
        raise FileNotFoundError(
            "未找到已保存的生产模型，请先运行一次 --mode weekly 训练并保存模型。"
        )
    run_step("第1步 刷新股票池", ["stock_pool_filter.py"])
    if not args.skip_download:
        # 增量更新：只拉最近窗口追加；遇除权(前复权整段缩放)自动回退全量重下该股。
        run_step("第2步 增量下载最新日线", ["download_stock_history.py", "--incremental"])
    run_step("第3步 计算因子", ["factor_engineering.py"])
    run_step("每日预测 加载模型打分", ["predict_daily.py"])
    signal_args = ["signal_filter.py"]
    if args.trend_filter:
        signal_args.append("--trend-filter")
    run_step("第6步 生成最终信号", signal_args)
    # 登记当日Top10到台账，并回填已满两个交易日的历史信号、输出跟踪报告。
    run_step("跟踪 登记信号+回填真实涨跌", ["track_performance.py"])


def weekly_pipeline(args: argparse.Namespace) -> None:
    """每周：刷新池 → 下载 → 因子 → 标签 → 训练保存 → 信号 → 回测。"""
    run_step("第1步 刷新股票池", ["stock_pool_filter.py"])
    if not args.skip_download:
        run_step("第2步 下载日线", ["download_stock_history.py", "--overwrite"])
    run_step("第3步 计算因子", ["factor_engineering.py"])
    run_step("第4步 生成标签", ["label_engineering.py"])
    run_step("第5步 滚动训练并保存模型", ["rolling_lightgbm_train.py"])
    signal_args = ["signal_filter.py"]
    if args.trend_filter:
        signal_args.append("--trend-filter")
    run_step("第6步 生成最终信号", signal_args)
    run_step("跟踪 登记信号+回填真实涨跌", ["track_performance.py"])
    if not args.skip_backtest:
        run_step("第7步 T+1 回测复盘", ["t1_backtest.py"])


def main() -> int:
    args = parse_args()
    setup_logging()
    started = time.time()
    LOGGER.info("=== 策略运行开始 mode=%s ===", args.mode)
    try:
        if args.mode == "daily":
            daily_pipeline(args)
        else:
            weekly_pipeline(args)
    except Exception as exc:
        LOGGER.exception("流水线中断: %s", exc)
        return 1
    LOGGER.info("=== 全部完成 mode=%s, 总耗时 %.1f 秒 ===", args.mode, time.time() - started)
    LOGGER.info("最终信号: %s", (ROOT / "final_signals" / "final_signals.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
