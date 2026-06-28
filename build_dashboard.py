#!/usr/bin/env python3
"""生成自包含的策略看板 dashboard.html。

读取 final_signals 与 tracking/ 下的台账、汇总、校准数据，把它们内嵌进一个
不依赖任何外部资源、无需服务器的 dashboard.html。双击即可在浏览器查看：
今日 Top10、近一月每日表现、整体/滚动指标、概率校准、明细台账。

接入 run_strategy.py 的日/周流程后每天自动刷新。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class DashboardConfig:
    signals_path: Path = Path("final_signals/final_signals.csv")
    ledger_path: Path = Path("tracking/signal_ledger.csv")
    summary_path: Path = Path("tracking/performance_summary.csv")
    by_day_path: Path = Path("tracking/performance_by_day.csv")
    calibration_path: Path = Path("tracking/calibration.csv")
    model_meta_path: Path = Path("model_outputs/production_model_meta.json")
    output_path: Path = Path("dashboard.html")
    recent_days: int = 30


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def read_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame()


def df_to_records(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    safe = df.replace([np.inf, -np.inf], np.nan).where(pd.notna(df), None)
    return safe.to_dict(orient="records")
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>A股选股策略看板</title>
<style>
:root{--bg:#0f1419;--card:#1a212b;--line:#2a3441;--txt:#e6edf3;--mut:#8b97a7;--up:#f0524c;--dn:#2ea043;--accent:#4493f8;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;font-size:14px;line-height:1.5;padding:20px;}
h1{font-size:20px;margin-bottom:4px;}
h2{font-size:15px;margin:0 0 12px;color:var(--accent);font-weight:600;}
.sub{color:var(--mut);font-size:12px;margin-bottom:20px;}
.grid{display:grid;gap:16px;}
.cards{grid-template-columns:repeat(auto-fit,minmax(150px,1fr));}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;}
.metric .label{color:var(--mut);font-size:12px;}
.metric .value{font-size:24px;font-weight:700;margin-top:4px;}
.section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:18px;margin-top:16px;}
table{width:100%;border-collapse:collapse;font-size:13px;}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap;}
th{color:var(--mut);font-weight:600;position:sticky;top:0;background:var(--card);}
td.l,th.l{text-align:left;}
.up{color:var(--up);} .dn{color:var(--dn);} .mut{color:var(--mut);}
.tag{display:inline-block;padding:2px 8px;border-radius:6px;font-size:12px;}
.tag.win{background:rgba(240,82,76,.15);color:var(--up);}
.tag.lose{background:rgba(46,160,67,.15);color:var(--dn);}
.tag.pend{background:rgba(139,151,167,.15);color:var(--mut);}
.scroll{max-height:520px;overflow:auto;}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:6px;font-size:12px;}
.bar-row .d{width:84px;color:var(--mut);}
.bar-track{flex:1;background:#0c1117;border-radius:4px;height:18px;position:relative;overflow:hidden;}
.bar-fill{height:100%;border-radius:4px;}
.bar-row .v{width:96px;text-align:right;}
.warn{background:rgba(240,82,76,.12);border:1px solid var(--up);border-radius:8px;padding:10px 14px;margin-top:12px;color:var(--up);font-size:13px;}
.ok{background:rgba(46,160,67,.10);border:1px solid var(--dn);border-radius:8px;padding:10px 14px;margin-top:12px;color:var(--dn);font-size:13px;}
.empty{color:var(--mut);padding:24px;text-align:center;}
.filters{margin-bottom:10px;}
select{background:#0c1117;color:var(--txt);border:1px solid var(--line);border-radius:6px;padding:6px 10px;font-size:13px;}
</style>
</head>
<body>
<h1>A股选股策略看板</h1>
<div class="sub" id="meta"></div>
<div id="alert"></div>
<div class="section"><h2 id="today-title">今日信号 Top10</h2><div id="today"></div></div>
<div class="grid cards" id="summary"></div>
<div class="section"><h2>近一月每日命中率</h2><div id="byday"></div></div>
<div class="section"><h2>概率校准（模型说的概率 vs 真实命中率）</h2><div id="calib"></div></div>
<div class="section">
  <h2>历史台账明细（每日10只 + 真实涨跌）</h2>
  <div class="filters">信号日：<select id="dayFilter"></select></div>
  <div class="scroll" id="ledger"></div>
</div>
<script>
const DATA = /*__DATA__*/;
// ---- helpers ----
const $ = id => document.getElementById(id);
const num = (v,d=2) => (v===null||v===undefined||v==="")?"-":Number(v).toFixed(d);
const pct = (v,d=1) => (v===null||v===undefined||v==="")?"-":(Number(v)*100).toFixed(d)+"%";
const cls = v => (v===null||v===undefined||v==="")?"":(Number(v)>0?"up":(Number(v)<0?"dn":""));
function code6(v){ const s=String(v==null?"":v).replace(/\D/g,""); return s.padStart(6,"0").slice(-6); }
function yi(v){ return (v===null||v===undefined||v==="")?"-":(Number(v)/1e8).toFixed(1)+"亿"; }
// ---- meta ----
(function(){
  const m = DATA.model_meta||{};
  let s = "生成时间 "+DATA.generated_at;
  if(m.train_end) s += " ｜ 模型训练截止 "+m.train_end;
  if(m.train_rows) s += " ｜ 训练样本 "+m.train_rows;
  $("meta").textContent = s;
})();

// ---- today's signals ----
(function(){
  const rows = DATA.signals||[];
  if(!rows.length){ $("today").innerHTML='<div class="empty">暂无今日信号</div>'; return; }
  $("today-title").textContent = "今日信号 Top10　（信号日 "+(rows[0]["日期"]||"")+"）";
  let h='<table><thead><tr><th>#</th><th class="l">代码</th><th class="l">名称</th><th>上涨概率</th><th>总市值</th><th>成交额</th></tr></thead><tbody>';
  rows.forEach((r,i)=>{
    h+='<tr><td class="mut">'+(i+1)+'</td><td class="l">'+code6(r["股票代码"])+'</td><td class="l">'+(r["名称"]||"")
      +'</td><td><b>'+pct(r["上涨概率"],1)+'</b></td><td>'+yi(r["总市值"])+'</td><td>'+yi(r["成交额"])+'</td></tr>';
  });
  $("today").innerHTML = h+'</tbody></table>';
})();

// ---- summary cards + decay alert ----
(function(){
  const s = (DATA.summary&&DATA.summary[0])||null;
  if(!s){ $("summary").innerHTML='<div class="card empty">暂无已结算的跟踪数据（信号需满两个交易日后才有真实涨跌）</div>'; return; }
  const items = [];
  for(const k in s){
    let v = s[k], disp;
    if(k.indexOf("命中率")>=0||k.indexOf("胜率")>=0) disp = pct(v,1);
    else if(k.indexOf("收益")>=0) disp = pct(v,2);
    else disp = (v===null?"-":v);
    items.push('<div class="card metric"><div class="label">'+k+'</div><div class="value">'+disp+'</div></div>');
  }
  $("summary").innerHTML = items.join("");
  // 衰减预警：找滚动命中率字段
  let rollKey=Object.keys(s).find(k=>k.indexOf("滚动命中率")>=0);
  if(rollKey && s[rollKey]!==null){
    const hit=Number(s[rollKey]);
    if(hit<0.25) $("alert").innerHTML='<div class="warn">⚠ 模型衰减预警：'+rollKey+' '+pct(hit)+' 低于 25%（回测基准约33%）。建议尽快 run_strategy.py --mode weekly 重训并复盘因子。</div>';
    else $("alert").innerHTML='<div class="ok">模型状态正常：'+rollKey+' '+pct(hit)+' ≥ 25%。</div>';
  }
})();
// ---- by-day hit rate bars ----
(function(){
  const rows = DATA.by_day||[];
  if(!rows.length){ $("byday").innerHTML='<div class="empty">暂无每日表现数据</div>'; return; }
  let h="";
  rows.slice().reverse().forEach(r=>{
    const hit = r["命中率"]==null?0:Number(r["命中率"]);
    const w = Math.max(2, Math.min(100, hit*100));
    const color = hit>=0.33 ? "var(--up)" : (hit>=0.2 ? "#d8a000" : "var(--dn)");
    h+='<div class="bar-row"><div class="d">'+r["信号日"]+'</div><div class="bar-track">'
      +'<div class="bar-fill" style="width:'+w+'%;background:'+color+'"></div></div>'
      +'<div class="v">'+pct(r["命中率"],1)+' ('+(r["只数"]||0)+'只)</div></div>';
  });
  $("byday").innerHTML = h;
})();

// ---- calibration ----
(function(){
  const rows = DATA.calibration||[];
  if(!rows.length){ $("calib").innerHTML='<div class="empty">样本不足，暂无校准曲线（已结算样本≥20才计算）</div>'; return; }
  let h='<table><thead><tr><th class="l">概率分桶</th><th>平均预测概率</th><th>真实命中率</th><th>样本数</th></tr></thead><tbody>';
  rows.forEach(r=>{
    h+='<tr><td class="l mut">'+(r["概率桶"]||"")+'</td><td>'+pct(r["平均预测概率"],1)+'</td><td><b>'
      +pct(r["真实命中率"],1)+'</b></td><td>'+(r["样本数"]||0)+'</td></tr>';
  });
  $("calib").innerHTML = h+'</tbody></table>';
})();

// ---- ledger with day filter ----
(function(){
  const all = DATA.ledger||[];
  const sel = $("dayFilter");
  if(!all.length){ $("ledger").innerHTML='<div class="empty">台账为空</div>'; return; }
  const days = [...new Set(all.map(r=>r["信号日"]))].sort().reverse();
  sel.innerHTML = '<option value="__all__">全部（'+all.length+'条）</option>'
    + days.map(d=>'<option value="'+d+'">'+d+'</option>').join("");
  function statusTag(r){
    if(r["状态"]!=="已结算") return '<span class="tag pend">待结算</span>';
    const hit = r["命中(标签>=3%)"];
    return hit==1 ? '<span class="tag win">命中</span>' : '<span class="tag lose">未中</span>';
  }
  function render(day){
    const rows = day==="__all__" ? all : all.filter(r=>r["信号日"]===day);
    let h='<table><thead><tr><th class="l">信号日</th><th class="l">代码</th><th class="l">名称</th><th>预测概率</th>'
      +'<th>标签2日涨幅</th><th>交易净收益</th><th class="l">结果</th></tr></thead><tbody>';
    rows.forEach(r=>{
      h+='<tr><td class="l mut">'+r["信号日"]+'</td><td class="l">'+code6(r["股票代码"])+'</td><td class="l">'+(r["名称"]||"")
        +'</td><td>'+pct(r["上涨概率"],1)+'</td>'
        +'<td class="'+cls(r["标签口径收益"])+'">'+pct(r["标签口径收益"],2)+'</td>'
        +'<td class="'+cls(r["交易口径净收益"])+'">'+pct(r["交易口径净收益"],2)+'</td>'
        +'<td class="l">'+statusTag(r)+'</td></tr>';
    });
    $("ledger").innerHTML = h+'</tbody></table>';
  }
  sel.addEventListener("change",()=>render(sel.value));
  render("__all__");
})();
</script>
</body>
</html>"""


def collect_data(config: DashboardConfig) -> dict:
    signals = read_csv(config.signals_path)
    ledger = read_csv(config.ledger_path)
    summary = read_csv(config.summary_path)
    by_day = read_csv(config.by_day_path)
    calibration = read_csv(config.calibration_path)

    meta = {}
    if config.model_meta_path.exists():
        try:
            meta = json.loads(config.model_meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

    # 近一月每日表现（按信号日倒序取最近 N 天）
    if not by_day.empty and "信号日" in by_day.columns:
        by_day = by_day.sort_values("信号日").tail(config.recent_days)

    # 台账：转成百分比、按信号日倒序，便于回看
    if not ledger.empty:
        for col in ("标签口径收益", "交易口径净收益"):
            if col in ledger.columns:
                ledger[col] = pd.to_numeric(ledger[col], errors="coerce")
        if "信号日" in ledger.columns:
            ledger = ledger.sort_values(["信号日", "上涨概率"], ascending=[False, False])

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_meta": meta,
        "signals": df_to_records(signals),
        "ledger": df_to_records(ledger),
        "summary": df_to_records(summary),
        "by_day": df_to_records(by_day),
        "calibration": df_to_records(calibration),
        "recent_days": config.recent_days,
    }


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False)
    return HTML_TEMPLATE.replace("/*__DATA__*/", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成自包含的策略看板 dashboard.html。")
    parser.add_argument("--signals", type=Path, default=DashboardConfig.signals_path)
    parser.add_argument("--ledger", type=Path, default=DashboardConfig.ledger_path)
    parser.add_argument("--summary", type=Path, default=DashboardConfig.summary_path)
    parser.add_argument("--by-day", type=Path, default=DashboardConfig.by_day_path)
    parser.add_argument("--calibration", type=Path, default=DashboardConfig.calibration_path)
    parser.add_argument("--output", type=Path, default=DashboardConfig.output_path)
    parser.add_argument("--recent-days", type=int, default=DashboardConfig.recent_days)
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    config = DashboardConfig(
        signals_path=args.signals,
        ledger_path=args.ledger,
        summary_path=args.summary,
        by_day_path=args.by_day,
        calibration_path=args.calibration,
        output_path=args.output,
        recent_days=args.recent_days,
    )
    try:
        data = collect_data(config)
        html = render_html(data)
        config.output_path.write_text(html, encoding="utf-8")
        LOGGER.info(
            "看板已生成: %s （今日信号=%d, 台账=%d, 近%d日表现=%d）",
            config.output_path.resolve(),
            len(data["signals"]),
            len(data["ledger"]),
            config.recent_days,
            len(data["by_day"]),
        )
    except Exception as exc:
        LOGGER.exception("生成看板失败: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

