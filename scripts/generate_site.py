#!/usr/bin/env python3
"""
scripts/generate_site.py
Reads registry/all_results.csv + registry/champion.json and writes docs/index.html.
Run after every backtest: python scripts/generate_site.py
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REGISTRY = Path("registry/all_results.csv")
CHAMPION = Path("registry/champion.json")
OUT      = Path("docs/index.html")
NY = ZoneInfo("America/New_York")

MIN_TRADES   = 15
MIN_EVAL_DAYS = 5


def load_champion() -> dict | None:
    if not CHAMPION.exists():
        return None
    return json.loads(CHAMPION.read_text(encoding="utf-8"))


def load_leaderboard() -> pd.DataFrame:
    if not REGISTRY.exists():
        return pd.DataFrame()
    df = pd.read_csv(REGISTRY)
    df = df[df["total_trades"] >= MIN_TRADES]
    df = df[df["eval_days"]    >= MIN_EVAL_DAYS]
    if "run_ts" in df.columns:
        df = df.sort_values("run_ts", ascending=False)
        key_cols = [c for c in ["model_id","version","symbol",
                                 "pivot_left","pivot_right","min_swing_range_pct",
                                 "stop_buffer_pct","max_wait_minutes_after_touch",
                                 "max_hold_minutes","rr_target","horizon","tp","sl","p_enter"]
                    if c in df.columns]
        if key_cols:
            df = df.drop_duplicates(subset=key_cols, keep="first")
    df = df.sort_values("avg_daily_roi_pct", ascending=False).reset_index(drop=True)
    return df.head(20)


def fmt(v, decimals=4) -> str:
    try:
        return f"{float(v):.{decimals}f}"
    except Exception:
        return str(v)


def champion_card(c: dict) -> str:
    if c is None:
        return "<p>No champion yet.</p>"
    crowned = c.get("crowned_at", "")[:10]
    return f"""
    <div class="champion-card">
      <div class="crown">👑 CHAMPION</div>
      <div class="champ-model">{c.get('model_id','')} <span class="ver">v{c.get('version','')}</span></div>
      <div class="champ-symbol">{c.get('symbol','')}</div>
      <div class="stats-grid">
        <div class="stat"><div class="stat-val">{fmt(c.get('avg_daily_roi_pct',0))}%</div><div class="stat-lbl">Avg Daily ROI</div></div>
        <div class="stat"><div class="stat-val">{fmt(float(c.get('positive_day_ratio',0))*100,1)}%</div><div class="stat-lbl">Positive Days</div></div>
        <div class="stat"><div class="stat-val">{fmt(float(c.get('win_rate',0))*100,1)}%</div><div class="stat-lbl">Win Rate</div></div>
        <div class="stat"><div class="stat-val">{int(c.get('total_trades',0))}</div><div class="stat-lbl">Total Trades</div></div>
        <div class="stat"><div class="stat-val">{fmt(c.get('max_drawdown_pct',0),2)}%</div><div class="stat-lbl">Max Drawdown</div></div>
        <div class="stat"><div class="stat-val">{int(c.get('eval_days',0))} days</div><div class="stat-lbl">Eval Period</div></div>
      </div>
      <div class="crowned-at">Crowned: {crowned}</div>
    </div>"""


def leaderboard_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "<p>No results meeting minimum criteria yet.</p>"
    cols = [c for c in ["model_id","version","symbol","avg_daily_roi_pct",
                         "positive_day_ratio","win_rate","total_trades",
                         "max_drawdown_pct","eval_days"] if c in df.columns]
    rows = ""
    for i, row in df[cols].iterrows():
        medal = ["🥇","🥈","🥉"][i] if i < 3 else f"{i+1}."
        cells = f"<td>{medal}</td>"
        for c in cols:
            v = row[c]
            if c == "avg_daily_roi_pct":
                cells += f'<td class="roi">{fmt(v)}%</td>'
            elif c in ("positive_day_ratio","win_rate"):
                cells += f"<td>{fmt(float(v)*100,1)}%</td>"
            elif c in ("total_trades","eval_days"):
                cells += f"<td>{int(v)}</td>"
            elif c == "max_drawdown_pct":
                cells += f"<td>{fmt(v,2)}%</td>"
            else:
                cells += f"<td>{v}</td>"
        rows += f"<tr>{cells}</tr>\n"

    headers = "<th>#</th>" + "".join(f"<th>{c}</th>" for c in cols)
    return f"""
    <table>
      <thead><tr>{headers}</tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_html(champion: dict | None, lb: pd.DataFrame, generated_at: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>TenEx Trading Bot — Model Research</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            background: #0d1117; color: #e6edf3; min-height: 100vh; padding: 2rem; }}
    h1 {{ font-size: 1.8rem; font-weight: 700; margin-bottom: .25rem; }}
    .subtitle {{ color: #8b949e; margin-bottom: 2rem; font-size: .95rem; }}
    h2 {{ font-size: 1.2rem; color: #58a6ff; margin: 2rem 0 1rem; border-bottom: 1px solid #21262d; padding-bottom: .5rem; }}
    .champion-card {{ background: #161b22; border: 1px solid #f0b429;
                      border-radius: 12px; padding: 1.5rem; max-width: 520px; }}
    .crown {{ color: #f0b429; font-size: .85rem; font-weight: 600; margin-bottom: .5rem; }}
    .champ-model {{ font-size: 1.6rem; font-weight: 700; }}
    .ver {{ color: #8b949e; font-size: 1rem; }}
    .champ-symbol {{ font-size: 1.1rem; color: #58a6ff; margin: .25rem 0 1rem; }}
    .stats-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; margin: 1rem 0; }}
    .stat {{ background: #0d1117; border-radius: 8px; padding: .75rem; text-align: center; }}
    .stat-val {{ font-size: 1.2rem; font-weight: 700; color: #3fb950; }}
    .stat-lbl {{ font-size: .75rem; color: #8b949e; margin-top: .25rem; }}
    .crowned-at {{ font-size: .8rem; color: #8b949e; }}
    table {{ width: 100%; border-collapse: collapse; font-size: .875rem; }}
    th {{ background: #161b22; color: #8b949e; padding: .6rem .75rem; text-align: left;
          font-weight: 600; border-bottom: 1px solid #21262d; }}
    td {{ padding: .55rem .75rem; border-bottom: 1px solid #21262d; }}
    tr:hover td {{ background: #161b22; }}
    .roi {{ color: #3fb950; font-weight: 600; }}
    .footer {{ margin-top: 3rem; color: #484f58; font-size: .8rem; }}
    a {{ color: #58a6ff; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
  </style>
</head>
<body>
  <h1>🤖 TenEx Trading Bot</h1>
  <p class="subtitle">Open model research — backtesting every strategy, finding the best alpha.</p>

  <h2>Current Champion</h2>
  {champion_card(champion)}

  <h2>Leaderboard <span style="color:#8b949e;font-size:.85rem">(min {MIN_TRADES} trades &amp; {MIN_EVAL_DAYS} eval days)</span></h2>
  {leaderboard_table(lb)}

  <div class="footer">
    Last updated: {generated_at} ET &nbsp;|&nbsp;
    <a href="https://github.com/tenexhan/TenEx" target="_blank">GitHub →</a>
  </div>
</body>
</html>"""


def main():
    champion = load_champion()
    lb       = load_leaderboard()
    now      = datetime.now(tz=NY).strftime("%Y-%m-%d %H:%M")
    html     = build_html(champion, lb, now)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print(f"Site generated: {OUT}  ({len(lb)} models in leaderboard)")


if __name__ == "__main__":
    main()
