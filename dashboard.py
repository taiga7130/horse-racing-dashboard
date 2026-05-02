#!/usr/bin/env python3
"""Generate dashboard: cumulative ROI / monthly stats / strategy breakdown.

Outputs:
  docs/index.html              # GitHub Pages 用
  docs/assets/cumulative.png   # 累積ROI チャート
  docs/assets/monthly.png      # 月次ROI バーチャート
  docs/assets/breakdown.png    # 戦略別 ROI / hit率

Reads from: data/features/features.parquet + payouts (for backtest)
            outputs/predict_dual_*.csv + outputs/results_*.csv (for live)

Run: uv run python scripts/dashboard.py
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lightgbm as lgb

from keiba.factors.nn_ranker import NNRanker, extract_raw_features
from keiba.log import get_logger

# Make plots readable in Japanese
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

logger = get_logger(__name__)


def score_v5_ensemble(members, feat_df):
    n = len(members)
    out = np.zeros((n, len(feat_df)), dtype=np.float64)
    for k, m in enumerate(members):
        X = m._to_matrix(extract_raw_features(feat_df, feature_cols=m.feature_cols), fit_stats=False)
        gids = feat_df["race_id"].astype(str).values
        unique = np.unique(gids); inv = np.searchsorted(unique, gids)
        with torch.no_grad():
            for i in range(len(unique)):
                rows = np.where(inv == i)[0]
                if len(rows) == 0: continue
                t = torch.from_numpy(X[rows]).unsqueeze(0)
                out[k, rows] = m.model(t).squeeze(0).cpu().numpy()
    return np.median(out, axis=0)


def score_lgbm(model, feat_names, feat_df):
    Xm = extract_raw_features(feat_df, feature_cols=feat_names).to_numpy(
        dtype=np.float32, na_value=np.nan
    )
    return model.predict(Xm)


def build_race_table(s_v5, s_gbm, feat_df, payouts):
    f2 = feat_df.copy(); f2["_v5"] = s_v5; f2["_gbm"] = s_gbm
    rows = []
    for rid, g in f2.groupby("race_id", sort=False):
        if len(g) < 6: continue
        try: key = (int(g.iloc[0]["日付"]), str(g.iloc[0]["開催"]), int(g.iloc[0]["Ｒ"]))
        except: continue
        pout = payouts.get(key)
        if pout is None or not pout["winners"]: continue
        gv = g.sort_values("_v5", ascending=False).reset_index(drop=True)
        gl = g.sort_values("_gbm", ascending=False).reset_index(drop=True)
        date = pd.to_datetime(g.iloc[0].get("date_iso"))
        rows.append({
            "race_id": rid,
            "date": date,
            "month": date.strftime("%Y-%m") if pd.notna(date) else None,
            "track_cond": str(g.iloc[0].get("馬場状態", "")),
            "v5_top1": int(gv.iloc[0]["馬番"]),
            "gbm_top1": int(gl.iloc[0]["馬番"]),
            "v5_floats4": [int(gv.iloc[i]["馬番"]) for i in range(1, min(5, len(gv)))],
            "gbm_floats4": [int(gl.iloc[i]["馬番"]) for i in range(1, min(5, len(gl)))],
            "top1_pop_v5": float(gv.iloc[0]["人気"]) if pd.notna(gv.iloc[0]["人気"]) else 99,
            "top1_pop_gbm": float(gl.iloc[0]["人気"]) if pd.notna(gl.iloc[0]["人気"]) else 99,
            "v5_mmd": float(gv["_v5"].iloc[0] - gv["_v5"].median()),
            "gbm_mmd": float(gl["_gbm"].iloc[0] - gl["_gbm"].median()),
            "winners": list(pout["winners"]),
            "trio": pout["trio"]/100.0 if not np.isnan(pout["trio"]) else 0,
        })
    return pd.DataFrame(rows)


def union_pnl(rt, q_v5, q_gbm):
    """Compute per-race PnL for UNION strategy. Returns df with race outcome."""
    rows = []
    for _, r in rt.iterrows():
        prime_ok = (r["track_cond"]=="良") and (r["top1_pop_v5"]>1) and (r["v5_mmd"]>=q_v5)
        gbm_ok = (r["track_cond"]=="重") and (r["top1_pop_gbm"]>1) and (r["gbm_mmd"]>=q_gbm)
        if not (prime_ok or gbm_ok): continue
        if prime_ok:
            top1 = r["v5_top1"]; floats4 = r["v5_floats4"]; label = "PRIME_良"
        else:
            top1 = r["gbm_top1"]; floats4 = r["gbm_floats4"]; label = "GBM_重"
        winners = set(r["winners"])
        won = (top1 in winners) and (len(winners)==3) and (winners-{top1}).issubset(set(floats4))
        ret = r["trio"] if won else 0
        # 1 unit total stake (6 points), payout = ret/6 per unit
        pnl = -1 + ret/6.0
        rows.append({"date": r["date"], "month": r["month"],
                     "label": label, "won": int(won), "ret": ret,
                     "pnl": pnl, "track": r["track_cond"]})
    return pd.DataFrame(rows)


def make_cumulative_chart(union_df, out_path):
    """Cumulative bankroll vs date."""
    df = union_df.sort_values("date").reset_index(drop=True)
    bankroll = 100000.0
    stake_pct = 2.0
    histories = []
    for _, r in df.iterrows():
        stake = bankroll * stake_pct/100
        bankroll += stake * r["pnl"]
        histories.append((r["date"], bankroll))
    if not histories: return
    dates, balances = zip(*histories)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(dates, balances, color="#2563eb", linewidth=1.5)
    ax.fill_between(dates, 100000, balances,
                    where=[b >= 100000 for b in balances],
                    alpha=0.2, color="#10b981", label="profit zone")
    ax.fill_between(dates, 100000, balances,
                    where=[b < 100000 for b in balances],
                    alpha=0.2, color="#ef4444", label="loss zone")
    ax.axhline(100000, color="#6b7280", linestyle="--", linewidth=0.8, label="initial bankroll")
    ax.set_xlabel("date")
    ax.set_ylabel("bankroll (yen)")
    ax.set_title(f"Cumulative bankroll: UNION strategy (stake={stake_pct}%, n={len(df)} bets)",
                 fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_monthly_chart(union_df, out_path):
    """Monthly ROI bar chart."""
    monthly = union_df.groupby("month").agg(
        n_races=("pnl", "size"),
        roi=("pnl", lambda x: (x + 1).mean() * 100),
        wins=("won", "sum"),
    ).reset_index()
    if len(monthly) == 0: return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#10b981" if r > 100 else "#ef4444" for r in monthly["roi"]]
    bars = ax.bar(monthly["month"], monthly["roi"], color=colors, alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.axhline(100, color="#6b7280", linestyle="--", linewidth=1, label="break-even")
    ax.set_xlabel("month")
    ax.set_ylabel("monthly ROI (%)")
    ax.set_title("Monthly ROI: UNION strategy (green = profit, red = loss)",
                 fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def make_breakdown_chart(union_df, out_path):
    """ROI by strategy label / track condition."""
    summary = union_df.groupby("label").agg(
        n=("pnl", "size"),
        hit_rate=("won", "mean"),
        roi=("pnl", lambda x: (x + 1).mean() * 100),
    ).reset_index()
    if len(summary) == 0: return
    # English labels for matplotlib (no JP font)
    label_map = {"PRIME_良": "PRIME (firm)", "GBM_重": "GBM (heavy)"}
    summary["label_en"] = summary["label"].map(label_map).fillna(summary["label"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors = ["#3b82f6", "#f59e0b"]
    bars = ax.bar(summary["label_en"], summary["roi"], color=colors[:len(summary)], alpha=0.85, edgecolor="black")
    ax.axhline(100, color="#6b7280", linestyle="--", linewidth=1)
    ax.set_ylabel("ROI (%)")
    ax.set_title("ROI by label", fontweight="bold")
    for i, (bar, n, r) in enumerate(zip(bars, summary["n"], summary["roi"])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{r:.1f}%\nn={n}", ha="center", fontsize=10)

    ax = axes[1]
    bars = ax.bar(summary["label_en"], summary["hit_rate"]*100, color=colors[:len(summary)], alpha=0.85, edgecolor="black")
    ax.set_ylabel("hit rate (%)")
    ax.set_title("Hit rate by label", fontweight="bold")
    for bar, h in zip(bars, summary["hit_rate"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{h*100:.1f}%", ha="center", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def render_html(union_df, out_path, forward_log=None):
    """Render docs/index.html with embedded chart references."""
    n_total = len(union_df)
    if n_total == 0:
        n_hits = 0; hit_rate = 0; cum_roi = 0; n_months = 0; win_months = 0
    else:
        n_hits = int(union_df["won"].sum())
        hit_rate = n_hits / n_total * 100
        cum_roi = (union_df["pnl"] + 1).mean() * 100
        monthly = union_df.groupby("month")["pnl"].apply(lambda x: (x+1).mean()*100)
        n_months = len(monthly)
        win_months = int((monthly > 100).sum())

    # Recent 10 races (backtest)
    recent = union_df.tail(10).iloc[::-1]
    recent_rows = ""
    for _, r in recent.iterrows():
        date_str = r["date"].strftime("%Y-%m-%d") if pd.notna(r["date"]) else ""
        result = "⭕ 的中" if r["won"] else "✗ 外れ"
        ret_str = f"¥{int(r['ret']*100):,}" if r["won"] else "-"
        bg = "#dcfce7" if r["won"] else "#fef2f2"
        recent_rows += f'<tr style="background:{bg}"><td>{date_str}</td><td>{r["label"]}</td><td>{r["track"]}</td><td>{result}</td><td>{ret_str}</td></tr>'

    # Forward log section (real-money bets after model deployment)
    fwd_html = ""
    if forward_log is not None and len(forward_log) > 0:
        # Split into two strategies
        mech = forward_log[forward_log["strategy"].str.contains("機械", na=False)]
        prot = forward_log[forward_log["strategy"].str.contains("協議", na=False)]

        def _section(df, title, subtitle, accent="blue"):
            n_total = len(df)
            n_hits = int((df["result"] == "hit").sum())
            # Stake assumed ¥100 / point (for ROI calc, but stake amount itself NOT shown)
            total_points = int(df["points"].fillna(0).sum())
            total_payout = int(df[df["result"]=="hit"]["payout"].fillna(0).sum())
            implied_stake = total_points * 100
            roi = total_payout / max(implied_stake, 1) * 100 if total_payout > 0 else 0

            rows_html = ""
            for _, r in df.iloc[::-1].iterrows():
                result_jp = "⭕ 的中" if r["result"] == "hit" else "✗ 外れ"
                bg = "#dcfce7" if r["result"] == "hit" else "#fef2f2"
                payout_str = f"¥{int(r['payout']):,}" if pd.notna(r['payout']) and r['payout'] > 0 else "—"
                pts = int(r['points']) if pd.notna(r['points']) else "?"
                rows_html += f'<tr style="background:{bg}"><td>{r["date"]}</td><td>{r["race_name"]}</td><td>{r.get("bet_type","")}</td><td>{pts}点</td><td>{result_jp}</td><td>{payout_str}</td></tr>'

            roi_color = "green" if roi > 100 else "red"
            return f"""
<div class="chart">
  <h2>{title}</h2>
  <p style="color:#6b7280;font-size:13px;margin-bottom:12px;">{subtitle}</p>
  <div class="stats-grid">
    <div class="stat {accent}"><div class="label">運用レース</div><div class="value">{n_total}</div></div>
    <div class="stat green"><div class="label">的中数</div><div class="value">{n_hits}/{n_total}</div></div>
    <div class="stat"><div class="label">的中率</div><div class="value">{n_hits/max(n_total,1)*100:.0f}%</div></div>
    <div class="stat {roi_color}"><div class="label">回収率</div><div class="value">{roi:.0f}%</div></div>
  </div>
  <table>
    <thead><tr><th>日付</th><th>レース</th><th>券種</th><th>点数</th><th>結果</th><th>配当</th></tr></thead>
    <tbody>{rows_html}</tbody>
  </table>
</div>"""

        if len(mech) > 0:
            fwd_html += _section(mech,
                "🤖 機械フロー (UNION戦略)",
                "v5 NN ensemble + GBM の自動判定で買った結果。本命=人気1番なので戦略的には警告ありだが結果は黒字。",
                accent="blue")

        if len(prot) > 0:
            fwd_html += _section(prot,
                "🧠 協議型プロトコル v1.0 (Claude × ML × 直感の三位一体)",
                "重要レースで使用。Claude との重み付け協議 → ML スコア参照 → 軸+穴の合議型。9ステップフローで進行。",
                accent="green")

    html = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>競馬予想 累積戦績 — UNION戦略</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", "Yu Gothic", sans-serif;
         max-width: 1100px; margin: 0 auto; padding: 24px; background: #fafafa; color: #111; }}
  h1 {{ font-size: 28px; margin-bottom: 8px; }}
  .subtitle {{ color: #6b7280; margin-bottom: 24px; font-size: 14px; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 32px; }}
  .stat {{ background: white; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .stat .label {{ font-size: 12px; color: #6b7280; margin-bottom: 4px; }}
  .stat .value {{ font-size: 24px; font-weight: 700; }}
  .stat.green .value {{ color: #10b981; }}
  .stat.red .value {{ color: #ef4444; }}
  .stat.blue .value {{ color: #2563eb; }}
  .chart {{ background: white; border-radius: 8px; padding: 16px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
  .chart img {{ max-width: 100%; height: auto; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
  th {{ background: #f3f4f6; text-align: left; padding: 8px 12px; font-size: 13px; }}
  td {{ padding: 8px 12px; font-size: 13px; border-top: 1px solid #e5e7eb; }}
  .footer {{ margin-top: 32px; font-size: 12px; color: #9ca3af; text-align: center; }}
  .disclaimer {{ background: #fef3c7; border-left: 4px solid #f59e0b; padding: 12px 16px; margin: 24px 0; font-size: 13px; }}
</style>
</head>
<body>

<h1>🐴 競馬予想 累積戦績 ダッシュボード</h1>
<p class="subtitle">UNION戦略 (PRIME_良 ∪ GBM_重) — AIアンサンブルモデル v5 + LightGBM</p>

<div class="stats-grid">
  <div class="stat blue">
    <div class="label">累計レース数 (backtest)</div>
    <div class="value">{n_total:,}</div>
  </div>
  <div class="stat green">
    <div class="label">累計ROI</div>
    <div class="value">{cum_roi:.1f}%</div>
  </div>
  <div class="stat">
    <div class="label">累計的中率</div>
    <div class="value">{hit_rate:.1f}%</div>
  </div>
  <div class="stat">
    <div class="label">月次黒字率</div>
    <div class="value">{win_months}/{n_months}</div>
  </div>
</div>

<div class="chart">
  <h2>累積バンクロール推移 (10万円スタート、毎レース 2%)</h2>
  <img src="cumulative.png" alt="cumulative bankroll">
</div>

<div class="chart">
  <h2>月次 ROI</h2>
  <img src="monthly.png" alt="monthly ROI">
</div>

<div class="chart">
  <h2>戦略別ブレイクダウン</h2>
  <img src="breakdown.png" alt="strategy breakdown">
</div>
{fwd_html}

<div class="disclaimer">
  ⚠️ 表示されている数値は過去 28ヶ月 (2024-01〜2026-04) の backtest 結果です。
  未来の利益は保証されません。投資/賭博は自己責任で。
</div>

<p class="footer">
  最終更新: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</p>

</body>
</html>"""
    out_path.write_text(html, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", default="data/features/features.parquet")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-04-19")
    ap.add_argument("--raw-csv", default="data/raw/2015-0101_plus_predict_0426.csv")
    ap.add_argument("--out-dir", default="docs")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backtest_real_payouts import load_raw_payouts, build_race_payouts

    out_dir = Path(args.out_dir)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)

    logger.info("Loading features...")
    df = pd.read_parquet(args.features)
    df["_dt"] = pd.to_datetime(df["date_iso"], errors="coerce")
    full = df[(df["_dt"]>=pd.Timestamp(args.start))&(df["_dt"]<=pd.Timestamp(args.end))].copy().reset_index(drop=True)
    logger.info(f"  {len(full)} rows / {full['race_id'].nunique()} races")

    raw = load_raw_payouts(args.raw_csv)
    payouts = build_race_payouts(raw)

    logger.info("Loading models...")
    ens = json.loads(open("models/nn_ranker_v5/ensemble.json").read())
    members = [NNRanker.load(f'models/nn_ranker_v5/{m["dir"]}') for m in ens["members"]]
    lgbm_meta = json.loads(open("models/raw_ranker/meta.json").read())
    lgbm = lgb.Booster(model_file="models/raw_ranker/model.txt")

    logger.info("Scoring...")
    s_v5 = score_v5_ensemble(members, full)
    s_g = score_lgbm(lgbm, lgbm_meta["feature_names"], full)
    rt = build_race_table(s_v5, s_g, full, payouts)
    logger.info(f"  {len(rt)} races scored")

    q_v5 = float(rt["v5_mmd"].quantile(0.85))
    q_gbm = float(rt["gbm_mmd"].quantile(0.85))
    logger.info(f"  v5_q85={q_v5:.3f}  gbm_q85={q_gbm:.3f}")

    union_df = union_pnl(rt, q_v5, q_gbm)
    logger.info(f"UNION races: {len(union_df)}  hits: {union_df['won'].sum()}  ROI: {(union_df['pnl']+1).mean()*100:.1f}%")

    # Load forward log if exists
    fwd_path = Path("data/forward_log.csv")
    forward_log = None
    if fwd_path.exists():
        forward_log = pd.read_csv(fwd_path)
        logger.info(f"Forward log: {len(forward_log)} bets recorded")

    logger.info("Generating charts...")
    make_cumulative_chart(union_df, out_dir / "cumulative.png")
    make_monthly_chart(union_df, out_dir / "monthly.png")
    make_breakdown_chart(union_df, out_dir / "breakdown.png")
    render_html(union_df, out_dir / "index.html", forward_log=forward_log)
    logger.info(f"Saved → {out_dir}/index.html and {out_dir}/*.png")


if __name__ == "__main__":
    main()
