#!/usr/bin/env python3
"""
終端機：比較「原策略累計價差」vs「進場收盤後，第一根收盤跌破進場×警戒倍數 的日K以收盤出場」
之累計價差。未觸發警戒則假設仍為原策略結果（與該段 stats 累計價差相同）。

不修改 tw_index_futur/fetch_daily_stock_data_W.py。

用法（在專案根目錄或本資料夾）：
  python3 long_underwater_web/run_alert_exit_backtest.py
  python3 long_underwater_web/run_alert_exit_backtest.py --ticker '^TWII' --start 2020-01-01 --alert-mult 0.96
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# 與 app 相同：可從 repo 根目錄或 long_underwater_web 執行
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tw_index_futur") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tw_index_futur"))

from logic import (  # noqa: E402
    build_daily_with_week_end,
    fetch_daily_chinese,
    calculate_long_position_stats,
    fetch_weekly_stock_data,
    segment_slice,
)


def main() -> None:
    p = argparse.ArgumentParser(description="警戒線(進場×倍數)收盤出場 vs 原策略 累計價差對照")
    p.add_argument("--ticker", default="^TWII", help="Yahoo 代號，預設 ^TWII")
    p.add_argument("--start", default="2020-01-01", help="資料起始日 YYYY-MM-DD")
    p.add_argument("--alert-mult", type=float, default=0.98, help="警戒倍數，預設 0.98")
    args = p.parse_args()

    ticker = args.ticker.strip()
    start = args.start.strip()
    alert_mult = float(args.alert_mult)
    if not (0 < alert_mult < 1):
        raise ValueError("--alert-mult 需介於 0 與 1 之間，例如 0.96")

    df_w = fetch_weekly_stock_data(ticker, start_date=start)
    stats = calculate_long_position_stats(df_w)
    if stats is None or len(stats) == 0:
        print("沒有做多段統計，結束。")
        return

    df_d = fetch_daily_chinese(ticker, start_date=start)
    daily_we, _ = build_daily_with_week_end(df_d)
    latest_week_end = pd.Timestamp(pd.to_datetime(daily_we["週結束日"]).max()).normalize()

    rows_detail = []
    sum_actual = 0.0
    sum_hypo = 0.0
    n_triggered = 0

    for _, row in stats.iterrows():
        entry_date = pd.Timestamp(row["進入日期"]).normalize()
        exit_date = pd.Timestamp(row["退出日期"]).normalize()
        entry_price = float(row["進入價格"])
        actual_pnl = float(row["累計價差"])
        is_ongoing = str(row.get("狀態", "")).strip() == "進行中"
        display_week_end = latest_week_end if is_ongoing else None

        res = segment_slice(
            daily_we,
            df_w,
            entry_date,
            exit_date,
            entry_price,
            display_week_end=display_week_end,
        )
        seg_daily = res["seg_daily"]
        alert = entry_price * alert_mult
        below = seg_daily[seg_daily["收盤價"].astype(float) < alert]

        if len(below) > 0:
            r0 = below.iloc[0]
            hypo_exit = float(r0["收盤價"])
            hypo_pnl = hypo_exit - entry_price
            n_triggered += 1
            trig_date = pd.Timestamp(r0["日期"]).strftime("%Y-%m-%d")
        else:
            hypo_pnl = actual_pnl
            hypo_exit = float(row["退出價格"]) if pd.notna(row.get("退出價格")) else float("nan")
            trig_date = "—"

        sum_actual += actual_pnl
        sum_hypo += hypo_pnl
        diff = hypo_pnl - actual_pnl
        rows_detail.append(
            {
                "進場週": entry_date.strftime("%Y-%m-%d"),
                "進場價": entry_price,
                "警戒線": round(alert, 2),
                "觸發日": trig_date,
                "原累計價差": round(actual_pnl, 2),
                "假設累計價差": round(hypo_pnl, 2),
                "差異(假設-原)": round(diff, 2),
            }
        )

    print("=== 警戒線收盤出場 vs 原策略（累計價差，點數）===")
    print(f"標的: {ticker}  起始日: {start}")
    print(f"做多段數: {len(stats)}  曾觸發收盤<進場×{alert_mult:.2f} 的段數: {n_triggered}")
    print()
    print(f"原策略 各段累計價差加總: {sum_actual:,.2f}")
    print(f"假設規則 各段累計價差加總: {sum_hypo:,.2f}")
    print(f"加總差異 (假設 − 原): {sum_hypo - sum_actual:,.2f}")
    if sum_actual != 0:
        rel = (sum_hypo - sum_actual) / abs(sum_actual) * 100
        print(f"（若以原加總絕對值為分母之相對變化: {rel:+.1f}%）")
    print()
    print("--- 各段明細（僅列有觸發警戒之段）---")
    for d in rows_detail:
        if d["觸發日"] == "—":
            continue
        print(
            f"  {d['進場週']} 進場{d['進場價']:.2f} 警戒{d['警戒線']:.2f} "
            f"觸發{d['觸發日']} | 原{d['原累計價差']:+.2f} 假設{d['假設累計價差']:+.2f} "
            f"差{d['差異(假設-原)']:+.2f}"
        )
    if n_triggered == 0:
        print("  （無）")
    print()
    print(f"說明：假設規則＝進場收盤日之後，第一個收盤價 < 進場×{alert_mult:.2f} 的交易日以該收盤出場；")
    print("      若該段從未觸發，假設累計價差＝該段 stats 累計價差（與原策略相同）。")
    print("      進行中段之 slice 上界與網頁一致（延伸至資料最新週結束日）。")


if __name__ == "__main__":
    main()
