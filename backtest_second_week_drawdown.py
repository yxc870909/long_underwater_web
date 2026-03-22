#!/usr/bin/env python3
"""
進場後「第二週」做多中：若該週最後收盤仍低於進場價（定義 A），
統計該週盤中最低相對進場價的跌幅% = (進場價 - 當週最低價) / 進場價 * 100。

週界：與 convert_to_weekly_data 一致（週四～下週三，週結束日＝週三）。
進場週＝統計表「進入日期」當週；第二週＝下一個週結束日所屬週。

用法：
  python3 long_underwater_web/backtest_second_week_drawdown.py [TICKER] [START_DATE]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
for p in (_ROOT / "tw_index_futur", _ROOT / "long_underwater_web"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fetch_daily_stock_data_W import calculate_long_position_stats, fetch_weekly_stock_data  # noqa: E402
from logic import build_daily_with_week_end, fetch_daily_chinese  # noqa: E402


def analyze(
    ticker: str,
    start_date: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_w = fetch_weekly_stock_data(ticker, start_date=start_date)
    stats = calculate_long_position_stats(df_w)
    if stats is None or len(stats) == 0:
        raise SystemExit("無做多段統計")

    daily = fetch_daily_chinese(ticker, start_date=start_date)
    daily_we, _ = build_daily_with_week_end(daily)
    daily_we = daily_we.copy()
    daily_we["日期"] = pd.to_datetime(daily_we["日期"])
    daily_we["週結束日"] = pd.to_datetime(daily_we["週結束日"])

    rows: list[dict] = []
    for _, r in stats.iterrows():
        entry = pd.Timestamp(r["進入日期"]).normalize()
        exit_w = pd.Timestamp(r["退出日期"]).normalize()
        entry_p = float(r["進入價格"])

        seg = daily_we[
            (daily_we["週結束日"] >= entry) & (daily_we["週結束日"] <= exit_w)
        ]
        if len(seg) == 0:
            rows.append(
                {
                    "進入日期": r["進入日期"],
                    "退出日期": r["退出日期"],
                    "進入價格": entry_p,
                    "有第二週": False,
                    "第二週週結束日": None,
                    "第二週最後收盤": None,
                    "第二週盤中最低": None,
                    "最深跌幅_pct": None,
                    "定義A_週收仍低於進場": None,
                }
            )
            continue

        weeks = sorted(seg["週結束日"].unique())
        wlist = [w for w in weeks if w >= entry]
        if len(wlist) < 2:
            rows.append(
                {
                    "進入日期": r["進入日期"],
                    "退出日期": r["退出日期"],
                    "進入價格": entry_p,
                    "有第二週": False,
                    "第二週週結束日": None,
                    "第二週最後收盤": None,
                    "第二週盤中最低": None,
                    "最深跌幅_pct": None,
                    "定義A_週收仍低於進場": None,
                }
            )
            continue

        second_w = wlist[1]
        wk = seg[seg["週結束日"] == second_w].sort_values("日期")
        min_low = float(wk["最低價"].min())
        last_close = float(wk.iloc[-1]["收盤價"])
        dd_pct = (entry_p - min_low) / entry_p * 100.0 if entry_p else np.nan
        def_a = last_close < entry_p

        rows.append(
            {
                "進入日期": r["進入日期"],
                "退出日期": r["退出日期"],
                "進入價格": entry_p,
                "有第二週": True,
                "第二週週結束日": second_w,
                "第二週最後收盤": round(last_close, 2),
                "第二週盤中最低": round(min_low, 2),
                "最深跌幅_pct": round(dd_pct, 4),
                "定義A_週收仍低於進場": def_a,
            }
        )

    out = pd.DataFrame(rows)
    has2 = out[out["有第二週"]]
    a_true = has2[has2["定義A_週收仍低於進場"] == True]  # noqa: E712

    return out, a_true


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ticker", nargs="?", default="^TWII")
    p.add_argument("start_date", nargs="?", default="2020-01-01")
    args = p.parse_args()

    full, a_only = analyze(args.ticker.strip(), args.start_date.strip())
    has2 = full[full["有第二週"]]

    print(f"標的: {args.ticker}  起始: {args.start_date}")
    print("定義：第二週＝進場週（進入日期當週）之後的下一個週界週。")
    print("盤中最深跌幅% = (進場價 - 第二週日線最低價) / 進場價 × 100")
    print("定義 A：第二週最後一個交易日收盤 < 進場價。\n")

    print(f"做多段總數: {len(full)}")
    print(f"含「第二週」資料之段數: {len(has2)}")
    print(f"其中符合定義 A（該週收不回進場價）: {len(a_only)}")
    if len(has2) > 0:
        print(
            f"符合 A 之占比（在有第二週的段中）: {100 * len(a_only) / len(has2):.2f}%"
        )

    if len(a_only) == 0:
        print("\n無符合定義 A 的樣本，不輸出跌幅分佈。")
        return

    s = a_only["最深跌幅_pct"]
    print("\n=== 符合定義 A 之「第二週盤中最深 vs 進場」跌幅% 分佈 ===")
    print(f"  樣本數: {len(s)}")
    print(f"  最小: {s.min():.4f}%")
    print(f"  最大: {s.max():.4f}%")
    print(f"  平均: {s.mean():.4f}%")
    print(f"  中位數: {s.median():.4f}%")
    for q in (25, 50, 75, 90, 95):
        print(f"  {q}% 分位: {s.quantile(q/100):.4f}%")

    print("\n=== 明細（僅定義 A）===")
    show = a_only[
        [
            "進入日期",
            "退出日期",
            "進入價格",
            "第二週週結束日",
            "第二週盤中最低",
            "第二週最後收盤",
            "最深跌幅_pct",
        ]
    ].copy()
    show["第二週週結束日"] = show["第二週週結束日"].apply(
        lambda x: x.strftime("%Y-%m-%d") if pd.notna(x) else ""
    )
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
