"""
週線做多區段 + 日線鑽取（週界與 fetch_daily_stock_data_W.convert_to_weekly_data 一致：週四～下週三）
進場價＝做多段第一週收盤，與 calculate_long_position_stats 一致。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# 專案內週線狀態機與做多統計（支援同層 repo 與上層 monorepo 兩種結構）
_APP_DIR = Path(__file__).resolve().parent
_MONO_ROOT = _APP_DIR.parent
_TW_CANDIDATES = [
    _APP_DIR / "tw_index_futur",
    _MONO_ROOT / "tw_index_futur",
]
_TW = next((p for p in _TW_CANDIDATES if p.exists()), _TW_CANDIDATES[0])
if str(_TW) not in sys.path:
    sys.path.insert(0, str(_TW))

# 快取目錄：standalone 用 repo 根；monorepo 用 Stk_Ops 根
if (_APP_DIR / "tw_index_futur").exists():
    _yf_cache_dir = _APP_DIR / ".yfinance_cache"
else:
    _yf_cache_dir = _MONO_ROOT / ".yfinance_cache"
_yf_cache_dir.mkdir(parents=True, exist_ok=True)
import yfinance as yf  # noqa: E402

yf.set_tz_cache_location(str(_yf_cache_dir))

from fetch_daily_stock_data_W import (  # noqa: E402
    calculate_long_position_stats,
    convert_to_weekly_data,
    fetch_weekly_stock_data,
)


def _strip_tz(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series)
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s


def fetch_daily_chinese(ticker: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """與 fetch_weekly_stock_data 相同來源的日線（中文欄位）。"""
    stock = yf.Ticker(ticker)
    df_daily = stock.history(start=start_date)
    if df_daily.empty:
        raise ValueError(f"無法取得 {ticker} 的資料")
    df_daily = df_daily.reset_index()
    df_daily.rename(
        columns={
            "Date": "日期",
            "High": "最高價",
            "Low": "最低價",
            "Open": "開盤價",
            "Close": "收盤價",
        },
        inplace=True,
    )
    df_daily["日期"] = _strip_tz(df_daily["日期"])
    df_daily = df_daily.drop_duplicates(subset=["日期"], keep="last").sort_values("日期").reset_index(drop=True)
    # ^TWII 最後一根日 K 改用 Yahoo 台股頁面即時值，避免 yfinance 最末筆延遲或缺漏。
    if ticker.strip().upper() == "^TWII":
        try:
            from yahoo_tw_twii_price import fetch_twii_quote_ohlc

            twii_ohlc, twii_err = fetch_twii_quote_ohlc()
            if twii_ohlc and twii_err is None:
                q_date = pd.Timestamp(twii_ohlc["日期"]).normalize()
                q_open = float(twii_ohlc["開盤價"])
                q_high = float(twii_ohlc["最高價"])
                q_low = float(twii_ohlc["最低價"])
                q_close = float(twii_ohlc["收盤價"])
                if len(df_daily) > 0 and pd.Timestamp(df_daily.iloc[-1]["日期"]).normalize() == q_date:
                    df_daily.loc[df_daily.index[-1], ["開盤價", "最高價", "最低價", "收盤價"]] = [
                        q_open,
                        q_high,
                        q_low,
                        q_close,
                    ]
                elif len(df_daily) == 0 or q_date > pd.Timestamp(df_daily.iloc[-1]["日期"]).normalize():
                    add = pd.DataFrame(
                        [
                            {
                                "日期": q_date,
                                "開盤價": q_open,
                                "最高價": q_high,
                                "最低價": q_low,
                                "收盤價": q_close,
                            }
                        ]
                    )
                    df_daily = pd.concat([df_daily, add], ignore_index=True)
        except Exception:
            # 即時來源失敗時回退 yfinance，避免中斷主流程。
            pass
    return df_daily


def assign_week_id_to_daily(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    與 convert_to_weekly_data 前半段相同：為每個交易日標上 周標識。
    """
    df = df_daily.copy()
    df["日期"] = pd.to_datetime(df["日期"])
    if getattr(df["日期"].dt, "tz", None) is not None:
        df["日期"] = df["日期"].dt.tz_localize(None)
    df = df.drop_duplicates(subset=["日期"], keep="last").sort_values("日期").reset_index(drop=True)
    df["星期"] = df["日期"].dt.dayofweek
    df["周標識"] = 0
    week_num = 0
    prev_date = None
    for i in range(len(df)):
        curr_date = df.loc[i, "日期"]
        curr_day = df.loc[i, "星期"]
        if curr_day == 3:
            week_num += 1
        elif prev_date is not None:
            days_diff = (curr_date - prev_date).days
            prev_day = df.loc[i - 1, "星期"]
            if days_diff > 7:
                week_num += 1
            elif days_diff > 1 and prev_day <= 2 and curr_day >= 4:
                week_num += 1
            elif prev_day == 2 and curr_day >= 4:
                week_num += 1
        df.loc[i, "周標識"] = week_num
        prev_date = curr_date
    return df


def build_daily_with_week_end(df_daily: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    日線附上 週結束日（週三）；並回傳對應周線表（含 周標識）。

    週結束日必須與 convert_to_weekly_data 內「周標識→週三」對照一致；若只用週線表 merge，
    在週線 drop_duplicates(日期) 後會丟失部分周標識，尾端交易日 週結束日 變成 NaN，日 K 會少最近幾根。
    """
    weekly, df_tagged, wmap = convert_to_weekly_data(df_daily, return_tagged_daily=True)
    out = df_tagged.merge(wmap, on="周標識", how="left")
    out["週結束日"] = pd.to_datetime(out["週結束日"])
    # 日Ｋ 20 日均線（依整段日線計算；之後切區間才不會被「起手不足 20 天」影響）
    out = out.sort_values("日期").reset_index(drop=True)
    out["20日均線"] = out["收盤價"].rolling(window=20, min_periods=20).mean().round(2)
    return out, weekly


def load_long_segments(ticker: str, start_date: str = "2020-01-01") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    回傳 (週線含狀態機, 做多段統計表, 日線含週結束日)
    """
    df_w = fetch_weekly_stock_data(ticker, start_date=start_date)
    stats = calculate_long_position_stats(df_w)
    df_d = fetch_daily_chinese(ticker, start_date=start_date)
    daily_we, _ = build_daily_with_week_end(df_d)
    return df_w, stats, daily_we


def segment_slice(
    daily_we: pd.DataFrame,
    weekly: pd.DataFrame,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
    entry_price: float,
    display_week_end: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """
    單一做多段：日線鑽取結果。

    注意：策略在該「週的收盤」才進場，因此跌破判定必須排除進場當天（依日線實際最後交易日計算）。

    display_week_end:
        圖表／週表顯示用的週結束日上界。用於「進行中」做多段：統計表退出日期為前一週，
        但仍要把最新一週（尚未收盤的週）畫進 K 線與週線表時傳入全資料最後一個週三日期。
    """
    entry_date = pd.Timestamp(entry_date).normalize()
    exit_date = pd.Timestamp(exit_date).normalize()
    slice_upper = (
        pd.Timestamp(display_week_end).normalize()
        if display_week_end is not None
        else exit_date
    )
    if slice_upper < entry_date:
        slice_upper = exit_date

    # 防呆：避免日期欄位混入字串，造成 Timestamp 比較錯誤
    daily_we = daily_we.copy()
    weekly = weekly.copy()
    daily_we["日期"] = pd.to_datetime(daily_we["日期"])
    daily_we["週結束日"] = pd.to_datetime(daily_we["週結束日"])
    weekly["日期"] = pd.to_datetime(weekly["日期"])

    seg_daily_all = daily_we[
        (daily_we["週結束日"] >= entry_date) & (daily_we["週結束日"] <= slice_upper)
    ].copy()

    seg_weekly = weekly[(weekly["日期"] >= entry_date) & (weekly["日期"] <= slice_upper)].copy()

    # 進場收盤當天：用「週結束日標籤=entry_date」那一週的最後一個交易日（避免遇到週三假日造成標籤日非交易日）
    entry_close_candidates = seg_daily_all.loc[seg_daily_all["週結束日"] == entry_date, "日期"]
    if len(entry_close_candidates) > 0:
        entry_close_day = entry_close_candidates.max()
    else:
        entry_close_day = entry_date

    # 僅看進場收盤後的日線（排除進場當天）— 用於跌破統計；單週同日進出場時可能為空
    seg_daily = seg_daily_all[seg_daily_all["日期"] > entry_close_day].copy()

    # 日線：區間內第一次盤中跌破進場價
    below = seg_daily[seg_daily["最低價"] < entry_price]
    first_break = None
    if len(below) > 0:
        row = below.iloc[0]
        first_break = {
            "日期": row["日期"],
            "最低價": float(row["最低價"]),
            "收盤價": float(row["收盤價"]),
        }

    min_low = float(seg_daily["最低價"].min()) if len(seg_daily) else None
    min_low_day = seg_daily.loc[seg_daily["最低價"].idxmin(), "日期"] if len(seg_daily) else None

    close_below = seg_daily[seg_daily["收盤價"] < entry_price]
    first_close_below = None
    if len(close_below) > 0:
        r = close_below.iloc[0]
        first_close_below = {"日期": r["日期"], "收盤價": float(r["收盤價"])}

    return {
        "seg_daily": seg_daily,
        # 圖表用：區間內完整日線（含進場當週），避免「僅一週且進場即收盤」時無 K 線可畫
        "seg_daily_chart": seg_daily_all,
        "seg_weekly": seg_weekly,
        "entry_price": entry_price,
        "first_intraday_break": first_break,
        "deepest_low": min_low,
        "deepest_low_date": min_low_day,
        "first_close_below": first_close_below,
    }


def segments_to_options(stats_df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []
    if stats_df is None or len(stats_df) == 0:
        return rows
    for i, r in stats_df.iterrows():
        label = (
            f"{pd.Timestamp(r['進入日期']).strftime('%Y-%m-%d')} → "
            f"{pd.Timestamp(r['退出日期']).strftime('%Y-%m-%d')} | "
            f"進場 {r['進入價格']:.2f}"
        )
        rows.append({"label": label, "index": int(i)})
    return rows
