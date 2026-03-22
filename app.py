"""
本地瀏覽：做多段週線區間 OHLC + 日線鑽取（進場收盤後才判斷跌破）
執行：cd 到本資料夾上層或本資料夾後
  streamlit run long_underwater_web/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np

# 確保可 import tw_index_futur
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tw_index_futur") not in sys.path:
    sys.path.insert(0, str(_ROOT / "tw_index_futur"))

from logic import (  # noqa: E402
    build_daily_with_week_end,
    fetch_daily_chinese,
    calculate_long_position_stats,
    fetch_weekly_stock_data,
    segment_slice,
    segments_to_options,
)


@st.cache_data(ttl=60, show_spinner=False)
def _cached_wtx_night_price():
    """Yahoo 台股 WTX&（台指期近一）嵌入價，60 秒快取。"""
    from yahoo_tw_wtx_price import fetch_wtx_quote_price

    return fetch_wtx_quote_price()


@st.cache_data(show_spinner="下載週線與日線…", ttl=600)
def load_market_data(ticker: str, start_date: str):
    """依代號與起始日抓取週線狀態機 + 日線（含週結束日）；結果快取避免重複請求。"""
    df_w = fetch_weekly_stock_data(ticker, start_date=start_date)
    stats = calculate_long_position_stats(df_w)
    df_d = fetch_daily_chinese(ticker, start_date=start_date)
    daily_we, _ = build_daily_with_week_end(df_d)
    return df_w, stats, daily_we


st.set_page_config(page_title="做多段日線鑽取", layout="wide")
st.title("做多段：週線區間 OHLC + 日線鑽取")
st.caption("週界：週四～下週三（與 fetch_daily_stock_data_W 一致）｜進場價＝第一週收盤｜跌破判定從進場收盤後開始")

with st.sidebar:
    ticker = st.text_input("股票代號", value="^TWII")
    start_date = st.text_input("資料起始日", value="2020-01-01")
    st.caption("開啟或變更代號／日期後會自動載入（約 10 分鐘內重複瀏覽使用快取）。")

_t = ticker.strip()
_sd = start_date.strip()
try:
    df_w, stats, daily_we = load_market_data(_t, _sd)
except Exception as e:
    st.error(f"資料載入失敗：{e}")
    st.stop()

if stats is None or len(stats) == 0:
    st.warning("沒有做多段統計資料。")
    st.stop()

opts = segments_to_options(stats)
labels = [o["label"] for o in opts]
_last_idx = max(0, len(labels) - 1)
# key 隨代號+起始日變化，新組合預設選最後一筆（通常為最新／進行中段）
choice = st.selectbox(
    "選擇做多段",
    options=list(range(len(labels))),
    index=_last_idx,
    format_func=lambda i: labels[i],
    key=f"做多段選擇_{_t}_{_sd}",
)
row = stats.iloc[choice]

entry_date = pd.Timestamp(row["進入日期"])
exit_date = pd.Timestamp(row["退出日期"])
entry_price = float(row["進入價格"])
alert_line = entry_price * 0.98
exit_price = float(row["退出價格"]) if "退出價格" in row and pd.notna(row["退出價格"]) else None
seg_brake_flag = str(row["做多煞車"]).strip() if "做多煞車" in row else ""

# 最後一段 + 週線最新列為「做多中」：對照 Yahoo 台指期近一 WTX&，跌破警戒線則警示
_latest_state = (
    str(df_w.iloc[-1]["交易狀態"]).strip()
    if len(df_w) > 0 and "交易狀態" in df_w.columns
    else ""
)
if choice == _last_idx and _latest_state == "做多中":
    wtx_px, wtx_err = _cached_wtx_night_price()
    if wtx_err:
        st.warning(f"台指期近一（[Yahoo WTX&](https://tw.stock.yahoo.com/quote/WTX&)）讀取失敗：{wtx_err}")
    elif wtx_px is not None and wtx_px < alert_line:
        st.error(
            f"台指期 **{wtx_px:,.2f}**，低於警戒 **{alert_line:,.2f}**。"
        )

# 進行中：統計表的「退出日期」是前一週（避免未收盤失真），圖表仍要延伸到資料最新一週（含進行中週）
latest_week_end = pd.Timestamp(pd.to_datetime(df_w["日期"]).max()).normalize()
is_ongoing = "狀態" in row.index and str(row.get("狀態", "")).strip() == "進行中"
display_week_end = latest_week_end if is_ongoing else None

res = segment_slice(
    daily_we, df_w, entry_date, exit_date, entry_price, display_week_end=display_week_end
)
seg_daily = res["seg_daily"]  # 進場收盤後：用於跌破統計（可能為空）
seg_daily_chart = res.get("seg_daily_chart", seg_daily)  # 完整區間日線：用於 K 線圖
seg_weekly = res["seg_weekly"]
brake_week_date = None

# 週對應出場價：本週使用「上週」計算出的出場價格（例：2026-03-25 對應 2026-03-18 的 33414.19）
seg_weekly = seg_weekly.sort_values("日期").copy()
# 每週做多煞車價位（即使做多煞車為 F 也存在）：
# 本週煞車價 = 上週收盤 - 0.2 * 上週做多累計漲幅
if {"收盤價", "做多累計漲幅"}.issubset(seg_weekly.columns):
    seg_weekly["本週煞車基準"] = (
        seg_weekly["收盤價"].astype(float) - 0.2 * seg_weekly["做多累計漲幅"].astype(float)
    )
    seg_weekly["每週做多煞車價位"] = seg_weekly["本週煞車基準"].shift(1)
else:
    seg_weekly["每週做多煞車價位"] = np.nan

# 每週對應出場價格梯線：本週沿用「上週」策略計算的出場價格（與週表欄位一致）
if "出場價格" in seg_weekly.columns:
    seg_weekly["週對應出場價"] = seg_weekly["出場價格"].shift(1)
else:
    seg_weekly["週對應出場價"] = np.nan

# 風險警示：本週煞車價位高於本週對應出場價，且本週已有 2 根以上日K收盤低於煞車價位
if len(seg_weekly) > 0 and {"每週做多煞車價位", "週對應出場價"}.issubset(seg_weekly.columns):
    latest_week_row = seg_weekly.iloc[-1]
    latest_week_end = pd.Timestamp(latest_week_row["日期"]).normalize()
    week_brake = (
        float(latest_week_row["每週做多煞車價位"])
        if pd.notna(latest_week_row["每週做多煞車價位"])
        else None
    )
    week_exit = (
        float(latest_week_row["週對應出場價"])
        if pd.notna(latest_week_row["週對應出場價"])
        else None
    )
    if week_brake is not None and week_exit is not None and week_brake > week_exit:
        week_daily = seg_daily_chart[seg_daily_chart["週結束日"] == latest_week_end].copy()
        close_below_brake_count = int(
            (week_daily["收盤價"].astype(float) < week_brake).sum()
        ) if len(week_daily) > 0 else 0
        if close_below_brake_count >= 2:
            st.error(
                f"做多煞車警戒：本週已連跌破 **{close_below_brake_count}** 根。"
            )

c1, c2 = st.columns(2)
c1.metric("進場價（第一週收盤）", f"{entry_price:,.2f}")
if res["first_intraday_break"]:
    c2.metric("本段首次盤中跌破日", res["first_intraday_break"]["日期"].strftime("%Y-%m-%d"))
else:
    c2.metric("本段首次盤中跌破日", "無")

st.subheader("日線 K 線（含進場/出場價線、煞車與跌破標記、20日均線）")
if len(seg_daily_chart) == 0:
    st.warning("此段無日線資料。")
else:
    if is_ongoing:
        st.caption(
            "本段進行中：週線表與日線已延伸至資料最新週（週結束日 "
            f"{latest_week_end.strftime('%Y-%m-%d')}）；"
            "統計列上的退出日仍為前一週（與週線腳本一致）。"
        )
    elif len(seg_daily) == 0 and len(seg_daily_chart) > 0:
        st.caption("本段僅進場當週（或進場後無交易日）：圖表仍顯示該週日線；跌破統計以進場收盤後為準，故可能為「無」。")
    fig = go.Figure()
    # 台股習慣：漲紅、跌綠
    fig.add_trace(
        go.Candlestick(
            x=seg_daily_chart["日期"],
            open=seg_daily_chart["開盤價"],
            high=seg_daily_chart["最高價"],
            low=seg_daily_chart["最低價"],
            close=seg_daily_chart["收盤價"],
            name="日K",
            customdata=np.column_stack(
                [
                    seg_daily_chart["開盤價"].astype(float).to_numpy(),
                    seg_daily_chart["最高價"].astype(float).to_numpy(),
                    seg_daily_chart["最低價"].astype(float).to_numpy(),
                    seg_daily_chart["收盤價"].astype(float).to_numpy(),
                ]
            ),
            hovertemplate=(
                "日期: %{x|%Y-%m-%d}<br>"
                "開盤: %{customdata[0]:.2f}<br>"
                "最高: %{customdata[1]:.2f}<br>"
                "最低: %{customdata[2]:.2f}<br>"
                "收盤: %{customdata[3]:.2f}<extra></extra>"
            ),
            increasing_line_color="#b91c1c",
            increasing_fillcolor="#b91c1c",
            decreasing_line_color="#15803d",
            decreasing_fillcolor="#15803d",
        )
    )
    fig.add_hline(
        y=entry_price,
        line_dash="dash",
        line_color="green",
        annotation_text=f"進場價 {entry_price:.2f}",
    )
    fig.add_hline(
        y=alert_line,
        line_dash="dash",
        line_color="#ca8a04",
        annotation_text=f"警戒線 進場×0.98 {alert_line:,.2f}",
    )
    # 每週出場價格梯線（上週出場價格）— 與煞車價位分開顯示
    weekly_exit_line = seg_weekly[["日期", "週對應出場價"]].copy()
    weekly_exit_line = weekly_exit_line[pd.notna(weekly_exit_line["週對應出場價"])]
    if len(weekly_exit_line) > 0:
        fig.add_trace(
            go.Scatter(
                x=weekly_exit_line["日期"],
                y=weekly_exit_line["週對應出場價"],
                mode="lines+markers",
                name="每週出場價格梯線",
                line=dict(color="#2563eb", width=2, dash="dot"),
                line_shape="hv",
                marker=dict(size=7, color="#2563eb", symbol="square"),
                hovertemplate=(
                    "週結束日: %{x|%Y-%m-%d}<br>"
                    "每週對應出場價: %{y:.2f}<extra></extra>"
                ),
            )
        )

    # 每週做多煞車價位：用階梯線呈現每週價位（F 也會有價位）
    weekly_brake_line = seg_weekly[["日期", "每週做多煞車價位"]].copy()
    weekly_brake_line = weekly_brake_line[pd.notna(weekly_brake_line["每週做多煞車價位"])]
    if len(weekly_brake_line) > 0:
        fig.add_trace(
            go.Scatter(
                x=weekly_brake_line["日期"],
                y=weekly_brake_line["每週做多煞車價位"],
                mode="lines+markers",
                name="每週做多煞車價位",
                line=dict(color="#dc2626", width=2, dash="dash"),
                line_shape="hv",
                marker=dict(size=7, color="#dc2626"),
                hovertemplate=(
                    "週結束日: %{x|%Y-%m-%d}<br>"
                    "每週做多煞車價位: %{y:.2f}<extra></extra>"
                ),
            )
        )

    exit_close_candidates = seg_daily_chart.loc[seg_daily_chart["週結束日"] == exit_date, "日期"]
    exit_close_day = pd.Timestamp(exit_close_candidates.max()) if len(exit_close_candidates) else None

    # 做多煞車（週層級 T）：標出煞車週的週結束日那一天
    brake_rows = seg_weekly.copy()
    if "做多煞車" in brake_rows.columns:
        brake_rows = brake_rows.loc[
            pd.notna(brake_rows["做多煞車"]) & (brake_rows["做多煞車"].astype(str).str.strip() == "T")
        ].copy()
    brake_week_date = None
    brake_week_dates = []
    if len(brake_rows) > 0:
        brake_rows_sorted = brake_rows.sort_values("日期")
        brake_week_dates = [pd.Timestamp(d).normalize() for d in brake_rows_sorted["日期"].unique()]
        brake_week_date = brake_week_dates[0]

    first_brake_legend = True
    for bw in brake_week_dates:
        brake_close_candidates = seg_daily_chart.loc[seg_daily_chart["週結束日"] == bw, "日期"]
        brake_close_day = pd.Timestamp(brake_close_candidates.max()) if len(brake_close_candidates) else None
        if brake_close_day is None:
            continue
        brake_close_price = float(
            seg_daily_chart.loc[seg_daily_chart["日期"] == brake_close_day, "收盤價"].iloc[0]
        )
        # 與出場日同一天時，下移一點避免標記重疊看不見
        if exit_close_day is not None and brake_close_day == exit_close_day:
            brake_close_price = brake_close_price * 0.995
        fig.add_trace(
            go.Scatter(
                x=[brake_close_day],
                y=[brake_close_price],
                mode="markers+text",
                name="做多煞車(T)",
                marker=dict(size=12, color="#f59e0b"),
                text=["煞車"],
                textposition="top center",
                showlegend=first_brake_legend,
            )
        )
        first_brake_legend = False
        fig.add_vline(
            x=brake_close_day,
            line_dash="dot",
            line_color="#f59e0b",
            opacity=0.7,
        )

    if "20日均線" in seg_daily_chart.columns:
        fig.add_trace(
            go.Scatter(
                x=seg_daily_chart["日期"],
                y=seg_daily_chart["20日均線"],
                mode="lines",
                name="20日均線",
                line=dict(color="#7c3aed", width=2),
            )
        )
    if res["first_intraday_break"]:
        fb = res["first_intraday_break"]
        fig.add_trace(
            go.Scatter(
                x=[fb["日期"]],
                y=[fb["最低價"]],
                mode="markers+text",
                name="首次盤中跌破",
                marker=dict(size=12, color="red"),
                text=["跌破"],
                textposition="top center",
            )
        )
    fig.update_layout(
        height=520,
        xaxis_title="日期",
        yaxis_title="價格",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("週線表（區間 OHLC）")
display_cols = ["日期", "開盤價", "最高價", "最低價", "收盤價"]
if "交易狀態" in seg_weekly.columns:
    display_cols.append("交易狀態")
if "做多煞車" in seg_weekly.columns:
    display_cols.append("做多煞車")
if "出場價格" in seg_weekly.columns:
    display_cols.append("出場價格")
if "每週做多煞車價位" in seg_weekly.columns:
    display_cols.append("每週做多煞車價位")
if "週對應出場價" in seg_weekly.columns:
    display_cols.append("週對應出場價")
display_w = seg_weekly[display_cols].copy()
display_w["日期"] = display_w["日期"].dt.strftime("%Y-%m-%d")
st.dataframe(display_w, use_container_width=True, hide_index=True)

st.subheader("日線鑽取摘要")
summary = {
    "進入日期": entry_date.strftime("%Y-%m-%d"),
    "退出日期": exit_date.strftime("%Y-%m-%d"),
    "進場價": entry_price,
    "出場價": exit_price,
    "本段做多煞車旗標": seg_brake_flag,
    "本段煞車(T)週結束日": str(brake_week_date.date()) if brake_week_date is not None else None,
    "首次盤中跌破": res["first_intraday_break"],
    "區間最低價": res["deepest_low"],
    "最低價所在日": (
        res["deepest_low_date"].strftime("%Y-%m-%d") if res["deepest_low_date"] is not None else None
    ),
    "首次收盤低於進場": res["first_close_below"],
}
st.json(summary)
