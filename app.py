"""
本地瀏覽：做多段週線區間 OHLC + 日線鑽取（進場收盤後才判斷跌破）
執行：cd 到本資料夾上層或本資料夾後
  streamlit run long_underwater_web/app.py
"""
from __future__ import annotations

import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
from streamlit_autorefresh import st_autorefresh

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


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_bottom_strategy_summary(_refresh_key: str) -> tuple[dict | None, str | None]:
    """固定指令執行 bottom_strategy，回傳摘要與錯誤訊息。"""
    script_dir = _ROOT / "tw_index_futur"
    cmd = [
        sys.executable,
        "bottom_strategy.py",
        "--target",
        "twi",
        "--alert-score",
        "6",
        "--watch-score",
        "4",
    ]
    try:
        cp = subprocess.run(
            cmd,
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except Exception as e:
        return None, str(e)

    out = (cp.stdout or "").strip()
    if cp.returncode != 0:
        err = (cp.stderr or "").strip()
        return None, f"exit={cp.returncode}; {err[:200] if err else '執行失敗'}"
    if not out:
        return None, "無輸出"

    def pick(pat: str) -> str | None:
        m = re.search(pat, out, flags=re.MULTILINE)
        return m.group(1).strip() if m else None

    summary = {
        "target": pick(r"^target:\s*(.+)$"),
        "date": pick(r"^date:\s*(.+)$"),
        "level": pick(r"^signal_level:\s*(.+)$"),
        "score": pick(r"^score:\s*(.+)$"),
        "hits": None,
        "snapshot": None,
    }
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("條件命中:"):
            summary["hits"] = line.replace("條件命中:", "", 1).strip()
        elif line.startswith("快照數值:"):
            summary["snapshot"] = line.replace("快照數值:", "", 1).strip()
    return summary, None


@st.cache_data(show_spinner="下載週線與日線…")
def load_market_data(ticker: str, start_date: str, _refresh_key: str):
    """依代號與起始日抓取週線狀態機 + 日線（含週結束日）；快取 60 秒與頁面每分鐘自動 rerun 對齊。"""
    df_w = fetch_weekly_stock_data(ticker, start_date=start_date)
    stats = calculate_long_position_stats(df_w)
    df_d = fetch_daily_chinese(ticker, start_date=start_date)
    daily_we, _ = build_daily_with_week_end(df_d)
    return df_w, stats, daily_we


def _is_market_open_now() -> bool:
    """台股一般盤時段：09:00~13:45（本機時間）。"""
    now_t = datetime.now().time()
    return time(9, 0) <= now_t <= time(13, 45)


def _market_refresh_key() -> str:
    """
    做多段資料刷新鍵：
    - 09:00~13:45：每分鐘換 key
    - 其他時段：每 30 分鐘換 key
    """
    now = datetime.now()
    if _is_market_open_now():
        return now.strftime("mkt-%Y%m%d-%H%M")
    slot = (now.minute // 30) * 30
    return now.replace(minute=slot, second=0, microsecond=0).strftime("mkt-%Y%m%d-%H%M")


def _bottom_refresh_key() -> str:
    """底部策略固定每 30 分鐘換 key。"""
    now = datetime.now()
    slot = (now.minute // 30) * 30
    return now.replace(minute=slot, second=0, microsecond=0).strftime("btm-%Y%m%d-%H%M")


def _is_after_1500() -> bool:
    return datetime.now().time() >= time(15, 0)


@st.cache_data(ttl=1800, show_spinner=False)
def _cached_bottom_strategy_partial_latest(_refresh_key: str) -> tuple[dict | None, str | None]:
    """
    讀取底部策略同源資料的「最新日期」快照（允許部分欄位缺值）。
    用於 15:00 後先顯示最新日，缺欄位標示尚未更新。
    """
    try:
        import bottom_strategy as bs_mod

        stock_id = bs_mod.resolve_stock_id("twi", "加權指數")
        urls = bs_mod.build_urls(stock_id)
        sess = bs_mod.requests.Session()
        html_k = bs_mod.fetch_goodinfo_html(sess, urls["k"], timeout=30)
        html_b = bs_mod.fetch_goodinfo_html(sess, urls["buy"], timeout=30)
        html_m = bs_mod.fetch_goodinfo_html(sess, urls["margin"], timeout=30)
        k_df = bs_mod.select_main_table(html_k)
        b_df = bs_mod.select_main_table(html_b)
        m_df = bs_mod.select_main_table(html_m)
        feat = bs_mod.build_feature_frame(k_df, b_df, m_df)
        if feat is None or len(feat) == 0:
            return None, "無最新快照資料"
        row = feat.sort_values("date").iloc[-1]

        def _fmt(v: float | int | None, suffix: str = "") -> str:
            if v is None or pd.isna(v):
                return "尚未更新"
            return f"{float(v):.2f}{suffix}"

        return {
            "date": pd.Timestamp(row["date"]).strftime("%Y-%m-%d") if pd.notna(row["date"]) else None,
            "snapshot_map": {
                "漲跌幅": _fmt(row.get("chg_pct"), "%"),
                "振幅": _fmt(row.get("amp_pct"), "%"),
                "成交量": _fmt(row.get("volume_b"), "億元"),
                "三大法人買賣超": _fmt(row.get("inst_net_100m"), "億元"),
                "融資增減": _fmt(row.get("mgn_chg_100m"), "億元"),
                "融券增減率": _fmt(row.get("short_chg_pct"), "%"),
            },
        }, None
    except Exception as e:
        return None, str(e)


def _render_bottom_strategy_panel(
    bs: dict | None,
    bs_err: str | None,
    partial_latest: dict | None = None,
) -> None:
    if bs_err:
        st.caption(f"底部策略：讀取失敗（{bs_err}）")
        return
    if not bs:
        return

    bs_date = pd.to_datetime(bs.get("date"), errors="coerce")
    partial_date = pd.to_datetime((partial_latest or {}).get("date"), errors="coerce")
    use_partial = bool(
        _is_after_1500()
        and partial_latest
        and pd.notna(partial_date)
        and (pd.isna(bs_date) or partial_date > bs_date)
    )

    score_text = bs.get("score") or "-"
    panel_date = bs.get("date") or "-"
    if use_partial:
        panel_date = partial_latest.get("date") or panel_date
        score_text = "尚未更新"

    st.info(
        "底部策略 | "
        f"日期 {panel_date} | "
        f"分數 {score_text}"
    )
    if use_partial and partial_latest:
        snap_map = partial_latest.get("snapshot_map", {})
        card_defs = [
            ("漲跌幅", "大跌"),
            ("振幅", "高振幅"),
            ("成交量", "高成交量"),
            ("三大法人買賣超", "法人偏空"),
            ("融資增減", "融資減少"),
            ("融券增減率", "融券減少"),
        ]
        cards = []
        for metric_name, _ in card_defs:
            val = str(snap_map.get(metric_name, "尚未更新"))
            is_ready = (val != "尚未更新")
            status = "已更新" if is_ready else "尚未更新"
            css_cls = "hit" if is_ready else "pending"
            cards.append(
                f'<div class="bs-card {css_cls}">'
                f'<div class="bs-title">{metric_name}｜{status}</div>'
                f'<div class="bs-value">{val}</div>'
                "</div>"
            )
        st.markdown('<div class="bs-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
        return

    if bs.get("hits") and bs.get("snapshot"):
        hit_map: dict[str, bool] = {}
        for part in str(bs["hits"]).split(","):
            if "=" not in part:
                continue
            k, v = [x.strip() for x in part.split("=", 1)]
            hit_map[k] = (v == "是")

        snap_map: dict[str, str] = {}
        for part in str(bs["snapshot"]).split(","):
            if "=" not in part:
                continue
            k, v = [x.strip() for x in part.split("=", 1)]
            snap_map[k] = v

        card_defs = [
            ("漲跌幅", "大跌"),
            ("振幅", "高振幅"),
            ("成交量", "高成交量"),
            ("三大法人買賣超", "法人偏空"),
            ("融資增減", "融資減少"),
            ("融券增減率", "融券減少"),
        ]
        cards = []
        for metric_name, hit_key in card_defs:
            val = snap_map.get(metric_name, "-")
            is_hit = bool(hit_map.get(hit_key, False))
            status = "達標" if is_hit else "未達"
            cards.append(
                f'<div class="bs-card {"hit" if is_hit else "miss"}">'
                f'<div class="bs-title">{metric_name}｜{status}</div>'
                f'<div class="bs-value">{val}</div>'
                "</div>"
            )
        st.markdown('<div class="bs-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)
    else:
        if bs.get("hits"):
            st.caption(f"條件命中：{bs['hits']}")
        if bs.get("snapshot"):
            st.caption(f"快照數值：{bs['snapshot']}")


st.set_page_config(page_title="做多段日線鑽取", layout="wide")
# 行動裝置優化：縮小字級與留白，讓單手滑動閱讀更順。
st.markdown(
    """
    <style>
    .kpi-row {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 1rem;
      margin: 0.2rem 0 0.5rem 0;
      max-width: 760px;
      width: 96%;
      margin-left: auto;
      margin-right: auto;
    }
    .kpi-col {
      flex: 1 1 0;
      min-width: 0;
    }
    .kpi-col.right {
      text-align: right;
    }
    .kpi-label {
      color: #111827;
      font-size: 0.96rem;
      line-height: 1.25;
      margin-bottom: 0.15rem;
      font-weight: 600;
      white-space: nowrap;
    }
    .kpi-value {
      color: #111827;
      font-size: 1.5rem;
      line-height: 1.25;
      font-weight: 600;
      white-space: nowrap;
    }
    .section-label {
      font-size: 1.05rem;
      font-weight: 600;
      line-height: 1.35;
      margin: 0;
      padding: 0;
    }
    .bs-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 0.45rem;
      margin: 0.25rem 0 0.35rem 0;
    }
    .bs-card {
      border-radius: 0.5rem;
      border: 1px solid #d1d5db;
      padding: 0.5rem 0.55rem;
      line-height: 1.25;
    }
    .bs-card.hit {
      background: #ecfdf5;
      border-color: #10b981;
    }
    .bs-card.miss {
      background: #f3f4f6;
      border-color: #d1d5db;
    }
    .bs-card.pending {
      background: #fffbeb;
      border-color: #f59e0b;
    }
    .bs-title {
      font-size: 0.78rem;
      font-weight: 600;
      color: #374151;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    .bs-value {
      font-size: 0.88rem;
      font-weight: 700;
      color: #111827;
      margin-top: 0.12rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }
    /* 自動刷新元件容器不需要可視空間，將留白壓到最小 */
    .st-key-long_underwater_dynamic_refresh {
      margin: 0 !important;
      padding: 0 !important;
      min-height: 0 !important;
    }
    .st-key-long_underwater_dynamic_refresh > div {
      margin: 0 !important;
      padding: 0 !important;
      min-height: 0 !important;
    }
    /* 隱藏右上角 Deploy 按鈕 */
    [data-testid="stAppDeployButton"] {
      display: none !important;
    }
    @media (max-width: 768px) {
      .block-container {
        padding-top: 0.8rem !important;
        padding-left: 0.8rem !important;
        padding-right: 0.8rem !important;
        padding-bottom: 1rem !important;
      }
      h1 {
        font-size: 1.35rem !important;
        line-height: 1.35 !important;
        margin-bottom: 0.35rem !important;
      }
      h2, h3 {
        font-size: 1.05rem !important;
        line-height: 1.35 !important;
        margin-top: 0.75rem !important;
        margin-bottom: 0.35rem !important;
      }
      p, label, .stCaption, .stMarkdown, .stAlert {
        font-size: 0.92rem !important;
      }
      .stSelectbox label, .stTextInput label {
        font-size: 0.85rem !important;
      }
      .kpi-row {
        gap: 0.55rem;
        width: 98%;
      }
      .kpi-label {
        font-size: 1.2rem;
      }
      .kpi-value {
        font-size: 1.5rem;
      }
      .section-label {
        font-size: 1rem;
      }
      .bs-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }
      .bs-title {
        font-size: 0.74rem;
      }
      .bs-value {
        font-size: 0.82rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# 交易時段每分鐘更新；其餘時段每 30 分鐘更新。
_refresh_sec = 60 if _is_market_open_now() else 1800
st_autorefresh(interval=_refresh_sec * 1000, limit=None, key="long_underwater_dynamic_refresh")

with st.sidebar:
    ticker = st.text_input("股票代號", value="^TWII")
    start_date = st.text_input("資料起始日", value="2020-01-01")
    st.caption("做多段：09:00~13:45 每分鐘更新，其餘時段每 30 分鐘更新；底部策略固定每 30 分鐘更新。")

_t = ticker.strip()
_sd = start_date.strip()
bs_slot = st.empty()
load_slot = st.empty()

_bs: dict | None = None
_bs_err: str | None = None
_bs_partial: dict | None = None
_bs_partial_err: str | None = None
df_w = stats = daily_we = None
market_err: Exception | None = None

with ThreadPoolExecutor(max_workers=2) as executor:
    futures = {
        executor.submit(_cached_bottom_strategy_summary, _bottom_refresh_key()): "bs",
        executor.submit(load_market_data, _t, _sd, _market_refresh_key()): "market",
    }
    if _is_after_1500():
        futures[executor.submit(_cached_bottom_strategy_partial_latest, _bottom_refresh_key())] = "bs_partial"
    for future in as_completed(futures):
        kind = futures[future]
        if kind == "bs":
            try:
                _bs, _bs_err = future.result()
            except Exception as e:
                _bs, _bs_err = None, str(e)
            with bs_slot.container():
                _render_bottom_strategy_panel(_bs, _bs_err, _bs_partial)
        elif kind == "bs_partial":
            try:
                _bs_partial, _bs_partial_err = future.result()
            except Exception as e:
                _bs_partial, _bs_partial_err = None, str(e)
            # partial 失敗不阻斷：僅回落完整訊號顯示
            with bs_slot.container():
                _render_bottom_strategy_panel(_bs, _bs_err, _bs_partial)
        else:
            try:
                df_w, stats, daily_we = future.result()
                load_slot.caption("做多段資料已載入")
            except Exception as e:
                market_err = e
                load_slot.error(f"資料載入失敗：{e}")

if market_err is not None:
    st.stop()
load_slot.empty()


st.caption("週界：週四~下週三｜進場=首週收盤")

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
# 統計列「進行中」或（選最後一段且週線最新列仍為做多中）都視為進行中，圖表需延伸到日線最後一週
is_ongoing = (
    ("狀態" in row.index and str(row.get("狀態", "")).strip() == "進行中")
    or (choice == _last_idx and _latest_state == "做多中")
)
# 日線可能比週線表多一週（yfinance 已更新、週線最後列仍停在上週三）：segment_slice 以「週結束日」篩選，
# 若 slice_upper 只跟週線走會整週被切掉（例如少 2026-03-23 這類尚未收週的交易日）。
if is_ongoing:
    _dwe_max = pd.to_datetime(daily_we["週結束日"], errors="coerce").max()
    if pd.notna(_dwe_max):
        _dwe_max = pd.Timestamp(_dwe_max).normalize()
        display_week_end = max(latest_week_end, _dwe_max)
    else:
        display_week_end = latest_week_end
else:
    display_week_end = None

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

first_break_text = (
    res["first_intraday_break"]["日期"].strftime("%Y-%m-%d")
    if res["first_intraday_break"]
    else "無"
)
st.markdown(
    f"""
    <div class="kpi-row">
      <div class="kpi-col">
        <div class="kpi-label">進場價</div>
        <div class="kpi-value">{entry_price:,.2f}</div>
      </div>
      <div class="kpi-col right">
        <div class="kpi-label">首次盤中跌破</div>
        <div class="kpi-value">{first_break_text}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">日線 K 線</div>', unsafe_allow_html=True)
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
        annotation_text=f"警戒線 {alert_line:,.2f}",
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
                name="出場梯線",
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
                name="煞車價位",
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
                name="煞車T",
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
                name="MA20",
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
                name="跌破",
                marker=dict(size=12, color="red"),
                text=["跌破"],
                textposition="top center",
            )
        )

    fig.update_layout(
        # 手機螢幕較短，預設用較緊湊高度；桌機仍可用容器寬度展開。
        height=420,
        xaxis_title=None,
        yaxis_title=None,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(size=11),
            entrywidthmode="pixels",
            entrywidth=72,
        ),
        margin=dict(l=8, r=8, t=30, b=8),
        xaxis_rangeslider_visible=False,
        # 關閉時間軸／價格軸拖曳與縮放（無下方滑軌、無框選縮放）
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        dragmode=False,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"scrollZoom": False, "doubleClick": False},
    )

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
