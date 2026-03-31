"""
本地瀏覽：做多段週線區間 OHLC + 日線鑽取（進場收盤後才判斷跌破）
執行：cd 到本資料夾上層或本資料夾後
  streamlit run long_underwater_web/app.py
"""
from __future__ import annotations

import html
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, time, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import numpy as np
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

# 確保可 import tw_index_futur（同層 clone：tw_index_futur 在 app.py 所在目錄；monorepo：在上層目錄）
_APP_DIR = Path(__file__).resolve().parent
if (_APP_DIR / "tw_index_futur").exists():
    _STK_ROOT = _APP_DIR
else:
    _STK_ROOT = _APP_DIR.parent
if str(_STK_ROOT / "tw_index_futur") not in sys.path:
    sys.path.insert(0, str(_STK_ROOT / "tw_index_futur"))

# yfinance 預設把 SQLite 快取放在 user_cache_dir；若目錄不可寫或異常會報 unable to open database file
_yf_cache_dir = _STK_ROOT / ".yfinance_cache"
_yf_cache_dir.mkdir(parents=True, exist_ok=True)
import yfinance as yf  # noqa: E402

yf.set_tz_cache_location(str(_yf_cache_dir))

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
def _cached_bottom_strategy_summary(
    _refresh_key: str,
    trade_date_str: str,
) -> tuple[dict | None, str | None]:
    """
    以指定日期計算 bottom strategy（取得最近一筆 <= 指定日期的交易日）。

    注意：_refresh_key 只用來讓快取可定期失效；真正的取值由 trade_date_str 決定。
    """
    try:
        import bottom_strategy as bs_mod

        target = "twi"
        stock_id = bs_mod.resolve_stock_id(target, "加權指數")
        cfg = bs_mod.StrategyConfig(
            stock_id=stock_id,
            score_alert=6,
            score_watch=4,
            timeout_sec=30,
        )

        trade_date = pd.Timestamp(trade_date_str).normalize()
        if pd.isna(trade_date):
            return None, f"invalid trade_date_str: {trade_date_str}"

        start_date_str = "2019-01-01"
        end_date_str = trade_date.strftime("%Y-%m-%d")
        requests_spec = bs_mod.build_post_requests(cfg.stock_id, start_date_str, end_date_str)
        legacy_urls = bs_mod.build_legacy_urls(cfg.stock_id)
        sess = bs_mod.requests.Session()
        html_k, _ = bs_mod.fetch_goodinfo_html_post(
            sess,
            requests_spec["k"]["url"],
            requests_spec["k"]["data"],
            timeout=cfg.timeout_sec,
            fallback_url=legacy_urls["k"],
            allow_legacy_fallback=False,
            query_url=requests_spec["k"]["query_url"],
            referer_url=requests_spec["k"]["referer_url"],
        )
        html_b, _ = bs_mod.fetch_goodinfo_html_post(
            sess,
            requests_spec["buy"]["url"],
            requests_spec["buy"]["data"],
            timeout=cfg.timeout_sec,
            fallback_url=legacy_urls["buy"],
            allow_legacy_fallback=False,
            query_url=requests_spec["buy"]["query_url"],
            referer_url=requests_spec["buy"]["referer_url"],
        )
        html_m, _ = bs_mod.fetch_goodinfo_html_post(
            sess,
            requests_spec["margin"]["url"],
            requests_spec["margin"]["data"],
            timeout=cfg.timeout_sec,
            fallback_url=legacy_urls["margin"],
            allow_legacy_fallback=False,
            query_url=requests_spec["margin"]["query_url"],
            referer_url=requests_spec["margin"]["referer_url"],
        )
        k_df = bs_mod.select_main_table(html_k)
        b_df = bs_mod.select_main_table(html_b)
        m_df = bs_mod.select_main_table(html_m)

        feat = bs_mod.build_feature_frame(k_df, b_df, m_df)
        feat = bs_mod.add_score(feat)
        if feat is None or len(feat) == 0:
            return None, "NO_DATA: bottom strategy feature is empty"

        feat = feat.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
        latest_available_date = None
        if len(feat) > 0 and feat.get("date") is not None:
            try:
                latest_available_date = pd.Timestamp(feat["date"].max()).strftime("%Y-%m-%d")
            except Exception:
                latest_available_date = None
        # 指定日期可能落在週末/假日：取「最近一筆 <= 指定日期」；若完全沒有則取最早一筆。
        cand = feat[feat["date"] <= trade_date]
        row = cand.iloc[-1] if len(cand) else feat.iloc[0]

        score = int(row["score"])
        if score >= cfg.score_alert:
            level = "ALERT"
        elif score >= cfg.score_watch:
            level = "WATCH"
        else:
            level = "NO_SIGNAL"

        yn = lambda v: "是" if bool(v) else "否"

        hits = (
            "大跌=" + yn(row["big_drop"]) + ", "
            "高振幅=" + yn(row["high_amp"]) + ", "
            "高成交量=" + yn(row["high_vol"]) + ", "
            "法人偏空=" + yn(row["inst_sell"]) + ", "
            "融資減少=" + yn(row["mgn_reduce"]) + ", "
            "融券減少=" + yn(row["short_reduce"])
        )

        snapshot = (
            f"漲跌幅={float(row['chg_pct']):.2f}%, "
            f"振幅={float(row['amp_pct']):.2f}%, "
            f"成交量={float(row['volume_b']):.2f}億元, "
            f"三大法人買賣超={float(row['inst_net_100m']):.2f}億元, "
            f"融資增減={float(row['mgn_chg_100m']):.2f}億元, "
            f"融券增減率={float(row['short_chg_pct']):.2f}%"
        )

        panel_date = pd.Timestamp(row["date"]).strftime("%Y-%m-%d")
        score_text = f"{score} / 6"

        return {
            "target": target,
            "date": panel_date,
            "latest_available_date": latest_available_date,
            "level": level,
            "score": score_text,
            "hits": hits,
            "snapshot": snapshot,
        }, None
    except Exception as e:
        return None, str(e)


@st.cache_data(show_spinner="下載週線與日線…")
def load_market_data(ticker: str, start_date: str, _refresh_key: str):
    """依代號與起始日抓取週線狀態機 + 日線（含週結束日）；以 _refresh_key 每分鐘失效快取。"""
    df_w = fetch_weekly_stock_data(ticker, start_date=start_date)
    stats = calculate_long_position_stats(df_w)
    df_d = fetch_daily_chinese(ticker, start_date=start_date)
    daily_we, _ = build_daily_with_week_end(df_d)
    return df_w, stats, daily_we


def _market_refresh_key() -> str:
    """做多段資料刷新鍵：全天每分鐘一個 key，與 st_autorefresh 對齊。"""
    return datetime.now().strftime("mkt-%Y%m%d-%H%M")


def _bottom_refresh_key() -> str:
    """底部策略固定每 30 分鐘換 key。"""
    now = datetime.now()
    slot = (now.minute // 30) * 30
    return now.replace(minute=slot, second=0, microsecond=0).strftime("btm-%Y%m%d-%H%M")


def _calc_week_end_from_trade_date(trade_date: pd.Timestamp) -> pd.Timestamp:
    """依既有規則（週四～下週三）計算某交易日對應的週結束日（週三）。"""
    d = pd.Timestamp(trade_date).normalize()
    wd = d.dayofweek
    if wd <= 2:  # 週一~週三
        return d + pd.Timedelta(days=(2 - wd))
    # 週四~週日歸到下週三（交易日通常不會出現週末，保留通用邏輯）
    return d + pd.Timedelta(days=(2 - wd + 7))


def _apply_live_last_candle(seg_daily_chart: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    保留整體資料快取（TTL=60），但每次 rerun 仍用即時來源覆蓋最後一根日K，改善「看起來不動」。
    目前僅對 ^TWII 啟用，避免改變其他標的既有行為。
    """
    if seg_daily_chart is None or len(seg_daily_chart) == 0:
        return seg_daily_chart
    if ticker.strip().upper() != "^TWII":
        return seg_daily_chart

    try:
        from yahoo_tw_twii_price import fetch_twii_quote_ohlc

        live_ohlc, live_err = fetch_twii_quote_ohlc()
        if live_err or not live_ohlc:
            return seg_daily_chart

        q_date = pd.Timestamp(live_ohlc["日期"]).normalize()
        q_open = float(live_ohlc["開盤價"])
        q_high = float(live_ohlc["最高價"])
        q_low = float(live_ohlc["最低價"])
        q_close = float(live_ohlc["收盤價"])
        q_week_end = _calc_week_end_from_trade_date(q_date)

        out = seg_daily_chart.copy()
        out["日期"] = pd.to_datetime(out["日期"]).dt.normalize()

        same_day_mask = out["日期"] == q_date
        if bool(same_day_mask.any()):
            out.loc[same_day_mask, ["開盤價", "最高價", "最低價", "收盤價", "週結束日"]] = [
                q_open,
                q_high,
                q_low,
                q_close,
                q_week_end,
            ]
        elif q_date > pd.Timestamp(out["日期"].max()).normalize():
            # 若快取日線尚未包含今日，僅為圖表補上一根最新日K（不影響統計主資料）
            extra = {c: np.nan for c in out.columns}
            extra.update(
                {
                    "日期": q_date,
                    "開盤價": q_open,
                    "最高價": q_high,
                    "最低價": q_low,
                    "收盤價": q_close,
                    "週結束日": q_week_end,
                }
            )
            out = pd.concat([out, pd.DataFrame([extra])], ignore_index=True)

        out = out.sort_values("日期").reset_index(drop=True)
        return out
    except Exception:
        return seg_daily_chart


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
        start_date_str = "2019-01-01"
        end_date_str = datetime.now().strftime("%Y-%m-%d")
        requests_spec = bs_mod.build_post_requests(stock_id, start_date_str, end_date_str)
        legacy_urls = bs_mod.build_legacy_urls(stock_id)
        sess = bs_mod.requests.Session()
        html_k, _ = bs_mod.fetch_goodinfo_html_post(
            sess,
            requests_spec["k"]["url"],
            requests_spec["k"]["data"],
            timeout=30,
            fallback_url=legacy_urls["k"],
            allow_legacy_fallback=False,
            query_url=requests_spec["k"]["query_url"],
            referer_url=requests_spec["k"]["referer_url"],
        )
        html_b, _ = bs_mod.fetch_goodinfo_html_post(
            sess,
            requests_spec["buy"]["url"],
            requests_spec["buy"]["data"],
            timeout=30,
            fallback_url=legacy_urls["buy"],
            allow_legacy_fallback=False,
            query_url=requests_spec["buy"]["query_url"],
            referer_url=requests_spec["buy"]["referer_url"],
        )
        html_m, _ = bs_mod.fetch_goodinfo_html_post(
            sess,
            requests_spec["margin"]["url"],
            requests_spec["margin"]["data"],
            timeout=30,
            fallback_url=legacy_urls["margin"],
            allow_legacy_fallback=False,
            query_url=requests_spec["margin"]["query_url"],
            referer_url=requests_spec["margin"]["referer_url"],
        )
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
    allow_partial_latest: bool = False,
) -> None:
    if bs_err:
        st.caption(f"底部策略：讀取失敗（{bs_err}）")
        return
    if not bs:
        return

    bs_date = pd.to_datetime(bs.get("date"), errors="coerce")
    partial_date = pd.to_datetime((partial_latest or {}).get("date"), errors="coerce")
    use_partial = bool(
        allow_partial_latest
        and partial_latest
        and pd.notna(partial_date)
        and (pd.isna(bs_date) or partial_date > bs_date)
    )

    score_text = bs.get("score") or "—"
    panel_date = bs.get("date") or "—"
    if use_partial:
        panel_date = partial_latest.get("date") or panel_date
        score_text = "…"

    # 底部策略標題 + 日期選擇器（calendar）+ 前後一天按鈕：同一列呈現，避免往下排列。
    today = datetime.now().date()
    min_bs_date = today - timedelta(days=365)
    max_bs_date = today

    # 預設日期：採 goodinfo 查到的「最新可用日期」（也就是 panel_date）。
    default_trade_date = pd.to_datetime(panel_date, errors="coerce")
    if pd.notna(default_trade_date):
        default_trade_date = default_trade_date.date()
    else:
        default_trade_date = max_bs_date

    # 可選的最大日期：以 goodinfo 查到的「全域最新可用日期」為上限
    # 注意：panel_date 會隨使用者調整而變動（例如從 25 切到 24），
    # 但這不應該影響日曆 max；因此使用 bs（底部策略同源）回傳的 latest_available_date。
    latest_available_raw = bs.get("latest_available_date") if isinstance(bs, dict) else None
    latest_available = pd.to_datetime(latest_available_raw, errors="coerce")
    max_bs_date = latest_available.date() if pd.notna(latest_available) else default_trade_date

    # 若尚未被設定（或 session_state 殘留了超出 goodinfo 最新日期的值），就改成 goodinfo 的最新日期。
    cur_ss_date = st.session_state.get("bs_trade_date")
    try:
        cur_ss_date_parsed = pd.to_datetime(cur_ss_date).date() if cur_ss_date is not None else None
    except Exception:
        cur_ss_date_parsed = None

    if cur_ss_date_parsed is None:
        st.session_state["bs_trade_date"] = default_trade_date
    else:
        if cur_ss_date_parsed < min_bs_date:
            st.session_state["bs_trade_date"] = min_bs_date
        elif cur_ss_date_parsed > max_bs_date:
            st.session_state["bs_trade_date"] = max_bs_date

    cur_date = st.session_state.get("bs_trade_date", default_trade_date)

    can_prev = cur_date > min_bs_date
    can_next = cur_date < max_bs_date

    # 右側（◀/▶ + calendar）更靠近標題文字：調整欄位權重並縮小列間距
    # 標題 + 控制列（◀ / calendar icon / ▶）：以自訂 flex row 呈現，確保緊鄰且手機不換行。
    st.markdown(
        f"""
        <div class="bs-header-row">
          <div class="bs-title">
            底部策略 — {html.escape(str(panel_date))}　分數 {html.escape(str(score_text))}
          </div>
          <div class="bs-controls" aria-label="底部策略日期控制">
            <button class="bs-ctrl-btn" type="button" id="bsPrev" aria-label="前一天" {("disabled" if not can_prev else "")}>◀</button>
            <button class="bs-ctrl-btn bs-ctrl-cal" type="button" id="bsCal" aria-label="選擇日期"></button>
            <button class="bs-ctrl-btn" type="button" id="bsNext" aria-label="後一天">▶</button>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 注意：避免在 `st.date_input(key="bs_trade_date")` 實例化之後再修改同一個 key。
    # 保留原本 widget 以維持狀態與邏輯，但移出畫面；由上面的 icon 按鈕觸發它們。
    hidden_prev = st.button("◀", key="bs_trade_date_prev")
    if hidden_prev and cur_date > min_bs_date:
        st.session_state["bs_trade_date"] = max(min_bs_date, cur_date - timedelta(days=1))
        st.rerun()

    hidden_next = st.button("▶", key="bs_trade_date_next")
    if hidden_next and cur_date < max_bs_date:
        st.session_state["bs_trade_date"] = min(max_bs_date, cur_date + timedelta(days=1))
        st.rerun()

    st.date_input(
        "底部策略日期（最近一年）",
        min_value=min_bs_date,
        max_value=max_bs_date,
        key="bs_trade_date",
        label_visibility="collapsed",
    )

    components.html(
        """
        <script>
        (function () {
          var DOC = window.parent.document;
          function qs(sel){ try { return DOC.querySelector(sel); } catch(e){ return null; } }
          function bindOnce(id, fn){
            var el = qs('#' + id);
            if (!el) return;
            if (el.__bsBound) return;
            el.__bsBound = true;
            el.addEventListener('click', fn);
          }

          function clickPrev(){
            var btn = qs('div.st-key-bs_trade_date_prev button');
            if (btn) btn.click();
          }
          function clickNext(){
            var btn = qs('div.st-key-bs_trade_date_next button');
            if (btn) btn.click();
          }
          function openCalendar(){
            // 依據不同 Streamlit/版本，input/層級可能略有差異：多嘗試幾種 selector。
            var input = qs('div.st-key-bs_trade_date input[data-testid="stDateInputField"]');
            if (!input) input = qs('div[data-testid="stDateInput"] input[data-testid="stDateInputField"]');
            if (input) {
              input.focus();
              input.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
              return;
            }

            // fallback：點擊 base input 容器（通常可觸發 datepicker 開啟）
            var base = qs('div.st-key-bs_trade_date div[data-baseweb="input"]');
            if (!base) base = qs('div[data-testid="stDateInput"] div[data-baseweb="input"]');
            if (base) base.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
          }

          bindOnce('bsPrev', clickPrev);
          bindOnce('bsNext', clickNext);
          bindOnce('bsCal', openCalendar);
        })();
        </script>
        """,
        height=0,
    )

    if use_partial and partial_latest:
        snap_map = partial_latest.get("snapshot_map", {})
        # (顯示標籤, snapshot_map 鍵)
        chip_specs = [
            ("漲跌幅", "漲跌幅"),
            ("振幅", "振幅"),
            ("成交量", "成交量"),
            ("三大法人", "三大法人買賣超"),
            ("融資增減", "融資增減"),
            ("融券增減率", "融券增減率"),
        ]
        chips = []
        for label, key in chip_specs:
            val = str(snap_map.get(key, "尚未更新"))
            is_ready = val != "尚未更新"
            css_cls = "hit" if is_ready else "pending"
            chips.append(
                f'<div class="v2-bs-chip {css_cls}">'
                f'<div class="v2-bs-chip-label">{html.escape(label)}</div>'
                f'<div class="v2-bs-chip-val">{html.escape(val)}</div>'
                "</div>"
            )
        st.markdown('<div class="v2-bs-row">' + "".join(chips) + "</div>", unsafe_allow_html=True)
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
        chips = []
        for metric_name, hit_key in card_defs:
            val = snap_map.get(metric_name, "—")
            is_hit = bool(hit_map.get(hit_key, False))
            css_cls = "hit" if is_hit else "miss"
            chips.append(
                f'<div class="v2-bs-chip {css_cls}">'
                f'<div class="v2-bs-chip-label">{html.escape(metric_name)}</div>'
                f'<div class="v2-bs-chip-val">{html.escape(str(val))}</div>'
                "</div>"
            )
        st.markdown('<div class="v2-bs-row">' + "".join(chips) + "</div>", unsafe_allow_html=True)
    else:
        if bs.get("hits"):
            st.caption(f"條件命中：{bs['hits']}")
        if bs.get("snapshot"):
            st.caption(f"快照數值：{bs['snapshot']}")


st.set_page_config(
    page_title="做多段日線鑽取",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# UI：沿用淺色主題與原版 K 線配色；版面參考 long_underwater_web_v2（緊湊留白、卡片化 KPI／底部策略）。
st.markdown(
    """
    <style>
    :root {
      --bg-card: #ffffff;
      --bg-card-hover: #f9fafb;
      --border: #e5e7eb;
      --text-primary: #111827;
      --text-secondary: #6b7280;
      --accent-green: #15803d;
      --accent-red: #b91c1c;
      --accent-amber: #b45309;
      --accent-blue: #2563eb;
    }
    [data-testid="stAppDeployButton"] { display: none !important; }
    [data-testid="stAppViewContainer"] .block-container { padding-top: 0 !important; }
    [data-testid="stMain"] { padding-top: 0 !important; }
    [data-testid="stMainBlockContainer"] { padding-top: 0 !important; }
    div.stElementContainer,
    div[data-testid="stElementContainer"] {
      margin: 0 !important;
      padding: 0 !important;
      min-height: 0 !important;
    }
    div.stElementContainer > div,
    div[data-testid="stElementContainer"] > div {
      margin: 0 !important;
      padding: 0 !important;
    }
    .v2-kpi-strip {
      display: flex;
      gap: 0.75rem;
      flex-wrap: wrap;
      margin-bottom: 1rem;
    }
    .v2-kpi {
      flex: 1 1 140px;
      min-width: 130px;
      background: var(--bg-card);
      border: 1px solid var(--border);
      border-radius: 0.75rem;
      padding: 0.75rem 1rem;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      transition: background 0.15s;
    }
    .v2-kpi:hover { background: var(--bg-card-hover); }
    .v2-kpi-label {
      font-size: 0.72rem;
      font-weight: 600;
      color: var(--text-secondary);
      text-transform: uppercase;
      letter-spacing: 0.06em;
      margin-bottom: 0.2rem;
    }
    .v2-kpi-value {
      font-size: 1.4rem;
      font-weight: 700;
      color: var(--text-primary);
      font-variant-numeric: tabular-nums;
    }
    .v2-kpi-value.green { color: var(--accent-green); }
    .v2-kpi-value.red   { color: var(--accent-red);   }
    .v2-kpi-value.amber  { color: var(--accent-amber); }
    .v2-bs-row {
      display: flex;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin: 0.5rem 0 1rem 0;
    }
    .v2-bs-chip {
      flex: 1 1 0;
      min-width: 100px;
      background: #f9fafb;
      border: 1px solid var(--border);
      border-radius: 0.6rem;
      padding: 0.55rem 0.7rem;
      text-align: center;
    }
    .v2-bs-chip.hit  { border-color: #10b981; background: #ecfdf5; }
    .v2-bs-chip.miss { border-color: var(--border); background: #f3f4f6; }
    .v2-bs-chip.pending { border-color: #f59e0b; background: #fffbeb; }
    .v2-bs-chip-label {
      font-size: 0.68rem;
      font-weight: 600;
      color: var(--text-secondary);
      margin-bottom: 0.15rem;
    }
    .v2-bs-chip-val {
      font-size: 0.92rem;
      font-weight: 700;
      color: var(--text-primary);
    }
    .v2-section {
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--accent-blue);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin: 1.5rem 0 0.5rem 0;
      border-left: 3px solid var(--accent-blue);
      padding-left: 0.6rem;
    }
    .v2-section--first { margin-top: 0.2rem !important; }
    .st-key-long_underwater_dynamic_refresh,
    .st-key-long_underwater_dynamic_refresh > div {
      margin: 0 !important;
      padding: 0 !important;
      min-height: 0 !important;
    }
    .v2-alert-bar {
      background: rgba(239,68,68,0.12);
      border: 1px solid var(--accent-red);
      border-radius: 0.6rem;
      padding: 0.65rem 1rem;
      margin-bottom: 0.75rem;
      font-weight: 600;
      color: var(--accent-red);
      font-size: 0.92rem;
    }
    .v2-alert-bar.amber {
      background: rgba(245,158,11,0.1);
      border-color: var(--accent-amber);
      color: var(--accent-amber);
    }
    .v2-alert-bar a { color: inherit; text-decoration: underline; }

    /* 底部策略：標題 + 控制（◀ calendar ▶）同列緊貼 */
    .bs-header-row{
      display:flex;
      align-items:center;
      justify-content:space-between;
      gap:8px;
      margin: 0.2rem 0 0.15rem 0;
      overflow: visible;
      flex-wrap: nowrap;
    }
    .bs-title{
      font-size: 0.9rem;
      font-weight: 700;
      color: var(--accent-blue);
      text-transform: uppercase;
      letter-spacing: 0.08em;
      border-left: 3px solid var(--accent-blue);
      padding-left: 0.6rem;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      min-width: 0;
      flex: 1 1 auto;
    }
    .bs-controls{
      display:flex;
      align-items:center;
      gap:4px;
      flex: 0 0 auto;
      white-space: nowrap;
    }
    .bs-ctrl-btn{
      width: 26px;
      height: 26px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: #fff;
      color: #111827;
      font-weight: 700;
      line-height: 1;
      padding: 0;
      margin: 0;
      display:flex;
      align-items:center;
      justify-content:center;
      box-shadow: 0 1px 2px rgba(0,0,0,0.06);
      cursor: pointer;
      user-select: none;
    }
    .bs-ctrl-btn:disabled{
      opacity: 0.45 !important;
      cursor: not-allowed !important;
      box-shadow: none !important;
    }
    .bs-ctrl-cal{
      position: relative;
      color: transparent; /* 只顯示 icon */
    }
    .bs-ctrl-cal::after{
      content:"";
      position:absolute;
      left:50%;
      top:50%;
      transform: translate(-50%, -50%);
      width: 14px;
      height: 14px;
      background-repeat:no-repeat;
      background-position:center;
      background-size:contain;
      background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='%2315803d'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='4'%20width='18'%20height='18'%20rx='2'%20/%3E%3Cline%20x1='16'%20y1='2'%20x2='16'%20y2='6'%20/%3E%3Cline%20x1='8'%20y1='2'%20x2='8'%20y2='6'%20/%3E%3Cline%20x1='3'%20y1='10'%20x2='21'%20y2='10'%20/%3E%3C/svg%3E");
      pointer-events:none;
    }

    /* 隱藏底部策略用的原生 widgets（仍保留於 DOM 供 JS 觸發） */
    div.st-key-bs_trade_date_prev,
    div.st-key-bs_trade_date_next,
    div.st-key-bs_trade_date{
      position: absolute !important;
      left: 0 !important;
      top: 0 !important;
      width: 1px !important;
      height: 1px !important;
      overflow: hidden !important;
      opacity: 0 !important;
      pointer-events: auto !important;
    }

    /* 底部策略日期：只留一顆 calendar icon（文字隱藏，icon 由 CSS 疊上） */
    div[data-testid="stDateInput"] {
      width: 26px !important;
      min-width: 26px !important;
      height: 26px !important;
      position: relative !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      overflow: visible !important;
    }
    /* 隱藏 label，避免佔位 */
    div[data-testid="stDateInput"] label[data-testid="stWidgetLabel"] {
      display: none !important;
    }
    div[data-testid="stDateInput"] div[data-baseweb="input"] {
      width: 26px !important;
      height: 26px !important;
      min-width: 26px !important;
      min-height: 26px !important;
    }
    div[data-testid="stDateInput"] div[data-baseweb="base-input"] {
      border: 0 !important;
      background: transparent !important;
      box-shadow: none !important;
      padding: 0 !important;
      margin: 0 !important;
      width: 26px !important;
      height: 26px !important;
    }
    div[data-testid="stDateInput"] input[type="text"] {
      color: transparent !important;
      caret-color: transparent !important;
      background: transparent !important;
      border: 0 !important;
      outline: 0 !important;
      padding: 0 !important;
      margin: 0 !important;
      width: 26px !important;
      height: 26px !important;
      min-width: 26px !important;
      min-height: 26px !important;
      opacity: 0 !important;
    }
    div[data-testid="stDateInput"] button {
      width: 26px !important;
      min-width: 26px !important;
      height: 26px !important;
      padding: 0 !important;
      margin: 0 !important;
      border: 0 !important;
      background: transparent !important;
      box-shadow: none !important;
      display: flex !important;
      align-items: center !important;
      justify-content: center !important;
      overflow: visible !important;
    }
    /* 隱藏 Streamlit 內建的 icon，我們改用 ::after 疊上（避免 svg 被裁掉/顏色錯誤） */
    div[data-testid="stDateInput"] button svg {
      opacity: 0 !important;
      visibility: hidden !important;
      display: none !important;
    }
    div[data-testid="stDateInput"]::after {
      content: "";
      position: absolute !important;
      left: 50% !important;
      top: 50% !important;
      transform: translate(-50%, -50%) !important;
      width: 14px !important;
      height: 14px !important;
      background-repeat: no-repeat !important;
      background-position: center !important;
      background-size: contain !important;
      /* 先用內建 calendar icon；若你後續貼 flaticon 的 base64/svg，我可把這段換成指定那顆。 */
      background-image: url("data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2024%2024'%20fill='none'%20stroke='%2315803d'%20stroke-width='2'%20stroke-linecap='round'%20stroke-linejoin='round'%3E%3Crect%20x='3'%20y='4'%20width='18'%20height='18'%20rx='2'%20/%3E%3Cline%20x1='16'%20y1='2'%20x2='16'%20y2='6'%20/%3E%3Cline%20x1='8'%20y1='2'%20x2='8'%20y2='6'%20/%3E%3Cline%20x1='3'%20y1='10'%20x2='21'%20y2='10'%20/%3E%3C/svg%3E");
      pointer-events: none !important;
      z-index: 2 !important;
    }

    /* 讓 ◀ / ▶ 按鈕更緊貼（減少 margin/padding 佔位） */
    div.st-key-bs_trade_date_prev,
    div.st-key-bs_trade_date_next {
      margin: 0 !important;
      padding: 0 !important;
    }
    div.st-key-bs_trade_date_prev button,
    div.st-key-bs_trade_date_next button {
      width: 26px !important;
      height: 26px !important;
      padding: 0 !important;
      margin: 0 !important;
    }
    @media (max-width: 768px) {
      [data-testid="stAppViewContainer"] .block-container {
        padding: 0 0.75rem 1rem 0.75rem !important;
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
      .v2-kpi-strip { gap: 0.45rem; }
      .v2-kpi { padding: 0.55rem 0.7rem; min-width: 110px; }
      .v2-kpi-value { font-size: 1.15rem; }
      .v2-bs-row { gap: 0.35rem; }
      .v2-bs-chip { min-width: 80px; padding: 0.4rem 0.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 行動端手勢：頁面最上方下拉可重新整理（Pull-to-Refresh）
components.html(
    """
    <script>
    (function () {
      if (window.__longUnderwaterPtrMounted) return;
      window.__longUnderwaterPtrMounted = true;

      let startY = 0;
      let pulling = false;
      let triggered = false;
      const threshold = 85;

      const hint = document.createElement("div");
      hint.textContent = "下拉更新中...";
      hint.style.position = "fixed";
      hint.style.top = "10px";
      hint.style.left = "50%";
      hint.style.transform = "translateX(-50%)";
      hint.style.background = "rgba(17, 24, 39, 0.86)";
      hint.style.color = "#fff";
      hint.style.fontSize = "13px";
      hint.style.padding = "6px 10px";
      hint.style.borderRadius = "999px";
      hint.style.zIndex = "9999";
      hint.style.opacity = "0";
      hint.style.transition = "opacity 0.16s ease";
      document.body.appendChild(hint);

      function pageAtTop() {
        return (window.scrollY || window.pageYOffset || 0) <= 0;
      }

      function showHint(show) {
        hint.style.opacity = show ? "1" : "0";
      }

      window.addEventListener("touchstart", function (e) {
        if (!e.touches || e.touches.length !== 1) return;
        if (!pageAtTop()) return;
        startY = e.touches[0].clientY;
        pulling = true;
        triggered = false;
      }, { passive: true });

      window.addEventListener("touchmove", function (e) {
        if (!pulling || triggered) return;
        if (!e.touches || e.touches.length !== 1) return;
        const currentY = e.touches[0].clientY;
        const deltaY = currentY - startY;
        if (deltaY > 18) showHint(true);
        if (deltaY > threshold && pageAtTop()) {
          triggered = true;
          showHint(true);
          window.setTimeout(function () {
            window.location.reload();
          }, 40);
        }
      }, { passive: true });

      window.addEventListener("touchend", function () {
        pulling = false;
        if (!triggered) showHint(false);
      }, { passive: true });
    })();
    </script>
    """,
    height=0,
    width=0,
)

# 做多段：全天每分鐘自動 rerun（底部策略仍依自身 refresh key，每 30 分鐘才換快取）。
st_autorefresh(interval=60_000, limit=None, key="long_underwater_dynamic_refresh")

with st.sidebar:
    st.markdown("#### 設定")
    ticker = st.text_input("股票代號", value="^TWII")
    start_date = st.text_input("資料起始日", value="2020-01-01")
    st.caption("做多段：全天每分鐘更新；底部策略固定每 30 分鐘更新。")

_t = ticker.strip()
_sd = start_date.strip()

# 底部策略日期（UI 會在渲染底部策略面板後出現，但抓取時需要這個值先存在）
_today = datetime.now().date()
_min_bs_date = _today - timedelta(days=365)
_max_bs_date = _today
st.session_state.setdefault("bs_trade_date", _max_bs_date)
_bs_trade_date = st.session_state.get("bs_trade_date", _max_bs_date)
_bs_trade_date_str = pd.Timestamp(_bs_trade_date).strftime("%Y-%m-%d")
_allow_partial_latest = _is_after_1500() and _bs_trade_date == _max_bs_date

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
        executor.submit(
            _cached_bottom_strategy_summary,
            _bottom_refresh_key(),
            _bs_trade_date_str,
        ): "bs",
        executor.submit(load_market_data, _t, _sd, _market_refresh_key()): "market",
    }
    if _allow_partial_latest:
        futures[executor.submit(_cached_bottom_strategy_partial_latest, _bottom_refresh_key())] = "bs_partial"
    for future in as_completed(futures):
        kind = futures[future]
        if kind == "bs":
            try:
                _bs, _bs_err = future.result()
            except Exception as e:
                _bs, _bs_err = None, str(e)
            with bs_slot.container():
                _render_bottom_strategy_panel(
                    _bs,
                    _bs_err,
                    _bs_partial,
                    allow_partial_latest=_allow_partial_latest,
                )
        elif kind == "bs_partial":
            try:
                _bs_partial, _bs_partial_err = future.result()
            except Exception as e:
                _bs_partial, _bs_partial_err = None, str(e)
            # partial 失敗不阻斷：僅回落完整訊號顯示
            with bs_slot.container():
                _render_bottom_strategy_panel(
                    _bs,
                    _bs_err,
                    _bs_partial,
                    allow_partial_latest=_allow_partial_latest,
                )
        else:
            try:
                df_w, stats, daily_we = future.result()
                load_slot.caption("做多段資料已載入")
            except Exception as e:
                market_err = e
                load_slot.empty()

if market_err is not None:
    st.markdown(
        f'<div class="v2-alert-bar">資料載入失敗：{html.escape(str(market_err))}</div>',
        unsafe_allow_html=True,
    )
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
# st.selectbox 無法為單一選項上色；以標籤中的「 · 虧損」為記號，在主文件為選項與已選文字套用下跌綠（與 K 線一致）。
components.html(
    """<script>
(function () {
  var GREEN = "#15803d";
  var DOC = window.parent.document;
  function paint() {
    try {
      DOC.querySelectorAll('[data-baseweb="popover"] li[role="option"]').forEach(function (li) {
        li.style.color = li.textContent.indexOf("虧損") !== -1 ? GREEN : "";
      });
      var sb = DOC.querySelector('[data-testid="stSelectbox"]');
      if (!sb) return;
      var combo = sb.querySelector('[role="combobox"]');
      if (combo) {
        combo.style.color = combo.textContent.indexOf("虧損") !== -1 ? GREEN : "";
      }
    } catch (e) {}
  }
  var obs = new MutationObserver(paint);
  obs.observe(DOC.body, { childList: true, subtree: true });
  paint();
})();
</script>""",
    height=0,
)
row = stats.iloc[choice]

entry_date = pd.Timestamp(row["進入日期"])
exit_date = pd.Timestamp(row["退出日期"])
entry_price = float(row["進入價格"])
alert_line = entry_price * 0.98
exit_price = float(row["退出價格"]) if "退出價格" in row and pd.notna(row["退出價格"]) else None

# 最後一段 + 週線最新列為「做多中」：對照 Yahoo 台指期近一 WTX&，跌破警戒線則警示
_latest_state = (
    str(df_w.iloc[-1]["交易狀態"]).strip()
    if len(df_w) > 0 and "交易狀態" in df_w.columns
    else ""
)
if choice == _last_idx and _latest_state == "做多中":
    wtx_px, wtx_err = _cached_wtx_night_price()
    if wtx_err:
        st.markdown(
            '<div class="v2-alert-bar amber">台指期近一（<a href="https://tw.stock.yahoo.com/quote/WTX&" '
            f'target="_blank" rel="noopener">Yahoo WTX&amp;</a>）讀取失敗：{html.escape(str(wtx_err))}</div>',
            unsafe_allow_html=True,
        )
    elif wtx_px is not None and wtx_px < alert_line:
        st.markdown(
            f'<div class="v2-alert-bar">台指期 {wtx_px:,.2f} 低於警戒線 {alert_line:,.2f}</div>',
            unsafe_allow_html=True,
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
seg_daily_chart = _apply_live_last_candle(seg_daily_chart, _t)
seg_weekly = res["seg_weekly"]

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
            st.markdown(
                f'<div class="v2-alert-bar">做多煞車警戒：本週已連跌破 {close_below_brake_count} 根</div>',
                unsafe_allow_html=True,
            )

first_break_text = (
    res["first_intraday_break"]["日期"].strftime("%Y-%m-%d")
    if res["first_intraday_break"]
    else "無"
)
chg_pct = (
    (float(seg_daily_chart.iloc[-1]["收盤價"]) / entry_price - 1) * 100
    if len(seg_daily_chart) > 0
    else 0.0
)
chg_cls = "green" if chg_pct >= 0 else "red"
ongoing_cls = "green" if is_ongoing else ""
ongoing_text = "● 進行中" if is_ongoing else "已結束"
st.markdown(
    f'<div class="v2-kpi-strip">'
    f'  <div class="v2-kpi"><div class="v2-kpi-label">進場價</div><div class="v2-kpi-value">{entry_price:,.2f}</div></div>'
    f'  <div class="v2-kpi"><div class="v2-kpi-label">警戒線 (−2%)</div><div class="v2-kpi-value amber">{alert_line:,.2f}</div></div>'
    f'  <div class="v2-kpi"><div class="v2-kpi-label">區間漲幅</div><div class="v2-kpi-value {chg_cls}">{chg_pct:+.2f}%</div></div>'
    f'  <div class="v2-kpi"><div class="v2-kpi-label">首次盤中跌破</div><div class="v2-kpi-value">{html.escape(first_break_text)}</div></div>'
    f'  <div class="v2-kpi"><div class="v2-kpi-label">狀態</div><div class="v2-kpi-value {ongoing_cls}">{ongoing_text}</div></div>'
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown('<div class="v2-section">日線 K 線</div>', unsafe_allow_html=True)
if len(seg_daily_chart) == 0:
    st.warning("此段無日線資料。")
else:
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

st.markdown('<div class="v2-section">週線表（區間 OHLC）</div>', unsafe_allow_html=True)
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

st.markdown('<div style="height:120px"></div>', unsafe_allow_html=True)

