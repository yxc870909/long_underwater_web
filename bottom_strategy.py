import re
import time
import urllib.parse as up
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import requests


BASE_URL = "https://goodinfo.tw/tw/"
DATA_BASE_URL = "https://goodinfo.tw/tw/data/"
QUICK_TARGETS = {
    "twi": "加權指數",
    "twii": "加權指數",
    "0050": "0050",
    "0056": "0056",
    "006208": "006208",
    "2330": "2330",
}


@dataclass
class StrategyConfig:
    stock_id: str = "加權指數"
    score_alert: int = 6
    score_watch: int = 4
    timeout_sec: int = 30


def build_post_requests(stock_id: str, start_dt: str, end_dt: str) -> Dict[str, Dict[str, object]]:
    common = {
        "STEP": "DATA",
        "STOCK_ID": stock_id,
        "CHT_CAT": "DATE",
        "PRICE_ADJ": "F",
        "START_DT": start_dt,
        "END_DT": end_dt,
        "IS_RELOAD_REPORT": "T",
    }

    def _make_spec(data_path: str, sheet: str, referer_path: str) -> Dict[str, str]:
        q = {**common, "SHEET": sheet}
        query = up.urlencode(q, safe=":/")
        return {
            "url": f"{DATA_BASE_URL}{data_path}",
            "query_url": f"{DATA_BASE_URL}{data_path}?{query}",
            "referer_url": f"{BASE_URL}{referer_path}?STOCK_ID={up.quote(stock_id)}",
        }

    return {
        "k": {
            **_make_spec("ShowK_Chart.asp", "大盤指數、法人買賣及融資券", "ShowK_Chart.asp"),
            "data": {**common, "SHEET": "大盤指數、法人買賣及融資券"},
        },
        "buy": {
            **_make_spec("ShowBuySaleChart.asp", "三大法人買賣金額", "ShowBuySaleChart.asp"),
            "data": {**common, "SHEET": "三大法人買賣金額"},
        },
        "margin": {
            **_make_spec("ShowMarginChart.asp", "融資融券餘額", "ShowMarginChart.asp"),
            "data": {**common, "SHEET": "融資融券餘額"},
        },
    }


def build_legacy_urls(stock_id: str) -> Dict[str, str]:
    sid = up.quote(stock_id)
    return {
        "k": f"{BASE_URL}ShowK_Chart.asp?STOCK_ID={sid}",
        "buy": f"{BASE_URL}ShowBuySaleChart.asp?STOCK_ID={sid}",
        "margin": f"{BASE_URL}ShowMarginChart.asp?STOCK_ID={sid}",
    }


def resolve_stock_id(target: str, stock_id: str) -> str:
    key = (target or "").strip().lower()
    if key and key in QUICK_TARGETS:
        return QUICK_TARGETS[key]
    return stock_id


def flatten_columns(cols) -> List[str]:
    out: List[str] = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(x).strip() for x in c if str(x).strip()]
            out.append(" | ".join(parts))
        else:
            out.append(str(c).strip())
    return out


def parse_date_yy_mm_dd(value: str) -> pd.Timestamp:
    s = str(value).strip().replace("'", "")
    if not s:
        return pd.NaT
    for fmt in ("%y/%m/%d", "%Y/%m/%d", "%Y-%m-%d"):
        try:
            return pd.to_datetime(s, format=fmt, errors="raise")
        except Exception:
            pass
    m = re.match(r"^(\d{3})/(\d{1,2})/(\d{1,2})$", s)
    if m:
        y = int(m.group(1)) + 1911
        mm = int(m.group(2))
        dd = int(m.group(3))
        return pd.to_datetime(f"{y:04d}-{mm:02d}-{dd:02d}", format="%Y-%m-%d", errors="coerce")
    return pd.NaT


def to_num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def _extract_cookie_prefix(html: str) -> str:
    marker = "setCookie('CLIENT_KEY',"
    if marker not in html:
        return "2.4|0|0|"
    text = html.split(marker, 1)[1]
    q1 = text.find("'")
    q2 = text.find("'", q1 + 1)
    if q1 >= 0 and q2 > q1:
        return text[q1 + 1 : q2]
    return "2.4|0|0|"


def ensure_goodinfo_client_key(session: requests.Session, timeout: int = 30) -> None:
    has_key = any(c.name == "CLIENT_KEY" and "goodinfo.tw" in (c.domain or "") for c in session.cookies)
    if has_key:
        return
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://goodinfo.tw/"}
    r = session.get(BASE_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    r.encoding = "utf-8"
    prefix = _extract_cookie_prefix(r.text)
    tz = -480
    day = time.time() / 86400 - tz / 1440
    client_key = f"{prefix}{tz}|{day}|{day}"
    session.cookies.set("CLIENT_KEY", client_key, domain="goodinfo.tw", path="/")


def fetch_goodinfo_html_post(
    session: requests.Session,
    url: str,
    data: Dict[str, str],
    timeout: int = 30,
    fallback_url: str = "",
    allow_legacy_fallback: bool = False,
    query_url: str = "",
    referer_url: str = "",
) -> tuple[str, str]:
    ensure_goodinfo_client_key(session, timeout=timeout)
    post_url = query_url or url
    referer = referer_url or "https://goodinfo.tw/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
        "Referer": referer,
        "Origin": "https://goodinfo.tw",
        "Accept": "text/html, */*; q=0.01",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }

    resp = session.post(post_url, headers=headers, data="", timeout=timeout)
    resp.raise_for_status()
    resp.encoding = "utf-8"
    html = resp.text
    if "<table" in html.lower():
        return html, "data_post"

    resp2 = session.get(
        post_url,
        headers={"User-Agent": headers["User-Agent"], "Referer": headers["Referer"]},
        timeout=timeout,
    )
    resp2.raise_for_status()
    resp2.encoding = "utf-8"
    html2 = resp2.text
    if "<table" in html2.lower():
        return html2, "data_get"

    if fallback_url and allow_legacy_fallback:
        legacy_headers = {"User-Agent": headers["User-Agent"], "Referer": "https://goodinfo.tw/"}
        resp3 = session.get(fallback_url, headers=legacy_headers, timeout=timeout)
        resp3.raise_for_status()
        resp3.encoding = "utf-8"
        html3 = resp3.text
        if "<table" in html3.lower():
            return html3, "legacy_get"
        preview = (html3 or html2 or html).strip().replace("\n", " ")[:220]
        raise ValueError(
            "抓取失敗：回應不含表格（"
            f"endpoint={url}, fallback={fallback_url}, preview={preview!r}）"
        )

    if fallback_url and not allow_legacy_fallback:
        preview = (html2 or html).strip().replace("\n", " ")[:220]
        raise ValueError(
            "抓取失敗：data 端點回應不含表格，且 strict 模式禁用 legacy fallback（"
            f"endpoint={url}, fallback={fallback_url}, preview={preview!r}）。"
            "若要允許 fallback，請加參數 --allow-legacy-fallback"
        )

    preview = (html2 or html).strip().replace("\n", " ")[:220]
    raise ValueError(f"抓取失敗：回應不含表格（endpoint={url}, preview={preview!r}）")


def select_main_table(html: str) -> pd.DataFrame:
    tables = pd.read_html(html)
    candidates = []
    for idx, t in enumerate(tables):
        cols = flatten_columns(t.columns)
        joined = " | ".join(cols)
        if ("期別" in joined or "交易 日期" in joined) and len(t) >= 20:
            candidates.append((idx, t))
    if not candidates:
        raise ValueError("找不到主資料表（期別/交易日期）")
    table = candidates[-1][1].copy()
    table.columns = flatten_columns(table.columns)
    return table


def pick_col(df: pd.DataFrame, must_have: List[str], must_not_have: List[str] = None) -> str:
    must_not_have = must_not_have or []
    for c in df.columns:
        if all(k in c for k in must_have) and all(k not in c for k in must_not_have):
            return c
    raise KeyError(f"找不到欄位: include={must_have}, exclude={must_not_have}")


def pick_col_any(df: pd.DataFrame, options: List[List[str]], must_not_have: List[str] = None) -> str:
    last_err = None
    for must_have in options:
        try:
            return pick_col(df, must_have, must_not_have=must_not_have)
        except KeyError as e:
            last_err = e
    raise last_err if last_err else KeyError("找不到欄位")


def build_feature_frame(k_df: pd.DataFrame, b_df: pd.DataFrame, m_df: pd.DataFrame) -> pd.DataFrame:
    k_date = pick_col(k_df, ["交易 日期"]) if any("交易 日期" in c for c in k_df.columns) else pick_col(k_df, ["期別"])
    k_chg_pct = pick_col(k_df, ["漲跌 (%)"])
    k_amp_pct = pick_col(k_df, ["振幅"])
    k_vol = pick_col(k_df, ["成交", "億元"])

    b_date = pick_col(b_df, ["期別"])
    b_inst_net = pick_col(b_df, ["三大法人", "買賣超"])

    m_date = pick_col(m_df, ["期別"])
    m_mgn_chg = pick_col(m_df, ["融資", "增減"], must_not_have=["(%)"])
    m_short_chg_pct = pick_col_any(
        m_df,
        [
            ["融券", "增減 (%)"],
            ["融券", "增減"],
        ],
    )

    out_k = pd.DataFrame(
        {
            "date": k_df[k_date].map(parse_date_yy_mm_dd),
            "chg_pct": to_num(k_df[k_chg_pct]),
            "amp_pct": to_num(k_df[k_amp_pct]),
            "volume_b": to_num(k_df[k_vol]),
        }
    )
    out_b = pd.DataFrame(
        {
            "date": b_df[b_date].map(parse_date_yy_mm_dd),
            "inst_net_100m": to_num(b_df[b_inst_net]),
        }
    )
    out_m = pd.DataFrame(
        {
            "date": m_df[m_date].map(parse_date_yy_mm_dd),
            "mgn_chg_100m": to_num(m_df[m_mgn_chg]),
            "short_chg_pct": to_num(m_df[m_short_chg_pct]),
        }
    )

    merged = (
        out_k.dropna(subset=["date"])
        .merge(out_b.dropna(subset=["date"]), on="date", how="left")
        .merge(out_m.dropna(subset=["date"]), on="date", how="left")
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    return merged


def add_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["p_chg"] = out["chg_pct"].rank(pct=True)
    out["p_amp"] = out["amp_pct"].rank(pct=True)
    out["p_vol"] = out["volume_b"].rank(pct=True)
    out["p_inst"] = out["inst_net_100m"].rank(pct=True)
    out["p_mgn"] = out["mgn_chg_100m"].rank(pct=True)
    out["p_short"] = out["short_chg_pct"].rank(pct=True)

    out["big_drop"] = out["p_chg"] <= 0.10
    out["high_amp"] = out["p_amp"] >= 0.90
    out["high_vol"] = out["p_vol"] >= 0.80
    out["inst_sell"] = out["p_inst"] <= 0.10
    out["mgn_reduce"] = out["p_mgn"] <= 0.20
    out["short_reduce"] = out["p_short"] <= 0.20
    out["score"] = (
        out["big_drop"].astype(int)
        + out["high_amp"].astype(int)
        + out["high_vol"].astype(int)
        + out["inst_sell"].astype(int)
        + out["mgn_reduce"].astype(int)
        + out["short_reduce"].astype(int)
    )
    return out
