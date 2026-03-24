"""
讀取 Yahoo 台股市「加權指數 ^TWII」頁面內嵌即時欄位。
來源頁面：https://tw.stock.yahoo.com/quote/%5ETWII
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

TWII_QUOTE_URL = "https://tw.stock.yahoo.com/quote/%5ETWII"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_DATE_RE = re.compile(r'"time"\s*:\s*"(\d{4}-\d{2}-\d{2})"')
_PRICE_RE = re.compile(r'"price"\s*:\s*\{\s*"raw"\s*:\s*"(\d+(?:\.\d+)?)"')
_OPEN_RE = re.compile(r'"open"\s*:\s*\{\s*"raw"\s*:\s*"(\d+(?:\.\d+)?)"')
_HIGH_RE = re.compile(r'"dayHigh"\s*:\s*\{\s*"raw"\s*:\s*"(\d+(?:\.\d+)?)"')
_LOW_RE = re.compile(r'"dayLow"\s*:\s*\{\s*"raw"\s*:\s*"(\d+(?:\.\d+)?)"')


def _parse_num(regex: re.Pattern[str], text: str) -> float | None:
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def fetch_twii_quote_ohlc(timeout: float = 12.0) -> tuple[dict[str, float | str] | None, str | None]:
    """
    回傳 ({日期, 開盤價, 最高價, 最低價, 收盤價}, 錯誤訊息)。
    成功時錯誤訊息為 None。
    """
    req = urllib.request.Request(TWII_QUOTE_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, str(e)

    m_date = _DATE_RE.search(raw)
    px = _parse_num(_PRICE_RE, raw)
    op = _parse_num(_OPEN_RE, raw)
    hi = _parse_num(_HIGH_RE, raw)
    lo = _parse_num(_LOW_RE, raw)
    if not m_date or px is None or op is None or hi is None or lo is None:
        return None, "頁面中找不到即時 OHLC 欄位（Yahoo 版面可能已改）"

    return (
        {
            "日期": m_date.group(1),
            "開盤價": op,
            "最高價": hi,
            "最低價": lo,
            "收盤價": px,
        },
        None,
    )
