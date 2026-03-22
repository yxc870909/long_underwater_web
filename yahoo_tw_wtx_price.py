"""
讀取 Yahoo 台股市「台指期近一」WTX& 頁面嵌入之即時價（與網頁
https://tw.stock.yahoo.com/quote/WTX& 同源 JSON 片段）。
僅供本機參考，頁面結構變更時解析可能失效。
"""
from __future__ import annotations

import re
import urllib.error
import urllib.request

WTX_QUOTE_URL = "https://tw.stock.yahoo.com/quote/WTX%26"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
# 頁面內嵌： "price":{"raw":"32737","fmt":"32,737.00",...}
_PRICE_RAW_RE = re.compile(r'"price"\s*:\s*\{\s*"raw"\s*:\s*"(\d+(?:\.\d+)?)"')


def fetch_wtx_quote_price(timeout: float = 12.0) -> tuple[float | None, str | None]:
    """
    回傳 (價格, 錯誤訊息)。成功時錯誤為 None。
    """
    req = urllib.request.Request(WTX_QUOTE_URL, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except urllib.error.URLError as e:
        return None, str(e.reason)
    except Exception as e:
        return None, str(e)

    m = _PRICE_RAW_RE.search(raw)
    if not m:
        return None, "頁面中找不到 price.raw（Yahoo 版面可能已改）"
    try:
        return float(m.group(1)), None
    except ValueError:
        return None, "價格格式無法解析"
