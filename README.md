# 做多段日線鑽取（本機）

以與 `tw_index_futur/fetch_daily_stock_data_W.py` **相同的週界（週四～下週三）**與 **進場價＝做多第一週收盤**，在網頁上：

- 列出每段做多區間
- **週線**：顯示區間 OHLC（不再計算「可能跌破週」標記）
- **日線**：鑽取首次盤中跌破、區間最低、首次收盤低於進場等
- 並統計整體策略在做多區段中「盤中/收盤跌破」機率（跌破判定從進場收盤後開始）

## 環境

**Monorepo（`Stk_Ops` 內含 `long_underwater_web` 與上層 `tw_index_futur`）**

```bash
cd /path/to/Stk_Ops
pip install -r long_underwater_web/requirements.txt
```

**獨立 clone（本 repo 根目錄已含 `tw_index_futur`）**

```bash
cd /path/to/long_underwater_web
pip install -r requirements.txt
```

## 啟動（localhost）

**Monorepo**

```bash
cd /path/to/Stk_Ops
streamlit run long_underwater_web/app.py
```

**獨立 clone**

```bash
cd /path/to/long_underwater_web
streamlit run app.py
```

瀏覽器開啟後會**自動載入**側欄代號與起始日（預設 `^TWII`）；做多段下拉選單預設為**最後一筆**。變更代號或日期後會重新抓取（約 10 分鐘內有快取）。

僅供本機檢視，無部署設定。

## 第二週「收不回進場價」與盤中最深跌幅統計

```bash
# Monorepo
python3 long_underwater_web/backtest_second_week_drawdown.py ^TWII 2020-01-01
# 獨立 clone（於 repo 根目錄）
python3 backtest_second_week_drawdown.py ^TWII 2020-01-01
```

進場後**第二個週界週**，若該週最後收盤仍低於進場價（定義 A），則統計該週盤中最低相對進場價之跌幅%。
