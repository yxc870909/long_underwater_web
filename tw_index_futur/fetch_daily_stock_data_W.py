"""
使用 yfinance 抓取台股加權指數日線資料，並轉換為周線資料
從 2020-01-01 開始，以每周三作為一周的結束（周四~下周三作為一周的單位）
輸出包含周結束日期、最高價、最低價、開盤價、收盤價、漲幅、漲跌點
"""

import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import json
import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from urllib import request as urllib_request
from urllib import error as urllib_error

# 設定 pandas 顯示選項，讓輸出對齊
pd.set_option('display.unicode.east_asian_width', True)  # 正確計算中文字符寬度
pd.set_option('display.max_columns', None)  # 顯示所有欄位
pd.set_option('display.width', None)  # 不限制顯示寬度
pd.set_option('display.max_colwidth', None)  # 不限制欄位寬度


def convert_to_weekly_data(df_daily: pd.DataFrame, return_tagged_daily: bool = False):
    """
    將日線資料轉換為周線資料（周四到下周三為一周）
    
    Parameters:
    -----------
    df_daily : pd.DataFrame
        日線資料，包含日期、開盤價、最高價、最低價、收盤價
    
    Returns:
    --------
    pd.DataFrame
        周線資料，每行代表一周（周四到下周三）
    """
    df = df_daily.copy()
    
    # 確保日期欄位為 datetime 類型
    df['日期'] = pd.to_datetime(df['日期'])
    
    # 日線依日期去重（保留最後一筆），避免 yfinance 回傳同一天兩筆造成周線重複
    df = df.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)
    
    # 計算星期幾（0=Monday, 2=Wednesday, 3=Thursday）
    df['星期'] = df['日期'].dt.dayofweek
    
    # 創建周標識：周四到下周三為一周
    # 如果遇到週四，開始新的一周
    # 如果週四是假日，則使用該周第一個交易日作為新周的開始
    df['周標識'] = 0
    week_num = 0
    prev_date = None
    
    for i in range(len(df)):
        curr_date = df.loc[i, '日期']
        curr_day = df.loc[i, '星期']
        
        # 如果是週四，開始新的一周
        if curr_day == 3:  # 周四
            week_num += 1
        # 如果與前一交易日相隔超過 7 天（例如春節長假），強制開始新的一周，避免整週被併入前一週
        elif prev_date is not None:
            days_diff = (curr_date - prev_date).days
            prev_day = df.loc[i-1, '星期']
            if days_diff > 7:
                week_num += 1
            # 如果前一個交易日是週三或更早，且當前是週四之後（週五、週一、週二），也開始新的一周
            # 這處理了週四是假日的情況
            elif days_diff > 1 and prev_day <= 2 and curr_day >= 4:
                week_num += 1
            # 如果上一個交易日是週三，且當前是週四之後（跳過了週四）
            elif prev_day == 2 and curr_day >= 4:
                week_num += 1
        
        df.loc[i, '周標識'] = week_num
        prev_date = curr_date
    
    # 與日線 merge 用：每個周標識對應的週結束日（必須在週線 drop_duplicates 前記錄，否則會丟周標識導致尾端日線 週結束日=NaN）
    week_id_to_week_end: dict = {}

    # 按周分組聚合
    weekly_data = []

    for week_id in sorted(df['周標識'].unique()):
        week_df = df[df['周標識'] == week_id].copy()
        
        if len(week_df) == 0:
            continue
        
        # 開盤價：周四的開盤價
        thursday_data = week_df[week_df['星期'] == 3]
        if len(thursday_data) > 0:
            open_price = thursday_data.iloc[0]['開盤價']
        else:
            # 如果沒有周四資料，使用該周第一天的開盤價
            open_price = week_df.iloc[0]['開盤價']
        
        # 最高價：周內最高價
        high_price = week_df['最高價'].max()
        
        # 最低價：周內最低價
        low_price = week_df['最低價'].min()
        
        # 第一根～第五根：當週第 1～5 個交易日的收盤價（不足則 NaN）
        bar_names = ['第一根', '第二根', '第三根', '第四根', '第五根']
        bars = {}
        for k, name in enumerate(bar_names, start=1):
            bars[name] = week_df.iloc[k - 1]['收盤價'] if len(week_df) >= k else np.nan
        first_day_close = bars['第一根']
        
        # 收盤價：周三的收盤價（如果沒有周三，使用該周最後一天的收盤價）
        wednesday_data = week_df[week_df['星期'] == 2]
        if len(wednesday_data) > 0:
            close_price = wednesday_data.iloc[-1]['收盤價']  # 取最後一個周三
            week_end_date = wednesday_data.iloc[-1]['日期']
        else:
            # 如果沒有周三資料（假日），使用該周最後一個交易日的收盤價
            # 但日期要計算為該周應該的週三日期
            close_price = week_df.iloc[-1]['收盤價']
            last_trading_date = week_df.iloc[-1]['日期']
            last_day_of_week = last_trading_date.dayofweek  # 0=週一, 1=週二, 2=週三, 3=週四, 4=週五
            
            # 計算該周應該的週三日期（周四~下周三為一周，週三為周結束日）
            if last_day_of_week < 2:  # 週一(0)或週二(1)：本週三尚未到，往後算
                days_to_wednesday = 2 - last_day_of_week
                week_end_date = last_trading_date + pd.Timedelta(days=days_to_wednesday)
            elif last_day_of_week == 2:  # 週三（不應該發生，因為已經檢查過沒有週三資料）
                week_end_date = last_trading_date
            else:  # 週四(3)或週五(4)：本週三尚未到，應為「下週三」（當前這周的結束日）
                # 週四 +6 天 = 下週三，週五 +5 天 = 下週三
                days_to_next_wednesday = 2 - last_day_of_week + 7  # 3->6, 4->5
                week_end_date = last_trading_date + pd.Timedelta(days=days_to_next_wednesday)
        
        week_id_to_week_end[week_id] = week_end_date

        row = {
            '日期': week_end_date,
            '開盤價': open_price,
            **bars,
            '最高價': high_price,
            '最低價': low_price,
            '收盤價': close_price,
            '周標識': week_id
        }
        weekly_data.append(row)

    weekly_df = pd.DataFrame(weekly_data)
    weekly_df = weekly_df.sort_values('日期').reset_index(drop=True)

    # 周線依日期去重（保留最後一筆），避免「無週三」推算的週三與下一週實際週三同一天造成兩筆同日期
    weekly_df = weekly_df.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)

    if return_tagged_daily:
        wmap = pd.DataFrame(
            [
                {"周標識": wid, "週結束日": week_id_to_week_end[wid]}
                for wid in sorted(week_id_to_week_end.keys())
            ]
        )
        return weekly_df, df, wmap

    return weekly_df


def fetch_weekly_stock_data(ticker: str, start_date: str = "2020-01-01") -> pd.DataFrame:
    """
    抓取股票日線資料並轉換為周線資料（周四到下周三為一周）
    
    Parameters:
    -----------
    ticker : str
        股票代號（例如：'^TWII' 為台股加權指數）
    start_date : str
        開始日期，格式為 'YYYY-MM-DD'
    
    Returns:
    --------
    pd.DataFrame
        包含以下欄位的周線 DataFrame：
        - 日期（周三，作為該周的結束日期）
        - 最高價（周內最高）
        - 最低價（周內最低）
        - 開盤價（周四開盤）
        - 收盤價（周三收盤）
        - 漲幅（%）
        - 漲跌點
    """
    # 下載股票日線資料
    stock = yf.Ticker(ticker)
    df_daily = stock.history(start=start_date)
    
    if df_daily.empty:
        raise ValueError(f"無法取得 {ticker} 的資料，請檢查股票代號是否正確")
    
    # 重置索引，將日期從索引轉為欄位
    df_daily.reset_index(inplace=True)
    
    # 重新命名欄位為中文
    df_daily.rename(columns={
        'Date': '日期',
        'High': '最高價',
        'Low': '最低價',
        'Open': '開盤價',
        'Close': '收盤價'
    }, inplace=True)
    
    # 日線依日期去重（保留最後一筆），避免 yfinance 回傳同一天兩筆
    df_daily = df_daily.drop_duplicates(subset=['日期'], keep='last').sort_values('日期').reset_index(drop=True)
    
    # 轉換為周線資料
    df = convert_to_weekly_data(df_daily)
    
    # 計算漲跌點（本周收盤價 - 上周收盤價）
    df['漲跌點'] = df['收盤價'].diff()
    
    # 計算漲幅（%）（(本周收盤價 - 上周收盤價) / 上周收盤價 * 100）
    df['漲幅'] = (df['收盤價'].pct_change() * 100).round(2)
    
    # 計算前一周收盤價
    df['前一周收盤價'] = df['收盤價'].shift(1)
    
    # 計算波動下限：前一周收盤價 * 0.02，以100為單位取整數（四捨五入）
    df['波動下限'] = (df['前一周收盤價'] * 0.02).apply(lambda x: np.round(x / 100) * 100 if pd.notna(x) else np.nan)
    
    # 計算波動上限：前一周收盤價 * 0.04，以50為單位取整數（四捨五入）
    df['波動上限'] = (df['前一周收盤價'] * 0.04).apply(lambda x: np.round(x / 50) * 50 if pd.notna(x) else np.nan)
    
    # 計算近N周總漲點（累積漲點，每周不同）
    # 這裡改為計算近4周的總漲點（約一個月）
    df['近四周總漲點'] = df['漲跌點'].rolling(window=4, min_periods=1).sum().round(2)
    
    # 計算兩周漲差：本周的「近四周總漲點」- 上周的「近四周總漲點」，取絕對值
    df['上周近四周總漲點'] = df['近四周總漲點'].shift(1)
    df['兩周漲差'] = (df['近四周總漲點'] - df['上周近四周總漲點']).abs().round(2)
    
    # 移除臨時的欄位
    df = df.drop(columns=['周標識', '上周近四周總漲點'])
    
    # 計算20周均線（以收盤價計算）
    df['20周均線'] = df['收盤價'].rolling(window=20, min_periods=1).mean().round(2)
    
    # 初始化反彈目標、出場價格、做多煞車和交易狀態欄位
    df['反彈目標'] = np.nan
    df['出場價格'] = np.nan
    df['做多煞車'] = pd.Series([np.nan] * len(df), dtype=object)  # 等待突破狀態下為 NaN，使用 object 類型以支援字串
    df['交易狀態'] = '等待突破'  # 初始狀態為等待突破
    df['做多累計漲幅'] = 0.0  # 從做多開始的累計漲幅（臨時欄位）
    
    def _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff):
        """反彈目標計算：prev_close + add_value，add_value 依 sum(當周+前周漲跌點) 與波動下限、兩周漲差決定。各處轉為等待突破時重算反彈目標皆用此公式。"""
        sum_changes = curr_change + prev_change
        if sum_changes > curr_vol_lower:
            week_diff_rounded = np.round(curr_week_diff / 100) * 100
            add_value = max(week_diff_rounded, curr_vol_lower)
        else:
            add_value = curr_vol_lower
        return prev_close + add_value
    
    # 狀態機計算（需要逐行計算，因為依賴前一行）
    for i in range(len(df)):
        if i == 0:
            # 第一筆資料無法計算（沒有前一周資料）
            continue
        
        # 取得當周和前一周的資料
        prev_close = df.loc[i-1, '收盤價']
        curr_close = df.loc[i, '收盤價']
        prev_rebound = df.loc[i-1, '反彈目標']
        prev_exit_price = df.loc[i-1, '出場價格']
        prev_state = df.loc[i-1, '交易狀態']
        prev_long_total = df.loc[i-1, '做多累計漲幅']  # 從做多開始的累計漲幅
        prev_brake = df.loc[i-1, '做多煞車']  # 前一周的做多煞車
        curr_change = df.loc[i, '漲跌點']
        prev_change = df.loc[i-1, '漲跌點']
        curr_vol_lower = df.loc[i, '波動下限']
        curr_vol_upper = df.loc[i, '波動上限']
        curr_week_total = df.loc[i, '近四周總漲點']
        curr_week_diff = df.loc[i, '兩周漲差']
        
        # 狀態機邏輯
        if prev_state == '等待突破':
            # ========== 狀態1：等待突破 ==========
            # 第一根起依序達標：當週依序檢查第一～第五根，最先達到上周反彈目標且 > 20周均線的那根即觸發做多中
            curr_ma20 = df.loc[i, '20周均線']
            trigger_price = None
            if pd.notna(prev_rebound) and pd.notna(curr_ma20):
                for col in ['第一根', '第二根', '第三根', '第四根', '第五根']:
                    if col not in df.columns:
                        continue
                    val = df.loc[i, col]
                    if pd.notna(val) and val >= prev_rebound and val > curr_ma20:
                        trigger_price = float(val)
                        break
            if trigger_price is not None:
                # 狀態轉換：等待突破 → 做多中
                df.loc[i, '交易狀態'] = '做多中'
                
                # 計算出場價格（第一次計算）
                if curr_week_total <= (curr_vol_upper * -1):
                    subtract_value = curr_vol_upper
                else:
                    subtract_value = curr_vol_lower
                
                df.loc[i, '出場價格'] = prev_close - subtract_value
                df.loc[i, '反彈目標'] = np.nan
                
                # 檢查進場當周是否已觸及出場價格（無效進場）：以觸發價與出場價格比較
                if trigger_price <= df.loc[i, '出場價格']:
                    # 視為無效進場，轉回等待突破狀態
                    df.loc[i, '交易狀態'] = '等待突破'
                    df.loc[i, '出場價格'] = np.nan
                    df.loc[i, '做多累計漲幅'] = 0.0
                    df.loc[i, '做多煞車'] = np.nan
                    # 重新計算反彈目標（與 _calc_rebound_target 公式一致）
                    df.loc[i, '反彈目標'] = _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff)
                else:
                    # 有效進場：僅在未因無效進場改回時才設定做多累計漲幅與做多煞車，避免等待突破列被覆寫
                    df.loc[i, '做多累計漲幅'] = curr_change  # 從做多開始的累計漲幅
                    df.loc[i, '做多煞車'] = 'F'  # 初始值為 F
                    
                    # 當週收盤未達上週反彈目標則當週改回等待突破（突破後未站穩視為無效）
                    if pd.notna(prev_rebound) and curr_close < prev_rebound:
                        df.loc[i, '交易狀態'] = '等待突破'
                        df.loc[i, '出場價格'] = np.nan
                        df.loc[i, '做多累計漲幅'] = 0.0
                        df.loc[i, '做多煞車'] = np.nan
                        df.loc[i, '反彈目標'] = _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff)
            else:
                # 繼續等待突破狀態
                df.loc[i, '交易狀態'] = '等待突破'
                df.loc[i, '出場價格'] = np.nan
                df.loc[i, '做多累計漲幅'] = 0.0  # 重置累計漲幅
                df.loc[i, '做多煞車'] = np.nan  # 等待突破狀態下為 NaN
                
                # 計算反彈目標（第一次與後續計算皆用 _calc_rebound_target 公式）
                df.loc[i, '反彈目標'] = _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff)
        
        elif prev_state == '做多中':
            # ========== 狀態2：做多中 ==========
            
            # 檢查止損條件：收盤價回落至出場價格以下
            # 使用出場價格作為止損參考
            if pd.notna(prev_exit_price):
                # 檢查是否回落至出場價格以下
                if curr_close <= prev_exit_price:
                    # 狀態轉換：做多中 → 等待突破（觸發出場價格）
                    df.loc[i, '交易狀態'] = '等待突破'
                    df.loc[i, '出場價格'] = np.nan
                    df.loc[i, '做多累計漲幅'] = 0.0  # 重置累計漲幅
                    df.loc[i, '做多煞車'] = np.nan  # 回到等待突破狀態，設為 NaN
                    df.loc[i, '反彈目標'] = _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff)
                else:
                    # 繼續做多狀態
                    # 檢查前一周的做多煞車是否為 T，如果是則在下一個交易周轉換為等待突破狀態
                    if prev_brake == 'T':
                        # 狀態轉換：做多中 → 等待突破（前一周做多煞車為 T，下一個交易周轉換）
                        df.loc[i, '交易狀態'] = '等待突破'
                        df.loc[i, '出場價格'] = np.nan
                        df.loc[i, '做多累計漲幅'] = 0.0  # 重置累計漲幅
                        df.loc[i, '做多煞車'] = np.nan  # 回到等待突破狀態，設為 NaN
                        df.loc[i, '反彈目標'] = _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff)
                    else:
                        # 繼續做多狀態
                        df.loc[i, '交易狀態'] = '做多中'
                        df.loc[i, '反彈目標'] = np.nan  # 確保反彈目標為 NaN，不能與出場價格同時有值
                        
                        # 更新做多累計漲幅（從做多開始累積）
                        df.loc[i, '做多累計漲幅'] = prev_long_total + curr_change
                        curr_long_total = df.loc[i, '做多累計漲幅']  # 當周的累計漲點
                        
                        # 計算做多煞車：當周的漲跌點 / 累計漲點(從做多開始累計的漲跌點) <= -0.2 則為 T
                        # 當周的漲跌點 = curr_change
                        # 累計漲點 = curr_long_total（從做多開始累計的漲跌點）
                        if curr_long_total != 0 and abs(curr_long_total) > 0.01:  # 避免除零
                            ratio = curr_change / curr_long_total
                            if ratio <= -0.2:
                                df.loc[i, '做多煞車'] = 'T'
                            else:
                                df.loc[i, '做多煞車'] = 'F'
                        else:
                            df.loc[i, '做多煞車'] = 'F'  # 如果累計漲點為0或接近0，設為 F
                        
                        # 計算出場價格（後續更新）
                        # 如果前一周收盤價 > 前一周出場價格 且 做多煞車 <> T：
                        #     計算新的出場價格：前一周收盤價 - IF(前一周近四周總漲點 <= (波動上限*-1), 波動上限, 波動下限)
                        # 否則：保持前一周的出場價格（如果前一周為空則為 NaN）
                        prev_week_total = df.loc[i-1, '近四周總漲點']
                        prev_vol_upper = df.loc[i-1, '波動上限']
                        prev_vol_lower = df.loc[i-1, '波動下限']
                        
                        # 檢查條件：前一周收盤價 > 前一周出場價格 且 做多煞車 <> T
                        condition1 = pd.notna(prev_exit_price) and prev_close > prev_exit_price
                        condition2 = prev_brake != 'T'  # 做多煞車 <> T
                        
                        if condition1 and condition2:
                            # 計算新的出場價格：前一周收盤價 - IF(前一周近四周總漲點 <= (波動上限*-1), 波動上限, 波動下限)
                            if prev_week_total <= (prev_vol_upper * -1):
                                subtract_value = prev_vol_upper
                            else:
                                subtract_value = prev_vol_lower
                            
                            df.loc[i, '出場價格'] = prev_close - subtract_value
                        else:
                            # 條件不滿足，保持前一周的出場價格（如果前一周為空則為 NaN）
                            if pd.notna(prev_exit_price):
                                df.loc[i, '出場價格'] = prev_exit_price  # 保持前一周的出場價格
                            else:
                                df.loc[i, '出場價格'] = np.nan  # 前一周為空，則為空值
            else:
                # 如果出場價格為空（異常情況），回到等待突破狀態
                df.loc[i, '交易狀態'] = '等待突破'
                df.loc[i, '出場價格'] = np.nan
                df.loc[i, '做多累計漲幅'] = 0.0  # 重置累計漲幅
                df.loc[i, '做多煞車'] = np.nan  # 回到等待突破狀態，設為 NaN
                
                df.loc[i, '反彈目標'] = _calc_rebound_target(prev_close, curr_change, prev_change, curr_vol_lower, curr_week_diff)
    
    # 後處理：當週為等待突破、上週為做多中且做多煞車=F 時，反推當週是否應觸發做多煞車，若是則將當週做多煞車標記為 T（獲利已在 calculate_long_position_stats 中以煞車價計算）
    for i in range(1, len(df)):
        curr_state = df.loc[i, '交易狀態']
        prev_state = df.loc[i - 1, '交易狀態']
        prev_brake = df.loc[i - 1, '做多煞車']
        if curr_state != '等待突破' or prev_state != '做多中' or prev_brake != 'F':
            continue
        prev_long_total = df.loc[i - 1, '做多累計漲幅']
        curr_change = df.loc[i, '漲跌點']
        if pd.isna(prev_long_total):
            prev_long_total = 0.0
        if pd.isna(curr_change):
            curr_change = 0.0
        prev_long_total = float(prev_long_total)
        curr_change = float(curr_change)
        sum_long = prev_long_total + curr_change
        if abs(sum_long) > 0.01 and (curr_change / sum_long <= -0.2):
            df.loc[i, '做多煞車'] = 'T'
    
    # 將日期轉換為只顯示年月日（YYYY-MM-DD 格式）
    df['日期'] = df['日期'].dt.strftime('%Y-%m-%d')
    
    # 選取需要的欄位，並按照指定順序排列（保留做多累計漲幅供 calculate_long_position_stats 反推做多煞車用）
    out_cols = ['日期', '最高價', '最低價', '開盤價', '第一根', '第二根', '第三根', '第四根', '第五根', '收盤價', '漲幅', '漲跌點', '波動下限', '波動上限', '近四周總漲點', '兩周漲差', '20周均線', '做多煞車', '做多累計漲幅', '交易狀態', '反彈目標', '出場價格']
    result_df = df[[c for c in out_cols if c in df.columns]].copy()
    
    # 將數值欄位四捨五入到小數點後2位
    numeric_cols = ['最高價', '最低價', '開盤價', '第一根', '第二根', '第三根', '第四根', '第五根', '收盤價', '漲跌點']
    numeric_cols = [c for c in numeric_cols if c in result_df.columns]
    result_df[numeric_cols] = result_df[numeric_cols].round(2)
    
    # 波動下限和波動上限已經是整數，但確保為整數格式
    result_df['波動下限'] = result_df['波動下限'].astype('Int64')  # 使用可空整數類型
    result_df['波動上限'] = result_df['波動上限'].astype('Int64')
    
    # 移除第一筆資料（因為漲幅和漲跌點無法計算）
    result_df = result_df.iloc[1:].reset_index(drop=True)
    
    return result_df


def get_stock_name(ticker: str) -> str:
    """
    獲取股票中文名稱
    
    Parameters:
    -----------
    ticker : str
        股票代號
    
    Returns:
    --------
    str
        股票中文名稱，如果獲取失敗則返回股票代號
    """
    # 台股加權指數對照表
    INDEX_NAMES = {
        '^TWII': '台股加權指數',
    }
    
    # 先檢查對照表
    if ticker in INDEX_NAMES:
        return INDEX_NAMES[ticker]
    
    # 嘗試從 yfinance 獲取
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        name = info.get('longName') or info.get('shortName') or info.get('name')
        
        if name and name != ticker:
            import re
            if re.search(r'[\u4e00-\u9fff]', name):
                return name
        
        return ticker
    except Exception as e:
        return ticker


def check_entry_signal(ticker: str, start_date: str = "2020-01-01") -> dict:
    """
    檢查指定股票是否為「做多中」狀態的第一周進場點
    
    Parameters:
    -----------
    ticker : str
        股票代號
    start_date : str
        開始日期
    
    Returns:
    --------
    dict
        包含進場信號信息的字典
    """
    try:
        # 獲取股票名稱
        stock_name = get_stock_name(ticker)
        
        df = fetch_weekly_stock_data(ticker, start_date=start_date)
        
        if len(df) == 0:
            return {
                '股票代號': ticker,
                '股票名稱': stock_name,
                '是否為進場點': False,
                '錯誤': '無數據'
            }
        
        # 獲取最新交易周（最後一行）
        latest_idx = len(df) - 1
        latest_row = df.iloc[latest_idx]
        prev_row = df.iloc[latest_idx - 1] if latest_idx > 0 else None
        
        latest_state = latest_row['交易狀態']
        latest_date = latest_row['日期']
        latest_close = latest_row['收盤價']
        latest_ma20 = latest_row['20周均線'] if '20周均線' in latest_row else None
        
        # 判斷是否為進場點
        is_entry = False
        entry_info = {}
        
        if latest_state == '做多中' and prev_row is not None:
            prev_state = prev_row['交易狀態']
            if prev_state == '等待突破':
                # 檢查進場當周是否已觸及出場價格（無效進場）
                latest_exit_price = latest_row['出場價格'] if pd.notna(latest_row['出場價格']) else None
                if latest_exit_price is not None and latest_close <= latest_exit_price:
                    # 進場當周已觸及出場價格，視為無效進場，不顯示進場信號
                    is_entry = False
                else:
                    # 這是有效進場點！
                    is_entry = True
                    entry_info = {
                        '進場日期': latest_date,
                        '進場價格': round(latest_close, 2),
                        '出場價格': round(latest_exit_price, 2) if latest_exit_price is not None else None,
                        '20周均線': round(latest_ma20, 2) if latest_ma20 is not None else None,
                        '前一周狀態': prev_state,
                        '前一周收盤價': round(prev_row['收盤價'], 2),
                        '前一周反彈目標': round(prev_row['反彈目標'], 2) if pd.notna(prev_row['反彈目標']) else None,
                    }
        
        result = {
            '股票代號': ticker,
            '股票名稱': stock_name,
            '是否為進場點': is_entry,
            '最新交易周': latest_date,
            '最新狀態': latest_state,
            '最新收盤價': round(latest_close, 2),
            '20周均線': round(latest_ma20, 2) if latest_ma20 is not None else None,
            '錯誤': None
        }
        
        if is_entry:
            result.update(entry_info)
        elif latest_state == '做多中':
            # 在做多中，但不是第一周
            result['做多中周數'] = None
            result['當前出場價格'] = round(latest_row['出場價格'], 2) if pd.notna(latest_row['出場價格']) else None
            result['做多煞車'] = latest_row['做多煞車'] if pd.notna(latest_row['做多煞車']) else None
        elif latest_state == '等待突破':
            # 進場信號檢查的反彈目標顯示為上一周的反彈目標（突破目標價）
            display_rebound = prev_row['反彈目標'] if prev_row is not None and pd.notna(prev_row.get('反彈目標')) else latest_row['反彈目標']
            result['反彈目標'] = round(display_rebound, 2) if pd.notna(display_rebound) else None
            result['距離反彈目標'] = round(display_rebound - latest_close, 2) if pd.notna(display_rebound) else None
            result['距離反彈目標(%)'] = round((display_rebound - latest_close) / latest_close * 100, 2) if pd.notna(display_rebound) and latest_close > 0 else None
        
        return result
        
    except Exception as e:
        # 即使出錯也嘗試獲取股票名稱
        stock_name = get_stock_name(ticker)
        return {
            '股票代號': ticker,
            '股票名稱': stock_name,
            '是否為進場點': False,
            '錯誤': str(e)
        }


def check_exit_signal(ticker: str, start_date: str = "2020-01-01") -> dict:
    """
    檢查指定股票是否為「做多中」狀態的出場點
    
    Parameters:
    -----------
    ticker : str
        股票代號
    start_date : str
        開始日期
    
    Returns:
    --------
    dict
        包含出場信號信息的字典
    """
    try:
        # 獲取股票名稱
        stock_name = get_stock_name(ticker)
        
        df = fetch_weekly_stock_data(ticker, start_date=start_date)
        
        if len(df) == 0:
            return {
                '股票代號': ticker,
                '股票名稱': stock_name,
                '是否為出場點': False,
                '錯誤': '無數據'
            }
        
        # 獲取最新交易周（最後一行）
        latest_idx = len(df) - 1
        latest_row = df.iloc[latest_idx]
        prev_row = df.iloc[latest_idx - 1] if latest_idx > 0 else None
        
        latest_state = latest_row['交易狀態']
        latest_date = latest_row['日期']
        latest_close = latest_row['收盤價']
        latest_exit_price = latest_row['出場價格'] if pd.notna(latest_row['出場價格']) else None
        latest_brake = latest_row['做多煞車'] if pd.notna(latest_row['做多煞車']) else None
        
        # 判斷是否為出場點
        is_exit = False
        exit_reason = None
        exit_info = {}
        
        if latest_state == '做多中' and latest_exit_price is not None:
            # 檢查是否回落至出場價格以下（出場條件）
            if latest_close <= latest_exit_price:
                is_exit = True
                exit_reason = '觸發出場價格'
                exit_info = {
                    '出場日期': latest_date,
                    '出場價格': round(latest_close, 2),
                    '設定出場價格': round(latest_exit_price, 2),
                    '說明': '收盤價回落至出場價格以下',
                }
            # 檢查做多煞車是否為 T（下一個交易周會轉為等待突破）
            elif latest_brake == 'T':
                is_exit = True
                exit_reason = '做多煞車觸發'
                exit_info = {
                    '出場日期': latest_date,
                    '出場價格': round(latest_close, 2),
                    '設定出場價格': round(latest_exit_price, 2),
                    '做多煞車': 'T',
                    '說明': '下一個交易周將轉為等待突破狀態',
                }
        
        result = {
            '股票代號': ticker,
            '股票名稱': stock_name,
            '是否為出場點': is_exit,
            '最新交易周': latest_date,
            '最新狀態': latest_state,
            '最新收盤價': round(latest_close, 2),
            '錯誤': None
        }
        
        if is_exit:
            result['出場原因'] = exit_reason
            result.update(exit_info)
        elif latest_state == '做多中':
            # 在做多中，但尚未達到出場條件
            # 當前出場價格與距離出場價格：使用「每段做多中」統計中狀態為「進行中」那一筆的退出價格（出場價格），與表格一致
            entry_price = None
            entry_idx = None
            i = latest_idx
            while i >= 0 and df.iloc[i]['交易狀態'] == '做多中':
                i -= 1
            if i >= 0:
                entry_idx = i + 1
                entry_price = float(df.iloc[entry_idx]['收盤價'])
            exit_ratio = 0.03  # TWII 出場 -3%
            long_brake = 0.10   # TWII 做多煞車 回撤 10%
            stats_df = calculate_long_position_stats(df)
            in_progress = stats_df[stats_df['狀態'] == '進行中'] if (stats_df is not None and len(stats_df) > 0 and '狀態' in stats_df.columns) else None
            if in_progress is not None and len(in_progress) > 0:
                in_progress_row = in_progress.iloc[-1]
                exit_price_from_stats = float(in_progress_row['退出價格'])
                result['當前出場價格'] = round(exit_price_from_stats, 2)
                result['距離出場價格'] = round(latest_close - exit_price_from_stats, 2)
                result['距離出場價格(%)'] = round((latest_close - exit_price_from_stats) / latest_close * 100, 2) if latest_close > 0 else None
            elif entry_price is not None and entry_price > 0:
                exit_price_by_entry = round(entry_price * (1 - exit_ratio), 2)
                result['當前出場價格'] = exit_price_by_entry
                result['距離出場價格'] = round(latest_close - exit_price_by_entry, 2)
                result['距離出場價格(%)'] = round((latest_close - exit_price_by_entry) / latest_close * 100, 2) if latest_close > 0 else None
            else:
                result['當前出場價格'] = round(latest_exit_price, 2) if latest_exit_price is not None else None
                if latest_exit_price is not None:
                    result['距離出場價格'] = round(latest_close - latest_exit_price, 2)
                    result['距離出場價格(%)'] = round((latest_close - latest_exit_price) / latest_close * 100, 2) if latest_close > 0 else None
            if entry_price is not None and entry_price > 0 and entry_idx is not None:
                peak_price = float(df.iloc[entry_idx:latest_idx + 1]['收盤價'].max())
                brake_price = round(entry_price + (peak_price - entry_price) * (1 - long_brake), 2)
                result['煞車價格'] = brake_price
                # 距離煞車價格 = 最新收盤價 - 煞車價格（點數）
                result['距離煞車價格'] = round(latest_close - brake_price, 2)
            result['做多煞車'] = latest_brake
        
        return result
        
    except Exception as e:
        # 即使出錯也嘗試獲取股票名稱
        stock_name = get_stock_name(ticker)
        return {
            '股票代號': ticker,
            '股票名稱': stock_name,
            '是否為出場點': False,
            '錯誤': str(e)
        }


def calculate_long_position_stats(df: pd.DataFrame):
    """
    統計每段「做多中」的累計價差。進場價＝做多中第一週收盤價。
    做多煞車為 T 時出場價＝煞車價位（最後一個做多中周的出場價格）；否則若轉為等待突破當週
    反推做多煞車應為 T（當週漲跌點/累計漲點<=-0.2）則出場價＝煞車價位；若當週收盤低於煞車價位亦以煞車價位出場。
    """
    stats = []
    in_long = False
    entry_date = None
    entry_price = None
    entry_idx = None
    last_long_brake = None
    last_long_date = None
    last_long_price = None
    last_long_exit_price = None  # 煞車價位：最後一個做多中周的出場價格
    
    for i in range(len(df)):
        curr_state = df.loc[i, '交易狀態']
        prev_state = df.loc[i-1, '交易狀態'] if i > 0 else None
        curr_close = df.loc[i, '收盤價']
        curr_date = df.loc[i, '日期']
        curr_brake = df.loc[i, '做多煞車']
        curr_exit_price = df.loc[i, '出場價格']
        
        # 進入做多中狀態：進場價＝做多中第一週收盤價
        if curr_state == '做多中' and (prev_state != '做多中' or prev_state is None):
            in_long = True
            entry_date = curr_date
            entry_price = float(curr_close)
            entry_idx = i
            last_long_brake = None
            last_long_date = None
            last_long_price = None
            last_long_exit_price = None
        
        if curr_state == '做多中':
            last_long_brake = curr_brake
            last_long_date = curr_date
            last_long_price = curr_close
            if pd.notna(curr_exit_price):
                last_long_exit_price = float(curr_exit_price)
        
        # 退出做多中狀態（轉為等待突破）
        elif curr_state == '等待突破' and prev_state == '做多中':
            if in_long and entry_date is not None and entry_price is not None:
                # 煞車價＝不含本週的做多中累計價差 -20% 的價位 ＝ 上週收盤 - 0.2*上週做多累計漲幅；先跌破此價則以煞車價出場，除非出場價格更高會先碰到出場價格
                prev_close = float(df.loc[i - 1, '收盤價']) if pd.notna(df.loc[i - 1, '收盤價']) else None
                prev_long_total = float(df.loc[i - 1, '做多累計漲幅']) if '做多累計漲幅' in df.columns and pd.notna(df.loc[i - 1, '做多累計漲幅']) else 0.0
                prev_exit_price = df.loc[i - 1, '出場價格'] if '出場價格' in df.columns else None
                prev_exit_price = float(prev_exit_price) if pd.notna(prev_exit_price) else None
                # 本週若先觸發做多煞車時的煞車價（上週收盤 - 20% 上週做多累計漲幅）
                brake_price_this_week = (prev_close - 0.2 * prev_long_total) if prev_close is not None else None
                # 上週已做多煞車 T 時，煞車價為「上上週收盤 - 0.2*上上週做多累計漲幅」
                if i >= 2 and last_long_brake == 'T':
                    prev2_close = float(df.loc[i - 2, '收盤價']) if pd.notna(df.loc[i - 2, '收盤價']) else None
                    prev2_long = float(df.loc[i - 2, '做多累計漲幅']) if '做多累計漲幅' in df.columns and pd.notna(df.loc[i - 2, '做多累計漲幅']) else 0.0
                    brake_price_prev_week = (prev2_close - 0.2 * prev2_long) if prev2_close is not None else None
                else:
                    brake_price_prev_week = None
                used_brake = False
                if last_long_brake == 'T':
                    used_brake = True
                    exit_date = last_long_date if last_long_date else curr_date
                    brake = brake_price_prev_week if brake_price_prev_week is not None else brake_price_this_week
                    if brake is not None and prev_exit_price is not None:
                        exit_price = max(brake, prev_exit_price)
                    elif brake is not None:
                        exit_price = brake
                    elif prev_exit_price is not None:
                        exit_price = prev_exit_price
                    else:
                        exit_price = last_long_price if last_long_price else curr_close
                elif str(curr_brake).strip() == 'T' and brake_price_this_week is not None:
                    used_brake = True
                    # 當週先觸發做多煞車：退出價＝max(煞車價, 出場價格)，出場價格較高則先碰到
                    exit_date = curr_date
                    exit_price = max(brake_price_this_week, prev_exit_price) if prev_exit_price is not None else brake_price_this_week
                else:
                    # 反推該週做多煞車是否應為 T（疊穿出場價格時先觸發做多煞車則以煞車價位出場）
                    curr_change = float(df.loc[i, '漲跌點']) if '漲跌點' in df.columns and pd.notna(df.loc[i, '漲跌點']) else 0.0
                    sum_long = prev_long_total + curr_change
                    would_brake_t = abs(sum_long) > 0.01 and (curr_change / sum_long <= -0.2)
                    if would_brake_t and brake_price_this_week is not None:
                        used_brake = True
                        exit_date = curr_date
                        exit_price = max(brake_price_this_week, prev_exit_price) if prev_exit_price is not None else brake_price_this_week
                    else:
                        exit_date = curr_date
                        exit_price = curr_close
                        if brake_price_this_week is not None and prev_exit_price is not None and curr_close < max(brake_price_this_week, prev_exit_price):
                            exit_price = max(brake_price_this_week, prev_exit_price)
                        elif brake_price_this_week is not None and curr_close < brake_price_this_week:
                            exit_price = brake_price_this_week
                        elif prev_exit_price is not None and curr_close < prev_exit_price:
                            exit_price = prev_exit_price
                if last_long_brake == 'T' and exit_price is None:
                    exit_price = last_long_price if last_long_price else curr_close
                seg_brake = 'T' if used_brake else (last_long_brake if last_long_brake in ['T', 'F'] else 'N/A')
                
                price_diff = exit_price - entry_price
                price_diff_pct = ((exit_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
                duration = i - entry_idx + 1  # 包含進入和退出的周數
                
                stats.append({
                    '進入日期': entry_date,
                    '退出日期': exit_date,
                    '進入價格': round(entry_price, 2),
                    '退出價格': round(float(exit_price), 2),
                    '累計價差': round(price_diff, 2),
                    '累計漲幅(%)': round(price_diff_pct, 2),
                    '持續周數': duration,
                    '做多煞車': seg_brake
                })
                
                in_long = False
                entry_date = None
                entry_price = None
                entry_idx = None
                last_long_brake = None
                last_long_date = None
                last_long_price = None
                last_long_exit_price = None
    
    # 如果最後還在做多中，退出價格與退出日期使用「前一周」的資料（進行中不顯示當周，避免未收盤失真）
    if in_long and entry_date is not None and entry_price is not None:
        if len(df) >= 2:
            prev_week_row = df.loc[len(df) - 2]
            exit_date = prev_week_row['日期']
            exit_price = round(float(prev_week_row['出場價格']), 2) if pd.notna(prev_week_row['出場價格']) else round(float(prev_week_row['收盤價']), 2)
        else:
            last_row = df.loc[len(df) - 1]
            exit_date = last_row['日期']
            exit_price = round(float(last_row['出場價格']), 2) if pd.notna(last_row['出場價格']) else round(float(last_row['收盤價']), 2)
        
        # 累計價差使用目前的市價（最新收盤價）計算
        latest_price = round(float(df.loc[len(df)-1, '收盤價']), 2)  # 最新市價
        price_diff = latest_price - entry_price  # 最新市價 - 進入價格
        price_diff_pct = ((latest_price - entry_price) / entry_price * 100) if entry_price != 0 else 0
        duration = len(df) - entry_idx
        
        stats.append({
            '進入日期': entry_date,
            '退出日期': exit_date,
            '進入價格': round(entry_price, 2),
            '退出價格': round(exit_price, 2),  # 確保取到小數點第二位
            '累計價差': round(price_diff, 2),
            '累計漲幅(%)': round(price_diff_pct, 2),
            '持續周數': duration,
            '狀態': '進行中',
            '做多煞車': last_long_brake if last_long_brake in ['T', 'F'] else 'N/A'
        })
    
    if stats:
        stats_df = pd.DataFrame(stats)
        return stats_df
    else:
        return pd.DataFrame(columns=['進入日期', '退出日期', '進入價格', '退出價格', '累計價差', '累計漲幅(%)', '持續周數', '做多煞車'])


def calculate_performance_metrics(stats_df: pd.DataFrame) -> dict:
    """
    計算操作績效指標
    
    Parameters:
    -----------
    stats_df : pd.DataFrame
        操作統計 DataFrame，包含 '累計價差' 欄位
    
    Returns:
    --------
    dict
        績效指標字典，包含：
        - 總交易次數
        - 累計價差總和
        - 平均每段累計價差
        - 盈利次數
        - 虧損次數
        - 盈利點數總和
        - 虧損點數總和
        - 淨盈虧
        - 勝率
        - 平均盈利
        - 平均虧損
        - 平均盈虧比
        - 最大單次虧損
        - 風險報酬比
    """
    if len(stats_df) == 0:
        return {
            '總交易次數': 0,
            '累計價差總和': 0.0,
            '平均每段累計價差': 0.0,
            '盈利次數': 0,
            '虧損次數': 0,
            '盈利點數總和': 0.0,
            '虧損點數總和': 0.0,
            '淨盈虧': 0.0,
            '勝率': 0.0,
            '平均盈利': 0.0,
            '平均虧損': 0.0,
            '平均盈虧比': 0.0,
            '最大單次虧損': 0.0,
            '風險報酬比': 0.0
        }
    
    # 基本統計
    total_trades = len(stats_df)
    total_diff = stats_df['累計價差'].sum()
    avg_diff = stats_df['累計價差'].mean()
    
    # 盈虧統計
    profit_trades = stats_df[stats_df['累計價差'] > 0]
    loss_trades = stats_df[stats_df['累計價差'] < 0]
    
    profit_count = len(profit_trades)
    loss_count = len(loss_trades)
    
    profit_points = profit_trades['累計價差'].sum() if len(profit_trades) > 0 else 0
    loss_points = loss_trades['累計價差'].sum() if len(loss_trades) > 0 else 0
    net_profit = profit_points + loss_points
    
    # 勝率
    win_rate = (profit_count / total_trades * 100) if total_trades > 0 else 0
    
    # 平均統計
    avg_profit = profit_trades['累計價差'].mean() if len(profit_trades) > 0 else 0
    avg_loss = abs(loss_trades['累計價差'].mean()) if len(loss_trades) > 0 else 1
    profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
    
    # 最大單次虧損
    max_loss = abs(loss_trades['累計價差'].min()) if len(loss_trades) > 0 else 0
    
    # 風險報酬比
    risk_reward_ratio = net_profit / max_loss if max_loss > 0 else 0
    
    return {
        '總交易次數': total_trades,
        '累計價差總和': round(total_diff, 2),
        '平均每段累計價差': round(avg_diff, 2),
        '盈利次數': profit_count,
        '虧損次數': loss_count,
        '盈利點數總和': round(profit_points, 2),
        '虧損點數總和': round(loss_points, 2),
        '淨盈虧': round(net_profit, 2),
        '勝率': round(win_rate, 2),
        '平均盈利': round(avg_profit, 2),
        '平均虧損': round(avg_loss, 2),
        '平均盈虧比': round(profit_loss_ratio, 2),
        '最大單次虧損': round(max_loss, 2),
        '風險報酬比': round(risk_reward_ratio, 2)
    }


def _run_status(ticker: str, start_date: str, mobile: bool = False) -> None:
    """參數 A：狀態查詢 — 出場信號檢查 + 後 20 筆周線資料。"""
    sep_len = 24 if mobile else 70
    stock_name = get_stock_name(ticker)
    df = fetch_weekly_stock_data(ticker, start_date=start_date)
    exit_result = check_exit_signal(ticker, start_date=start_date)
    print("=" * sep_len)
    print("出場信號檢查（周線）")
    print("=" * sep_len)
    print(f"【{exit_result['股票代號']} - {exit_result['股票名稱']}】")
    print(f"最新狀態：{exit_result['最新狀態']}")
    print(f"最新收盤價：{exit_result['最新收盤價']}")
    if exit_result.get('當前出場價格') is not None:
        print(f"當前出場價格：{exit_result['當前出場價格']}")
        print(f"距離出場價格：{exit_result.get('距離出場價格', 0):.2f} 點")
    if exit_result.get('煞車價格') is not None:
        print(f"煞車價格：{exit_result['煞車價格']}")
        print(f"距離煞車價格：{exit_result.get('距離煞車價格', 0):.2f} 點")
    print()
    print("後20筆資料預覽：")
    if mobile:
        display_df = df[['日期', '開盤價', '收盤價', '做多煞車', '反彈目標', '出場價格']].tail(20).copy()
        for col in ['開盤價', '收盤價', '反彈目標', '出場價格']:
            s = pd.to_numeric(display_df[col], errors='coerce').round(0)
            display_df[col] = s.astype('Int64')
        display_df['日期'] = display_df['日期'].astype(str).str[5:10]
        display_df['做多煞車'] = display_df['做多煞車'].fillna('').astype(str).str.strip()
        w_date, w_num, w_brake = 5, 6, 2
        fmt = f"{{:<{w_date}}} {{:>{w_num}}} {{:>{w_num}}} {{:>{w_brake}}} {{:>{w_num}}} {{:>{w_num}}}"
        print(fmt.format("日期", "開盤", "收盤", "煞車", "反彈", "出場"))
        for _, row in display_df.iterrows():
            d = str(row['日期'])[:w_date].ljust(w_date)
            o = '' if pd.isna(row['開盤價']) else str(int(row['開盤價']))
            c = '' if pd.isna(row['收盤價']) else str(int(row['收盤價']))
            b = str(row['做多煞車']).strip()[:w_brake].rjust(w_brake)
            r = '' if pd.isna(row['反彈目標']) else str(int(row['反彈目標']))
            e = '' if pd.isna(row['出場價格']) else str(int(row['出場價格']))
            print(fmt.format(d, o, c, b, r, e))
    else:
        # 顯示時不包含第一根～第五根（df 內仍保留）
        display_cols = ['日期', '最高價', '最低價', '開盤價', '收盤價', '漲幅', '漲跌點', '波動下限', '波動上限', '近四周總漲點', '兩周漲差', '20周均線', '做多煞車']
        display_cols = [c for c in display_cols if c in df.columns]
        display_df = df[display_cols].tail(20).copy()
        print(display_df.to_string(index=True))


def _run_year(ticker: str, start_date: str, year: int, mobile: bool = False) -> None:
    """參數 B：指定年度績效 — 該年度做多期間表格。"""
    sep_len = 24 if mobile else 70
    df = fetch_weekly_stock_data(ticker, start_date=start_date)
    long_stats = calculate_long_position_stats(df)
    if len(long_stats) == 0:
        print(f"【{year}年】無做多期間資料")
        return
    long_stats['年份'] = pd.to_datetime(long_stats['進入日期']).dt.year
    year_data = long_stats[long_stats['年份'] == year]
    if len(year_data) == 0:
        print(f"【{year}年】無資料")
        return
    print(f"【{year}年】")
    print("-" * sep_len)
    if mobile:
        for idx, row in year_data.iterrows():
            entry_d = row.get('進入日期', '')
            exit_d = row.get('退出日期', '')
            entry_p = row.get('進入價格', '')
            exit_p = row.get('退出價格', '')
            diff = row.get('累計價差', '')
            pct = row.get('累計漲幅(%)', '')
            weeks = row.get('持續周數', '')
            brake = row.get('做多煞車', '')
            print(f"進入 {entry_d}→退出 {exit_d}")
            print(f"進入價 {entry_p} | 退出價 {exit_p}")
            print(f"價差 {diff} | 漲幅 {pct}")
            print(f"周數 {weeks} | 煞車 {brake}")
            print()
        year_total = year_data['累計價差'].sum()
        print(f"{year}年小計：{len(year_data)} 段，價差總和：{year_total:.2f}")
    else:
        display_data = year_data.drop(columns=['年份'])
        print(display_data.to_string(index=False))
        year_total = year_data['累計價差'].sum()
        print(f"\n{year}年小計：{len(year_data)} 段，價差總和：{year_total:.2f}")


def _run_strategy(ticker: str, start_date: str, mobile: bool = False) -> None:
    """參數 C：策略資訊 — 盈虧統計與策略評估。"""
    sep_len = 24 if mobile else 70
    df = fetch_weekly_stock_data(ticker, start_date=start_date)
    long_stats = calculate_long_position_stats(df)
    if len(long_stats) == 0:
        print("無做多期間，無法計算策略評估")
        return
    metrics = calculate_performance_metrics(long_stats)
    print("=" * sep_len)
    print("盈虧統計")
    print("=" * sep_len)
    print(f"總計：{metrics['總交易次數']} 段做多期間")
    print(f"累計價差總和：{metrics['累計價差總和']}")
    print(f"平均每段累計價差：{metrics['平均每段累計價差']}")
    print()
    print(f"盈利次數：{metrics['盈利次數']} 次")
    print(f"虧損次數：{metrics['虧損次數']} 次")
    print(f"盈利點數總和：{metrics['盈利點數總和']}")
    print(f"虧損點數總和：{metrics['虧損點數總和']}")
    print(f"淨盈虧：{metrics['淨盈虧']}")
    print(f"勝率：{metrics['勝率']}%")
    print()
    print("=" * sep_len)
    print("策略評估")
    print("=" * sep_len)
    print(f"平均盈利：{metrics['平均盈利']}")
    print(f"平均虧損：{metrics['平均虧損']}")
    print(f"平均盈虧比：{metrics['平均盈虧比']}")
    print(f"最大單次虧損：{metrics['最大單次虧損']}")
    print(f"風險報酬比：{metrics['風險報酬比']}")


def _run_exit_trigger(ticker: str, start_date: str, mobile: bool = False) -> None:
    """參數 D：價格觸發 — 出場信號與進場信號檢查。"""
    sep_len = 24 if mobile else 70
    stock_name = get_stock_name(ticker)
    exit_result = check_exit_signal(ticker, start_date=start_date)
    entry_result = check_entry_signal(ticker, start_date=start_date)

    print("=" * sep_len)
    print("出場信號檢查（周線）")
    print("=" * sep_len)
    print(f"【{exit_result['股票代號']} - {exit_result['股票名稱']}】")
    if exit_result['是否為出場點']:
        print(f"出場日期（周結束日）：{exit_result['出場日期']}")
        print(f"出場價格：{exit_result['出場價格']}")
        print(f"出場原因：{exit_result['出場原因']}")
    else:
        print(f"最新狀態：{exit_result['最新狀態']}")
        print(f"最新收盤價：{exit_result['最新收盤價']}")
        print("尚未觸發出場條件")

    print()
    print("=" * sep_len)
    print("進場信號檢查（周線）")
    print("=" * sep_len)
    print(f"【{entry_result['股票代號']} - {entry_result['股票名稱']}】")
    if entry_result['是否為進場點']:
        print(f"進場日期（周結束日）：{entry_result['進場日期']}")
        print(f"進場價格：{entry_result['進場價格']}")
        print(f"出場價格：{entry_result['出場價格']}")
    else:
        print(f"最新狀態：{entry_result['最新狀態']}")
        print(f"最新收盤價：{entry_result['最新收盤價']}")
        if entry_result['最新狀態'] == '等待突破' and entry_result.get('反彈目標') is not None:
            print(f"反彈目標：{entry_result['反彈目標']}")
            print(f"距離反彈目標：{entry_result.get('距離反彈目標', 0):.2f} 點")
        elif entry_result['最新狀態'] == '做多中' and entry_result.get('當前出場價格') is not None:
            print(f"當前出場價格：{entry_result['當前出場價格']}")
            print(f"做多煞車：{entry_result.get('做多煞車', '')}")




def _send_to_discord_webhook(webhook_url: str, content: str) -> bool:
    """
    將 content 以 POST 送至 Discord Webhook URL。
    content 會被切分成每段最多 1990 字元（Discord 上限 2000），逐段送出。
    使用 urllib.request（標準庫，不新增依賴）。
    成功回傳 True，失敗回傳 False，錯誤輸出至 stderr。
    """
    if not content.strip():
        return True
    chunk_size = 1990
    chunks = [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]
    for chunk in chunks:
        try:
            data = json.dumps({"content": chunk}).encode("utf-8")
            req = urllib_request.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json", "User-Agent": "Discord-Webhook-Client/1.0"},
                method="POST",
            )
            with urllib_request.urlopen(req, timeout=30) as resp:
                if resp.status not in (200, 204):
                    print(f"Discord webhook HTTP {resp.status}", file=sys.stderr)
                    return False
        except urllib_error.URLError as e:
            print(f"Discord webhook error: {e}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Discord webhook error: {e}", file=sys.stderr)
            return False
    return True


if __name__ == "__main__":
    import argparse
    import io
    parser = argparse.ArgumentParser(description="台股加權指數周線資料與策略查詢（供 Agent Bot 呼叫）")
    parser.add_argument("--ticker", "-t", default="^TWII", help="標的代號，預設 ^TWII")
    parser.add_argument("--start-date", "-s", default="2020-01-01", help="資料起始日")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="A：狀態查詢（出場信號檢查 + 後20筆周線）")
    group.add_argument("--year", type=int, metavar="YYYY", help="B：列出指定年度績效表")
    group.add_argument("--strategy", action="store_true", help="C：列出策略資訊（盈虧統計與策略評估）")
    group.add_argument("--exit-trigger", action="store_true", help="D：價格觸發（出場日期/價格/原因）")
    parser.add_argument("--mobile", action="store_true", help="手機版精簡輸出（僅於使用 --status/--year/--strategy/--exit-trigger 時有效）")
    parser.add_argument("--webhook-url", "-w", default="", help="Discord Webhook URL（可透過環境變數 DISCORD_WEBHOOK_URL 設定）")
    args = parser.parse_args()

    # 解析 webhook URL：優先使用參數，其次環境變數
    webhook_url = (args.webhook_url or os.environ.get("DISCORD_WEBHOOK_URL", "")).strip()

    ticker = args.ticker
    start_date = args.start_date
    mobile = args.mobile

    # 判斷是否為帶參數的查詢模式（--status / --year / --strategy / --exit-trigger）
    has_report_mode = args.status or args.year is not None or args.strategy or args.exit_trigger

    if has_report_mode and webhook_url:
        # Webhook 模式：將輸出導向 StringIO，執行後送至 Discord
        buf = StringIO()
        old_stdout = sys.stdout
        sys.stdout = buf
        try:
            if args.status:
                _run_status(ticker, start_date, mobile=mobile)
            elif args.year is not None:
                _run_year(ticker, start_date, args.year, mobile=mobile)
            elif args.strategy:
                _run_strategy(ticker, start_date, mobile=mobile)
            else:
                _run_exit_trigger(ticker, start_date, mobile=mobile)
        finally:
            sys.stdout = old_stdout
        content = buf.getvalue()
        ok = _send_to_discord_webhook(webhook_url, content)
        raise SystemExit(0 if ok else 1)
    elif has_report_mode:
        # 無 webhook：輸出至 stdout
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            if hasattr(sys.stdout, "buffer"):
                sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        if args.status:
            _run_status(ticker, start_date, mobile=mobile)
            raise SystemExit(0)
        if args.year is not None:
            _run_year(ticker, start_date, args.year, mobile=mobile)
            raise SystemExit(0)
        if args.strategy:
            _run_strategy(ticker, start_date, mobile=mobile)
            raise SystemExit(0)
        if args.exit_trigger:
            _run_exit_trigger(ticker, start_date, mobile=mobile)
            raise SystemExit(0)

    # 無參數 + webhook：將完整輸出導向 Discord（與 --status 等模式一致）
    if webhook_url and not has_report_mode:
        _full_buf = StringIO()
        _old_stdout = sys.stdout
        sys.stdout = _full_buf

    # 無參數時：原有完整輸出
    try:
        print(f"正在抓取 {ticker} 的日線資料並轉換為周線資料（周四~下周三為一周）...")
        df = fetch_weekly_stock_data(ticker, start_date=start_date)
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weekly_result.csv")
        df_to_csv = df.drop(columns=['做多累計漲幅'], errors='ignore') if '做多累計漲幅' in df.columns else df
        df_to_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"已輸出周線 df 至 {csv_path}")
        print(f"\n成功抓取 {len(df)} 筆周線資料")
        # 預覽不顯示第一根～第五根與做多累計漲幅（df 內仍保留）
        _hide_bar_cols = ['第一根', '第二根', '第三根', '第四根', '第五根', '做多累計漲幅']
        _disp = df.drop(columns=[c for c in _hide_bar_cols if c in df.columns], errors='ignore')
        print("\n前5筆資料預覽：")
        print(_disp.head(5))
        print("\n後5筆資料預覽：")
        print(_disp.tail(20))
        print(f"\n資料日期範圍：{df['日期'].min()} 至 {df['日期'].max()}")
        
        # 統計每段做多中的累計價差
        print("\n" + "="*70)
        print("每段「做多中」累計價差統計（周線）")
        print("="*70)
        long_stats = calculate_long_position_stats(df)
        if len(long_stats) > 0:
            # 從進入日期提取年份
            long_stats['年份'] = pd.to_datetime(long_stats['進入日期']).dt.year
            
            # 按照年份分組顯示
            years = sorted(long_stats['年份'].unique())
            
            for year_idx, year in enumerate(years):
                year_data = long_stats[long_stats['年份'] == year].copy()
                
                # 顯示年份標題
                print(f"\n【{year}年】")
                print("-" * 70)
                
                # 顯示該年的數據（不包含年份欄位，保持原始欄位順序）
                original_cols = [col for col in long_stats.columns if col != '年份']
                display_data = year_data[original_cols].copy()
                print(display_data.to_string(index=False))
                
                # 計算該年的價差總和
                year_total = year_data['累計價差'].sum()
                year_count = len(year_data)
                print(f"\n{year}年小計：{year_count} 段，價差總和：{year_total:.2f}")
                
                # 在年份之間加上分隔線（最後一年不加）
                if year_idx < len(years) - 1:
                    print("\n" + "="*70)
            
            print(f"\n總計：{len(long_stats)} 段做多期間")
            if '累計價差' in long_stats.columns:
                total_diff = long_stats['累計價差'].sum()
                avg_diff = long_stats['累計價差'].mean()
                print(f"累計價差總和：{total_diff:.2f}")
                print(f"平均每段累計價差：{avg_diff:.2f}")
                
                # 計算盈/虧次數和點數
                profit_trades = long_stats[long_stats['累計價差'] > 0]
                loss_trades = long_stats[long_stats['累計價差'] < 0]
                even_trades = long_stats[long_stats['累計價差'] == 0]
                
                profit_count = len(profit_trades)
                loss_count = len(loss_trades)
                even_count = len(even_trades)
                
                profit_points = profit_trades['累計價差'].sum() if len(profit_trades) > 0 else 0
                loss_points = loss_trades['累計價差'].sum() if len(loss_trades) > 0 else 0
                
                # 計算勝率（盈利次數 / 總次數 * 100%）
                total_trades = len(long_stats)
                win_rate = (profit_count / total_trades * 100) if total_trades > 0 else 0
                
                print("\n" + "="*70)
                print("盈虧統計")
                print("="*70)
                print(f"盈利次數：{profit_count} 次")
                print(f"虧損次數：{loss_count} 次")
                if even_count > 0:
                    print(f"持平次數：{even_count} 次")
                print(f"盈利點數總和：{profit_points:.2f}")
                print(f"虧損點數總和：{loss_points:.2f}")
                print(f"淨盈虧：{profit_points + loss_points:.2f}")
                print(f"勝率：{win_rate:.2f}%")
                
                # 計算額外評估指標
                avg_profit = profit_trades['累計價差'].mean() if len(profit_trades) > 0 else 0
                avg_loss = abs(loss_trades['累計價差'].mean()) if len(loss_trades) > 0 else 1
                profit_loss_ratio = avg_profit / avg_loss if avg_loss > 0 else 0
                
                max_loss = abs(loss_trades['累計價差'].min()) if len(loss_trades) > 0 else 0
                net_profit = profit_points + loss_points
                risk_reward_ratio = net_profit / max_loss if max_loss > 0 else 0
                
                print("\n" + "="*70)
                print("策略評估")
                print("="*70)
                print(f"平均盈利：{avg_profit:.2f}")
                print(f"平均虧損：{-avg_loss:.2f}")
                print(f"平均盈虧比：{profit_loss_ratio:.2f}")
                print(f"最大單次虧損：{max_loss:.2f}")
                print(f"風險報酬比：{risk_reward_ratio:.2f}")
        else:
            print("沒有找到做多期間")
        
        # 檢查進場和出場信號
        print("\n" + "="*70)
        print("進場信號檢查（周線）")
        print("="*70)
        entry_result = check_entry_signal(ticker)
        if entry_result['是否為進場點']:
            print(f"【{entry_result['股票代號']} - {entry_result['股票名稱']}】")
            print(f"進場日期（周結束日）：{entry_result['進場日期']}")
            print(f"進場價格：{entry_result['進場價格']}")
            print(f"出場價格：{entry_result['出場價格']}")
        else:
            print(f"【{entry_result['股票代號']} - {entry_result['股票名稱']}】")
            print(f"最新狀態：{entry_result['最新狀態']}")
            print(f"最新收盤價：{entry_result['最新收盤價']}")
            if entry_result['最新狀態'] == '等待突破' and entry_result.get('反彈目標'):
                print(f"反彈目標：{entry_result['反彈目標']}")
                print(f"距離反彈目標：{entry_result.get('距離反彈目標', 0):.2f} 點")
        
        print("\n" + "="*70)
        print("出場信號檢查（周線）")
        print("="*70)
        exit_result = check_exit_signal(ticker)
        if exit_result['是否為出場點']:
            print(f"【{exit_result['股票代號']} - {exit_result['股票名稱']}】")
            print(f"出場日期（周結束日）：{exit_result['出場日期']}")
            print(f"出場價格：{exit_result['出場價格']}")
            print(f"出場原因：{exit_result['出場原因']}")
        else:
            print(f"【{exit_result['股票代號']} - {exit_result['股票名稱']}】")
            print(f"最新狀態：{exit_result['最新狀態']}")
            print(f"最新收盤價：{exit_result['最新收盤價']}")
            if exit_result['最新狀態'] == '做多中' and exit_result.get('當前出場價格'):
                print(f"當前出場價格：{exit_result['當前出場價格']}")
                print(f"距離出場價格：{exit_result.get('距離出場價格', 0):.2f} 點")
                if exit_result.get('煞車價格') is not None:
                    print(f"煞車價格：{exit_result['煞車價格']}")
                    print(f"距離煞車價格：{exit_result.get('距離煞車價格', 0):.2f} 點")
        
    except Exception as e:
        print(f"發生錯誤：{e}")
        import traceback
        traceback.print_exc()

    # 無參數 + webhook：送出至 Discord 後結束
    if webhook_url and not has_report_mode:
        try:
            sys.stdout = _old_stdout
        except NameError:
            pass
        content = _full_buf.getvalue()
        ok = _send_to_discord_webhook(webhook_url, content)
        raise SystemExit(0 if ok else 1)
