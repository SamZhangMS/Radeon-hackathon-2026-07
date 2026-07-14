import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .config import CACHE_DIR


class ETFDataFetcher:
    """ETF数据获取器"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
    
    def get_etf_quote(self, symbol: str) -> Optional[Dict]:
        """获取实时行情"""
        try:
            df = ak.stock_zh_a_spot_em()
            row = df[df['代码'] == symbol]
            if not row.empty:
                return {
                    'symbol': symbol,
                    'name': row.iloc[0]['名称'],
                    'price': float(row.iloc[0]['最新价']),
                    'change': float(row.iloc[0]['涨跌幅']),
                    'volume': float(row.iloc[0]['成交量']),
                    'high': float(row.iloc[0]['最高']),
                    'low': float(row.iloc[0]['最低']),
                    'open': float(row.iloc[0]['今开']),
                }
        except:
            pass
        return None
    
    def get_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """获取历史数据"""
        try:
            ticker = yf.Ticker(f"{symbol}.SS" if symbol.startswith('6') else f"{symbol}.SZ")
            df = ticker.history(period=period)
            if not df.empty:
                return df
        except:
            pass
        
        # 备用方案：使用akshare
        try:
            end = datetime.now()
            start = end - timedelta(days=365)
            df = ak.fund_etf_hist_em(
                symbol=symbol,
                start_date=start.strftime('%Y%m%d'),
                end_date=end.strftime('%Y%m%d')
            )
            if not df.empty:
                df = df.rename(columns={
                    '日期': 'Date',
                    '开盘': 'Open',
                    '收盘': 'Close',
                    '最高': 'High',
                    '最低': 'Low',
                    '成交量': 'Volume'
                })
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                return df
        except:
            pass
        
        return pd.DataFrame()
    def get_etf_list(self) -> List[str]:
        """获取默认ETF列表"""
        from .config import DEFAULT_ETF_POOL
        return DEFAULT_ETF_POOL