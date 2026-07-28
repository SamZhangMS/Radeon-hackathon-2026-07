import os
import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .config import CACHE_DIR

script_path = os.path.dirname(os.path.abspath(__file__))
# raw_data_path=f'{script_path}/../data/1D'
raw_data_path=f'{script_path}/../../datatemp/raw_data/1D'

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
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume'
                })
                df['Date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        except:
            pass

        try:
            raw_data_file_path=raw_data_path+'/'+symbol+'.txt'
            df_raw=pd.read_csv(raw_data_file_path
                ,encoding='gb2312',
                    skipfooter=1,
                    names=['date','open', 'high', 'low', 'close', 'volume', 'money'],
                    dtype={'date': str,'open': float,'high': float, 'low': float, 'close': float}
                    ,engine='python')
            df_raw['date'] = pd.to_datetime(df_raw['date'].astype(str) , format='%Y/%m/%d')
            return df_raw
        except Exception as e:
            pass
            
        return pd.DataFrame()
    def get_etf_list(self) -> List[str]:
        """获取默认ETF列表"""

        data_path = Path(raw_data_path)
        if not data_path.exists():
            return []
        
        stock_list = [
            p.stem for p in data_path.iterdir() 
            if p.is_file() and p.suffix in ['.txt', '.csv']
        ]
        return stock_list