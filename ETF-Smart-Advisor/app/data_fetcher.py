import os
import glob

import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .config import CACHE_DIR

script_path = os.path.dirname(os.path.abspath(__file__))
raw_data_path=f'{script_path}/../data/1D'
parquet_path=f'{script_path}/../data/history'
# raw_data_path=f'{script_path}/../../datatemp/raw_data/1D'


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
        
        base_dir=raw_data_path
        pattern = os.path.join(base_dir, "**", f"*{symbol}*.parquet")
        
        # 2. 使用 glob 递归查找匹配的文件
        matched_files = glob.glob(pattern, recursive=True)
        
        if not matched_files:
            print(f"⚠️ 警告: 在 {base_dir} 目录下未找到包含 '{symbol}' 的 parquet 文件<websource>source_group_web_2</websource>。")
            return None
        
        # 如果找到多个文件，这里默认读取第一个（您可以根据需要修改为读取最新修改的文件）
        file_path = matched_files
        print(f"✅ 找到文件: {file_path}")
        
        # 3. 使用 pandas 读取 parquet 文件
        try:
            df = pd.read_parquet(file_path)
            print(f"📊 成功读取数据，共 {len(df)} 行, {len(df.columns)} 列<websource>source_group_web_3</websource>。")
            return df
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return None
    def get_etf_list(self) -> List[str]:
        """获取默认ETF列表"""

        data_path = Path(raw_data_path)
        if not data_path.exists():
            return []
        
        stock_list = [
            p.stem for p in data_path.iterdir() 
            if p.is_file() and p.suffix in ['.txt', '.csv', '.parquet']
        ]
        return stock_list