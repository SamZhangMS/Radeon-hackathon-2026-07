# app/utils.py
"""
工具函数
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Union, List
from datetime import datetime, timedelta
import traceback
import sys
holiday_list_cn = [
        '2026-01-01', # 元旦
        '2026-02-15', '2026-02-16', '2026-02-17', '2026-02-18', '2026-02-19', 
        '2026-02-20', '2026-02-21', '2026-02-22', '2026-02-23', # 春节连休
        '2026-04-04', '2026-04-05', '2026-04-06', # 清明
        '2026-05-01', '2026-05-02', '2026-05-03', '2026-05-04', '2026-05-05', # 五一 (当前时间附近的假期)
        '2026-06-19', '2026-06-20', '2026-06-21', # 端午
        '2026-09-25', '2026-09-26', '2026-09-27', # 中秋
        '2026-10-01', '2026-10-02', '2026-10-03', '2026-10-04', 
        '2026-10-05', '2026-10-06', '2026-10-07', # 国庆
        '2027-01-01'
    ]


def format_exception(e):
    exception_list = traceback.format_stack()
    exception_list = exception_list[:-2]
    exception_list.extend(traceback.format_tb(sys.exc_info()[2]))
    exception_list.extend(traceback.format_exception_only(sys.exc_info()[0], sys.exc_info()[1]))
    
    exception_str = "Traceback (most recent call last):\n"
    exception_str += "".join(exception_list)
    # Removing the last \n
    exception_str = exception_str[:-1]

    return exception_str
def get_last_date(df: pd.DataFrame) -> Any:
    """从 DataFrame 获取最后日期（兼容索引和列）"""
    if df is None or df.empty:
        return None
    
    # 检查索引是否为日期类型
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[-1]
    
    # 检查是否有 'date' 列
    if 'date' in df.columns:
        return df['date'].iloc[-1]
    
    return None


def generate_future_daily_dates( start_date, pred_len, end_date=None):
 
    def _to_timestamp(date_obj):
        if not isinstance(date_obj, pd.Timestamp):
            return pd.Timestamp(date_obj)
        return date_obj
    
    start_date =  _to_timestamp(start_date)
    
    if end_date is not None:
        end_date =  _to_timestamp(end_date)
        # 计算从 start_date 到 end_date 之间的交易日数量
        temp_date = start_date + timedelta(days=1)
        max_days = 0
        while temp_date <= end_date:
            if temp_date.weekday() < 5 and temp_date.strftime('%Y-%m-%d') not in holiday_list_cn:
                max_days += 1
            temp_date += timedelta(days=1)
        
        actual_len = min(pred_len, max_days)
    else:
        actual_len = pred_len
    
    future_dates = []
    current_date = start_date + timedelta(days=1)
    
    while len(future_dates) < actual_len:
        if end_date is not None and current_date > end_date:
            break
            
        if current_date.weekday() < 5 and current_date.strftime('%Y-%m-%d') not in holiday_list_cn:
            future_dates.append(pd.Timestamp(current_date))
        current_date += timedelta(days=1)
    
    return future_dates[:pred_len]

# def get_latest_date(df: pd.DataFrame) -> str:
#     """从 DataFrame 获取最新日期字符串"""
#     if df is None or df.empty:
#         return None
    
#     try:
#         last = get_last_date(df)
#         if last is None:
#             return None
#         if hasattr(last, 'strftime'):
#             return last.strftime('%Y-%m-%d')
#         return str(last)
#     except Exception:
#         return None


    
# def get_latest_date_from_df_list(df_list: list) -> str:
#     """
#     从 DataFrame 列表中获取最新的日期
    
#     Args:
#         df_list: DataFrame 列表
    
#     Returns:
#         最新的日期字符串 (YYYY-MM-DD)
#     """
#     latest_date = None
#     latest_dt = None
    
#     print(f'df_list:{df_list}')
#     for df in df_list:
#         if df is not None and not df.empty:
#             try:
#                 last = df.index[-1]
#                 if hasattr(last, 'strftime'):
#                     dt = last
#                 elif isinstance(last, (int, float)):
#                     dt = pd.to_datetime(last, unit='s')
#                 else:
#                     continue
                
#                 if latest_dt is None or dt > latest_dt:
#                     latest_dt = dt
#                     latest_date = dt.strftime('%Y-%m-%d')
#             except Exception:
#                 continue
    
#     if latest_date is None:
#         return datetime.now().strftime('%Y-%m-%d')
#     return latest_date


# def add_latest_date_to_data(data: dict, df: Optional[pd.DataFrame] = None, 
#                             default: Optional[str] = None) -> dict:
#     """
#     向数据字典添加最新日期字段
    
#     Args:
#         data: 要添加日期的数据字典
#         df: 用于获取日期的 DataFrame
#         default: 默认日期
    
#     Returns:
#         包含 latest_date 的数据字典
#     """
#     if 'latest_date' not in data:
#         data['latest_date'] = get_last_date(df)
#     return data


# def add_timestamp_to_data(data: dict) -> dict:
#     """
#     向数据字典添加时间戳字段
    
#     Args:
#         data: 要添加时间戳的数据字典
    
#     Returns:
#         包含 timestamp 的数据字典
#     """
#     if 'timestamp' not in data:
#         data['timestamp'] = datetime.now().isoformat()
#     return data


# def ensure_date_fields(data: dict, df: Optional[pd.DataFrame] = None, 
#                        add_timestamp: bool = True) -> dict:
#     """
#     确保数据字典包含日期字段
    
#     Args:
#         data: 数据字典
#         df: 用于获取日期的 DataFrame
#         add_timestamp: 是否添加时间戳
    
#     Returns:
#         包含日期字段的数据字典
#     """
#     add_latest_date_to_data(data, df)
#     if add_timestamp:
#         add_timestamp_to_data(data)
#     return data





# ============================================================
# 未来日期生成函数
# ============================================================

# def parse_last_date(last_date: Union[pd.Timestamp, datetime, int, float, str]) -> datetime:
#     """
#     解析各种格式的日期为 datetime 对象
    
#     Args:
#         last_date: 日期数据（pd.Timestamp, datetime, int, float, str）
    
#     Returns:
#         datetime 对象
#     """
#     if isinstance(last_date, pd.Timestamp):
#         return last_date.to_pydatetime()
#     elif isinstance(last_date, datetime):
#         return last_date
#     elif isinstance(last_date, (int, float)):
#         try:
#             return pd.to_datetime(last_date, unit='s').to_pydatetime()
#         except:
#             return datetime.now()
#     elif isinstance(last_date, str):
#         try:
#             return pd.to_datetime(last_date).to_pydatetime()
#         except:
#             return datetime.now()
#     else:
#         return datetime.now()


# def is_weekend(date: datetime) -> bool:
#     """
#     检查日期是否为周末
    
#     Args:
#         date: datetime 对象
    
#     Returns:
#         True 如果是周末（周六或周日）
#     """
#     return date.weekday() in [5, 6]  # 5=周六, 6=周日


# def get_next_trading_day(date: datetime) -> datetime:
#     """
#     获取下一个交易日（跳过周末）
    
#     Args:
#         date: 当前日期
    
#     Returns:
#         下一个交易日（如果是周末则顺延到周一）
#     """
#     next_date = date + timedelta(days=1)
#     while is_weekend(next_date):
#         next_date += timedelta(days=1)
#     return next_date


# def generate_future_dates(
#     last_date: Any,
#     n_days: int = 20,
#     skip_weekends: bool = True
# ) -> List[datetime]:
#     """生成未来日期列表"""
#     if n_days <= 0:
#         return []
    
#     # 解析日期
#     current = parse_date_to_datetime(last_date)
#     if current is None:
#         return []
    
#     # 如果当前是周末，先移到下一个交易日
#     if skip_weekends and current.weekday() in [5, 6]:
#         current += timedelta(days=(7 - current.weekday()))
    
#     future_dates = []
#     for _ in range(n_days):
#         current += timedelta(days=1)
#         if skip_weekends:
#             while current.weekday() in [5, 6]:
#                 current += timedelta(days=1)
#         future_dates.append(current)
    
#     return future_dates


# def generate_future_date_strings(
#     last_date: Any,
#     n_days: int = 20,
#     skip_weekends: bool = True,
#     date_format: str = '%Y-%m-%d'
# ) -> List[str]:
#     """生成未来日期字符串列表"""
#     dates = generate_future_dates(last_date, n_days, skip_weekends)
#     return [d.strftime(date_format) for d in dates]

# def get_trading_days_between(start_date: Any, end_date: Any) -> List[datetime]:
#     """获取两个日期之间的所有交易日"""
#     start = parse_date_to_datetime(start_date)
#     end = parse_date_to_datetime(end_date)
    
#     if start is None or end is None:
#         return []
    
#     trading_days = []
#     current = start
#     while current <= end:
#         if not is_weekend(current):
#             trading_days.append(current)
#         current += timedelta(days=1)
#     return trading_days

# def get_trading_days_between(start_date: datetime, end_date: datetime) -> List[datetime]:
#     """
#     获取两个日期之间的所有交易日（跳过周末）
    
#     Args:
#         start_date: 开始日期
#         end_date: 结束日期
    
#     Returns:
#         交易日列表
#     """
#     trading_days = []
#     current = start_date
#     while current <= end_date:
#         if not is_weekend(current):
#             trading_days.append(current)
#         current += timedelta(days=1)
#     return trading_days