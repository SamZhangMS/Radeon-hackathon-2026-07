# app/utils.py
"""
工具函数
"""

import numpy as np
import pandas as pd
from typing import Any, Optional, Union, List
from datetime import datetime, timedelta

__all__ = [
    'to_python',
    'get_latest_date',
    'get_latest_date_from_df_list',
    'add_latest_date_to_data',
    'add_timestamp_to_data',
    'ensure_date_fields',
    'get_quote_with_date',
    'format_response',
    # 日期生成函数
    'parse_last_date',
    'is_weekend',
    'get_next_trading_day',
    'generate_future_dates',
    'generate_future_date_strings',
    'get_trading_days_between',
]

def to_python(obj: Any) -> Any:
    """转换 numpy 类型为 Python 原生类型，处理 NaN"""
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        # ✅ 处理 NaN 和 Inf
        if np.isnan(obj):
            return None
        if np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return to_python(obj.tolist())
    elif isinstance(obj, pd.Series):
        return to_python(obj.tolist())
    elif isinstance(obj, pd.DataFrame):
        return to_python(obj.to_dict('records'))
    return obj


def get_latest_date(df: pd.DataFrame, default: Optional[str] = None) -> str:
    """从 DataFrame 索引获取最新日期"""
    if default is None:
        default = datetime.now().strftime('%Y-%m-%d')
    
    if df is None or df.empty:
        return default
    
    try:
        latest = df.index[-1]
        if isinstance(latest, pd.Timestamp):
            return latest.strftime('%Y-%m-%d')
        elif isinstance(latest, datetime):
            return latest.strftime('%Y-%m-%d')
        elif isinstance(latest, (int, float)):
            try:
                return pd.to_datetime(latest, unit='s').strftime('%Y-%m-%d')
            except:
                return str(latest)
        else:
            return str(latest)
    except Exception:
        return default

def parse_date_to_datetime(date_val: Any) -> Optional[datetime]:
    """将各种日期格式转换为 datetime"""
    if date_val is None:
        return None
    
    try:
        print(f'parse_date_to_datetime:{date_val}')
        if isinstance(date_val, pd.Timestamp):
            return date_val.to_pydatetime()
        elif isinstance(date_val, datetime):
            return date_val
        elif isinstance(date_val, pd.DatetimeIndex):
            return date_val[0].to_pydatetime()
        elif isinstance(date_val, (int, float)):
            # 检查是否为无效时间戳（如 1970-01-01 附近）
            if date_val < 1000000000:  # 小于 2001-09-09 的时间戳
                return datetime.now()
            # 判断是毫秒还是秒时间戳
            if date_val > 1e10:  # 毫秒时间戳
                return datetime.fromtimestamp(date_val / 1000)
            else:  # 秒时间戳
                return datetime.fromtimestamp(date_val)
        elif isinstance(date_val, str):
            try:
                return pd.to_datetime(date_val).to_pydatetime()
            except:
                return None
        else:
            return None
    except Exception as e:
        return None
    
def get_latest_date_from_df_list(df_list: list) -> str:
    """
    从 DataFrame 列表中获取最新的日期
    
    Args:
        df_list: DataFrame 列表
    
    Returns:
        最新的日期字符串 (YYYY-MM-DD)
    """
    latest_date = None
    latest_dt = None
    
    print(f'df_list:{df_list}')
    for df in df_list:
        if df is not None and not df.empty:
            try:
                last = df.index[-1]
                if hasattr(last, 'strftime'):
                    dt = last
                elif isinstance(last, (int, float)):
                    dt = pd.to_datetime(last, unit='s')
                else:
                    continue
                
                if latest_dt is None or dt > latest_dt:
                    latest_dt = dt
                    latest_date = dt.strftime('%Y-%m-%d')
            except Exception:
                continue
    
    if latest_date is None:
        return datetime.now().strftime('%Y-%m-%d')
    return latest_date


def add_latest_date_to_data(data: dict, df: Optional[pd.DataFrame] = None, 
                            default: Optional[str] = None) -> dict:
    """
    向数据字典添加最新日期字段
    
    Args:
        data: 要添加日期的数据字典
        df: 用于获取日期的 DataFrame
        default: 默认日期
    
    Returns:
        包含 latest_date 的数据字典
    """
    if 'latest_date' not in data:
        data['latest_date'] = get_latest_date(df, default)
    return data


def add_timestamp_to_data(data: dict) -> dict:
    """
    向数据字典添加时间戳字段
    
    Args:
        data: 要添加时间戳的数据字典
    
    Returns:
        包含 timestamp 的数据字典
    """
    if 'timestamp' not in data:
        data['timestamp'] = datetime.now().isoformat()
    return data


def ensure_date_fields(data: dict, df: Optional[pd.DataFrame] = None, 
                       add_timestamp: bool = True) -> dict:
    """
    确保数据字典包含日期字段
    
    Args:
        data: 数据字典
        df: 用于获取日期的 DataFrame
        add_timestamp: 是否添加时间戳
    
    Returns:
        包含日期字段的数据字典
    """
    add_latest_date_to_data(data, df)
    if add_timestamp:
        add_timestamp_to_data(data)
    return data


def get_quote_with_date(quote_data: dict) -> dict:
    """
    为行情数据添加日期信息
    
    Args:
        quote_data: 行情数据
    
    Returns:
        包含日期的行情数据
    """
    result = quote_data.copy() if quote_data else {}
    result['latest_date'] = datetime.now().strftime('%Y-%m-%d')
    if 'timestamp' not in result:
        result['timestamp'] = datetime.now().isoformat()
    return result


def format_response(data: Any, df: Optional[pd.DataFrame] = None, 
                    add_timestamp: bool = True) -> dict:
    """
    格式化 API 响应，自动添加日期字段
    
    Args:
        data: 响应数据
        df: 用于获取日期的 DataFrame
        add_timestamp: 是否添加时间戳
    
    Returns:
        包含日期字段的响应数据
    """
    if isinstance(data, dict):
        ensure_date_fields(data, df, add_timestamp)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                ensure_date_fields(item, None, add_timestamp)
        return {
            'data': data,
            'count': len(data),
            'latest_date': datetime.now().strftime('%Y-%m-%d'),
            'timestamp': datetime.now().isoformat() if add_timestamp else None
        }
    return data


# ============================================================
# 未来日期生成函数
# ============================================================

def parse_last_date(last_date: Union[pd.Timestamp, datetime, int, float, str]) -> datetime:
    """
    解析各种格式的日期为 datetime 对象
    
    Args:
        last_date: 日期数据（pd.Timestamp, datetime, int, float, str）
    
    Returns:
        datetime 对象
    """
    if isinstance(last_date, pd.Timestamp):
        return last_date.to_pydatetime()
    elif isinstance(last_date, datetime):
        return last_date
    elif isinstance(last_date, (int, float)):
        try:
            return pd.to_datetime(last_date, unit='s').to_pydatetime()
        except:
            return datetime.now()
    elif isinstance(last_date, str):
        try:
            return pd.to_datetime(last_date).to_pydatetime()
        except:
            return datetime.now()
    else:
        return datetime.now()


def is_weekend(date: datetime) -> bool:
    """
    检查日期是否为周末
    
    Args:
        date: datetime 对象
    
    Returns:
        True 如果是周末（周六或周日）
    """
    return date.weekday() in [5, 6]  # 5=周六, 6=周日


def get_next_trading_day(date: datetime) -> datetime:
    """
    获取下一个交易日（跳过周末）
    
    Args:
        date: 当前日期
    
    Returns:
        下一个交易日（如果是周末则顺延到周一）
    """
    next_date = date + timedelta(days=1)
    while is_weekend(next_date):
        next_date += timedelta(days=1)
    return next_date


def generate_future_dates(
    last_date: Any,
    n_days: int = 20,
    skip_weekends: bool = True
) -> List[datetime]:
    """
    生成未来日期列表
    
    Args:
        last_date: 起始日期（数据中的最后一天）
        n_days: 需要生成的天数
        skip_weekends: 是否跳过周末（交易日模式）
    
    Returns:
        未来日期列表（datetime 对象）
    """
    if n_days <= 0:
        return []
    
    # 解析日期
    current = parse_date_to_datetime(last_date)
    if current is None:
        current = datetime.now()
    
    # 如果当前是周末，先移到下一个交易日
    if skip_weekends and is_weekend(current):
        current = get_next_trading_day(current)
    
    future_dates = []
    for _ in range(n_days):
        current = get_next_trading_day(current) if skip_weekends else current + timedelta(days=1)
        future_dates.append(current)
    
    return future_dates


def generate_future_date_strings(
    last_date: Union[pd.Timestamp, datetime, int, float, str],
    n_days: int = 20,
    skip_weekends: bool = True,
    date_format: str = '%Y-%m-%d'
) -> List[str]:
    """
    生成未来日期字符串列表
    
    Args:
        last_date: 起始日期（数据中的最后一天）
        n_days: 需要生成的天数
        skip_weekends: 是否跳过周末（交易日模式）
        date_format: 日期格式
    
    Returns:
        未来日期字符串列表
    """
    dates = generate_future_dates(last_date, n_days, skip_weekends)
    return [d.strftime(date_format) for d in dates]

def get_trading_days_between(start_date: Any, end_date: Any) -> List[datetime]:
    """获取两个日期之间的所有交易日"""
    start = parse_date_to_datetime(start_date)
    end = parse_date_to_datetime(end_date)
    
    if start is None or end is None:
        return []
    
    trading_days = []
    current = start
    while current <= end:
        if not is_weekend(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days

def get_trading_days_between(start_date: datetime, end_date: datetime) -> List[datetime]:
    """
    获取两个日期之间的所有交易日（跳过周末）
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
    
    Returns:
        交易日列表
    """
    trading_days = []
    current = start_date
    while current <= end_date:
        if not is_weekend(current):
            trading_days.append(current)
        current += timedelta(days=1)
    return trading_days