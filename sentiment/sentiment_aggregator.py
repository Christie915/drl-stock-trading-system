"""
情感时间聚合模块 - 情感信号的时间聚合

根据提案要求，实现以下功能：
1. 情感信号的时间聚合以匹配交易决策周期
2. 多时间框架情感聚合
3. 情感时间序列的统计特征提取

作者：大狗（电子兄弟）
日期：2026-03-12
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy import stats


class SentimentAggregator:
    """
    情感时间聚合器
    
    提案要求：
    - 情感信号的时间聚合以匹配交易决策周期
    - 多时间框架情感聚合
    - 集成到DRL框架中的注意力机制
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化情感聚合器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 默认聚合窗口
        self.default_windows = self.config.get('aggregation_windows', ['1H', '4H', '1D', '1W'])
        self.sentiment_columns = self.config.get('sentiment_columns', ['sentiment_score'])
        
        self.logger.info("情感时间聚合器初始化完成")
    
    def aggregate_by_time(self, 
                         df: pd.DataFrame,
                         time_column: str = 'published_at',
                         value_columns: Optional[List[str]] = None,
                         frequency: str = '1D',
                         aggregation_method: str = 'mean') -> pd.DataFrame:
        """
        按时间频率聚合情感数据
        
        Args:
            df: 输入DataFrame
            time_column: 时间列名
            value_columns: 要聚合的值列（None表示自动检测情感列）
            frequency: 聚合频率（'1H', '4H', '1D', '1W'等）
            aggregation_method: 聚合方法（'mean', 'median', 'sum', 'last', 'first'）
            
        Returns:
            聚合后的DataFrame
        """
        self.logger.info(f"按时间聚合情感数据，频率: {frequency}, 方法: {aggregation_method}")
        
        if df.empty:
            self.logger.warning("输入DataFrame为空")
            return pd.DataFrame()
        
        # 确保时间列存在
        if time_column not in df.columns:
            self.logger.error(f"时间列不存在: {time_column}")
            return df
        
        # 复制数据并设置时间索引
        aggregated_df = df.copy()
        aggregated_df[time_column] = pd.to_datetime(aggregated_df[time_column], errors='coerce')
        aggregated_df = aggregated_df.set_index(time_column)
        
        # 确定要聚合的列
        if value_columns is None:
            # 自动检测情感列
            sentiment_patterns = ['sentiment', 'positive', 'negative', 'neutral', 'intensity', 'subjectivity']
            value_columns = [col for col in aggregated_df.columns
                             if any(pattern in col.lower() for pattern in sentiment_patterns)
                             and pd.api.types.is_numeric_dtype(aggregated_df[col])]
        
        if not value_columns:
            self.logger.warning("未找到情感列，使用所有数值列")
            value_columns = aggregated_df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"将聚合 {len(value_columns)} 个列: {value_columns[:5]}...")
        
        # 确保只聚合数值列（额外的安全检查）
        numeric_columns = [col for col in value_columns if pd.api.types.is_numeric_dtype(aggregated_df[col])]
        if len(numeric_columns) < len(value_columns):
            non_numeric = [col for col in value_columns if not pd.api.types.is_numeric_dtype(aggregated_df[col])]
            self.logger.warning(f"跳过 {len(non_numeric)} 个非数值列: {non_numeric[:3]}")
            value_columns = numeric_columns
        
        if not value_columns:
            self.logger.error("没有数值列可供聚合")
            return pd.DataFrame()
        
        # 按频率聚合
        if aggregation_method == 'mean':
            aggregated = aggregated_df[value_columns].resample(frequency).mean(numeric_only=True)
        elif aggregation_method == 'median':
            aggregated = aggregated_df[value_columns].resample(frequency).median(numeric_only=True)
        elif aggregation_method == 'sum':
            aggregated = aggregated_df[value_columns].resample(frequency).sum(numeric_only=True)
        elif aggregation_method == 'last':
            aggregated = aggregated_df[value_columns].resample(frequency).last()
        elif aggregation_method == 'first':
            aggregated = aggregated_df[value_columns].resample(frequency).first()
        else:
            self.logger.warning(f"未知聚合方法: {aggregation_method}，使用平均值")
            aggregated = aggregated_df[value_columns].resample(frequency).mean(numeric_only=True)
        
        # 填充缺失值
        aggregated = aggregated.ffill().bfill()
        
        # 添加聚合统计信息
        aggregated = self._add_aggregation_stats(aggregated, aggregated_df[value_columns], frequency)
        
        self.logger.info(f"时间聚合完成: {len(aggregated)} 个时间点")
        return aggregated.reset_index()
    
    def _add_aggregation_stats(self, 
                              aggregated_df: pd.DataFrame,
                              original_df: pd.DataFrame,
                              frequency: str) -> pd.DataFrame:
        """
        添加聚合统计信息
        
        Args:
            aggregated_df: 聚合后的DataFrame
            original_df: 原始DataFrame
            frequency: 聚合频率
            
        Returns:
            添加统计信息后的DataFrame
        """
        result_df = aggregated_df.copy()
        
        # 为每个聚合窗口添加原始数据点数量
        for col in aggregated_df.columns:
            # 计算每个聚合窗口中的原始数据点数量
            counts = original_df[col].resample(frequency).count()
            result_df[f'{col}_count'] = counts
            
            # 计算每个聚合窗口中的标准差
            stds = original_df[col].resample(frequency).std()
            result_df[f'{col}_std'] = stds
            
            # 计算变异系数（标准差/均值）
            with np.errstate(divide='ignore', invalid='ignore'):
                cv = stds / aggregated_df[col]
                cv = cv.replace([np.inf, -np.inf], np.nan)
                result_df[f'{col}_cv'] = cv
        
        return result_df
    
    def create_multi_timeframe_features(self,
                                       df: pd.DataFrame,
                                       time_column: str = 'date',
                                       value_column: str = 'sentiment_score',
                                       windows: Optional[List[str]] = None) -> pd.DataFrame:
        """
        创建多时间框架特征
        
        提案要求：多时间框架情感聚合
        
        Args:
            df: 输入DataFrame
            time_column: 时间列名
            value_column: 要聚合的值列
            windows: 时间窗口列表（如['1H', '4H', '1D', '1W']）
            
        Returns:
            添加多时间框架特征后的DataFrame
        """
        self.logger.info(f"创建多时间框架特征，值列: {value_column}")
        
        if df.empty or value_column not in df.columns:
            self.logger.error(f"输入数据为空或缺少列: {value_column}")
            return df
        
        if windows is None:
            windows = self.default_windows
        
        result_df = df.copy()
        
        # 确保时间列正确
        result_df[time_column] = pd.to_datetime(result_df[time_column], errors='coerce')
        result_df = result_df.set_index(time_column)
        result_df = result_df.sort_index()
        
        # 为每个时间窗口创建特征
        for window in windows:
            self.logger.info(f"  处理时间窗口: {window}")
            
            # 滚动统计特征
            try:
                # 滚动平均值
                result_df[f'{value_column}_ma_{window}'] = (
                    result_df[value_column].rolling(window).mean()
                )
                
                # 滚动标准差
                result_df[f'{value_column}_std_{window}'] = (
                    result_df[value_column].rolling(window).std()
                )
                
                # 滚动最小值
                result_df[f'{value_column}_min_{window}'] = (
                    result_df[value_column].rolling(window).min()
                )
                
                # 滚动最大值
                result_df[f'{value_column}_max_{window}'] = (
                    result_df[value_column].rolling(window).max()
                )
                
                # 滚动分位数
                result_df[f'{value_column}_q25_{window}'] = (
                    result_df[value_column].rolling(window).quantile(0.25)
                )
                result_df[f'{value_column}_q75_{window}'] = (
                    result_df[value_column].rolling(window).quantile(0.75)
                )
                
                # 滚动变化率
                result_df[f'{value_column}_pct_change_{window}'] = (
                    result_df[value_column].pct_change(periods=self._get_periods_from_freq(window))
                )
                
            except Exception as e:
                self.logger.warning(f"创建窗口 {window} 的特征时出错: {e}")
        
        # 填充缺失值
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"多时间框架特征创建完成: 添加了 {len(windows) * 7} 个特征")
        return result_df.reset_index()
    
    def _get_periods_from_freq(self, frequency: str) -> int:
        """
        从频率字符串获取周期数
        
        Args:
            frequency: 频率字符串（如'1D', '1W'）
            
        Returns:
            周期数
        """
        # 简单映射，实际应用可能需要更复杂的逻辑
        freq_map = {
            '1H': 1, '4H': 4, '8H': 8, '12H': 12,
            '1D': 1, '2D': 2, '3D': 3, '5D': 5,
            '1W': 5, '2W': 10, '4W': 20,
            '1M': 21, '3M': 63, '6M': 126
        }
        
        return freq_map.get(frequency, 1)
    
    def extract_temporal_features(self,
                                 df: pd.DataFrame,
                                 time_column: str = 'date',
                                 value_columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        提取时间序列特征
        
        Args:
            df: 输入DataFrame
            time_column: 时间列名
            value_columns: 值列列表
            
        Returns:
            添加时间特征后的DataFrame
        """
        self.logger.info("提取时间序列特征")
        
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # 确保时间列正确
        if time_column in result_df.columns:
            result_df[time_column] = pd.to_datetime(result_df[time_column], errors='coerce')
            
            # 提取时间特征
            result_df['year'] = result_df[time_column].dt.year
            result_df['month'] = result_df[time_column].dt.month
            result_df['day'] = result_df[time_column].dt.day
            result_df['dayofweek'] = result_df[time_column].dt.dayofweek
            result_df['dayofyear'] = result_df[time_column].dt.dayofyear
            result_df['weekofyear'] = result_df[time_column].dt.isocalendar().week
            result_df['quarter'] = result_df[time_column].dt.quarter
            result_df['is_month_start'] = result_df[time_column].dt.is_month_start.astype(int)
            result_df['is_month_end'] = result_df[time_column].dt.is_month_end.astype(int)
            result_df['is_quarter_start'] = result_df[time_column].dt.is_quarter_start.astype(int)
            result_df['is_quarter_end'] = result_df[time_column].dt.is_quarter_end.astype(int)
            result_df['is_year_start'] = result_df[time_column].dt.is_year_start.astype(int)
            result_df['is_year_end'] = result_df[time_column].dt.is_year_end.astype(int)
        
        # 如果指定了值列，提取统计特征
        if value_columns:
            for col in value_columns:
                if col in result_df.columns:
                    # 滞后特征
                    for lag in [1, 2, 3, 5, 10]:
                        result_df[f'{col}_lag{lag}'] = result_df[col].shift(lag)
                    
                    # 差分特征
                    for diff in [1, 2, 3]:
                        result_df[f'{col}_diff{diff}'] = result_df[col].diff(diff)
                    
                    # 滚动统计（短、中、长窗口）
                    for window in [5, 10, 20]:
                        result_df[f'{col}_ma{window}'] = result_df[col].rolling(window).mean()
                        result_df[f'{col}_std{window}'] = result_df[col].rolling(window).std()
                        result_df[f'{col}_min{window}'] = result_df[col].rolling(window).min()
                        result_df[f'{col}_max{window}'] = result_df[col].rolling(window).max()
                    
                    # 扩展窗口统计
                    result_df[f'{col}_expanding_mean'] = result_df[col].expanding().mean()
                    result_df[f'{col}_expanding_std'] = result_df[col].expanding().std()
        
        # 填充缺失值
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"时间序列特征提取完成: 添加了 {len(result_df.columns) - len(df.columns)} 个特征")
        return result_df
    
    def align_with_trading_schedule(self,
                                   sentiment_df: pd.DataFrame,
                                   trading_schedule: pd.DataFrame,
                                   sentiment_time_column: str = 'published_at',
                                   trading_time_column: str = 'date',
                                   method: str = 'nearest') -> pd.DataFrame:
        """
        将情感数据与交易时间表对齐
        
        提案要求：情感信号的时间聚合以匹配交易决策周期
        
        Args:
            sentiment_df: 情感DataFrame
            trading_schedule: 交易时间表DataFrame
            sentiment_time_column: 情感时间列
            trading_time_column: 交易时间列
            method: 对齐方法（'nearest', 'forward', 'backward'）
            
        Returns:
            对齐后的情感DataFrame
        """
        self.logger.info("将情感数据与交易时间表对齐")
        
        if sentiment_df.empty or trading_schedule.empty:
            self.logger.error("输入数据为空")
            return sentiment_df if not trading_schedule.empty else pd.DataFrame()
        
        # 确保时间列正确
        sentiment_df = sentiment_df.copy()
        trading_schedule = trading_schedule.copy()
        
        sentiment_df[sentiment_time_column] = pd.to_datetime(
            sentiment_df[sentiment_time_column], errors='coerce'
        )
        trading_schedule[trading_time_column] = pd.to_datetime(
            trading_schedule[trading_time_column], errors='coerce'
        )
        
        # 设置索引
        sentiment_df = sentiment_df.set_index(sentiment_time_column)
        trading_times = trading_schedule[trading_time_column].values
        
        # 对齐数据
        aligned_data = []
        
        for trading_time in trading_times:
            if method == 'nearest':
                # 找到最近的情感数据点
                time_diff = abs(sentiment_df.index - trading_time)
                nearest_idx = time_diff.argmin() if len(time_diff) > 0 else None
                
            elif method == 'forward':
                # 找到之后的情感数据点
                future_sentiment = sentiment_df[sentiment_df.index >= trading_time]
                nearest_idx = future_sentiment.index[0] if len(future_sentiment) > 0 else None
                
            elif method == 'backward':
                # 找到之前的情感数据点
                past_sentiment = sentiment_df[sentiment_df.index <= trading_time]
                nearest_idx = past_sentiment.index[-1] if len(past_sentiment) > 0 else None
                
            else:
                self.logger.warning(f"未知对齐方法: {method}，使用最近方法")
                time_diff = abs(sentiment_df.index - trading_time)
                nearest_idx = time_diff.argmin() if len(time_diff) > 0 else None
            
            if nearest_idx is not None:
                # 获取情感数据
                sentiment_row = sentiment_df.loc[nearest_idx].copy()
                sentiment_row[trading_time_column] = trading_time
                sentiment_row['time_diff_hours'] = (trading_time - nearest_idx).total_seconds() / 3600
                aligned_data.append(sentiment_row)
        
        aligned_df = pd.DataFrame(aligned_data)
        
        if not aligned_df.empty:
            aligned_df = aligned_df.sort_values(trading_time_column)
            self.logger.info(f"对齐完成: {len(aligned_df)}/{len(trading_times)} 个交易时间点")
        else:
            self.logger.warning("对齐后无数据")
        
        return aligned_df
    
    def calculate_sentiment_momentum(self,
                                    df: pd.DataFrame,
                                    value_column: str = 'sentiment_score',
                                    windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        计算情感动量指标
        
        Args:
            df: 输入DataFrame
            value_column: 值列
            windows: 动量窗口列表
            
        Returns:
            添加动量特征后的DataFrame
        """
        self.logger.info(f"计算情感动量指标，值列: {value_column}")
        
        if df.empty or value_column not in df.columns:
            return df
        
        result_df = df.copy()
        
        for window in windows:
            # 简单动量（当前值 - N天前值）
            result_df[f'{value_column}_momentum_{window}'] = (
                result_df[value_column] - result_df[value_column].shift(window)
            )
            
            # 动量百分比变化
            result_df[f'{value_column}_mom_pct_{window}'] = (
                result_df[value_column].pct_change(periods=window)
            )
            
            # 动量方向（1上升，-1下降，0持平）
            result_df[f'{value_column}_mom_dir_{window}'] = np.sign(
                result_df[f'{value_column}_momentum_{window}']
            )
            
            # 动量加速度（动量的变化）
            if window > 1:
                result_df[f'{value_column}_accel_{window}'] = (
                    result_df[f'{value_column}_momentum_{window}'] - 
                    result_df[f'{value_column}_momentum_{window}'].shift(1)
                )
        
        # 相对强度指数（RSI）风格指标
        for window in [14, 21, 28]:
            delta = result_df[value_column].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
            
            result_df[f'{value_column}_rsi_{window}'] = rsi
        
        # 填充缺失值
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"情感动量计算完成: 添加了 {len(windows)*3 + 3} 个动量特征")
        return result_df


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建情感聚合器
    aggregator = SentimentAggregator(logger=logger)
    
    # 创建示例情感数据
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    sentiment_data = pd.DataFrame({
        'published_at': dates,
        'sentiment_score': np.random.uniform(-1, 1, len(dates)),
        'sentiment_intensity': np.random.uniform(0, 1, len(dates))
    })
    
    print("情感时间聚合器测试")
    print("=" * 50)
    
    print(f"原始数据形状: {sentiment_data.shape}")
    print(f"时间范围: {sentiment_data['published_at'].min()} 到 {sentiment_data['published_at'].max()}")
    
    # 按天聚合
    daily_aggregated = aggregator.aggregate_by_time(
        sentiment_data,
        time_column='published_at',
        frequency='1D',
        aggregation_method='mean'
    )
    
    print(f"\n按天聚合后形状: {daily_aggregated.shape}")
    print(f"聚合后时间点数量: {len(daily_aggregated)}")
    print(f"聚合后列: {list(daily_aggregated.columns)}")
    
    # 多时间框架特征
    multi_timeframe = aggregator.create_multi_timeframe_features(
        daily_aggregated,
        time_column='published_at',
        value_column='sentiment_score',
        windows=['1D', '3D', '7D']
    )
    
    print(f"\n多时间框架特征后形状: {multi_timeframe.shape}")
    print(f"添加的特征示例:")
    new_cols = [col for col in multi_timeframe.columns if col not in daily_aggregated.columns]
    for col in new_cols[:10]:
        print(f"  {col}")
    
    # 时间序列特征
    temporal_features = aggregator.extract_temporal_features(
        daily_aggregated,
        time_column='published_at',
        value_columns=['sentiment_score', 'sentiment_intensity']
    )
    
    print(f"\n时间序列特征后形状: {temporal_features.shape}")
    print(f"时间特征示例: year, month, dayofweek 等")
    
    # 创建交易时间表（示例）
    trading_dates = pd.date_range(start='2024-01-01', end='2024-01-15', freq='B')  # 工作日
    trading_schedule = pd.DataFrame({
        'date': trading_dates,
        'market_open': True
    })
    
    # 对齐情感数据与交易时间表
    aligned_sentiment = aggregator.align_with_trading_schedule(
        sentiment_data,
        trading_schedule,
        sentiment_time_column='published_at',
        trading_time_column='date',
        method='nearest'
    )
    
    print(f"\n对齐后情感数据形状: {aligned_sentiment.shape if aligned_sentiment is not None else 0}")
    if aligned_sentiment is not None and not aligned_sentiment.empty:
        print(f"对齐后时间范围: {aligned_sentiment['date'].min()} 到 {aligned_sentiment['date'].max()}")
    
    # 计算情感动量
    momentum_features = aggregator.calculate_sentiment_momentum(
        daily_aggregated,
        value_column='sentiment_score',
        windows=[3, 7, 14]
    )
    
    print(f"\n情感动量特征后形状: {momentum_features.shape}")
    momentum_cols = [col for col in momentum_features.columns if 'momentum' in col or 'rsi' in col]
    print(f"动量特征示例: {momentum_cols[:5]}")
    
    print("\n情感时间聚合模块测试完成!")