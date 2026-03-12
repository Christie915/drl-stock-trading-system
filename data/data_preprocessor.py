"""
数据预处理模块 - 金融数据清洗与转换

根据提案要求，实现以下功能：
1. 稳健的数据清理、缺失值处理和时间对齐
2. 适用于金融时间序列的平稳性转换和归一化
3. 异常值检测和处理
4. 时间对齐，确保情感和价格数据同步

作者：大狗（电子兄弟）
日期：2026-03-12
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    金融数据预处理器
    
    提案要求：
    - 使用前向填充和插值方法进行缺失数据插补
    - 使用稳健统计方法进行异常值检测和处理
    - 适用于金融时间序列的平稳性转换和归一化
    - 时间对齐，确保情感和价格数据同步
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化数据预处理器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 配置参数
        self.missing_threshold = self.config.get('missing_threshold', 0.3)
        self.outlier_zscore = self.config.get('outlier_zscore', 3.0)
        self.min_required_data = self.config.get('min_required_data', 0.7)
        
        self.logger.info("数据预处理器初始化完成")
    
    def clean_stock_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理股票数据
        
        Args:
            df: 原始股票数据
            
        Returns:
            清理后的股票数据
        """
        self.logger.info(f"清理股票数据: 原始形状 {df.shape}")
        
        if df.empty:
            self.logger.warning("输入数据为空")
            return df
        
        # 创建副本以避免修改原始数据
        cleaned_df = df.copy()
        
        # 1. 检查并处理缺失值
        cleaned_df = self._handle_missing_values(cleaned_df, data_type='stock')
        
        # 2. 检查并处理重复值
        cleaned_df = self._handle_duplicates(cleaned_df)
        
        # 3. 检查并处理异常值
        cleaned_df = self._handle_outliers(cleaned_df, data_type='stock')
        
        # 4. 验证数据完整性
        self._validate_data(cleaned_df, data_type='stock')
        
        self.logger.info(f"股票数据清理完成: 最终形状 {cleaned_df.shape}")
        return cleaned_df
    
    def clean_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理新闻数据
        
        Args:
            df: 原始新闻数据
            
        Returns:
            清理后的新闻数据
        """
        self.logger.info(f"清理新闻数据: 原始形状 {df.shape}")
        
        if df.empty:
            self.logger.warning("输入数据为空")
            return df
        
        # 创建副本
        cleaned_df = df.copy()
        
        # 1. 确保日期列存在且正确
        date_cols = ['published_at', 'date', 'timestamp', 'created_at']
        date_col = None
        for col in date_cols:
            if col in cleaned_df.columns:
                date_col = col
                break
        
        if date_col:
            # 转换为datetime
            cleaned_df[date_col] = pd.to_datetime(cleaned_df[date_col], errors='coerce')
            # 按日期排序
            cleaned_df = cleaned_df.sort_values(date_col)
        else:
            # 添加默认日期列
            cleaned_df['processed_date'] = pd.Timestamp.now()
            date_col = 'processed_date'
            self.logger.warning("未找到日期列，添加默认日期")
        
        # 2. 处理缺失的文本内容
        text_cols = ['title', 'description', 'content', 'text']
        for col in text_cols:
            if col in cleaned_df.columns:
                # 填充缺失的文本
                cleaned_df[col] = cleaned_df[col].fillna('')
        
        # 3. 移除完全空白的记录
        before_len = len(cleaned_df)
        if 'title' in cleaned_df.columns and 'content' in cleaned_df.columns:
            mask = (cleaned_df['title'].str.strip() != '') | (cleaned_df['content'].str.strip() != '')
            cleaned_df = cleaned_df[mask]
        
        # 4. 移除重复新闻（基于标题和日期）
        duplicate_cols = ['title', date_col] if 'title' in cleaned_df.columns else [date_col]
        cleaned_df = cleaned_df.drop_duplicates(subset=duplicate_cols, keep='first')
        
        self.logger.info(f"新闻数据清理完成: 原始 {before_len} → 最终 {len(cleaned_df)} 条")
        return cleaned_df
    
    def clean_social_media_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清理社交媒体数据
        
        Args:
            df: 原始社交媒体数据
            
        Returns:
            清理后的社交媒体数据
        """
        self.logger.info(f"清理社交媒体数据: 原始形状 {df.shape}")
        
        if df.empty:
            self.logger.warning("输入数据为空")
            return df
        
        # 创建副本
        cleaned_df = df.copy()
        
        # 1. 确保日期列存在
        if 'created_at' in cleaned_df.columns:
            cleaned_df['created_at'] = pd.to_datetime(cleaned_df['created_at'], errors='coerce')
            cleaned_df = cleaned_df.sort_values('created_at')
        
        # 2. 处理缺失文本
        if 'text' in cleaned_df.columns:
            cleaned_df['text'] = cleaned_df['text'].fillna('')
        
        # 3. 移除空文本
        before_len = len(cleaned_df)
        if 'text' in cleaned_df.columns:
            cleaned_df = cleaned_df[cleaned_df['text'].str.strip() != '']
        
        # 4. 移除重复帖子（基于文本和日期）
        if 'text' in cleaned_df.columns and 'created_at' in cleaned_df.columns:
            cleaned_df = cleaned_df.drop_duplicates(subset=['text', 'created_at'], keep='first')
        
        self.logger.info(f"社交媒体数据清理完成: 原始 {before_len} → 最终 {len(cleaned_df)} 条")
        return cleaned_df
    
    def _handle_missing_values(self, df: pd.DataFrame, data_type: str = 'stock') -> pd.DataFrame:
        """
        处理缺失值
        
        提案要求：使用前向填充和插值方法进行缺失数据插补
        
        Args:
            df: 输入数据
            data_type: 数据类型（'stock', 'news', 'social'）
            
        Returns:
            处理缺失值后的数据
        """
        missing_before = df.isnull().sum().sum()
        
        if missing_before == 0:
            return df
        
        self.logger.info(f"处理缺失值: 缺失值数量 {missing_before}")
        
        if data_type == 'stock':
            # 对于股票数据，使用时间序列方法
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # 先尝试前向填充（适用于连续时间序列）
                df[col] = df[col].ffill()
                # 然后后向填充
                df[col] = df[col].bfill()
                # 如果仍有缺失，使用线性插值
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            
            # 对于非数值列，使用众数或特定值填充
            for col in df.columns:
                if col not in numeric_cols and df[col].isnull().any():
                    if df[col].dtype == 'object':
                        # 使用前一个有效值填充
                        df[col] = df[col].ffill().bfill()
        
        else:
            # 对于其他类型数据，简单填充
            df = df.ffill().bfill()
        
        missing_after = df.isnull().sum().sum()
        self.logger.info(f"缺失值处理完成: {missing_before} → {missing_after}")
        
        # 如果仍有缺失值，删除包含缺失值的行
        if missing_after > 0:
            before_len = len(df)
            df = df.dropna()
            self.logger.warning(f"删除包含缺失值的行: {before_len} → {len(df)}")
        
        return df
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理重复值
        
        Args:
            df: 输入数据
            
        Returns:
            去重后的数据
        """
        duplicates = df.duplicated().sum()
        
        if duplicates > 0:
            self.logger.info(f"发现 {duplicates} 个重复值，正在处理...")
            
            # 对于有时间索引的数据，优先保持时间序列完整性
            if isinstance(df.index, pd.DatetimeIndex):
                # 按时间排序后去重
                df = df[~df.index.duplicated(keep='first')]
            else:
                # 普通去重
                df = df.drop_duplicates()
            
            self.logger.info(f"重复值处理完成，移除 {duplicates} 个重复值")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, data_type: str = 'stock') -> pd.DataFrame:
        """
        处理异常值
        
        提案要求：使用稳健统计方法进行异常值检测和处理
        
        Args:
            df: 输入数据
            data_type: 数据类型
            
        Returns:
            处理异常值后的数据
        """
        if data_type != 'stock':
            # 对于非股票数据，不处理异常值
            return df
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_counts = {}
        
        for col in numeric_cols:
            if col in ['Volume', 'volume']:
                # 对于成交量，使用对数处理
                if (df[col] <= 0).any():
                    # 避免对数负值或零值
                    continue
                
                log_values = np.log(df[col])
                z_scores = np.abs((log_values - log_values.mean()) / log_values.std())
                outliers = z_scores > self.outlier_zscore
                
            else:
                # 使用Z-score方法检测异常值
                z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
                outliers = z_scores > self.outlier_zscore
            
            outlier_count = outliers.sum()
            if outlier_count > 0:
                outlier_counts[col] = outlier_count
                
                # 处理异常值：使用移动平均填充
                if outlier_count < len(df) * 0.1:  # 异常值少于10%
                    # 使用前后值的平均值填充
                    df.loc[outliers, col] = np.nan
                    df[col] = df[col].interpolate(method='linear')
                else:
                    # 异常值太多，使用中位数
                    median_val = df[col].median()
                    df.loc[outliers, col] = median_val
        
        if outlier_counts:
            self.logger.info(f"异常值处理: {outlier_counts}")
        
        return df
    
    def _validate_data(self, df: pd.DataFrame, data_type: str = 'stock'):
        """
        验证数据完整性
        
        Args:
            df: 要验证的数据
            data_type: 数据类型
        """
        if df.empty:
            self.logger.error(f"{data_type}数据验证失败: 数据为空")
            return
        
        # 检查基本统计信息
        self.logger.info(f"{data_type}数据验证:")
        self.logger.info(f"  数据形状: {df.shape}")
        self.logger.info(f"  时间范围: {df.index.min() if hasattr(df.index, 'min') else 'N/A'} 到 "
                        f"{df.index.max() if hasattr(df.index, 'max') else 'N/A'}")
        
        # 检查缺失值
        missing = df.isnull().sum().sum()
        if missing > 0:
            self.logger.warning(f"  仍有 {missing} 个缺失值")
        
        # 检查数值列的统计信息
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            self.logger.info(f"  数值列: {len(numeric_cols)}个")
            
            # 检查极端值
            for col in numeric_cols[:3]:  # 只显示前3个列
                if len(df[col]) > 0:
                    self.logger.info(f"    {col}: 均值={df[col].mean():.4f}, "
                                    f"标准差={df[col].std():.4f}, 范围=[{df[col].min():.4f}, {df[col].max():.4f}]")
    
    def align_time_series(self, 
                         price_df: pd.DataFrame,
                         sentiment_df: pd.DataFrame,
                         frequency: str = '1D') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        时间序列对齐
        
        提案要求：时间对齐，确保情感和价格数据同步
        
        Args:
            price_df: 价格数据
            sentiment_df: 情感数据
            frequency: 重采样频率
            
        Returns:
            对齐后的价格和情感数据
        """
        self.logger.info("时间序列对齐...")
        
        # 确保两个DataFrame都有时间索引
        price_aligned = price_df.copy()
        sentiment_aligned = sentiment_df.copy()
        
        # 识别时间列
        price_time_col = None
        sentiment_time_col = None
        
        # 查找价格数据的时间列
        if isinstance(price_aligned.index, pd.DatetimeIndex):
            price_time_col = 'index'
        else:
            # 查找日期列
            date_cols = ['Date', 'date', 'timestamp', 'time']
            for col in date_cols:
                if col in price_aligned.columns:
                    price_time_col = col
                    price_aligned[col] = pd.to_datetime(price_aligned[col], errors='coerce')
                    price_aligned.set_index(col, inplace=True)
                    break
        
        # 查找情感数据的时间列
        if isinstance(sentiment_aligned.index, pd.DatetimeIndex):
            sentiment_time_col = 'index'
        else:
            date_cols = ['published_at', 'created_at', 'date', 'timestamp']
            for col in date_cols:
                if col in sentiment_aligned.columns:
                    sentiment_time_col = col
                    sentiment_aligned[col] = pd.to_datetime(sentiment_aligned[col], errors='coerce')
                    sentiment_aligned.set_index(col, inplace=True)
                    break
        
        if not price_time_col or not sentiment_time_col:
            self.logger.error("无法找到时间列进行对齐")
            return price_df, sentiment_df
        
        self.logger.info(f"价格时间列: {price_time_col}, 情感时间列: {sentiment_time_col}")
        
        # 统一时区为UTC以便比较
        def unify_timezone_to_utc(df_index):
            """将时间索引统一为UTC时区"""
            if df_index.tz is None:
                # 如果没有时区信息，假设为UTC
                return df_index.tz_localize('UTC', ambiguous='infer', nonexistent='shift_forward')
            else:
                # 如果有时区信息，转换为UTC
                return df_index.tz_convert('UTC')
        
        # 转换价格数据索引为UTC
        try:
            price_aligned.index = unify_timezone_to_utc(price_aligned.index)
            self.logger.debug(f"价格数据索引转换为UTC时区")
        except Exception as e:
            self.logger.warning(f"价格数据时区转换失败，使用原有时区: {e}")
        
        # 转换情感数据索引为UTC
        try:
            sentiment_aligned.index = unify_timezone_to_utc(sentiment_aligned.index)
            self.logger.debug(f"情感数据索引转换为UTC时区")
        except Exception as e:
            self.logger.warning(f"情感数据时区转换失败，使用原有时区: {e}")
        
        # 获取时间范围（转换后）
        price_start = price_aligned.index.min()
        price_end = price_aligned.index.max()
        sentiment_start = sentiment_aligned.index.min()
        sentiment_end = sentiment_aligned.index.max()
        
        # 处理时区问题：转换为naive时间（去掉时区）以便比较
        def make_timezone_naive(dt):
            if hasattr(dt, 'tz') and dt.tz is not None:
                return dt.tz_localize(None) if dt.tz is not None else dt
            return dt
        
        price_start_naive = make_timezone_naive(price_start)
        price_end_naive = make_timezone_naive(price_end)
        sentiment_start_naive = make_timezone_naive(sentiment_start)
        sentiment_end_naive = make_timezone_naive(sentiment_end)
        
        self.logger.info(f"价格时间范围: {price_start} 到 {price_end}")
        self.logger.info(f"情感时间范围: {sentiment_start} 到 {sentiment_end}")
        
        # 确定共同时间范围（使用naive时间比较）
        common_start = max(price_start_naive, sentiment_start_naive)
        common_end = min(price_end_naive, sentiment_end_naive)
        
        if common_start >= common_end:
            self.logger.error("时间范围无重叠")
            return price_df, sentiment_df
        
        self.logger.info(f"共同时间范围: {common_start} 到 {common_end}")
        
        # 将共同时间范围转换回原始时区用于裁剪
        # 对于裁剪，我们需要使用原始时区的时间戳
        # 所以需要找到原始时间索引中对应的实际时间戳
        def find_closest_time(naive_time, time_index):
            """在时间索引中找到最接近naive_time的时间戳"""
            # 首先尝试直接匹配（忽略时区）
            for ts in time_index:
                ts_naive = make_timezone_naive(ts)
                if ts_naive == naive_time:
                    return ts
            # 如果找不到精确匹配，找到最接近的
            return min(time_index, key=lambda x: abs(make_timezone_naive(x) - naive_time))
        
        # 找到实际的时间戳进行裁剪
        price_common_start = find_closest_time(common_start, price_aligned.index)
        price_common_end = find_closest_time(common_end, price_aligned.index)
        sentiment_common_start = find_closest_time(common_start, sentiment_aligned.index)
        sentiment_common_end = find_closest_time(common_end, sentiment_aligned.index)
        
        # 裁剪到共同时间范围（使用实际找到的时间戳）
        price_aligned = price_aligned[(price_aligned.index >= price_common_start) & 
                                      (price_aligned.index <= price_common_end)]
        sentiment_aligned = sentiment_aligned[(sentiment_aligned.index >= sentiment_common_start) & 
                                              (sentiment_aligned.index <= sentiment_common_end)]
        
        # 重采样到相同频率
        self.logger.info(f"重采样到频率: {frequency}")
        
        # 价格数据重采样（使用最后值）
        price_resampled = price_aligned.resample(frequency).last()
        
        # 情感数据重采样（使用平均值）
        sentiment_numeric = sentiment_aligned.select_dtypes(include=[np.number])
        if not sentiment_numeric.empty:
            sentiment_resampled = sentiment_numeric.resample(frequency).mean()
            
            # 合并非数值列（使用最后值）
            sentiment_non_numeric = sentiment_aligned.select_dtypes(exclude=[np.number])
            if not sentiment_non_numeric.empty:
                non_numeric_resampled = sentiment_non_numeric.resample(frequency).last()
                sentiment_resampled = pd.concat([sentiment_resampled, non_numeric_resampled], axis=1)
        else:
            # 如果没有数值列，直接重采样
            sentiment_resampled = sentiment_aligned.resample(frequency).last()
        
        # 对齐索引
        aligned_index = price_resampled.index.intersection(sentiment_resampled.index)
        
        price_final = price_resampled.loc[aligned_index]
        sentiment_final = sentiment_resampled.loc[aligned_index]
        
        self.logger.info(f"对齐完成: 价格形状 {price_final.shape}, 情感形状 {sentiment_final.shape}")
        
        return price_final, sentiment_final
    
    def normalize_series(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        序列标准化
        
        提案要求：适用于金融时间序列的平稳性转换和归一化
        
        Args:
            df: 输入数据
            method: 标准化方法（'zscore', 'minmax', 'robust', 'log', 'return'）
            
        Returns:
            标准化后的数据
        """
        self.logger.info(f"标准化数据，方法: {method}")
        
        if df.empty:
            return df
        
        normalized_df = df.copy()
        numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            # Z-score标准化（均值0，标准差1）
            for col in numeric_cols:
                mean_val = normalized_df[col].mean()
                std_val = normalized_df[col].std()
                if std_val > 0:
                    normalized_df[col] = (normalized_df[col] - mean_val) / std_val
        
        elif method == 'minmax':
            # 最小-最大标准化（范围0-1）
            for col in numeric_cols:
                min_val = normalized_df[col].min()
                max_val = normalized_df[col].max()
                if max_val > min_val:
                    normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
        
        elif method == 'robust':
            # 稳健标准化（基于中位数和IQR）
            for col in numeric_cols:
                median_val = normalized_df[col].median()
                iqr_val = normalized_df[col].quantile(0.75) - normalized_df[col].quantile(0.25)
                if iqr_val > 0:
                    normalized_df[col] = (normalized_df[col] - median_val) / iqr_val
        
        elif method == 'log':
            # 对数转换（适用于价格数据）
            for col in numeric_cols:
                if (normalized_df[col] > 0).all():
                    normalized_df[col] = np.log(normalized_df[col])
                else:
                    # 如果有非正值，使用偏移
                    min_val = normalized_df[col].min()
                    if min_val <= 0:
                        offset = abs(min_val) + 1
                        normalized_df[col] = np.log(normalized_df[col] + offset)
        
        elif method == 'return':
            # 收益率转换（金融时间序列常用）
            for col in numeric_cols:
                normalized_df[col] = normalized_df[col].pct_change()
                # 填充第一个NaN值
                normalized_df[col] = normalized_df[col].fillna(0)
        
        else:
            self.logger.warning(f"未知标准化方法: {method}，使用Z-score")
            return self.normalize_series(df, method='zscore')
        
        self.logger.info(f"标准化完成，方法: {method}")
        return normalized_df
    
    def create_lagged_features(self, df: pd.DataFrame, lags: List[int], columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        创建滞后特征
        
        Args:
            df: 输入数据
            lags: 滞后阶数列表
            columns: 要创建滞后特征的列（None表示所有数值列）
            
        Returns:
            包含滞后特征的数据
        """
        self.logger.info(f"创建滞后特征，滞后阶数: {lags}")
        
        if df.empty:
            return df
        
        result_df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col not in df.columns:
                continue
            
            for lag in lags:
                if lag > 0:
                    lag_col_name = f"{col}_lag{lag}"
                    result_df[lag_col_name] = df[col].shift(lag)
        
        # 删除因滞后产生的NaN行
        before_len = len(result_df)
        result_df = result_df.dropna()
        after_len = len(result_df)
        
        self.logger.info(f"滞后特征创建完成: 添加了 {len(lags) * len(columns)} 个特征, "
                        f"数据 {before_len} → {after_len}")
        
        return result_df
    
    def split_time_series(self, 
                         df: pd.DataFrame,
                         train_ratio: float = 0.7,
                         val_ratio: float = 0.15,
                         test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        时间序列数据划分
        
        提案要求：
        - 训练期（2010-2019年）：模型开发和超参数优化
        - 验证期（2020-2021年）：样本外测试，包括COVID-19市场压力期
        - 测试期（2022-2024年）：对最新数据的最终评估
        
        Args:
            df: 输入数据（应按时间排序）
            train_ratio: 训练集比例
            val_ratio: 验证集比例
            test_ratio: 测试集比例
            
        Returns:
            (训练集, 验证集, 测试集)
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            self.logger.error("数据必须具有DatetimeIndex才能进行时间序列划分")
            return df, pd.DataFrame(), pd.DataFrame()
        
        # 按时间排序
        df = df.sort_index()
        
        total_len = len(df)
        train_end = int(total_len * train_ratio)
        val_end = train_end + int(total_len * val_ratio)
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        self.logger.info(f"时间序列划分完成:")
        self.logger.info(f"  训练集: {train_df.index.min()} 到 {train_df.index.max()} ({len(train_df)} 行)")
        self.logger.info(f"  验证集: {val_df.index.min()} 到 {val_df.index.max()} ({len(val_df)} 行)")
        self.logger.info(f"  测试集: {test_df.index.min()} 到 {test_df.index.max()} ({len(test_df)} 行)")
        
        return train_df, val_df, test_df


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建预处理器
    preprocessor = DataPreprocessor(logger=logger)
    
    # 创建示例数据
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, len(dates)),
        'High': np.random.uniform(100, 200, len(dates)),
        'Low': np.random.uniform(100, 200, len(dates)),
        'Close': np.random.uniform(100, 200, len(dates)),
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })
    
    # 添加一些缺失值和异常值
    price_data.loc[10:15, 'Close'] = np.nan
    price_data.loc[50, 'Close'] = 1000  # 异常值
    
    print("原始数据形状:", price_data.shape)
    print("缺失值数量:", price_data.isnull().sum().sum())
    
    # 清理数据
    cleaned_data = preprocessor.clean_stock_data(price_data)
    
    print("\n清理后数据形状:", cleaned_data.shape)
    print("缺失值数量:", cleaned_data.isnull().sum().sum())
    
    # 创建示例情感数据
    sentiment_dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='H')
    sentiment_data = pd.DataFrame({
        'published_at': sentiment_dates,
        'sentiment_score': np.random.uniform(-1, 1, len(sentiment_dates)),
        'title': [f"News {i}" for i in range(len(sentiment_dates))]
    })
    
    print("\n情感数据形状:", sentiment_data.shape)
    
    # 对齐时间序列
    price_data_ts = price_data.set_index('Date')
    sentiment_data_ts = sentiment_data.set_index('published_at')
    
    price_aligned, sentiment_aligned = preprocessor.align_time_series(
        price_data_ts, sentiment_data_ts, frequency='1D'
    )
    
    print(f"\n对齐后价格数据形状: {price_aligned.shape}")
    print(f"对齐后情感数据形状: {sentiment_aligned.shape}")
    
    # 标准化数据
    normalized_price = preprocessor.normalize_series(price_aligned[['Close']], method='zscore')
    print(f"\n标准化后Close列统计:")
    print(f"  均值: {normalized_price['Close'].mean():.4f}")
    print(f"  标准差: {normalized_price['Close'].std():.4f}")
    
    # 创建滞后特征
    lagged_data = preprocessor.create_lagged_features(
        price_aligned[['Close']], 
        lags=[1, 2, 3, 5, 10]
    )
    print(f"\n滞后特征数据形状: {lagged_data.shape}")
    print("列名:", list(lagged_data.columns))
    
    # 时间序列划分
    if isinstance(price_aligned.index, pd.DatetimeIndex):
        train_df, val_df, test_df = preprocessor.split_time_series(
            price_aligned, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        print(f"\n数据划分:")
        print(f"  训练集: {len(train_df)} 行")
        print(f"  验证集: {len(val_df)} 行")
        print(f"  测试集: {len(test_df)} 行")
    
    print("\n数据预处理模块测试完成!")