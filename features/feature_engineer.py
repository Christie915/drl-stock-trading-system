"""
Feature Engineering Module - Technical indicators and sentiment features

According to proposal requirements, implement:
1. Technical indicators calculation
2. Sentiment feature engineering  
3. Attention mechanism for dynamic feature weighting
4. State space construction for DRL

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Financial feature engineering module
    
    Proposal requirements:
    - Technical indicators (moving averages, RSI, volatility, etc.)
    - Sentiment feature engineering
    - Attention mechanism for dynamic feature weighting
    - State space construction for DRL
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize feature engineer
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Default parameters
        self.tech_indicators = self.config.get('technical_indicators', [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'stochastic',
            'atr', 'volatility', 'momentum', 'volume_indicators'
        ])
        
        self.window_sizes = self.config.get('window_sizes', [5, 10, 20, 50])
        
        self.logger.info("Feature engineering module initialized")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate comprehensive technical indicators
        
        Proposal requirement: Technical indicators for time series analysis
        
        Args:
            df: DataFrame with OHLCV data (must have 'Open', 'High', 'Low', 'Close', 'Volume')
            
        Returns:
            DataFrame with added technical indicators
        """
        self.logger.info("Calculating technical indicators")
        
        if df.empty:
            self.logger.warning("Input DataFrame is empty")
            return df
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            self.logger.error(f"Missing required columns: {missing_cols}")
            return df
        
        result_df = df.copy()
        
        # 1. Price-based indicators
        result_df = self._calculate_price_indicators(result_df)
        
        # 2. Volume-based indicators
        result_df = self._calculate_volume_indicators(result_df)
        
        # 3. Volatility indicators
        result_df = self._calculate_volatility_indicators(result_df)
        
        # 4. Momentum indicators
        result_df = self._calculate_momentum_indicators(result_df)
        
        # 5. Trend indicators
        result_df = self._calculate_trend_indicators(result_df)
        
        # Fill NaN values
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"Technical indicators calculated: added {len(result_df.columns) - len(df.columns)} features")
        
        return result_df
    
    def _calculate_price_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate price-based technical indicators"""
        result_df = df.copy()
        
        # Simple Moving Averages
        for window in self.window_sizes:
            result_df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
        
        # Exponential Moving Averages
        for window in [9, 12, 26]:
            result_df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Weighted Moving Average
        for window in [10, 20]:
            weights = np.arange(1, window + 1)
            result_df[f'WMA_{window}'] = df['Close'].rolling(window).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            )
        
        # Price channels
        for window in [20, 50]:
            result_df[f'High_{window}'] = df['High'].rolling(window).max()
            result_df[f'Low_{window}'] = df['Low'].rolling(window).min()
            result_df[f'Mid_{window}'] = (result_df[f'High_{window}'] + result_df[f'Low_{window}']) / 2
        
        # Typical price and weighted close
        result_df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
        result_df['Weighted_Close'] = (df['High'] + df['Low'] + 2 * df['Close']) / 4
        
        return result_df
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume-based technical indicators"""
        result_df = df.copy()
        
        # Volume moving averages
        for window in [5, 10, 20]:
            result_df[f'Volume_SMA_{window}'] = df['Volume'].rolling(window).mean()
        
        # Volume Price Trend (VPT)
        result_df['VPT'] = 0.0
        for i in range(1, len(df)):
            price_change = (df['Close'].iloc[i] - df['Close'].iloc[i-1]) / df['Close'].iloc[i-1]
            result_df['VPT'].iloc[i] = result_df['VPT'].iloc[i-1] + (price_change * df['Volume'].iloc[i])
        
        # On-Balance Volume (OBV)
        result_df['OBV'] = 0.0
        for i in range(1, len(df)):
            if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
                result_df['OBV'].iloc[i] = result_df['OBV'].iloc[i-1] + df['Volume'].iloc[i]
            elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
                result_df['OBV'].iloc[i] = result_df['OBV'].iloc[i-1] - df['Volume'].iloc[i]
            else:
                result_df['OBV'].iloc[i] = result_df['OBV'].iloc[i-1]
        
        # Volume Ratio
        for window in [5, 10]:
            result_df[f'Volume_Ratio_{window}'] = df['Volume'] / df['Volume'].rolling(window).mean()
        
        # Money Flow Index (simplified)
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        money_flow = typical_price * df['Volume']
        
        positive_flow = money_flow.copy()
        negative_flow = money_flow.copy()
        
        for i in range(1, len(df)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                negative_flow.iloc[i] = 0
            else:
                positive_flow.iloc[i] = 0
        
        for window in [14, 21]:
            positive_sum = positive_flow.rolling(window).sum()
            negative_sum = negative_flow.rolling(window).sum()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                money_ratio = positive_sum / negative_sum
                mfi = 100 - (100 / (1 + money_ratio))
                mfi = mfi.replace([np.inf, -np.inf], np.nan)
            
            result_df[f'MFI_{window}'] = mfi
        
        return result_df
    
    def _calculate_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility indicators"""
        result_df = df.copy()
        
        # True Range
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        result_df['True_Range'] = tr
        
        # Average True Range (ATR)
        for window in [14, 20, 21]:
            result_df[f'ATR_{window}'] = tr.rolling(window).mean()
        
        # Historical Volatility
        returns = df['Close'].pct_change()
        for window in [10, 20, 30]:
            result_df[f'Volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
        
        # Bollinger Bands
        for window in [20, 50]:
            sma = df['Close'].rolling(window).mean()
            std = df['Close'].rolling(window).std()
            
            result_df[f'BB_Middle_{window}'] = sma
            result_df[f'BB_Upper_{window}'] = sma + (2 * std)
            result_df[f'BB_Lower_{window}'] = sma - (2 * std)
            result_df[f'BB_Width_{window}'] = (result_df[f'BB_Upper_{window}'] - result_df[f'BB_Lower_{window}']) / sma
            result_df[f'BB_%B_{window}'] = (df['Close'] - result_df[f'BB_Lower_{window}']) / (
                result_df[f'BB_Upper_{window}'] - result_df[f'BB_Lower_{window}']
            )
        
        # Keltner Channels (simplified)
        for window in [20]:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            kc_middle = typical_price.ewm(span=window).mean()
            kc_range = result_df[f'ATR_{window}']
            
            result_df[f'KC_Middle_{window}'] = kc_middle
            result_df[f'KC_Upper_{window}'] = kc_middle + (2 * kc_range)
            result_df[f'KC_Lower_{window}'] = kc_middle - (2 * kc_range)
        
        return result_df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum indicators"""
        result_df = df.copy()
        
        # Rate of Change (ROC)
        for window in [5, 10, 20]:
            result_df[f'ROC_{window}'] = df['Close'].pct_change(periods=window) * 100
        
        # Relative Strength Index (RSI)
        for window in [14, 21]:
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                rsi = rsi.replace([np.inf, -np.inf], np.nan)
            
            result_df[f'RSI_{window}'] = rsi
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        
        result_df['MACD'] = ema_12 - ema_26
        result_df['MACD_Signal'] = result_df['MACD'].ewm(span=9, adjust=False).mean()
        result_df['MACD_Histogram'] = result_df['MACD'] - result_df['MACD_Signal']
        
        # Stochastic Oscillator
        for window in [14]:
            low_min = df['Low'].rolling(window).min()
            high_max = df['High'].rolling(window).max()
            
            result_df[f'Stoch_K_{window}'] = 100 * ((df['Close'] - low_min) / (high_max - low_min))
            result_df[f'Stoch_D_{window}'] = result_df[f'Stoch_K_{window}'].rolling(3).mean()
        
        # Commodity Channel Index (CCI)
        for window in [20]:
            typical_price = (df['High'] + df['Low'] + df['Close']) / 3
            sma = typical_price.rolling(window).mean()
            mean_dev = (typical_price - sma).abs().rolling(window).mean()
            
            with np.errstate(divide='ignore', invalid='ignore'):
                cci = (typical_price - sma) / (0.015 * mean_dev)
                cci = cci.replace([np.inf, -np.inf], np.nan)
            
            result_df[f'CCI_{window}'] = cci
        
        # Williams %R
        for window in [14]:
            highest_high = df['High'].rolling(window).max()
            lowest_low = df['Low'].rolling(window).min()
            
            result_df[f'Williams_R_{window}'] = -100 * ((highest_high - df['Close']) / (highest_high - lowest_low))
        
        return result_df
    
    def _calculate_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend indicators"""
        result_df = df.copy()
        
        # Average Directional Index (ADX)
        # Simplified version - full ADX calculation is complex
        for window in [14]:
            # Directional Movement
            up_move = df['High'].diff()
            down_move = -df['Low'].diff()
            
            pos_dm = up_move.where((up_move > down_move) & (up_move > 0), 0)
            neg_dm = down_move.where((down_move > up_move) & (down_move > 0), 0)
            
            # True Range (already calculated)
            if 'True_Range' in result_df.columns:
                tr = result_df['True_Range']
            else:
                high_low = df['High'] - df['Low']
                high_close = np.abs(df['High'] - df['Close'].shift())
                low_close = np.abs(df['Low'] - df['Close'].shift())
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            
            # Directional Indicators
            pos_di = 100 * (pos_dm.rolling(window).mean() / tr.rolling(window).mean())
            neg_di = 100 * (neg_dm.rolling(window).mean() / tr.rolling(window).mean())
            
            # ADX (simplified)
            dx = 100 * np.abs(pos_di - neg_di) / (pos_di + neg_di)
            result_df[f'ADX_{window}'] = dx.rolling(window).mean()
            result_df[f'+DI_{window}'] = pos_di
            result_df[f'-DI_{window}'] = neg_di
        
        # Parabolic SAR (simplified)
        # This is a complex indicator, simplified for demo
        result_df['Parabolic_SAR'] = df['Close'].rolling(5).min()
        
        # Ichimoku Cloud (simplified key components)
        high_9 = df['High'].rolling(9).max()
        low_9 = df['Low'].rolling(9).min()
        result_df['Ichimoku_Conversion'] = (high_9 + low_9) / 2
        
        high_26 = df['High'].rolling(26).max()
        low_26 = df['Low'].rolling(26).min()
        result_df['Ichimoku_Base'] = (high_26 + low_26) / 2
        
        result_df['Ichimoku_Lead_A'] = ((result_df['Ichimoku_Conversion'] + result_df['Ichimoku_Base']) / 2).shift(26)
        
        high_52 = df['High'].rolling(52).max()
        low_52 = df['Low'].rolling(52).min()
        result_df['Ichimoku_Lead_B'] = ((high_52 + low_52) / 2).shift(26)
        
        return result_df
    
    def engineer_sentiment_features(self, 
                                   sentiment_df: pd.DataFrame,
                                   price_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Engineer sentiment features
        
        Proposal requirement: Sentiment feature engineering
        
        Args:
            sentiment_df: DataFrame with sentiment scores
            price_df: Optional price data for alignment
            
        Returns:
            DataFrame with engineered sentiment features
        """
        self.logger.info("Engineering sentiment features")
        
        if sentiment_df.empty:
            self.logger.warning("Input sentiment DataFrame is empty")
            return pd.DataFrame()
        
        result_df = sentiment_df.copy()
        
        # Basic sentiment features (if not already present)
        sentiment_cols = [col for col in result_df.columns if 'sentiment' in col.lower()]
        
        if not sentiment_cols:
            self.logger.warning("No sentiment columns found in input")
            return result_df
        
        # 1. Sentiment momentum features
        for col in sentiment_cols:
            if result_df[col].dtype in [np.float64, np.float32, np.int64]:
                # Lagged sentiment
                for lag in [1, 2, 3, 5, 10]:
                    result_df[f'{col}_lag{lag}'] = result_df[col].shift(lag)
                
                # Sentiment changes
                for window in [1, 3, 7]:
                    result_df[f'{col}_change_{window}'] = result_df[col].pct_change(window)
                    result_df[f'{col}_diff_{window}'] = result_df[col].diff(window)
                
                # Rolling statistics
                for window in [5, 10, 20]:
                    result_df[f'{col}_ma_{window}'] = result_df[col].rolling(window).mean()
                    result_df[f'{col}_std_{window}'] = result_df[col].rolling(window).std()
                    result_df[f'{col}_min_{window}'] = result_df[col].rolling(window).min()
                    result_df[f'{col}_max_{window}'] = result_df[col].rolling(window).max()
        
        # 2. Sentiment interactions (if multiple sentiment columns)
        if len(sentiment_cols) > 1:
            # Combine multiple sentiment scores
            for i in range(len(sentiment_cols)):
                for j in range(i + 1, len(sentiment_cols)):
                    col1 = sentiment_cols[i]
                    col2 = sentiment_cols[j]
                    
                    if result_df[col1].dtype in [np.float64, np.float32] and \
                       result_df[col2].dtype in [np.float64, np.float32]:
                        
                        result_df[f'{col1}_{col2}_ratio'] = result_df[col1] / (result_df[col2] + 1e-8)
                        result_df[f'{col1}_{col2}_diff'] = result_df[col1] - result_df[col2]
                        result_df[f'{col1}_{col2}_product'] = result_df[col1] * result_df[col2]
        
        # 3. Sentiment volatility and regime detection
        for col in sentiment_cols:
            if result_df[col].dtype in [np.float64, np.float32]:
                # Volatility
                for window in [10, 20]:
                    result_df[f'{col}_volatility_{window}'] = result_df[col].pct_change().rolling(window).std()
                
                # Regime detection (using z-score)
                for window in [20]:
                    z_score = (result_df[col] - result_df[col].rolling(window).mean()) / \
                              (result_df[col].rolling(window).std() + 1e-8)
                    result_df[f'{col}_zscore_{window}'] = z_score
                    result_df[f'{col}_regime_{window}'] = np.where(z_score > 1, 1, np.where(z_score < -1, -1, 0))
        
        # 4. Sentiment-price alignment (if price data provided)
        if price_df is not None and not price_df.empty:
            if 'Close' in price_df.columns:
                # Align sentiment with price returns
                price_returns = price_df['Close'].pct_change()
                
                for col in sentiment_cols:
                    if result_df[col].dtype in [np.float64, np.float32]:
                        # Correlation with price
                        for window in [10, 20]:
                            # Rolling correlation
                            corr_series = pd.Series(index=result_df.index, dtype=np.float64)
                            
                            for i in range(window, len(result_df)):
                                if i < len(price_returns):
                                    sentiment_window = result_df[col].iloc[i-window:i]
                                    price_window = price_returns.iloc[i-window:i]
                                    
                                    if len(sentiment_window) == len(price_window):
                                        corr = sentiment_window.corr(price_window)
                                        corr_series.iloc[i] = corr if not np.isnan(corr) else 0
                            
                            result_df[f'{col}_price_corr_{window}'] = corr_series
        
        # Fill NaN values
        result_df = result_df.fillna(method='ffill').fillna(method='bfill')
        
        self.logger.info(f"Sentiment features engineered: {len(result_df.columns) - len(sentiment_df.columns)} features added")
        
        return result_df
    
    def create_attention_weights(self, 
                               features_df: pd.DataFrame,
                               attention_type: str = 'dynamic') -> pd.DataFrame:
        """
        Create attention weights for features
        
        Proposal requirement: Attention mechanism for dynamic feature weighting
        
        Args:
            features_df: DataFrame with features
            attention_type: Type of attention ('dynamic', 'static', 'learned')
            
        Returns:
            DataFrame with attention weights
        """
        self.logger.info(f"Creating attention weights (type: {attention_type})")
        
        if features_df.empty:
            self.logger.warning("Input features DataFrame is empty")
            return pd.DataFrame()
        
        # Select numeric columns for attention
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("No numeric columns for attention weights")
            return features_df
        
        weights_df = pd.DataFrame(index=features_df.index)
        
        if attention_type == 'static':
            # Static weights based on feature importance (simulated)
            for col in numeric_cols:
                # Simple heuristic: more recent features get higher weight
                if 'lag' in col:
                    # Lagged features get lower weight
                    lag_value = int(col.split('lag')[-1]) if 'lag' in col else 1
                    weights_df[f'{col}_weight'] = 1.0 / (1 + lag_value)
                elif 'ma' in col or 'sma' in col or 'ema' in col:
                    # Moving averages get medium weight
                    weights_df[f'{col}_weight'] = 0.7
                elif 'rsi' in col or 'macd' in col or 'stoch' in col:
                    # Momentum indicators get high weight
                    weights_df[f'{col}_weight'] = 0.9
                elif 'sentiment' in col.lower():
                    # Sentiment features get high weight
                    weights_df[f'{col}_weight'] = 0.8
                else:
                    # Default weight
                    weights_df[f'{col}_weight'] = 0.5
        
        elif attention_type == 'dynamic':
            # Dynamic weights based on recent volatility and correlation
            for col in numeric_cols:
                # Weight based on recent volatility (high volatility = high attention)
                volatility = features_df[col].pct_change().rolling(10).std().fillna(0)
                normalized_vol = (volatility - volatility.min()) / (volatility.max() - volatility.min() + 1e-8)
                
                # Weight based on recent trend (strong trend = high attention)
                trend = features_df[col].diff(5).fillna(0)
                normalized_trend = np.abs(trend) / (np.abs(trend).max() + 1e-8)
                
                # Combine volatility and trend
                combined_weight = 0.6 * normalized_vol + 0.4 * normalized_trend
                
                # Apply smoothing
                weights_df[f'{col}_weight'] = combined_weight.rolling(5).mean().fillna(0.5)
        
        elif attention_type == 'learned':
            # Placeholder for learned attention (would require training)
            self.logger.info("Learned attention requires training, using dynamic as default")
            return self.create_attention_weights(features_df, attention_type='dynamic')
        
        else:
            self.logger.warning(f"Unknown attention type: {attention_type}, using dynamic")
            return self.create_attention_weights(features_df, attention_type='dynamic')
        
        # Normalize weights per time step
        for idx in weights_df.index:
            row_weights = weights_df.loc[idx]
            total_weight = row_weights.sum()
            if total_weight > 0:
                weights_df.loc[idx] = row_weights / total_weight
        
        self.logger.info(f"Attention weights created: {len(weights_df.columns)} weight columns")
        
        return weights_df
    
    def construct_state_space(self, 
                            price_features: pd.DataFrame,
                            sentiment_features: pd.DataFrame,
                            portfolio_state: Optional[pd.DataFrame] = None,
                            window_size: int = 10) -> pd.DataFrame:
        """
        Construct state space for DRL
        
        Proposal requirement: State space construction for DRL
        
        State space = Market state + Sentiment parameters + Portfolio state
        
        Args:
            price_features: Price and technical indicators
            sentiment_features: Sentiment features
            portfolio_state: Portfolio state (balance, shares, etc.)
            window_size: Historical window size
            
        Returns:
            DataFrame with state vectors
        """
        self.logger.info(f"Constructing state space (window: {window_size})")
        
        # Check inputs
        if price_features.empty:
            self.logger.error("Price features are empty")
            return pd.DataFrame()
        
        # Align all features by index
        aligned_features = price_features.copy()
        
        if not sentiment_features.empty:
            # Merge sentiment features
            common_idx = aligned_features.index.intersection(sentiment_features.index)
            if len(common_idx) > 0:
                aligned_features = pd.concat([
                    aligned_features.loc[common_idx],
                    sentiment_features.loc[common_idx]
                ], axis=1)
        
        if portfolio_state is not None and not portfolio_state.empty:
            # Merge portfolio state
            common_idx = aligned_features.index.intersection(portfolio_state.index)
            if len(common_idx) > 0:
                aligned_features = pd.concat([
                    aligned_features.loc[common_idx],
                    portfolio_state.loc[common_idx]
                ], axis=1)
        
        if aligned_features.empty:
            self.logger.error("No aligned features after merging")
            return pd.DataFrame()
        
        # Create state vectors with sliding window
        state_vectors = []
        state_indices = []
        
        numeric_cols = aligned_features.select_dtypes(include=[np.number]).columns
        
        for i in range(window_size, len(aligned_features)):
            # Extract window
            window_data = aligned_features.iloc[i-window_size:i][numeric_cols]
            
            # Flatten window
            state_vector = window_data.values.flatten()
            
            # Add current portfolio state (if available)
            if portfolio_state is not None and not portfolio_state.empty:
                current_portfolio = portfolio_state.iloc[i][portfolio_state.select_dtypes(include=[np.number]).columns]
                state_vector = np.concatenate([state_vector, current_portfolio.values])
            
            state_vectors.append(state_vector)
            state_indices.append(aligned_features.index[i])
        
        # Create state DataFrame
        state_columns = []
        for col in numeric_cols:
            for t in range(window_size):
                state_columns.append(f"{col}_t-{window_size-t-1}")
        
        if portfolio_state is not None and not portfolio_state.empty:
            portfolio_numeric = portfolio_state.select_dtypes(include=[np.number]).columns
            for col in portfolio_numeric:
                state_columns.append(f"{col}_current")
        
        state_df = pd.DataFrame(state_vectors, index=state_indices, columns=state_columns)
        
        self.logger.info(f"State space constructed: {state_df.shape}")
        self.logger.info(f"  State dimension: {state_df.shape[1]}")
        self.logger.info(f"  Time steps: {state_df.shape[0]}")
        
        return state_df
    
    def select_important_features(self, 
                                features_df: pd.DataFrame,
                                target_column: str = 'Close',
                                n_features: int = 20,
                                method: str = 'correlation') -> List[str]:
        """
        Select important features
        
        Args:
            features_df: Features DataFrame
            target_column: Target column for feature selection
            n_features: Number of features to select
            method: Selection method ('correlation', 'variance', 'mutual_info')
            
        Returns:
            List of selected feature names
        """
        self.logger.info(f"Selecting important features (method: {method})")
        
        if features_df.empty or target_column not in features_df.columns:
            self.logger.error(f"Target column {target_column} not found")
            return []
        
        # Select numeric features (excluding target)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != target_column]
        
        if len(feature_cols) == 0:
            self.logger.warning("No feature columns available")
            return []
        
        if method == 'correlation':
            # Select based on correlation with target
            correlations = []
            for col in feature_cols:
                corr = features_df[col].corr(features_df[target_column])
                if not np.isnan(corr):
                    correlations.append((col, abs(corr)))
            
            # Sort by absolute correlation
            correlations.sort(key=lambda x: x[1], reverse=True)
            selected = [col for col, _ in correlations[:n_features]]
            
        elif method == 'variance':
            # Select based on variance
            variances = []
            for col in feature_cols:
                var = features_df[col].var()
                if not np.isnan(var):
                    variances.append((col, var))
            
            # Sort by variance
            variances.sort(key=lambda x: x[1], reverse=True)
            selected = [col for col, _ in variances[:n_features]]
            
        elif method == 'mutual_info':
            # Simplified mutual information (using correlation as proxy)
            self.logger.info("Using correlation as proxy for mutual information")
            return self.select_important_features(features_df, target_column, n_features, method='correlation')
        
        else:
            self.logger.warning(f"Unknown method: {method}, using correlation")
            return self.select_important_features(features_df, target_column, n_features, method='correlation')
        
        self.logger.info(f"Selected {len(selected)} important features")
        if len(selected) > 0:
            self.logger.info(f"Top 5: {selected[:5]}")
        
        return selected


# Quick demo
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("Feature Engineering Module - Quick Demo")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }).set_index('Date')
    
    print(f"Sample data shape: {sample_data.shape}")
    
    # Create feature engineer
    engineer = FeatureEngineer(logger=logger)
    
    # Test technical indicators
    print("\n1. Calculating technical indicators...")
    tech_features = engineer.calculate_technical_indicators(sample_data)
    print(f"   Technical features shape: {tech_features.shape}")
    print(f"   Added features: {len(tech_features.columns) - len(sample_data.columns)}")
    
    # Show some example features
    tech_cols = [col for col in tech_features.columns if col not in sample_data.columns]
    print(f"   Example features: {tech_cols[:10]}")
    
    # Test sentiment feature engineering
    print("\n2. Engineering sentiment features...")
    sentiment_data = pd.DataFrame({
        'Date': dates,
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'sentiment_intensity': np.random.uniform(0, 1, 100)
    }).set_index('Date')
    
    sentiment_features = engineer.engineer_sentiment_features(sentiment_data, sample_data)
    print(f"   Sentiment features shape: {sentiment_features.shape}")
    
    # Test attention weights
    print("\n3. Creating attention weights...")
    combined_features = pd.concat([tech_features, sentiment_features], axis=1).dropna()
    attention_weights = engineer.create_attention_weights(combined_features, attention_type='dynamic')
    print(f"   Attention weights shape: {attention_weights.shape}")
    
    # Test state space construction
    print("\n4. Constructing state space...")
    portfolio_state = pd.DataFrame({
        'Date': dates,
        'balance': np.full(100, 10000.0),
        'shares': np.random.randint(0, 100, 100),
        'pnl': np.random.uniform(-1000, 1000, 100)
    }).set_index('Date')
    
    state_space = engineer.construct_state_space(
        tech_features,
        sentiment_features,
        portfolio_state,
        window_size=5
    )
    
    print(f"   State space shape: {state_space.shape}")
    print(f"   State dimension: {state_space.shape[1]}")
    
    # Test feature selection
    print("\n5. Selecting important features...")
    important_features = engineer.select_important_features(
        combined_features,
        target_column='Close',
        n_features=10,
        method='correlation'
    )
    
    print(f"   Selected {len(important_features)} important features")
    print(f"   Features: {important_features}")
    
    print("\nFeature engineering module demo completed!")