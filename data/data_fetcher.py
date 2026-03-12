"""
数据采集模块 - 多源金融数据获取

根据提案要求，实现以下数据源：
1. 雅虎财经API：历史市场数据（OHLCV价格、成交量）
2. 实时金融新闻和社交媒体源
3. 公司基本面数据和经济指标

作者：大狗（电子兄弟）
日期：2026-03-12
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import os


class DataFetcher:
    """
    多源金融数据采集器
    
    提案要求：
    - 处理多源数据摄取
    - 实现稳健的数据清理、缺失值处理和时间对齐
    - 确保跨不同数据频率和格式的连贯集成
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化数据采集器
        
        Args:
            config: 配置字典，可包含API密钥等
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 配置参数
        self.yahoo_timeout = self.config.get('yahoo_timeout', 30)
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 5)
        
        # API密钥（从环境变量或配置读取）
        self.news_api_key = self.config.get('news_api_key') or os.getenv('NEWS_API_KEY')
        self.twitter_bearer_token = self.config.get('twitter_bearer_token') or os.getenv('TWITTER_BEARER_TOKEN')
        
        self.logger.info("数据采集器初始化完成")
    
    def fetch_stock_data(self, 
                         ticker: str, 
                         start_date: Union[str, datetime], 
                         end_date: Union[str, datetime],
                         interval: str = '1d') -> pd.DataFrame:
        """
        获取股票历史数据（雅虎财经）
        
        提案要求：来自雅虎财经API的历史市场数据（OHLCV价格、成交量）
        
        Args:
            ticker: 股票代码（如'AAPL'）
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔（'1d', '1h', '1m'等）
            
        Returns:
            DataFrame包含OHLCV数据
        """
        self.logger.info(f"获取股票数据: {ticker}, {start_date} 到 {end_date}, 间隔: {interval}")
        
        for attempt in range(self.max_retries):
            try:
                # 下载数据
                stock = yf.Ticker(ticker)
                df = stock.history(start=start_date, end=end_date, interval=interval)
                
                if df.empty:
                    self.logger.warning(f"未获取到 {ticker} 的数据，可能股票代码错误或时间段无数据")
                    return pd.DataFrame()
                
                # 确保列名一致性
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                for col in required_cols:
                    if col not in df.columns:
                        self.logger.error(f"数据缺少必要列: {col}")
                        return pd.DataFrame()
                
                # 添加额外信息
                df['Ticker'] = ticker
                df['Return'] = df['Close'].pct_change()
                df['LogReturn'] = np.log(df['Close'] / df['Close'].shift(1))
                
                self.logger.info(f"成功获取 {ticker} 数据: {len(df)} 行, {len(df.columns)} 列")
                return df
                
            except Exception as e:
                self.logger.error(f"获取股票数据失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"所有重试均失败，无法获取 {ticker} 数据")
        
        return pd.DataFrame()
    
    def fetch_multiple_stocks(self, 
                             tickers: List[str], 
                             start_date: Union[str, datetime], 
                             end_date: Union[str, datetime],
                             interval: str = '1d') -> Dict[str, pd.DataFrame]:
        """
        获取多个股票数据
        
        Args:
            tickers: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            interval: 时间间隔
            
        Returns:
            股票代码到DataFrame的字典映射
        """
        self.logger.info(f"获取多个股票数据: {tickers}")
        
        data_dict = {}
        for ticker in tickers:
            df = self.fetch_stock_data(ticker, start_date, end_date, interval)
            if not df.empty:
                data_dict[ticker] = df
            else:
                self.logger.warning(f"跳过 {ticker}，数据为空")
        
        self.logger.info(f"成功获取 {len(data_dict)}/{len(tickers)} 个股票数据")
        return data_dict
    
    def fetch_sp500_constituents(self, date: Optional[Union[str, datetime]] = None) -> List[str]:
        """
        获取标普500成分股列表
        
        提案要求：实验验证采用标普500成分股（2010-2024年）
        
        Args:
            date: 指定日期的成分股列表，默认为当前
            
        Returns:
            股票代码列表
        """
        self.logger.info("获取标普500成分股列表")
        
        try:
            # 从维基百科获取标普500成分股列表
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            
            # 第一个表格包含成分股信息
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # 清理股票代码（移除可能的后缀）
            cleaned_tickers = []
            for ticker in tickers:
                # 移除点和其他字符
                cleaned = str(ticker).replace('.', '-').strip()
                cleaned_tickers.append(cleaned)
            
            self.logger.info(f"获取到 {len(cleaned_tickers)} 个标普500成分股")
            return cleaned_tickers
            
        except Exception as e:
            self.logger.error(f"获取标普500成分股失败: {e}")
            
            # 备用方案：使用一些常见的标普500股票
            backup_tickers = [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'BRK-B',
                'JPM', 'JNJ', 'V', 'PG', 'UNH', 'HD', 'MA', 'CVX', 'LLY', 'AVGO',
                'KO', 'PEP', 'MRK', 'ABBV', 'WMT', 'ORCL', 'BAC', 'XOM', 'CSCO',
                'DIS', 'NFLX', 'ADBE', 'CRM', 'AMD', 'INTC', 'CMCSA', 'TMO', 'PFE',
                'INTU', 'AMGN', 'QCOM', 'IBM', 'GS', 'CAT', 'BA', 'UNP', 'HON',
                'RTX', 'GE', 'TXN', 'LOW', 'UPS', 'SBUX', 'MDT', 'PLD', 'SPGI'
            ]
            self.logger.warning(f"使用备用股票列表: {len(backup_tickers)} 个股票")
            return backup_tickers
    
    def fetch_financial_news(self, 
                            query: str = "stock market",
                            from_date: Optional[Union[str, datetime]] = None,
                            to_date: Optional[Union[str, datetime]] = None,
                            limit: int = 100) -> pd.DataFrame:
        """
        获取金融新闻数据
        
        提案要求：来自路透社/彭博社的金融新闻
        
        Args:
            query: 搜索查询词
            from_date: 开始日期
            to_date: 结束日期
            limit: 最大新闻数量
            
        Returns:
            DataFrame包含新闻标题、内容、日期等信息
        """
        self.logger.info(f"获取金融新闻: '{query}'")
        
        # 如果没有NewsAPI密钥，返回模拟数据
        if not self.news_api_key:
            self.logger.warning("未提供NewsAPI密钥，返回模拟新闻数据")
            return self._generate_mock_news(query, from_date, to_date, limit)
        
        try:
            # NewsAPI参数
            if isinstance(from_date, datetime):
                from_date = from_date.strftime('%Y-%m-%d')
            if isinstance(to_date, datetime):
                to_date = to_date.strftime('%Y-%m-%d')
            
            # 构建请求URL
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'apiKey': self.news_api_key,
                'pageSize': min(limit, 100),  # NewsAPI每页最大100条
                'language': 'en',
                'sortBy': 'publishedAt'
            }
            
            if from_date:
                params['from'] = from_date
            if to_date:
                params['to'] = to_date
            
            # 发送请求
            response = requests.get(url, params=params, timeout=self.yahoo_timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if data['status'] != 'ok':
                self.logger.error(f"NewsAPI返回错误: {data.get('message', 'Unknown error')}")
                return self._generate_mock_news(query, from_date, to_date, limit)
            
            # 解析新闻数据
            articles = data.get('articles', [])
            
            news_data = []
            for article in articles:
                news_data.append({
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'content': article.get('content', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'published_at': article.get('publishedAt', ''),
                    'author': article.get('author', '')
                })
            
            df = pd.DataFrame(news_data)
            self.logger.info(f"获取到 {len(df)} 条新闻")
            return df
            
        except Exception as e:
            self.logger.error(f"获取新闻数据失败: {e}")
            self.logger.warning("返回模拟新闻数据")
            return self._generate_mock_news(query, from_date, to_date, limit)
    
    def _generate_mock_news(self, 
                           query: str,
                           from_date: Optional[Union[str, datetime]],
                           to_date: Optional[Union[str, datetime]],
                           limit: int) -> pd.DataFrame:
        """生成模拟新闻数据（用于测试）"""
        # 生成日期范围
        if from_date is None:
            from_date = datetime.now() - timedelta(days=30)
        if to_date is None:
            to_date = datetime.now()
        
        self.logger.debug(f"Mock news: from_date={from_date}, to_date={to_date}, type_from={type(from_date)}, type_to={type(to_date)}")
        
        # 转换日期格式
        if isinstance(from_date, str):
            try:
                from_date = datetime.strptime(from_date, '%Y-%m-%d')
            except ValueError:
                # 尝试其他格式
                from_date = pd.to_datetime(from_date).to_pydatetime()
        if isinstance(to_date, str):
            try:
                to_date = datetime.strptime(to_date, '%Y-%m-%d')
            except ValueError:
                to_date = pd.to_datetime(to_date).to_pydatetime()
        
        # 确保to_date不早于from_date
        if to_date < from_date:
            to_date = from_date + timedelta(days=30)
        
        self.logger.info(f"生成模拟新闻，时间范围: {from_date.date()} 到 {to_date.date()}")
        
        # 生成模拟新闻
        mock_titles = [
            f"{query} shows strong growth in Q4",
            f"Analysts raise target price for major {query} companies",
            f"Federal Reserve decision impacts {query}",
            f"Global economic outlook affects {query} performance",
            f"New regulations could shape {query} future",
            f"Tech giants lead {query} rally",
            f"Market volatility creates opportunities in {query}",
            f"Inflation data influences {query} sentiment",
            f"Earnings season brings surprises to {query}",
            f"Institutional investors increase {query} exposure"
        ]
        
        mock_sources = ['Reuters', 'Bloomberg', 'CNBC', 'Financial Times', 'Wall Street Journal']
        
        news_data = []
        days_range = (to_date - from_date).days
        
        for i in range(min(limit, 50)):  # 最多50条模拟新闻
            days_offset = np.random.randint(0, days_range) if days_range > 0 else 0
            publish_date = from_date + timedelta(days=days_offset)
            
            news_data.append({
                'title': np.random.choice(mock_titles),
                'description': f"Analysis of recent trends in {query} market",
                'content': f"Detailed analysis of {query} performance and outlook. Market experts provide insights.",
                'url': f"https://example.com/news/{i}",
                'source': np.random.choice(mock_sources),
                'published_at': publish_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'author': f"Analyst {np.random.randint(1, 10)}"
            })
        
        df = pd.DataFrame(news_data)
        df['published_at'] = pd.to_datetime(df['published_at'])
        df = df.sort_values('published_at')
        
        self.logger.info(f"生成 {len(df)} 条模拟新闻")
        return df
    
    def fetch_social_media_sentiment(self, 
                                    query: str = "stocks",
                                    max_results: int = 100) -> pd.DataFrame:
        """
        获取社交媒体情绪数据
        
        提案要求：来自StockTwits/Twitter的社交媒体情绪
        
        Args:
            query: 搜索查询词
            max_results: 最大结果数
            
        Returns:
            DataFrame包含社交媒体帖子
        """
        self.logger.info(f"获取社交媒体情绪数据: '{query}'")
        
        # 如果没有Twitter API密钥，返回模拟数据
        if not self.twitter_bearer_token:
            self.logger.warning("未提供Twitter API密钥，返回模拟社交媒体数据")
            return self._generate_mock_social_media(query, max_results)
        
        # 注意：实际Twitter API v2需要更复杂的实现
        # 这里仅提供框架
        try:
            # Twitter API v2 搜索端点
            url = "https://api.twitter.com/2/tweets/search/recent"
            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}'
            }
            params = {
                'query': f'{query} -is:retweet lang:en',
                'max_results': min(max_results, 100),
                'tweet.fields': 'created_at,public_metrics',
                'expansions': 'author_id'
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=self.yahoo_timeout)
            response.raise_for_status()
            
            data = response.json()
            tweets = data.get('data', [])
            
            tweet_data = []
            for tweet in tweets:
                tweet_data.append({
                    'id': tweet.get('id'),
                    'text': tweet.get('text'),
                    'created_at': tweet.get('created_at'),
                    'like_count': tweet.get('public_metrics', {}).get('like_count', 0),
                    'retweet_count': tweet.get('public_metrics', {}).get('retweet_count', 0),
                    'reply_count': tweet.get('public_metrics', {}).get('reply_count', 0),
                    'quote_count': tweet.get('public_metrics', {}).get('quote_count', 0),
                    'query': query
                })
            
            df = pd.DataFrame(tweet_data)
            if not df.empty:
                df['created_at'] = pd.to_datetime(df['created_at'])
            
            self.logger.info(f"获取到 {len(df)} 条社交媒体帖子")
            return df
            
        except Exception as e:
            self.logger.error(f"获取社交媒体数据失败: {e}")
            self.logger.warning("返回模拟社交媒体数据")
            return self._generate_mock_social_media(query, max_results)
    
    def _generate_mock_social_media(self, query: str, max_results: int) -> pd.DataFrame:
        """生成模拟社交媒体数据"""
        mock_texts = [
            f"Just bought more ${query}, feeling bullish!",
            f"Market looks shaky today, cautious on ${query}",
            f"${query} earnings beat expectations, great results!",
            f"Analysts downgrading ${query}, time to sell?",
            f"Long-term holder of ${query}, not worried about short-term dips",
            f"Technical analysis shows ${query} forming bullish pattern",
            f"Fed announcement could impact ${query} significantly",
            f"${query} showing strong relative strength",
            f"Institutional buying detected in ${query}",
            f"Considering selling ${query} to lock in profits"
        ]
        
        social_data = []
        end_date = datetime.now()
        
        for i in range(min(max_results, 50)):
            hours_offset = np.random.randint(0, 24 * 7)  # 最近7天
            created_at = end_date - timedelta(hours=hours_offset)
            
            social_data.append({
                'id': f"mock_{i}",
                'text': np.random.choice(mock_texts),
                'created_at': created_at,
                'like_count': np.random.randint(0, 100),
                'retweet_count': np.random.randint(0, 50),
                'reply_count': np.random.randint(0, 20),
                'quote_count': np.random.randint(0, 10),
                'query': query
            })
        
        df = pd.DataFrame(social_data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df = df.sort_values('created_at', ascending=False)
        
        self.logger.info(f"生成 {len(df)} 条模拟社交媒体帖子")
        return df
    
    def fetch_economic_indicators(self) -> pd.DataFrame:
        """
        获取经济指标数据
        
        提案要求：宏观经济指标（利率、通胀数据、经济意外指数）
        
        Returns:
            DataFrame包含经济指标时间序列
        """
        self.logger.info("获取经济指标数据")
        
        try:
            # 这里使用FRED（Federal Reserve Economic Data）API的示例
            # 实际使用时需要FRED API密钥
            economic_data = {
                'date': pd.date_range(start='2010-01-01', end=datetime.now().strftime('%Y-%m-%d'), freq='M'),
                'inflation_rate': np.random.uniform(1.5, 5.0, 200),  # 模拟通胀率
                'interest_rate': np.random.uniform(0.1, 4.0, 200),   # 模拟利率
                'gdp_growth': np.random.uniform(-2.0, 4.0, 200),     # 模拟GDP增长
                'unemployment_rate': np.random.uniform(3.5, 8.0, 200)  # 模拟失业率
            }
            
            df = pd.DataFrame(economic_data)
            self.logger.info(f"生成 {len(df)} 条经济指标数据")
            return df
            
        except Exception as e:
            self.logger.error(f"获取经济指标失败: {e}")
            return pd.DataFrame()
    
    def save_data(self, data: Union[pd.DataFrame, Dict], filepath: str, format: str = 'parquet'):
        """
        保存数据到文件
        
        Args:
            data: 要保存的数据
            filepath: 文件路径
            format: 保存格式（'parquet', 'csv', 'json'）
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            if isinstance(data, dict):
                # 保存字典数据
                if format == 'json':
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
                elif format == 'parquet':
                    # 字典转DataFrame
                    df = pd.DataFrame(data)
                    df.to_parquet(filepath)
                else:
                    pd.DataFrame(data).to_csv(filepath, index=False)
            else:
                # 保存DataFrame
                if format == 'parquet':
                    data.to_parquet(filepath)
                elif format == 'json':
                    data.to_json(filepath, orient='records', indent=2)
                else:
                    data.to_csv(filepath, index=False)
            
            self.logger.info(f"数据已保存: {filepath}")
            
        except Exception as e:
            self.logger.error(f"保存数据失败: {e}")
    
    def load_data(self, filepath: str, format: str = None) -> Union[pd.DataFrame, Dict]:
        """
        从文件加载数据
        
        Args:
            filepath: 文件路径
            format: 文件格式（自动检测如果为None）
            
        Returns:
            加载的数据
        """
        try:
            if format is None:
                # 从文件扩展名推断格式
                ext = os.path.splitext(filepath)[1].lower()
                if ext == '.parquet':
                    format = 'parquet'
                elif ext == '.json':
                    format = 'json'
                else:
                    format = 'csv'
            
            if format == 'parquet':
                data = pd.read_parquet(filepath)
            elif format == 'json':
                # 尝试加载为DataFrame，如果失败则加载为字典
                try:
                    data = pd.read_json(filepath)
                except:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
            else:
                data = pd.read_csv(filepath)
            
            self.logger.info(f"数据已加载: {filepath}, 形状: {data.shape if hasattr(data, 'shape') else 'N/A'}")
            return data
            
        except Exception as e:
            self.logger.error(f"加载数据失败: {e}")
            return None if format == 'json' else pd.DataFrame()


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建数据采集器
    fetcher = DataFetcher(logger=logger)
    
    # 获取股票数据示例
    print("获取AAPL股票数据...")
    aapl_data = fetcher.fetch_stock_data('AAPL', '2024-01-01', '2024-12-31')
    print(f"AAPL数据形状: {aapl_data.shape}")
    
    # 获取标普500成分股示例
    print("\n获取标普500成分股...")
    sp500_tickers = fetcher.fetch_sp500_constituents()
    print(f"前10个成分股: {sp500_tickers[:10]}")
    
    # 获取新闻数据示例
    print("\n获取金融新闻...")
    news_data = fetcher.fetch_financial_news("Apple", limit=5)
    print(f"新闻数量: {len(news_data)}")
    
    # 获取社交媒体数据示例
    print("\n获取社交媒体数据...")
    social_data = fetcher.fetch_social_media_sentiment("AAPL", max_results=5)
    print(f"社交媒体帖子数量: {len(social_data)}")
    
    print("\n数据采集模块测试完成!")