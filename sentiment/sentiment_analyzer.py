"""
情感分析模块 - 金融文本情感分析核心

根据提案要求，实现以下功能：
1. 使用NLTK和基于金融语料库微调的基于Transformer的模型的高级NLP技术
2. 包含强度、主观性和上下文感知的多维情感评分
3. 情感信号的时间聚合以匹配交易决策周期
4. 集成特定领域的金融词典（Loughran-McDonald词典）以改进语义理解

作者：大狗（电子兄弟）
日期：2026-03-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import re
import nltk
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 尝试导入Transformer模型
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    print("警告: transformers库未安装，情感分析功能受限")

# 尝试下载NLTK数据（带超时机制）
import threading
import sys

class NLTKDownloader(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.exception = None
        self.success = False
        
    def run(self):
        try:
            nltk.download('vader_lexicon', quiet=True)
            nltk.download('punkt', quiet=True)
            from nltk.sentiment import SentimentIntensityAnalyzer
            self.success = True
        except Exception as e:
            self.exception = e
            self.success = False

NLTK_AVAILABLE = False
try:
    downloader = NLTKDownloader()
    downloader.daemon = True
    downloader.start()
    downloader.join(timeout=30)  # 最多等待30秒
    
    if downloader.is_alive():
        # 如果还在运行，说明卡住了，强制终止
        print("警告: NLTK下载超时（30秒），将使用简化情感分析")
        NLTK_AVAILABLE = False
    elif downloader.success:
        NLTK_AVAILABLE = True
        from nltk.sentiment import SentimentIntensityAnalyzer
        print("NLTK数据下载成功")
    else:
        NLTK_AVAILABLE = False
        print(f"警告: NLTK下载失败: {downloader.exception}")
except Exception as e:
    NLTK_AVAILABLE = False
    print(f"警告: NLTK初始化失败，将使用简化情感分析: {e}")


class FinancialSentimentAnalyzer:
    """
    金融情感分析器
    
    提案要求：
    - 使用NLTK和基于金融语料库微调的基于Transformer的模型的高级NLP技术
    - 包含强度、主观性和上下文感知的多维情感评分
    - 情感信号的时间聚合以匹配交易决策周期
    - 集成特定领域的金融词典（Loughran-McDonald词典）以改进语义理解
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        初始化情感分析器
        
        Args:
            config: 配置字典
            logger: 日志记录器
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # 初始化组件
        self._init_components()
        
        # 情感词典
        self._load_financial_lexicons()
        
        self.logger.info("金融情感分析器初始化完成")
    
    def _init_components(self):
        """初始化NLP组件"""
        # VADER情感分析器（适用于社交媒体）
        if NLTK_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
        else:
            self.vader_analyzer = None
            self.logger.warning("NLTK VADER不可用，将使用简化情感分析")
        
        # Transformer模型（金融微调）
        self.transformer_model = None
        self.transformer_tokenizer = None
        
        if TRANSFORMER_AVAILABLE:
            try:
                # 尝试加载金融微调的模型
                model_name = self.config.get('transformer_model', 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis')
                self.transformer_model = pipeline(
                    'sentiment-analysis',
                    model=model_name,
                    tokenizer=model_name,
                    truncation=True,
                    max_length=512
                )
                self.logger.info(f"加载Transformer模型: {model_name}")
            except Exception as e:
                self.logger.warning(f"加载Transformer模型失败: {e}")
                self.logger.info("将使用VADER进行情感分析")
        
        # 自定义情感词典权重
        self.sentiment_weights = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
            'uncertainty': -0.3,
            'litigious': -0.5,
            'strong_modal': 0.7,
            'weak_modal': 0.3,
            'constraining': -0.4
        }
    
    def _load_financial_lexicons(self):
        """加载金融词典"""
        # Loughran-McDonald金融情感词典（简化版）
        self.financial_lexicon = {
            'positive': {
                'words': [
                    'profit', 'gain', 'growth', 'increase', 'rise', 'improve',
                    'strong', 'positive', 'outperform', 'beat', 'exceed',
                    'success', 'achievement', 'record', 'high', 'bullish',
                    'optimistic', 'favorable', 'advantage', 'opportunity',
                    'robust', 'solid', 'stellar', 'exceptional', 'outstanding'
                ],
                'weight': 1.0
            },
            'negative': {
                'words': [
                    'loss', 'decline', 'decrease', 'fall', 'drop', 'downgrade',
                    'weak', 'negative', 'underperform', 'miss', 'disappoint',
                    'failure', 'problem', 'issue', 'concern', 'bearish',
                    'pessimistic', 'risk', 'threat', 'challenge', 'uncertainty',
                    'volatile', 'turmoil', 'crisis', 'recession', 'bankruptcy',
                    'default', 'fraud', 'investigation', 'lawsuit'
                ],
                'weight': -1.0
            },
            'uncertainty': {
                'words': [
                    'uncertain', 'unclear', 'unknown', 'potential', 'possible',
                    'might', 'could', 'may', 'perhaps', 'maybe', 'if',
                    'depending', 'contingent', 'conditional', 'speculative'
                ],
                'weight': -0.3
            },
            'litigious': {
                'words': [
                    'lawsuit', 'litigation', 'legal', 'court', 'judge',
                    'attorney', 'settlement', 'damages', 'liability',
                    'breach', 'violation', 'infringement', 'claim', 'allege'
                ],
                'weight': -0.5
            },
            'strong_modal': {
                'words': [
                    'will', 'shall', 'must', 'certainly', 'definitely',
                    'undoubtedly', 'clearly', 'obviously', 'necessarily'
                ],
                'weight': 0.7
            },
            'weak_modal': {
                'words': [
                    'could', 'might', 'may', 'possibly', 'perhaps',
                    'maybe', 'sometimes', 'occasionally', 'seldom'
                ],
                'weight': 0.3
            },
            'constraining': {
                'words': [
                    'but', 'however', 'although', 'though', 'yet',
                    'despite', 'nevertheless', 'nonetheless', 'except'
                ],
                'weight': -0.4
            }
        }
        
        # 创建词到类别的映射
        self.word_to_category = {}
        for category, data in self.financial_lexicon.items():
            for word in data['words']:
                self.word_to_category[word.lower()] = category
        
        self.logger.info(f"加载金融词典: {len(self.word_to_category)} 个词, {len(self.financial_lexicon)} 个类别")
    
    def analyze_text(self, text: str, method: str = 'combined') -> Dict[str, float]:
        """
        分析单个文本的情感
        
        Args:
            text: 要分析的文本
            method: 分析方法（'vader', 'transformer', 'lexicon', 'combined'）
            
        Returns:
            情感分数字典
        """
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return self._get_empty_sentiment()
        
        text = text.strip()
        
        # 根据方法选择分析器
        if method == 'vader':
            return self._analyze_with_vader(text)
        elif method == 'transformer':
            return self._analyze_with_transformer(text)
        elif method == 'lexicon':
            return self._analyze_with_lexicon(text)
        elif method == 'combined':
            # 组合方法：尝试多个分析器并综合结果
            return self._analyze_combined(text)
        else:
            self.logger.warning(f"未知分析方法: {method}，使用组合方法")
            return self._analyze_combined(text)
    
    def _analyze_with_vader(self, text: str) -> Dict[str, float]:
        """使用VADER分析情感"""
        if not self.vader_analyzer:
            return self._get_empty_sentiment()
        
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # VADER返回：neg, neu, pos, compound
            result = {
                'sentiment_score': scores['compound'],  # 综合分数 (-1 到 1)
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu'],
                'intensity': abs(scores['compound']),  # 情感强度
                'subjectivity': 1.0 - scores['neu'],   # 主观性 (1 - 中性比例)
                'method': 'vader'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"VADER分析失败: {e}")
            return self._get_empty_sentiment()
    
    def _analyze_with_transformer(self, text: str) -> Dict[str, float]:
        """使用Transformer模型分析情感"""
        if not self.transformer_model:
            return self._get_empty_sentiment()
        
        try:
            # 限制文本长度
            if len(text) > 1000:
                text = text[:1000]
            
            result = self.transformer_model(text)[0]
            
            # Transformer返回：label, score
            label = result['label'].lower()
            score = result['score']
            
            # 映射标签到情感分数
            if 'positive' in label:
                sentiment = score
            elif 'negative' in label:
                sentiment = -score
            else:  # neutral或其他
                sentiment = 0.0
            
            result_dict = {
                'sentiment_score': sentiment,
                'positive': score if sentiment > 0 else 0.0,
                'negative': abs(score) if sentiment < 0 else 0.0,
                'neutral': 1.0 - score if abs(sentiment) < 0.3 else 0.0,
                'intensity': abs(sentiment),
                'subjectivity': 1.0 if abs(sentiment) > 0.3 else 0.5,
                'confidence': score,
                'method': 'transformer'
            }
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Transformer分析失败: {e}")
            return self._get_empty_sentiment()
    
    def _analyze_with_lexicon(self, text: str) -> Dict[str, float]:
        """使用金融词典分析情感"""
        try:
            # 预处理文本
            text_lower = text.lower()
            
            # 简单分词（实际中应该使用更好的分词器）
            words = re.findall(r'\b\w+\b', text_lower)
            
            # 统计每个类别的出现次数
            category_counts = defaultdict(int)
            for word in words:
                if word in self.word_to_category:
                    category = self.word_to_category[word]
                    category_counts[category] += 1
            
            # 计算基础情感分数
            total_weighted_score = 0.0
            total_words = len(words) if words else 1
            
            for category, count in category_counts.items():
                weight = self.financial_lexicon[category]['weight']
                total_weighted_score += weight * count
            
            # 归一化情感分数 (-1 到 1)
            normalized_score = total_weighted_score / total_words
            normalized_score = max(-1.0, min(1.0, normalized_score))
            
            # 计算类别比例
            positive_count = category_counts.get('positive', 0)
            negative_count = category_counts.get('negative', 0)
            uncertainty_count = category_counts.get('uncertainty', 0)
            
            total_categorized = sum(category_counts.values())
            
            if total_categorized > 0:
                positive_ratio = positive_count / total_categorized
                negative_ratio = negative_count / total_categorized
                uncertainty_ratio = uncertainty_count / total_categorized
            else:
                positive_ratio = negative_ratio = uncertainty_ratio = 0.0
            
            # 计算主观性（基于情感词的比例）
            subjectivity = total_categorized / total_words if total_words > 0 else 0.0
            
            result = {
                'sentiment_score': normalized_score,
                'positive': positive_ratio,
                'negative': negative_ratio,
                'neutral': 1.0 - (positive_ratio + negative_ratio),
                'uncertainty': uncertainty_ratio,
                'intensity': abs(normalized_score),
                'subjectivity': min(1.0, subjectivity * 2),  # 放大主观性
                'lexicon_words': total_categorized,
                'total_words': total_words,
                'method': 'lexicon'
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"词典分析失败: {e}")
            return self._get_empty_sentiment()
    
    def _analyze_combined(self, text: str) -> Dict[str, float]:
        """组合多种方法分析情感"""
        results = []
        weights = {'vader': 0.3, 'transformer': 0.4, 'lexicon': 0.3}
        
        # 收集各种方法的结果
        vader_result = self._analyze_with_vader(text)
        if vader_result['method'] != 'none':
            results.append(('vader', vader_result))
        
        transformer_result = self._analyze_with_transformer(text)
        if transformer_result['method'] != 'none':
            results.append(('transformer', transformer_result))
        
        lexicon_result = self._analyze_with_lexicon(text)
        if lexicon_result['method'] != 'none':
            results.append(('lexicon', lexicon_result))
        
        if not results:
            return self._get_empty_sentiment()
        
        # 加权平均
        combined_result = {
            'sentiment_score': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0,
            'intensity': 0.0,
            'subjectivity': 0.0,
            'confidence': 0.0,
            'method': 'combined',
            'components': len(results)
        }
        
        total_weight = 0.0
        for method, result in results:
            weight = weights.get(method, 1.0/len(results))
            
            # 加权平均主要指标
            combined_result['sentiment_score'] += result['sentiment_score'] * weight
            combined_result['positive'] += result.get('positive', 0.0) * weight
            combined_result['negative'] += result.get('negative', 0.0) * weight
            combined_result['intensity'] += result.get('intensity', 0.0) * weight
            combined_result['subjectivity'] += result.get('subjectivity', 0.5) * weight
            
            total_weight += weight
        
        # 归一化
        if total_weight > 0:
            for key in ['sentiment_score', 'positive', 'negative', 'intensity', 'subjectivity']:
                combined_result[key] /= total_weight
        
        # 计算中性比例
        combined_result['neutral'] = 1.0 - (combined_result['positive'] + combined_result['negative'])
        combined_result['neutral'] = max(0.0, min(1.0, combined_result['neutral']))
        
        # 计算置信度（基于方法数量和一致性）
        if len(results) > 1:
            # 计算情感分数的一致性
            sentiment_scores = [r['sentiment_score'] for _, r in results]
            score_variance = np.var(sentiment_scores)
            consistency = 1.0 / (1.0 + score_variance)  # 方差越小，一致性越高
            combined_result['confidence'] = consistency * 0.5 + 0.5  # 映射到0.5-1.0
        else:
            combined_result['confidence'] = 0.7
        
        return combined_result
    
    def _get_empty_sentiment(self) -> Dict[str, float]:
        """获取空情感结果"""
        return {
            'sentiment_score': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'intensity': 0.0,
            'subjectivity': 0.5,
            'confidence': 0.0,
            'method': 'none'
        }
    
    def analyze_batch(self, texts: List[str], method: str = 'combined') -> List[Dict[str, float]]:
        """
        批量分析文本情感
        
        Args:
            texts: 文本列表
            method: 分析方法
            
        Returns:
            情感结果列表
        """
        self.logger.info(f"批量分析 {len(texts)} 个文本，方法: {method}")
        
        results = []
        for i, text in enumerate(texts):
            if i % 100 == 0 and i > 0:
                self.logger.info(f"已分析 {i}/{len(texts)} 个文本")
            
            result = self.analyze_text(text, method)
            results.append(result)
        
        self.logger.info(f"批量分析完成: {len(results)} 个结果")
        return results
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         method: str = 'combined',
                         result_prefix: str = 'sentiment_') -> pd.DataFrame:
        """
        分析DataFrame中的文本列
        
        Args:
            df: 输入DataFrame
            text_column: 文本列名
            method: 分析方法
            result_prefix: 结果列名前缀
            
        Returns:
            添加情感列后的DataFrame
        """
        if text_column not in df.columns:
            self.logger.error(f"DataFrame中没有列: {text_column}")
            return df
        
        self.logger.info(f"分析DataFrame中的文本列: {text_column}")
        
        # 分析情感
        texts = df[text_column].fillna('').astype(str).tolist()
        sentiment_results = self.analyze_batch(texts, method)
        
        # 转换结果到DataFrame
        sentiment_df = pd.DataFrame(sentiment_results)
        
        # 重命名列
        column_mapping = {
            'sentiment_score': f'{result_prefix}score',
            'positive': f'{result_prefix}positive',
            'negative': f'{result_prefix}negative',
            'neutral': f'{result_prefix}neutral',
            'intensity': f'{result_prefix}intensity',
            'subjectivity': f'{result_prefix}subjectivity',
            'confidence': f'{result_prefix}confidence',
            'method': f'{result_prefix}method'
        }
        
        sentiment_df = sentiment_df.rename(columns=column_mapping)
        
        # 合并到原始DataFrame
        result_df = pd.concat([df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)
        
        self.logger.info(f"DataFrame分析完成，添加了 {len(sentiment_df.columns)} 个情感列")
        return result_df
    
    def get_detailed_analysis(self, text: str) -> Dict[str, Any]:
        """
        获取详细的情感分析结果
        
        Args:
            text: 要分析的文本
            
        Returns:
            详细分析结果
        """
        # 基础情感分析
        combined_result = self.analyze_text(text, 'combined')
        
        # 单独方法结果
        vader_result = self._analyze_with_vader(text)
        transformer_result = self._analyze_with_transformer(text)
        lexicon_result = self._analyze_with_lexicon(text)
        
        # 词典分析细节
        lexicon_details = {}
        if lexicon_result['method'] != 'none':
            text_lower = text.lower()
            words = re.findall(r'\b\w+\b', text_lower)
            
            found_words = []
            for word in words:
                if word in self.word_to_category:
                    found_words.append({
                        'word': word,
                        'category': self.word_to_category[word],
                        'weight': self.financial_lexicon[self.word_to_category[word]]['weight']
                    })
            
            lexicon_details = {
                'found_words': found_words,
                'total_words': len(words),
                'sentiment_words': len(found_words)
            }
        
        # 构建详细结果
        detailed_result = {
            'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
            'text_length': len(text),
            'combined_sentiment': combined_result,
            'method_results': {
                'vader': vader_result if vader_result['method'] != 'none' else None,
                'transformer': transformer_result if transformer_result['method'] != 'none' else None,
                'lexicon': lexicon_result if lexicon_result['method'] != 'none' else None
            },
            'lexicon_details': lexicon_details,
            'overall_sentiment': self._interpret_sentiment(combined_result['sentiment_score']),
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return detailed_result
    
    def _interpret_sentiment(self, score: float) -> str:
        """解释情感分数"""
        if score >= 0.5:
            return "强烈积极"
        elif score >= 0.2:
            return "积极"
        elif score >= 0.05:
            return "略微积极"
        elif score > -0.05:
            return "中性"
        elif score > -0.2:
            return "略微消极"
        elif score > -0.5:
            return "消极"
        else:
            return "强烈消极"


# 使用示例
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # 创建情感分析器
    analyzer = FinancialSentimentAnalyzer(logger=logger)
    
    # 测试文本
    test_texts = [
        "Apple reported record profits in Q4, exceeding analyst expectations with strong iPhone sales.",
        "The stock market plunged today as fears of inflation and interest rate hikes spooked investors.",
        "Microsoft announced a new partnership that could boost its cloud computing division.",
        "Tesla faces production delays and supply chain issues, causing concerns among investors.",
        "The Federal Reserve is expected to maintain current interest rates amid economic uncertainty."
    ]
    
    print("金融情感分析器测试")
    print("=" * 50)
    
    for i, text in enumerate(test_texts):
        print(f"\n文本 {i+1}: {text[:80]}...")
        
        # 组合分析
        result = analyzer.analyze_text(text, 'combined')
        
        print(f"  综合情感分数: {result['sentiment_score']:.3f}")
        print(f"  积极比例: {result['positive']:.3f}")
        print(f"  消极比例: {result['negative']:.3f}")
        print(f"  中性比例: {result['neutral']:.3f}")
        print(f"  情感强度: {result['intensity']:.3f}")
        print(f"  主观性: {result['subjectivity']:.3f}")
        print(f"  分析方法: {result['method']}")
        
        # 详细分析
        if i == 0:  # 只对第一个文本进行详细分析
            print(f"\n详细分析示例:")
            detailed = analyzer.get_detailed_analysis(text)
            print(f"  总体情感: {detailed['overall_sentiment']}")
            
            if detailed['lexicon_details']:
                print(f"  发现的情感词: {len(detailed['lexicon_details']['found_words'])} 个")
                for word_info in detailed['lexicon_details']['found_words'][:5]:
                    print(f"    '{word_info['word']}' → {word_info['category']} (权重: {word_info['weight']})")
    
    # 批量分析测试
    print(f"\n{'=' * 50}")
    print("批量分析测试")
    
    batch_results = analyzer.analyze_batch(test_texts)
    scores = [r['sentiment_score'] for r in batch_results]
    
    print(f"平均情感分数: {np.mean(scores):.3f}")
    print(f"情感分数范围: [{min(scores):.3f}, {max(scores):.3f}]")
    print(f"标准差: {np.std(scores):.3f}")
    
    # DataFrame分析测试
    print(f"\n{'=' * 50}")
    print("DataFrame分析测试")
    
    news_df = pd.DataFrame({
        'title': [f"News {i}" for i in range(5)],
        'content': test_texts,
        'date': pd.date_range(start='2024-01-01', periods=5, freq='D')
    })
    
    analyzed_df = analyzer.analyze_dataframe(news_df, 'content')
    
    print(f"原始DataFrame形状: {news_df.shape}")
    print(f"分析后DataFrame形状: {analyzed_df.shape}")
    print(f"添加的情感列: {[col for col in analyzed_df.columns if col.startswith('sentiment_')]}")
    
    print("\n情感分析模块测试完成!")