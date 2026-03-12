"""
Sentiment Analyzer - Offline Version

This is an offline-capable version of the sentiment analyzer
that doesn't require internet connection.

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import re
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class FinancialSentimentAnalyzerOffline:
    """
    Financial sentiment analyzer - offline version
    
    This version uses only local resources and doesn't require
    internet connection for sentiment analysis.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize offline sentiment analyzer
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Financial lexicon (simplified version of Loughran-McDonald)
        self.financial_lexicon = self._create_financial_lexicon()
        
        # Sentiment weights
        self.positive_words = self.financial_lexicon['positive']
        self.negative_words = self.financial_lexicon['negative']
        self.uncertainty_words = self.financial_lexicon['uncertainty']
        self.litigious_words = self.financial_lexicon['litigious']
        self.strong_modal_words = self.financial_lexicon['strong_modal']
        self.weak_modal_words = self.financial_lexicon['weak_modal']
        self.constraining_words = self.financial_lexicon['constraining']
        
        # Sentiment scoring weights
        self.weights = {
            'positive': 1.0,
            'negative': -1.0,
            'uncertainty': -0.5,
            'litigious': -0.3,
            'strong_modal': 0.3,
            'weak_modal': -0.2,
            'constraining': -0.4
        }
        
        # Initialize VADER-like sentiment dictionary
        self.sentiment_dict = self._create_sentiment_dictionary()
        
        self.logger.info("FinancialSentimentAnalyzerOffline initialized (offline mode)")
    
    def _create_financial_lexicon(self) -> Dict[str, List[str]]:
        """Create a simplified financial lexicon"""
        return {
            'positive': [
                'profit', 'gain', 'growth', 'increase', 'rise', 'up', 'strong',
                'positive', 'bullish', 'optimistic', 'outperform', 'beat',
                'exceed', 'surpass', 'record', 'high', 'success', 'win'
            ],
            'negative': [
                'loss', 'decline', 'decrease', 'fall', 'down', 'weak',
                'negative', 'bearish', 'pessimistic', 'underperform', 'miss',
                'fail', 'drop', 'plunge', 'crash', 'low', 'trouble', 'risk'
            ],
            'uncertainty': [
                'uncertain', 'unknown', 'unclear', 'ambiguous', 'volatile',
                'unpredictable', 'risky', 'speculative', 'maybe', 'perhaps',
                'possible', 'potential', 'might', 'could', 'may'
            ],
            'litigious': [
                'lawsuit', 'legal', 'court', 'sue', 'litigation', 'claim',
                'allege', 'accuse', 'violation', 'breach', 'fraud', 'settle'
            ],
            'strong_modal': [
                'must', 'shall', 'will', 'should', 'require', 'necessity',
                'obligation', 'mandatory', 'essential', 'critical'
            ],
            'weak_modal': [
                'can', 'could', 'may', 'might', 'possible', 'potential',
                'optional', 'discretionary', 'choice', 'option'
            ],
            'constraining': [
                'limit', 'restrict', 'constrain', 'boundary', 'cap',
                'maximum', 'minimum', 'threshold', 'ceiling', 'floor'
            ]
        }
    
    def _create_sentiment_dictionary(self) -> Dict[str, float]:
        """Create a simple sentiment dictionary"""
        sentiment_dict = {}
        
        # Add positive words
        for word in self.positive_words:
            sentiment_dict[word] = 1.0
            # Add variations
            sentiment_dict[word + 's'] = 1.0
            sentiment_dict[word + 'ed'] = 1.0
            sentiment_dict[word + 'ing'] = 1.0
        
        # Add negative words
        for word in self.negative_words:
            sentiment_dict[word] = -1.0
            # Add variations
            sentiment_dict[word + 's'] = -1.0
            sentiment_dict[word + 'ed'] = -1.0
            sentiment_dict[word + 'ing'] = -1.0
        
        # Add intensifiers
        intensifiers = {
            'very': 1.5, 'extremely': 2.0, 'highly': 1.8, 'greatly': 1.7,
            'slightly': 0.5, 'somewhat': 0.7, 'moderately': 0.8,
            'not': -1.0, 'never': -1.5, 'no': -1.0
        }
        
        for word, weight in intensifiers.items():
            sentiment_dict[word] = weight
        
        return sentiment_dict
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment scores and metadata
        """
        if not text or not isinstance(text, str):
            return self._get_default_sentiment()
        
        # Clean text
        clean_text = self._clean_text(text)
        
        # Calculate lexicon-based sentiment
        lexicon_score = self._calculate_lexicon_sentiment(clean_text)
        
        # Calculate financial category scores
        category_scores = self._calculate_category_scores(clean_text)
        
        # Combine scores
        sentiment_score = lexicon_score * 0.7 + category_scores['overall'] * 0.3
        
        # Calculate intensity
        intensity = self._calculate_intensity(clean_text, sentiment_score)
        
        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = 'POSITIVE'
        elif sentiment_score < -0.2:
            sentiment_label = 'NEGATIVE'
        else:
            sentiment_label = 'NEUTRAL'
        
        return {
            'sentiment_score': float(sentiment_score),
            'sentiment_label': sentiment_label,
            'sentiment_intensity': float(intensity),
            'lexicon_score': float(lexicon_score),
            'category_scores': category_scores,
            'text_length': len(clean_text),
            'word_count': len(clean_text.split()),
            'source': 'offline_analyzer'
        }
    
    def analyze_dataframe(self, 
                         df: pd.DataFrame, 
                         text_column: str,
                         result_column: str = 'sentiment') -> pd.DataFrame:
        """
        Analyze sentiment for a DataFrame column
        
        Args:
            df: DataFrame containing text
            text_column: Column name with text
            result_column: Column name for sentiment results
            
        Returns:
            DataFrame with added sentiment columns
        """
        if df.empty or text_column not in df.columns:
            self.logger.warning("Invalid DataFrame or text column")
            return df
        
        # Create copy to avoid warnings
        result_df = df.copy()
        
        # Initialize sentiment columns
        result_df[f'{result_column}_score'] = 0.0
        result_df[f'{result_column}_label'] = 'NEUTRAL'
        result_df[f'{result_column}_intensity'] = 0.0
        
        # Analyze each row
        for idx, row in result_df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ''
            
            if text:
                sentiment_result = self.analyze_text(text)
                
                result_df.at[idx, f'{result_column}_score'] = sentiment_result['sentiment_score']
                result_df.at[idx, f'{result_column}_label'] = sentiment_result['sentiment_label']
                result_df.at[idx, f'{result_column}_intensity'] = sentiment_result['sentiment_intensity']
        
        self.logger.info(f"Analyzed {len(result_df)} texts in DataFrame")
        
        return result_df
    
    def _clean_text(self, text: str) -> str:
        """Clean text for analysis"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _calculate_lexicon_sentiment(self, text: str) -> float:
        """Calculate sentiment using simple lexicon"""
        words = text.split()
        
        if not words:
            return 0.0
        
        total_score = 0.0
        word_count = 0
        
        for word in words:
            if word in self.sentiment_dict:
                total_score += self.sentiment_dict[word]
                word_count += 1
        
        # Normalize score
        if word_count > 0:
            score = total_score / word_count
            # Clip to [-1, 1]
            return max(-1.0, min(1.0, score))
        
        return 0.0
    
    def _calculate_category_scores(self, text: str) -> Dict[str, Any]:
        """Calculate scores for each financial category"""
        words = text.split()
        word_count = len(words)
        
        if word_count == 0:
            return self._get_default_category_scores()
        
        # Initialize category counts
        category_counts = {category: 0 for category in self.financial_lexicon.keys()}
        
        # Count words in each category
        for word in words:
            for category, word_list in self.financial_lexicon.items():
                if word in word_list:
                    category_counts[category] += 1
        
        # Calculate scores
        category_scores = {}
        for category, count in category_counts.items():
            # Normalize by word count
            score = (count / word_count) * self.weights[category]
            category_scores[category] = float(score)
        
        # Calculate overall category score
        overall_score = sum(category_scores.values()) / len(category_scores)
        category_scores['overall'] = float(overall_score)
        
        # Add raw counts
        category_scores['word_count'] = word_count
        category_scores.update({f'{cat}_count': count for cat, count in category_counts.items()})
        
        return category_scores
    
    def _calculate_intensity(self, text: str, sentiment_score: float) -> float:
        """Calculate sentiment intensity"""
        words = text.split()
        
        if not words:
            return 0.0
        
        # Intensity based on sentiment score magnitude
        intensity = abs(sentiment_score)
        
        # Boost intensity for strong words
        strong_words = ['very', 'extremely', 'highly', 'greatly', 'massive', 'huge']
        strong_count = sum(1 for word in words if word in strong_words)
        
        if strong_count > 0:
            intensity = min(1.0, intensity + (strong_count * 0.1))
        
        # Boost for exclamation marks
        exclamation_count = text.count('!')
        if exclamation_count > 0:
            intensity = min(1.0, intensity + (exclamation_count * 0.05))
        
        # Boost for question marks (uncertainty)
        question_count = text.count('?')
        if question_count > 0:
            intensity = min(1.0, intensity + (question_count * 0.03))
        
        return float(intensity)
    
    def _get_default_sentiment(self) -> Dict[str, Any]:
        """Get default sentiment for empty/invalid text"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'NEUTRAL',
            'sentiment_intensity': 0.0,
            'lexicon_score': 0.0,
            'category_scores': self._get_default_category_scores(),
            'text_length': 0,
            'word_count': 0,
            'source': 'offline_analyzer'
        }
    
    def _get_default_category_scores(self) -> Dict[str, Any]:
        """Get default category scores"""
        return {
            'positive': 0.0,
            'negative': 0.0,
            'uncertainty': 0.0,
            'litigious': 0.0,
            'strong_modal': 0.0,
            'weak_modal': 0.0,
            'constraining': 0.0,
            'overall': 0.0,
            'word_count': 0
        }
    
    def get_lexicon_info(self) -> Dict[str, Any]:
        """Get information about the lexicon"""
        return {
            'total_words': sum(len(words) for words in self.financial_lexicon.values()),
            'categories': list(self.financial_lexicon.keys()),
            'words_per_category': {cat: len(words) for cat, words in self.financial_lexicon.items()},
            'weights': self.weights.copy()
        }


# Quick test
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("Financial Sentiment Analyzer - Offline Version")
    print("="*60)
    
    # Create analyzer
    analyzer = FinancialSentimentAnalyzerOffline(logger=logger)
    
    # Test texts
    test_texts = [
        "Apple reported strong earnings growth and exceeded expectations.",
        "The market declined sharply due to economic uncertainty.",
        "Investors are optimistic about future prospects.",
        "Company faces legal challenges and potential lawsuits."
    ]
    
    print("\nTesting sentiment analysis:")
    print("-" * 60)
    
    for i, text in enumerate(test_texts, 1):
        result = analyzer.analyze_text(text)
        print(f"\nText {i}: {text[:50]}...")
        print(f"  Score: {result['sentiment_score']:.3f}")
        print(f"  Label: {result['sentiment_label']}")
        print(f"  Intensity: {result['sentiment_intensity']:.3f}")
    
    # Test lexicon info
    lexicon_info = analyzer.get_lexicon_info()
    print(f"\nLexicon information:")
    print(f"  Total words: {lexicon_info['total_words']}")
    print(f"  Categories: {', '.join(lexicon_info['categories'])}")
    
    print("\n" + "="*60)
    print("Offline sentiment analyzer test completed!")
    print("="*60)