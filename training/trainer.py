"""
Training System - Complete DRL training pipeline

According to proposal requirements:
1. End-to-end training pipeline with validation and early stopping
2. Hyperparameter optimization
3. Model checkpointing and logging
4. Integration of all components (data, sentiment, features, DRL)

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
try:
    from data.data_fetcher import DataFetcher
    from data.data_preprocessor import DataPreprocessor
    from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer
    from sentiment.sentiment_aggregator import SentimentAggregator
    from features.feature_engineer import FeatureEngineer
    from drl.trading_env import create_trading_environment
    from drl.a2c_agent import A2CAgent
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all modules are in the correct location")
    sys.exit(1)


class DRLTrainer:
    """
    Complete DRL training system
    
    Proposal requirement: End-to-end training pipeline with validation
    and early stopping
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize trainer
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        # Merge provided config with defaults (user config overrides defaults)
        default_config = self._get_default_config()
        if config:
            # Ensure all required keys exist by merging defaults with user config
            self.config = {**default_config, **config}
        else:
            self.config = default_config
        
        self.logger = logger or logging.getLogger(__name__)
        
        # Components
        self.data_fetcher = None
        self.data_preprocessor = None
        self.sentiment_analyzer = None
        self.sentiment_aggregator = None
        self.feature_engineer = None
        
        # Training state
        self.training_data = None
        self.validation_data = None
        self.test_data = None
        self.env_train = None
        self.env_val = None
        self.env_test = None
        self.agent = None
        
        # Results
        self.training_history = []
        self.validation_history = []
        self.best_model_path = None
        self.best_validation_reward = -np.inf
        
        # Setup
        self._setup_directories()
        self._init_components()
        
        self.logger.info("DRL Trainer initialized")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # Data configuration
            'ticker': 'AAPL',
            'start_date': '2010-01-01',
            'end_date': '2024-12-31',
            'train_ratio': 0.7,
            'val_ratio': 0.15,
            'test_ratio': 0.15,
            
            # Feature engineering
            'window_size': 10,
            'technical_indicators': True,
            'sentiment_features': True,
            'attention_mechanism': True,
            
            # Environment configuration
            'initial_balance': 10000,
            'transaction_cost': 0.001,
            'alpha': 1.0,    # Return weight
            'beta': 0.1,     # Risk penalty weight
            'gamma': 0.5,    # Sentiment alignment weight
            'delta': 0.01,   # Transaction cost weight
            'max_position_pct': 0.05,
            'max_daily_trades': 10,
            
            # Training configuration
            'episodes': 300,
            'early_stop_patience': 30,
            'batch_size': 64,
            'learning_rate': 1e-4,
            'gamma_discount': 0.99,
            'entropy_coef': 0.01,
            'use_replay': True,
            'replay_capacity': 10000,
            
            # Output configuration
            'output_dir': 'training_results',
            'save_checkpoints': True,
            'checkpoint_interval': 10,
            'log_interval': 10,
            'visualize_training': True
        }
    
    def _setup_directories(self):
        """Setup output directories"""
        self.output_dir = self.config.get('output_dir', 'training_results')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.model_dir = os.path.join(self.output_dir, 'models')
        
        for directory in [self.output_dir, self.checkpoint_dir, self.log_dir, self.model_dir]:
            os.makedirs(directory, exist_ok=True)
        
        self.logger.info(f"Output directories created: {self.output_dir}")
    
    def _init_components(self):
        """Initialize all components"""
        self.logger.info("Initializing components...")
        
        # Data fetcher
        self.data_fetcher = DataFetcher(logger=self.logger)
        
        # Data preprocessor
        self.data_preprocessor = DataPreprocessor(logger=self.logger)
        
        # Sentiment analyzer
        self.sentiment_analyzer = FinancialSentimentAnalyzer(logger=self.logger)
        
        # Sentiment aggregator
        self.sentiment_aggregator = SentimentAggregator(logger=self.logger)
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer(logger=self.logger)
        
        self.logger.info("All components initialized")
    
    def load_and_prepare_data(self) -> bool:
        """
        Load and prepare all data
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Loading and preparing data...")
        
        try:
            # 1. Fetch stock data
            ticker = self.config['ticker']
            start_date = self.config['start_date']
            end_date = self.config['end_date']
            
            self.logger.info(f"Fetching {ticker} data from {start_date} to {end_date}")
            stock_data = self.data_fetcher.fetch_stock_data(ticker, start_date, end_date)
            
            if stock_data.empty:
                self.logger.error(f"Failed to fetch {ticker} data")
                return False
            
            self.logger.info(f"Fetched {len(stock_data)} days of stock data")
            
            # 2. Clean stock data
            cleaned_stock_data = self.data_preprocessor.clean_stock_data(stock_data)
            
            # 3. Fetch and analyze news data
            self.logger.info("Fetching and analyzing news data...")
            news_data = self.data_fetcher.fetch_financial_news(
                ticker, 
                from_date=start_date,
                to_date=end_date,
                limit=100
            )
            
            if not news_data.empty:
                # Analyze sentiment
                news_data = self.sentiment_analyzer.analyze_dataframe(news_data, 'title')
                
                # Aggregate sentiment by time
                aggregated_sentiment = self.sentiment_aggregator.aggregate_by_time(
                    news_data, 
                    time_column='published_at',
                    frequency='1D'
                )
            else:
                # Create mock sentiment data
                self.logger.warning("No news data, creating mock sentiment")
                dates = pd.date_range(start=start_date, end=end_date, freq='D')
                aggregated_sentiment = pd.DataFrame({
                    'date': dates,
                    'sentiment_score': np.random.uniform(-1, 1, len(dates)),
                    'sentiment_intensity': np.random.uniform(0, 1, len(dates))
                }).set_index('date')
            
            # 4. Align stock and sentiment data
            stock_data_ts = cleaned_stock_data.set_index('Date') if 'Date' in cleaned_stock_data.columns else cleaned_stock_data
            sentiment_data_ts = aggregated_sentiment
            
            aligned_price, aligned_sentiment = self.data_preprocessor.align_time_series(
                stock_data_ts, sentiment_data_ts, frequency='1D'
            )
            
            if aligned_price.empty or aligned_sentiment.empty:
                self.logger.error("Failed to align price and sentiment data")
                return False
            
            self.logger.info(f"Aligned data: price shape={aligned_price.shape}, sentiment shape={aligned_sentiment.shape}")
            
            # 5. Calculate technical indicators
            self.logger.info("Calculating technical indicators...")
            price_features = self.feature_engineer.calculate_technical_indicators(aligned_price)
            
            # 6. Engineer sentiment features
            self.logger.info("Engineering sentiment features...")
            sentiment_features = self.feature_engineer.engineer_sentiment_features(
                aligned_sentiment, price_features
            )
            
            # 7. Split into train/val/test sets
            self.logger.info("Splitting data into train/val/test sets...")
            
            # Combine features for splitting
            combined_features = pd.concat([price_features, sentiment_features], axis=1)
            
            # Fill NaN values instead of dropping
            combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Ensure all columns are numeric (convert object types to float)
            for col in combined_features.columns:
                if combined_features[col].dtype == 'object':
                    combined_features[col] = pd.to_numeric(combined_features[col], errors='coerce')
            
            # Fill NaN again after conversion
            combined_features = combined_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            # Check if we have enough data for splitting
            min_data_for_split = 20  # Minimum data points for proper split
            window_size = self.config.get('window_size', 10)
            
            if len(combined_features) < max(min_data_for_split, window_size * 2):
                # Not enough data for proper split, use all data for training
                self.logger.warning(f"Insufficient data for split ({len(combined_features)} rows). Using all data for training.")
                self.training_data = combined_features
                self.validation_data = pd.DataFrame()
                self.test_data = pd.DataFrame()
            else:
                # Time series split
                train_ratio = self.config['train_ratio']
                val_ratio = self.config['val_ratio']
                test_ratio = self.config.get('test_ratio', 0.15)  # Default to 0.15 if not specified
                
                train_idx = int(len(combined_features) * train_ratio)
                val_idx = train_idx + int(len(combined_features) * val_ratio)
                
                # Handle test ratio of 0 or very small
                if test_ratio <= 0.001:  # Effectively 0
                    # All remaining data goes to validation if test_ratio is 0
                    self.training_data = combined_features.iloc[:train_idx]
                    self.validation_data = combined_features.iloc[train_idx:]
                    self.test_data = pd.DataFrame()
                else:
                    self.training_data = combined_features.iloc[:train_idx]
                    self.validation_data = combined_features.iloc[train_idx:val_idx]
                    self.test_data = combined_features.iloc[val_idx:]
                    
                    # If test data is too small (less than window_size), merge it into validation
                    window_size = self.config.get('window_size', 10)
                    if len(self.test_data) < window_size:
                        self.logger.warning(f"Test data too small ({len(self.test_data)} rows < window_size {window_size}). Merging into validation data.")
                        self.validation_data = pd.concat([self.validation_data, self.test_data])
                        self.test_data = pd.DataFrame()
            
            # Save feature configuration for consistent backtesting
            try:
                import torch
                import os
                
                # Create feature configuration
                feature_config = {
                    'feature_columns': combined_features.columns.tolist(),
                    'feature_count': len(combined_features.columns),
                    'window_size': self.config.get('window_size', 10),
                    'state_dimension': (len(combined_features.columns) * self.config.get('window_size', 10)) + 5,
                    'price_feature_columns': price_features.columns.tolist() if 'price_features' in locals() else [],
                    'sentiment_feature_columns': sentiment_features.columns.tolist() if 'sentiment_features' in locals() else []
                }
                
                # Save to model directory
                output_dir = self.config.get('output_dir', 'output')
                feature_config_path = os.path.join(output_dir, 'feature_config.pth')
                
                torch.save(feature_config, feature_config_path)
                self.logger.info(f"Feature configuration saved to {feature_config_path}")
                self.logger.info(f"Total features: {len(combined_features.columns)}")
                self.logger.info(f"Expected state dimension: {feature_config['state_dimension']}")
                
            except Exception as e:
                self.logger.warning(f"Failed to save feature configuration: {e}")
            
            self.logger.info(f"Data split: Train={len(self.training_data)}, Val={len(self.validation_data)}, Test={len(self.test_data)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading and preparing data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def create_environments(self) -> bool:
        """
        Create training, validation, and test environments
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Creating trading environments...")
        
        try:
            if self.training_data is None or self.validation_data is None:
                self.logger.error("Data not loaded. Call load_and_prepare_data() first.")
                return False
            
            # Separate price and sentiment features
            price_cols = [col for col in self.training_data.columns 
                         if not col.startswith('sentiment_') and col != 'sentiment_score']
            sentiment_cols = [col for col in self.training_data.columns 
                            if col.startswith('sentiment_') or col == 'sentiment_score']
            
            # Training environment
            self.logger.info("Creating training environment...")
            price_features_train = self.training_data[price_cols]
            sentiment_features_train = self.training_data[sentiment_cols] if sentiment_cols else pd.DataFrame()
            
            self.env_train = create_trading_environment(
                price_features=price_features_train,
                sentiment_features=sentiment_features_train,
                config={
                    'initial_balance': self.config['initial_balance'],
                    'window_size': self.config['window_size'],
                    'transaction_cost': self.config['transaction_cost'],
                    'alpha': self.config['alpha'],
                    'beta': self.config['beta'],
                    'gamma': self.config['gamma'],
                    'delta': self.config['delta'],
                    'max_position_pct': self.config['max_position_pct'],
                    'max_daily_trades': self.config['max_daily_trades']
                },
                logger=self.logger
            )
            
            # Validation environment (only create if enough data)
            window_size = self.config.get('window_size', 10)
            if len(self.validation_data) > window_size:
                self.logger.info("Creating validation environment...")
                price_features_val = self.validation_data[price_cols]
                sentiment_features_val = self.validation_data[sentiment_cols] if sentiment_cols else pd.DataFrame()
                
                self.env_val = create_trading_environment(
                    price_features=price_features_val,
                    sentiment_features=sentiment_features_val,
                    config={
                        'initial_balance': self.config['initial_balance'],
                        'window_size': window_size,
                        'transaction_cost': self.config['transaction_cost'],
                        'alpha': self.config['alpha'],
                        'beta': self.config['beta'],
                        'gamma': self.config['gamma'],
                        'delta': self.config['delta'],
                        'max_position_pct': self.config['max_position_pct'],
                        'max_daily_trades': self.config['max_daily_trades']
                    },
                    logger=self.logger
                )
            else:
                self.logger.warning(f"Validation data too small ({len(self.validation_data)} rows <= window_size {window_size}). Skipping validation environment.")
                self.env_val = None
            
            # Test environment (only create if enough data)
            if self.test_data is not None and len(self.test_data) > window_size:
                self.logger.info("Creating test environment...")
                price_features_test = self.test_data[price_cols]
                sentiment_features_test = self.test_data[sentiment_cols] if sentiment_cols else pd.DataFrame()
                
                self.env_test = create_trading_environment(
                    price_features=price_features_test,
                    sentiment_features=sentiment_features_test,
                    config={
                        'initial_balance': self.config['initial_balance'],
                        'window_size': window_size,
                        'transaction_cost': self.config['transaction_cost'],
                        'alpha': self.config['alpha'],
                        'beta': self.config['beta'],
                        'gamma': self.config['gamma'],
                        'delta': self.config['delta'],
                        'max_position_pct': self.config['max_position_pct'],
                        'max_daily_trades': self.config['max_daily_trades']
                    },
                    logger=self.logger
                )
            elif self.test_data is not None and len(self.test_data) > 0:
                self.logger.warning(f"Test data too small ({len(self.test_data)} rows <= window_size {window_size}). Skipping test environment.")
                self.env_test = None
            else:
                self.env_test = None
            
            # Create agent
            state_dim = self.env_train.observation_space.shape[0]
            
            # Get network architecture parameters with defaults
            hidden_dim = self.config.get('hidden_dim', 256)
            num_layers = self.config.get('num_layers', 3)
            dropout_rate = self.config.get('dropout_rate', 0.2)
            weight_decay = self.config.get('weight_decay', 1e-5)
            
            self.agent = A2CAgent(
                state_dim=state_dim,
                action_dim=3,
                learning_rate=self.config['learning_rate'],
                gamma=self.config['gamma_discount'],
                entropy_coef=self.config['entropy_coef'],
                use_replay=self.config['use_replay'],
                replay_capacity=self.config['replay_capacity'],
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                logger=self.logger
            )
            
            self.logger.info(f"Environments created. State dimension: {state_dim}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating environments: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def train_episode(self, env, training: bool = True) -> Tuple[float, Dict[str, Any]]:
        """
        Train one episode
        
        Args:
            env: Environment to train on
            training: Whether to update network parameters
            
        Returns:
            total_reward: Total reward for episode
            episode_info: Additional episode information
        """
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        action_counts = {'sell': 0, 'hold': 0, 'buy': 0}
        
        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(state, deterministic=not training)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            if training:
                self.agent.store_transition(state, action, reward, log_prob, value, done)
            
            # Update statistics
            total_reward += reward
            step_count += 1
            
            # Count actions
            if action == 0:
                action_counts['sell'] += 1
            elif action == 1:
                action_counts['hold'] += 1
            else:
                action_counts['buy'] += 1
            
            # Move to next state
            state = next_state
        
        # Update network if training
        if training and len(self.agent.states) > 0:
            # Get value of next state for final update
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.agent.device)
                _, next_value = self.agent.model(next_state_tensor)
            
            # Update agent
            self.agent.update(next_value)
            
            # Update from replay buffer
            if self.agent.use_replay:
                self.agent.update_from_replay(batch_size=self.config['batch_size'])
        
        episode_info = {
            'steps': step_count,
            'action_counts': action_counts,
            'final_net_worth': info.get('net_worth', 0),
            'total_trades': info.get('total_trades', 0)
        }
        
        return total_reward, episode_info
    
    def validate_episode(self, env) -> Tuple[float, Dict[str, Any]]:
        """
        Validate one episode (no training)
        
        Args:
            env: Environment to validate on
            
        Returns:
            total_reward: Total reward for episode
            episode_info: Additional episode information
        """
        return self.train_episode(env, training=False)
    
    def train(self) -> bool:
        """
        Main training loop
        
        Returns:
            True if training successful, False otherwise
        """
        self.logger.info("Starting training...")
        
        if self.env_train is None or self.env_val is None or self.agent is None:
            self.logger.error("Environments or agent not initialized")
            return False
        
        episodes = self.config['episodes']
        early_stop_patience = self.config['early_stop_patience']
        checkpoint_interval = self.config['checkpoint_interval']
        log_interval = self.config['log_interval']
        
        patience_counter = 0
        best_episode = 0
        
        # Training loop
        for episode in range(episodes):
            # Training episode
            self.agent.model.train()
            train_reward, train_info = self.train_episode(self.env_train, training=True)
            self.training_history.append(train_reward)
            
            # Validation episode
            self.agent.model.eval()
            val_reward, val_info = self.validate_episode(self.env_val)
            self.validation_history.append(val_reward)
            
            # Check for improvement
            if val_reward > self.best_validation_reward:
                self.best_validation_reward = val_reward
                best_episode = episode
                patience_counter = 0
                
                # Save best model
                self._save_best_model(episode)
                self.logger.info(f"Episode {episode}: New best validation reward: {val_reward:.2f}")
            else:
                patience_counter += 1
            
            # Logging
            if episode % log_interval == 0:
                self._log_training_progress(episode, train_reward, val_reward, train_info, val_info, best_episode)
            
            # Checkpoint
            if self.config['save_checkpoints'] and episode % checkpoint_interval == 0:
                self._save_checkpoint(episode)
            
            # Early stopping
            if patience_counter >= early_stop_patience:
                self.logger.info(f"Early stopping triggered at episode {episode}")
                self.logger.info(f"Best validation reward: {self.best_validation_reward:.2f} (episode {best_episode})")
                break
        
        # Final logging
        self._log_final_results(best_episode)
        
        # Save final model
        self._save_final_model()
        
        # Save training history
        self._save_training_history()
        
        # Visualize training (if enabled)
        if self.config['visualize_training']:
            self._visualize_training()
        
        return True
    
    def _save_best_model(self, episode: int):
        """Save best model"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f"best_model_ep{episode}_{timestamp}.pth"
        model_path = os.path.join(self.model_dir, model_name)
        
        self.agent.save_checkpoint(model_path)
        self.best_model_path = model_path
        
        # Also save config
        config_path = os.path.join(self.model_dir, f"config_ep{episode}_{timestamp}.json")
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint_name = f"checkpoint_ep{episode}.pth"
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        self.agent.save_checkpoint(checkpoint_path)
        self.logger.debug(f"Checkpoint saved: {checkpoint_path}")
    
    def _save_final_model(self):
        """Save final model"""
        final_model_path = os.path.join(self.model_dir, "final_model.pth")
        self.agent.save_checkpoint(final_model_path)
        self.logger.info(f"Final model saved: {final_model_path}")
    
    def _log_training_progress(self, 
                              episode: int, 
                              train_reward: float, 
                              val_reward: float,
                              train_info: Dict[str, Any],
                              val_info: Dict[str, Any],
                              best_episode: int):
        """Log training progress"""
        log_message = (
            f"Episode {episode:4d} | "
            f"Train Reward: {train_reward:8.2f} | "
            f"Val Reward: {val_reward:8.2f} | "
            f"Best Val: {self.best_validation_reward:8.2f} | "
            f"Actions: B{train_info['action_counts']['buy']:3d}/"
            f"S{train_info['action_counts']['sell']:3d}/"
            f"H{train_info['action_counts']['hold']:3d} | "
            f"Patience: {len(self.validation_history) - best_episode:2d}/"
            f"{self.config['early_stop_patience']}"
        )
        
        self.logger.info(log_message)
        
        # Also write to file
        log_file = os.path.join(self.log_dir, 'training_log.txt')
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now().isoformat()} - {log_message}\n")
    
    def _log_final_results(self, best_episode: int):
        """Log final training results"""
        if len(self.training_history) == 0:
            return
        
        final_results = {
            'total_episodes': len(self.training_history),
            'best_episode': best_episode,
            'best_validation_reward': self.best_validation_reward,
            'final_training_reward': self.training_history[-1],
            'final_validation_reward': self.validation_history[-1],
            'training_reward_mean': np.mean(self.training_history),
            'training_reward_std': np.std(self.training_history),
            'validation_reward_mean': np.mean(self.validation_history),
            'validation_reward_std': np.std(self.validation_history),
            'config': self.config
        }
        
        # Log to console
        self.logger.info("\n" + "="*60)
        self.logger.info("Training Results")
        self.logger.info("="*60)
        self.logger.info(f"Total Episodes: {final_results['total_episodes']}")
        self.logger.info(f"Best Episode: {final_results['best_episode']}")
        self.logger.info(f"Best Validation Reward: {final_results['best_validation_reward']:.4f}")
        self.logger.info(f"Final Training Reward: {final_results['final_training_reward']:.4f}")
        self.logger.info(f"Final Validation Reward: {final_results['final_validation_reward']:.4f}")
        self.logger.info(f"Training Reward Mean±Std: {final_results['training_reward_mean']:.4f}±{final_results['training_reward_std']:.4f}")
        self.logger.info(f"Validation Reward Mean±Std: {final_results['validation_reward_mean']:.4f}±{final_results['validation_reward_std']:.4f}")
        
        # Save to file
        results_file = os.path.join(self.output_dir, 'training_results.json')
        with open(results_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.logger.info(f"Results saved: {results_file}")
    
    def _save_training_history(self):
        """Save training history to file"""
        history = {
            'training_rewards': self.training_history,
            'validation_rewards': self.validation_history,
            'timestamps': [datetime.now().isoformat()] * len(self.training_history)
        }
        
        history_file = os.path.join(self.output_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        self.logger.info(f"Training history saved: {history_file}")
    
    def _visualize_training(self):
        """Visualize training progress"""
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            # Training and validation rewards
            plt.subplot(1, 2, 1)
            plt.plot(self.training_history, label='Training Reward', alpha=0.7)
            plt.plot(self.validation_history, label='Validation Reward', alpha=0.7)
            plt.xlabel('Episode')
            plt.ylabel('Total Reward')
            plt.title('Training Progress')
            plt.legend()
            plt.grid(True)
            
            # Smoothed rewards
            plt.subplot(1, 2, 2)
            window = min(20, len(self.training_history) // 10)
            if window < 2:
                window = 2  # 最小窗口为2以确保平滑效果
            
            # 确保有足够数据点
            if len(self.training_history) >= window:
                train_smooth = pd.Series(self.training_history).rolling(window=window).mean()
                val_smooth = pd.Series(self.validation_history).rolling(window=window).mean()
                plt.plot(train_smooth, label=f'Training (MA{window})', alpha=0.7)
                plt.plot(val_smooth, label=f'Validation (MA{window})', alpha=0.7)
            else:
                # 数据不足时显示原始数据
                plt.plot(self.training_history, label='Training (Raw)', alpha=0.7)
                plt.plot(self.validation_history, label='Validation (Raw)', alpha=0.7)
            
            plt.xlabel('Episode')
            plt.ylabel('Smoothed Reward')
            plt.title('Smoothed Training Progress')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plot_path = os.path.join(self.output_dir, 'training_curve.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Training curve saved: {plot_path}")
            
        except ImportError:
            self.logger.warning("Matplotlib not available, skipping visualization")
        except Exception as e:
            self.logger.error(f"Error visualizing training: {e}")
    
    def test(self) -> Dict[str, Any]:
        """
        Test trained model on test set
        
        Returns:
            Test results dictionary
        """
        self.logger.info("Testing trained model...")
        
        if self.env_test is None:
            self.logger.warning("Test environment not available")
            return {}
        
        if self.best_model_path is None:
            self.logger.warning("No best model found, using current model")
        else:
            # Load best model
            self.logger.info(f"Loading best model: {self.best_model_path}")
            self.agent.load_checkpoint(self.best_model_path)
        
        # Test episode
        self.agent.model.eval()
        test_reward, test_info = self.validate_episode(self.env_test)
        
        # Get portfolio history
        portfolio_history = self.env_test.get_portfolio_history()
        
        # Calculate additional metrics
        if not portfolio_history.empty:
            initial_balance = self.config['initial_balance']
            final_net_worth = portfolio_history['net_worth'].iloc[-1]
            max_net_worth = portfolio_history['net_worth'].max()
            min_net_worth = portfolio_history['net_worth'].min()
            
            total_return = (final_net_worth - initial_balance) / initial_balance * 100
            max_drawdown = (max_net_worth - min_net_worth) / max_net_worth * 100 if max_net_worth > 0 else 0
        else:
            total_return = 0
            max_drawdown = 0
        
        # Build results
        test_results = {
            'test_reward': test_reward,
            'total_return_pct': total_return,
            'max_drawdown_pct': max_drawdown,
            'total_trades': test_info.get('total_trades', 0),
            'final_net_worth': test_info.get('final_net_worth', 0),
            'action_counts': test_info.get('action_counts', {}),
            'portfolio_history': portfolio_history.to_dict() if not portfolio_history.empty else {}
        }
        
        # Log results
        self.logger.info("\n" + "="*60)
        self.logger.info("Test Results")
        self.logger.info("="*60)
        self.logger.info(f"Test Reward: {test_reward:.4f}")
        self.logger.info(f"Total Return: {total_return:.2f}%")
        self.logger.info(f"Max Drawdown: {max_drawdown:.2f}%")
        self.logger.info(f"Total Trades: {test_info.get('total_trades', 0)}")
        self.logger.info(f"Final Net Worth: ${test_info.get('final_net_worth', 0):.2f}")
        
        # Save results
        results_file = os.path.join(self.output_dir, 'test_results.json')
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        self.logger.info(f"Test results saved: {results_file}")
        
        return test_results


# Quick demo
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("DRL Training System - Quick Demo")
    print("="*60)
    
    # Create trainer with demo config
    demo_config = {
        'ticker': 'AAPL',
        'start_date': '2024-01-01',
        'end_date': '2024-03-01',
        'episodes': 50,  # Short training for demo
        'early_stop_patience': 20,
        'output_dir': 'demo_training_results'
    }
    
    trainer = DRLTrainer(config=demo_config, logger=logger)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    if not trainer.load_and_prepare_data():
        print("   Failed to load data, exiting")
        sys.exit(1)
    
    print("   Data loaded successfully")
    
    # Create environments
    print("\n2. Creating environments...")
    if not trainer.create_environments():
        print("   Failed to create environments, exiting")
        sys.exit(1)
    
    print("   Environments created successfully")
    
    # Train
    print("\n3. Training model (short demo)...")
    success = trainer.train()
    
    if success:
        print("   Training completed successfully")
        
        # Test
        print("\n4. Testing trained model...")
        test_results = trainer.test()
        
        if test_results:
            print(f"   Test reward: {test_results.get('test_reward', 0):.4f}")
            print(f"   Total return: {test_results.get('total_return_pct', 0):.2f}%")
        
        print("\n🎉 Training system demo completed!")
        print(f"   Results saved to: {demo_config['output_dir']}")
    else:
        print("   Training failed")
    
    print("\nTraining system demo completed!")