"""
Main Entry Point - Complete DRL trading system

This is the main entry point for the proposal-based DRL trading system.
It integrates all modules and provides a command-line interface.

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import os
import sys
import logging
import argparse
import json
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    # Also log to file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"drl_trading_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(log_format))
    
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to file: {log_file}")
    
    return logger


def run_full_training(config_path: str = None, logger: logging.Logger = None):
    """
    Run complete training pipeline
    
    Args:
        config_path: Path to configuration file
        logger: Logger instance
    """
    from training.trainer import DRLTrainer
    
    logger.info("="*60)
    logger.info("Starting Complete Training Pipeline")
    logger.info("="*60)
    
    # Load configuration
    config = {}
    if config_path and os.path.exists(config_path):
        logger.info(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        logger.info("Using default configuration")
    
    # Create trainer
    trainer = DRLTrainer(config=config, logger=logger)
    
    # Step 1: Load and prepare data
    logger.info("\n" + "="*60)
    logger.info("Step 1: Loading and Preparing Data")
    logger.info("="*60)
    
    if not trainer.load_and_prepare_data():
        logger.error("Failed to load and prepare data")
        return False
    
    # Step 2: Create environments
    logger.info("\n" + "="*60)
    logger.info("Step 2: Creating Trading Environments")
    logger.info("="*60)
    
    if not trainer.create_environments():
        logger.error("Failed to create environments")
        return False
    
    # Step 3: Train model
    logger.info("\n" + "="*60)
    logger.info("Step 3: Training DRL Model")
    logger.info("="*60)
    
    success = trainer.train()
    if not success:
        logger.error("Training failed")
        return False
    
    # Step 4: Test model
    logger.info("\n" + "="*60)
    logger.info("Step 4: Testing Trained Model")
    logger.info("="*60)
    
    test_results = trainer.test()
    
    logger.info("\n" + "="*60)
    logger.info("Training Pipeline Completed Successfully!")
    logger.info("="*60)
    
    return True


def run_quick_demo(logger: logging.Logger):
    """
    Run a quick demo of the system
    
    Args:
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("Running Quick Demo")
    logger.info("="*60)
    
    # Create a minimal configuration for demo
    demo_config = {
        'ticker': 'AAPL',
        'start_date': '2024-01-01',
        'end_date': '2024-03-01',
        'episodes': 10,  # Very short for demo
        'early_stop_patience': 5,
        'output_dir': 'demo_results'
    }
    
    logger.info(f"Demo configuration: {json.dumps(demo_config, indent=2)}")
    
    # Run training with demo config
    return run_full_training_with_config(demo_config, logger)


def run_full_training_with_config(config: dict, logger: logging.Logger):
    """
    Run training with specific configuration
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    from training.trainer import DRLTrainer
    
    try:
        # Create trainer
        trainer = DRLTrainer(config=config, logger=logger)
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        if not trainer.load_and_prepare_data():
            logger.error("Failed to load data")
            return False
        
        # Create environments
        logger.info("Creating environments...")
        if not trainer.create_environments():
            logger.error("Failed to create environments")
            return False
        
        # Train
        logger.info("Training model...")
        success = trainer.train()
        
        if success:
            logger.info("Training completed successfully")
            return True
        else:
            logger.error("Training failed")
            return False
            
    except Exception as e:
        logger.error(f"Error in training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_modules(logger: logging.Logger):
    """
    Test individual modules
    
    Args:
        logger: Logger instance
    """
    logger.info("="*60)
    logger.info("Testing Individual Modules")
    logger.info("="*60)
    
    try:
        # Test data modules
        from data.data_fetcher import DataFetcher
        from data.data_preprocessor import DataPreprocessor
        
        logger.info("1. Testing data modules...")
        fetcher = DataFetcher(logger=logger)
        preprocessor = DataPreprocessor(logger=logger)
        
        # Test fetching data
        data = fetcher.fetch_stock_data('AAPL', '2024-01-01', '2024-01-10')
        logger.info(f"   Fetched data shape: {data.shape}")
        
        # Test preprocessing
        if not data.empty:
            cleaned = preprocessor.clean_stock_data(data)
            logger.info(f"   Cleaned data shape: {cleaned.shape}")
        
        # Test sentiment modules
        from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer
        from sentiment.sentiment_aggregator import SentimentAggregator
        
        logger.info("2. Testing sentiment modules...")
        analyzer = FinancialSentimentAnalyzer(logger=logger)
        aggregator = SentimentAggregator(logger=logger)
        
        # Test sentiment analysis
        test_text = "Apple reported strong earnings growth."
        sentiment = analyzer.analyze_text(test_text)
        logger.info(f"   Sentiment analysis: score={sentiment.get('sentiment_score', 0):.3f}")
        
        # Test feature engineering
        from features.feature_engineer import FeatureEngineer
        
        logger.info("3. Testing feature engineering...")
        engineer = FeatureEngineer(logger=logger)
        
        if not data.empty and 'Close' in data.columns:
            features = engineer.calculate_technical_indicators(data)
            logger.info(f"   Technical features: {len(features.columns)} columns")
        
        # Test DRL modules
        from drl.trading_env import create_trading_environment
        from drl.a2c_agent import A2CAgent
        
        logger.info("4. Testing DRL modules...")
        
        # Create sample data for environment
        import pandas as pd
        import numpy as np
        
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        price_data = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, 50),
            'Volume': np.random.randint(1000000, 10000000, 50)
        }).set_index('Date')
        
        sentiment_data = pd.DataFrame({
            'Date': dates,
            'sentiment_score': np.random.uniform(-1, 1, 50)
        }).set_index('Date')
        
        # Create environment
        env = create_trading_environment(price_data, sentiment_data, logger=logger)
        
        # Create agent
        state_dim = env.observation_space.shape[0]
        agent = A2CAgent(state_dim=state_dim, logger=logger)
        
        logger.info(f"   Environment state dimension: {state_dim}")
        logger.info(f"   Agent created successfully")
        
        # Test training module
        from training.trainer import DRLTrainer
        
        logger.info("5. Testing training module...")
        trainer = DRLTrainer(logger=logger)
        logger.info("   Trainer created successfully")
        
        logger.info("\nAll modules tested successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Module test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def generate_config_template(output_path: str = "config_template.json"):
    """
    Generate a configuration template
    
    Args:
        output_path: Path to save template
    """
    template = {
        "ticker": "AAPL",
        "start_date": "2010-01-01",
        "end_date": "2024-12-31",
        "train_ratio": 0.7,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "window_size": 10,
        "initial_balance": 10000,
        "transaction_cost": 0.001,
        "alpha": 1.0,
        "beta": 0.1,
        "gamma": 0.5,
        "delta": 0.01,
        "max_position_pct": 0.05,
        "max_daily_trades": 10,
        "episodes": 300,
        "early_stop_patience": 30,
        "batch_size": 64,
        "learning_rate": 0.0001,
        "gamma_discount": 0.99,
        "entropy_coef": 0.01,
        "use_replay": True,
        "replay_capacity": 10000,
        "output_dir": "training_results",
        "save_checkpoints": True,
        "checkpoint_interval": 10,
        "log_interval": 10,
        "visualize_training": True
    }
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Configuration template saved to: {output_path}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="DRL Trading System")
    
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'demo', 'test', 'config', 'modules'],
        default='demo',
        help='Mode to run: train (full training), demo (quick demo), test (test modules), config (generate config template), modules (test individual modules)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to configuration file (required for train mode)'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    logger.info("="*60)
    logger.info("DRL Trading System - Proposal Based Implementation")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Log level: {args.log_level}")
    
    try:
        if args.mode == 'train':
            # Full training
            if args.config and os.path.exists(args.config):
                run_full_training(args.config, logger)
            else:
                logger.error(f"Configuration file not found: {args.config}")
                logger.info("Please provide a valid config file or use --mode demo for a quick demo")
                logger.info("You can generate a template with: python main.py --mode config")
        
        elif args.mode == 'demo':
            # Quick demo
            run_quick_demo(logger)
        
        elif args.mode == 'test':
            # Run module tests
            test_modules(logger)
        
        elif args.mode == 'config':
            # Generate config template
            generate_config_template()
        
        elif args.mode == 'modules':
            # Test individual modules
            test_modules(logger)
        
        else:
            logger.error(f"Unknown mode: {args.mode}")
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info("\n" + "="*60)
    logger.info("DRL Trading System - Completed")
    logger.info("="*60)


if __name__ == "__main__":
    main()