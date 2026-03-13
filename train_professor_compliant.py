"""
Training script compliant with professor requirements

Train on 2022-2024 data, save model for independent backtest on 2025 data.
"""

import os
import sys
import json
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def setup_logging():
    """Setup logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main function"""
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("PROFESSOR-COMPLIANT TRAINING: Train on 2022-2024 data")
    logger.info("="*80)
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load configuration
    config_file = "config_professor_requirements.json"
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded from: {config_file}")
    
    # Import modules
    try:
        from data.data_fetcher import DataFetcher
        from data.data_preprocessor import DataPreprocessor
        from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer
        from sentiment.sentiment_aggregator import SentimentAggregator
        from features.feature_engineer import FeatureEngineer
        from drl.trading_env import create_trading_environment
        from drl.a2c_agent import A2CAgent
        from training.trainer import DRLTrainer
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return
    
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Training (2022-2024)")
    logger.info("="*80)
    
    # Create training-specific config
    train_config = {
        'ticker': config['ticker'],
        'start_date': config['train_start_date'],  # 2022-01-01
        'end_date': config['train_end_date'],      # 2024-12-31
        'train_ratio': 0.8,
        'val_ratio': 0.2,
        'test_ratio': 0.0,  # No test during training - use separate backtest period
        'window_size': config['window_size'],
        'initial_balance': config['initial_balance'],
        'transaction_cost': config['transaction_cost'],
        'alpha': config['reward_params']['alpha'],
        'beta': config['reward_params']['beta'],
        'gamma': config['reward_params']['gamma'],
        'delta': config['reward_params']['delta'],
        'sentiment_threshold': config['trading_params']['sentiment_threshold'],
        'max_position_pct': config['trading_params']['max_position_pct'],
        'max_daily_trades': config['trading_params']['max_daily_trades'],
        'episodes': config['training_params']['episodes'],
        'early_stop_patience': config['training_params']['early_stop_patience'],
        'batch_size': config['training_params']['batch_size'],
        'learning_rate': config['training_params']['learning_rate'],
        'gamma_discount': config['training_params']['gamma_discount'],
        'entropy_coef': config['training_params']['entropy_coef'],
        'use_replay': config['training_params']['use_replay'],
        'replay_capacity': config['training_params']['replay_capacity'],
        'hidden_dim': config['model_params']['hidden_dim'],
        'num_layers': config['model_params']['num_layers'],
        'dropout_rate': config['model_params']['dropout_rate'],
        'weight_decay': config['model_params']['weight_decay'],
        'output_dir': config['output_params']['output_dir'],
        'save_checkpoints': config['output_params']['save_checkpoints'],
        'checkpoint_interval': config['output_params']['checkpoint_interval'],
        'log_interval': config['output_params']['log_interval'],
        'visualize_training': config['output_params']['visualize_training']
    }
    
    # Create output directory
    output_dir = config['output_params']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
    
    # Save training config
    train_config_path = os.path.join(output_dir, 'train_config.json')
    with open(train_config_path, 'w') as f:
        json.dump(train_config, f, indent=2)
    logger.info(f"Training configuration saved: {train_config_path}")
    
    # Create and run trainer
    try:
        trainer = DRLTrainer(config=train_config, logger=logger)
        
        # Step 1: Load training data (2022-2024 only)
        logger.info("Loading training data (2022-2024)...")
        if not trainer.load_and_prepare_data():
            logger.error("Failed to load training data")
            return
        
        # Step 2: Create training environment
        logger.info("Creating training environment...")
        if not trainer.create_environments():
            logger.error("Failed to create training environment")
            return
        
        # Step 3: Train model
        logger.info("Starting DRL training...")
        success = trainer.train()
        
        if not success:
            logger.error("Training failed")
            return
        
        # Step 4: Test on training period (optional)
        logger.info("Testing on training period...")
        test_results = trainer.test()
        
        # Save test results
        test_results_path = os.path.join(output_dir, 'train_period_test_results.json')
        with open(test_results_path, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Training period test results saved: {test_results_path}")
        
        # Get final model path
        if trainer.best_model_path:
            final_model_path = trainer.best_model_path
        else:
            final_model_path = os.path.join(output_dir, 'models', 'final_model.pth')
            trainer._save_final_model()
        
        logger.info(f"Final model saved: {final_model_path}")
        
        # Save full training summary
        summary = {
            'training_period': f"{config['train_start_date']} to {config['train_end_date']}",
            'backtest_period': f"{config['backtest_start_date']} to {config['backtest_end_date']}",
            'final_model_path': final_model_path,
            'training_results': test_results,
            'config_used': train_config,
            'training_completed': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        summary_path = os.path.join(output_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Training summary saved: {summary_path}")
        
        logger.info("\n" + "="*80)
        logger.info("TRAINING COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Model ready for independent backtest on 2025 data")
        logger.info(f"Model saved at: {final_model_path}")
        logger.info(f"Next: Run backtest_professor_compliant.py for independent evaluation")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        import traceback
        traceback.print_exc()
        return

if __name__ == "__main__":
    main()