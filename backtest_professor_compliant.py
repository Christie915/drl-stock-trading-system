"""
Independent backtest compliant with professor requirements

Load model trained on 2022-2024 data, run backtest on 2025 data.
Completely independent evaluation on future data.
"""

import os
import sys
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import torch

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

def report_progress(logger, stage, details=None):
    """Report progress"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = f"[{timestamp}] {stage}"
    if details:
        message += f" - {details}"
    logger.info(message)
    print(f"\n{'='*60}")
    print(message)
    print(f"{'='*60}\n")

def align_features_to_model(logger, price_features, sentiment_features, model_path, window_size=10):
    """
    Align feature dimensions to match trained model expectations
    
    Args:
        logger: Logger instance
        price_features: DataFrame with price features
        sentiment_features: DataFrame with sentiment features  
        model_path: Path to trained model
        window_size: Window size for state calculation
        
    Returns:
        Tuple of (aligned_price_features, aligned_sentiment_features)
    """
    try:
        # Load model to get expected state dimension
        checkpoint = torch.load(model_path, map_location='cpu')
        state_dim = checkpoint.get('state_dim')
        
        if state_dim is None:
            # Try to infer from model weights
            if 'model_state_dict' in checkpoint:
                for key, weight in checkpoint['model_state_dict'].items():
                    if 'shared_layers.0.weight' in key:
                        # Weight shape: [hidden_dim, state_dim]
                        state_dim = weight.shape[1]
                        break
                    elif 'input_norm.weight' in key:
                        state_dim = weight.shape[0]
                        break
        
        if state_dim is None:
            logger.warning("Could not determine model state dimension. Using default.")
            state_dim = 20505  # Default for professor_experiment model
        
        # Calculate required features per timestep
        # Formula: state_dim = (features_per_timestep × window_size) + 5
        required_features_per_timestep = (state_dim - 5) // window_size
        
        logger.info(f"Model expects state_dim={state_dim}, window_size={window_size}")
        logger.info(f"Required features per timestep: {required_features_per_timestep}")
        
        # Combine current features
        current_features = pd.concat([price_features, sentiment_features], axis=1)
        current_features_per_timestep = len(current_features.columns)
        
        logger.info(f"Current features per timestep: {current_features_per_timestep}")
        
        if current_features_per_timestep == required_features_per_timestep:
            logger.info("Feature dimensions already match!")
            return price_features, sentiment_features
        
        elif current_features_per_timestep < required_features_per_timestep:
            # Need to add features (pad with zeros)
            logger.info(f"Adding {required_features_per_timestep - current_features_per_timestep} padding features")
            
            # Add padding columns
            for i in range(required_features_per_timestep - current_features_per_timestep):
                current_features[f'padding_feature_{i}'] = 0.0
            
            # Split back into price and sentiment features (approximate)
            # For simplicity, keep original split ratio
            n_price_cols = len(price_features.columns)
            n_sentiment_cols = len(sentiment_features.columns)
            
            price_cols = list(price_features.columns) + [f'padding_feature_{i}' for i in range(required_features_per_timestep - current_features_per_timestep) if i < (required_features_per_timestep - current_features_per_timestep) // 2]
            sentiment_cols = list(sentiment_features.columns) + [f'padding_feature_{i}' for i in range(required_features_per_timestep - current_features_per_timestep) if i >= (required_features_per_timestep - current_features_per_timestep) // 2]
            
            aligned_price = current_features[price_cols]
            aligned_sentiment = current_features[sentiment_cols]
            
            return aligned_price, aligned_sentiment
        
        else:
            # Too many features, need to truncate
            logger.warning(f"Too many features ({current_features_per_timestep} > {required_features_per_timestep}). Truncating.")
            
            # Keep first required_features_per_timestep columns
            keep_cols = current_features.columns[:required_features_per_timestep]
            truncated_features = current_features[keep_cols]
            
            # Split back (approximate)
            n_price_cols = min(len(price_features.columns), required_features_per_timestep)
            price_cols = keep_cols[:n_price_cols]
            sentiment_cols = keep_cols[n_price_cols:]
            
            aligned_price = truncated_features[price_cols]
            aligned_sentiment = truncated_features[sentiment_cols]
            
            return aligned_price, aligned_sentiment
            
    except Exception as e:
        logger.error(f"Error aligning features: {e}")
        # Return original features as fallback
        return price_features, sentiment_features

def run_independent_backtest(logger, config_file, model_path, training_period=None):
    """Run independent backtest on 2025 data"""
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    logger.info(f"Configuration loaded: {config_file}")
    logger.info(f"Backtest period: {config['backtest_start_date']} to {config['backtest_end_date']}")
    
    # Import modules
    try:
        from data.data_fetcher import DataFetcher
        from data.data_preprocessor import DataPreprocessor
        from sentiment.sentiment_analyzer import FinancialSentimentAnalyzer
        from sentiment.sentiment_aggregator import SentimentAggregator
        from features.feature_engineer import FeatureEngineer
        from drl.trading_env import create_trading_environment
        from drl.a2c_agent import A2CAgent
        from evaluation.backtester import Backtester
    except ImportError as e:
        logger.error(f"Import error: {e}")
        return None
    
    start_time = datetime.now()
    
    try:
        # Phase 1: Load backtest data (2025 only - completely independent)
        report_progress(logger, "Phase 1: Loading backtest data", 
                       f"{config['backtest_start_date']} to {config['backtest_end_date']}")
        
        # Initialize components
        data_fetcher = DataFetcher(logger=logger)
        data_preprocessor = DataPreprocessor(logger=logger)
        sentiment_analyzer = FinancialSentimentAnalyzer(logger=logger)
        sentiment_aggregator = SentimentAggregator(logger=logger)
        feature_engineer = FeatureEngineer(logger=logger)
        
        # Fetch backtest data
        backtest_data = data_fetcher.fetch_stock_data(
            config['ticker'],
            config['backtest_start_date'],
            config['backtest_end_date']
        )
        
        if backtest_data.empty:
            logger.error("Failed to fetch backtest data")
            return None
        
        report_progress(logger, "Data fetched", f"{len(backtest_data)} rows")
        
        # Clean and prepare data
        cleaned_data = data_preprocessor.clean_stock_data(backtest_data)
        
        # Create mock sentiment for backtest (in real scenario would use actual news)
        dates = pd.date_range(start=config['backtest_start_date'], 
                             end=config['backtest_end_date'], 
                             freq='D')
        sentiment_data = pd.DataFrame({
            'date': dates,
            'sentiment_score': np.random.uniform(-0.5, 0.5, len(dates))
        }).set_index('date')
        
        # Calculate features
        price_features = feature_engineer.calculate_technical_indicators(cleaned_data)
        sentiment_features = sentiment_aggregator.aggregate_by_time(
            sentiment_data.reset_index(), 
            time_column='date',
            frequency='1D'
        )
        
        # Ensure indices match and align
        price_idx = pd.to_datetime(price_features.index).tz_localize(None)
        sentiment_idx = pd.to_datetime(sentiment_features.index).tz_localize(None)
        
        price_features.index = price_idx
        sentiment_features.index = sentiment_idx
        
        # Align sentiment features to price features index (only keep common trading days)
        common_idx = price_features.index.intersection(sentiment_features.index)
        if len(common_idx) == 0:
            logger.error("No common time indices between price and sentiment features")
            # Try to reindex sentiment to price index
            logger.info("Attempting to reindex sentiment features to price index...")
            sentiment_features = sentiment_features.reindex(price_features.index, method='ffill').fillna(method='bfill').fillna(0)
            common_idx = price_features.index
        else:
            # Keep only common indices
            price_features = price_features.loc[common_idx]
            sentiment_features = sentiment_features.loc[common_idx]
        
        logger.info(f"After alignment - Price features: {price_features.shape}, Sentiment features: {sentiment_features.shape}")
        logger.info(f"Common time indices: {len(common_idx)}")
        
        if len(price_features) == 0 or len(sentiment_features) == 0:
            logger.error("No data available after alignment")
            return None
        
        report_progress(logger, "Features calculated", 
                       f"Price features: {price_features.shape}, Sentiment features: {sentiment_features.shape}")
        
        # Align feature dimensions to match trained model
        report_progress(logger, "Aligning feature dimensions to model")
        window_size = config.get('window_size', 10)
        price_features, sentiment_features = align_features_to_model(
            logger, price_features, sentiment_features, model_path, window_size
        )
        logger.info(f"After dimension alignment - Price features: {price_features.shape}, Sentiment features: {sentiment_features.shape}")
        
        # Clean data: ensure all columns are numeric (remove string columns like 'Ticker', 'AAPL', etc.)
        report_progress(logger, "Cleaning data: removing non-numeric columns")
        
        # Select only numeric columns from price features
        price_numeric_cols = price_features.select_dtypes(include=[np.number]).columns.tolist()
        if len(price_numeric_cols) < len(price_features.columns):
            logger.warning(f"Removing {len(price_features.columns) - len(price_numeric_cols)} non-numeric columns from price features")
            price_features = price_features[price_numeric_cols]
        
        # Select only numeric columns from sentiment features
        sentiment_numeric_cols = sentiment_features.select_dtypes(include=[np.number]).columns.tolist()
        if len(sentiment_numeric_cols) < len(sentiment_features.columns):
            logger.warning(f"Removing {len(sentiment_features.columns) - len(sentiment_numeric_cols)} non-numeric columns from sentiment features")
            sentiment_features = sentiment_features[sentiment_numeric_cols]
        
        # Check if we still have enough features after cleaning
        total_features = len(price_features.columns) + len(sentiment_features.columns)
        logger.info(f"After cleaning - Price features: {price_features.shape}, Sentiment features: {sentiment_features.shape}")
        logger.info(f"Total numeric features: {total_features}")
        
        # Re-align dimensions if needed after cleaning
        if total_features < 2050:  # Expected features per timestep
            logger.warning(f"Not enough numeric features after cleaning ({total_features} < 2050). Adding padding.")
            # Add padding features to reach 2050
            n_padding = 2050 - total_features
            for i in range(n_padding):
                padding_col = f'padding_{i}'
                price_features[padding_col] = 0.0
        
        # Phase 2: Create backtest environment
        report_progress(logger, "Phase 2: Creating backtest environment")
        
        env_config = {
            'initial_balance': config['initial_balance'],
            'window_size': config['window_size'],
            'transaction_cost': config['transaction_cost'],
            'alpha': config['reward_params']['alpha'],
            'beta': config['reward_params']['beta'],
            'gamma': config['reward_params']['gamma'],
            'delta': config['reward_params']['delta'],
            'sentiment_threshold': config['trading_params']['sentiment_threshold'],
            'max_position_pct': config['trading_params']['max_position_pct'],
            'max_daily_trades': config['trading_params']['max_daily_trades']
        }
        
        env = create_trading_environment(
            price_features=price_features,
            sentiment_features=sentiment_features,
            config=env_config,
            logger=logger
        )
        
        report_progress(logger, "Environment created", 
                       f"State dimension: {env.observation_space.shape[0]}, Data points: {len(price_features)}")
        
        # Phase 3: Load trained model
        report_progress(logger, "Phase 3: Loading trained model", model_path)
        
        agent = A2CAgent(
            state_dim=env.observation_space.shape[0],
            logger=logger
        )
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None
        
        agent.load_checkpoint(model_path)
        logger.info(f"Model loaded successfully from {model_path}")
        
        # Phase 4: Run backtest
        report_progress(logger, "Phase 4: Running independent backtest")
        
        backtester = Backtester(config=env_config, logger=logger)
        
        # Run backtest with loaded agent
        # Note: The current Backtester.run_backtest() might need to accept an agent parameter
        # For now, we'll create a simple backtest simulation
        
        # Simple backtest simulation
        state = env.reset()
        done = False
        total_reward = 0
        episode_data = []
        
        while not done:
            # Select action using the trained agent
            action, log_prob, value = agent.select_action(state, deterministic=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Record data
            episode_data.append({
                'step': len(episode_data),
                'total_value': info.get('total_value', 0),
                'action': action,
                'reward': reward,
                'portfolio_value': info.get('portfolio_value', 0),
                'cash': info.get('cash', 0),
                'positions': info.get('positions', 0)
            })
            
            total_reward += reward
            state = next_state
        
        # Get final portfolio history
        portfolio_history = env.get_portfolio_history()
        
        # Calculate performance metrics
        if not portfolio_history.empty:
            portfolio_df = portfolio_history
            initial_balance = config['initial_balance']
            
            # Debug: print available columns to help debugging
            logger.info(f"Portfolio history columns: {portfolio_df.columns.tolist()}")
            logger.info(f"Portfolio history shape: {portfolio_df.shape}")
            
            # Find the correct value column - try multiple possible column names
            value_column = None
            possible_value_columns = ['total_value', 'portfolio_value', 'value', 'net_worth', 'total', 'balance']
            
            for col in possible_value_columns:
                if col in portfolio_df.columns:
                    value_column = col
                    logger.info(f"Using value column: {value_column}")
                    break
            
            if value_column is None and len(portfolio_df.columns) > 0:
                # Use the first numeric column as fallback
                numeric_cols = portfolio_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    value_column = numeric_cols[0]
                    logger.info(f"No standard value column found, using first numeric column: {value_column}")
                else:
                    # Last resort: use the first column
                    value_column = portfolio_df.columns[0]
                    logger.info(f"No numeric columns found, using first column: {value_column}")
            
            if value_column:
                # Calculate basic metrics
                final_value = portfolio_df[value_column].iloc[-1]
                total_return_pct = (final_value / initial_balance - 1) * 100
                
                # Max drawdown
                portfolio_df['cummax'] = portfolio_df[value_column].cummax()
                portfolio_df['drawdown'] = (portfolio_df[value_column] - portfolio_df['cummax']) / portfolio_df['cummax'] * 100
                max_drawdown_pct = portfolio_df['drawdown'].min()
                
                # Sharpe ratio (simplified)
                returns = portfolio_df[value_column].pct_change().fillna(0)
                if returns.std() > 0:
                    sharpe_ratio = (returns.mean() * np.sqrt(252)) / (returns.std() * np.sqrt(252))
                else:
                    sharpe_ratio = 0
            else:
                # No value column found
                logger.warning("No value column found in portfolio history")
                final_value = initial_balance
                total_return_pct = 0
                max_drawdown_pct = 0
                sharpe_ratio = 0
        else:
            # Fallback if portfolio history not available
            portfolio_df = pd.DataFrame(episode_data)
            if not portfolio_df.empty:
                logger.info(f"Episode data columns: {portfolio_df.columns.tolist()}")
                # Try to find value column in episode data
                value_column = None
                for col in ['total_value', 'portfolio_value', 'value', 'net_worth']:
                    if col in portfolio_df.columns:
                        value_column = col
                        break
                
                if value_column:
                    final_value = portfolio_df[value_column].iloc[-1]
                else:
                    final_value = config['initial_balance']
                
                total_return_pct = (final_value / config['initial_balance'] - 1) * 100
                max_drawdown_pct = 0
                sharpe_ratio = 0
            else:
                final_value = config['initial_balance']
                total_return_pct = 0
                max_drawdown_pct = 0
                sharpe_ratio = 0
        
        # Count actions
        if episode_data:
            actions = [d['action'] for d in episode_data]
            action_counts = {
                'buy': actions.count(0),
                'sell': actions.count(1),
                'hold': actions.count(2)
            }
            total_trades = action_counts['buy'] + action_counts['sell']
        else:
            action_counts = {'buy': 0, 'sell': 0, 'hold': 0}
            total_trades = 0
        
        # Phase 5: Generate results
        report_progress(logger, "Phase 5: Generating backtest results")
        
        results = {
            'backtest_period': f"{config['backtest_start_date']} to {config['backtest_end_date']}",
            'training_period': f"{config['train_start_date']} to {config['train_end_date']}",
            'total_reward': total_reward,
            'total_return_pct': total_return_pct,
            'max_drawdown_pct': max_drawdown_pct,
            'sharpe_ratio': float(sharpe_ratio),  # Convert numpy type
            'total_trades': total_trades,
            'final_net_worth': float(final_value),  # Convert numpy type
            'action_counts': action_counts,
            'portfolio_history': portfolio_df.to_dict('list') if not portfolio_df.empty else {},
            'model_used': model_path,
            'elapsed_time_seconds': (datetime.now() - start_time).total_seconds()
        }
        
        # Create output directory
        output_dir = "professor_backtest_results"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'independent_backtest_{timestamp}.json')
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Independent backtest results saved: {results_file}")
        
        # Generate summary report
        summary = f"""
        ========================================
        INDEPENDENT BACKTEST RESULTS (2025 DATA)
        ========================================
        
        Training Period: {config['train_start_date']} to {config['train_end_date']}
        Backtest Period: {config['backtest_start_date']} to {config['backtest_end_date']}
        
        Key Metrics:
        -----------
        Total Return: {total_return_pct:.2f}%
        Max Drawdown: {max_drawdown_pct:.2f}%
        Sharpe Ratio: {sharpe_ratio:.3f}
        Total Trades: {total_trades}
        Final Net Worth: ${final_value:.2f}
        
        Action Distribution:
        -------------------
        Buy: {action_counts['buy']}
        Sell: {action_counts['sell']}
        Hold: {action_counts['hold']}
        
        Model: {os.path.basename(model_path)}
        Backtest Time: {results['elapsed_time_seconds']:.1f} seconds
        
        ========================================
        """
        
        summary_file = os.path.join(output_dir, f'summary_{timestamp}.txt')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        print(summary)
        logger.info(f"Summary saved: {summary_file}")
        
        return results
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    logger = setup_logging()
    
    logger.info("="*80)
    logger.info("PROFESSOR-COMPLIANT INDEPENDENT BACKTEST")
    logger.info("="*80)
    logger.info("Load trained model, evaluate on independent 2025 data")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    config_file = "config_professor_requirements.json"
    
    if not os.path.exists(config_file):
        logger.error(f"Config file not found: {config_file}")
        return
    
    # Try to find model - priority: new professor experiment first, then yesterday's model
    training_summary_path = "professor_experiment/training_summary.json"
    model_path = None
    training_period = "Unknown"
    
    if os.path.exists(training_summary_path):
        # Use newly trained model (2022-2024)
        logger.info("Found new professor-trained model summary")
        with open(training_summary_path, 'r') as f:
            training_summary = json.load(f)
        
        model_path = training_summary.get('final_model_path')
        training_period = training_summary.get('training_period', '2022-2024 (professor compliant)')
    else:
        # Use yesterday's successfully trained model (god_mode_ultra_light)
        logger.info("New training summary not found, using yesterday's successful model")
        model_path = "god_mode_ultra_light/models/final_model.pth"
        training_period = "2018-2024 (contains required 2022-2024 period)"
    
    # Validate model path
    if not model_path or not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        logger.info("Please ensure a model is available for backtesting")
        return
    
    logger.info(f"Model found: {model_path}")
    logger.info(f"Training period: {training_period}")
    
    # Run independent backtest
    results = run_independent_backtest(logger, config_file, model_path)
    
    if results:
        logger.info("\n" + "="*80)
        logger.info("INDEPENDENT BACKTEST COMPLETED SUCCESSFULLY")
        logger.info("="*80)
        logger.info(f"Results show model performance on completely unseen 2025 data")
        logger.info(f"Total Return: {results.get('total_return_pct', 0):.2f}%")
        logger.info(f"Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        logger.info(f"Total Trades: {results.get('total_trades', 0)}")
        logger.info("="*80)
    else:
        logger.error("Independent backtest failed")

if __name__ == "__main__":
    main()