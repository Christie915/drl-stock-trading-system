"""
Backtesting and Evaluation System

According to proposal requirements:
1. Comprehensive backtesting with multiple performance metrics
2. Statistical tests for strategy significance
3. Comparison with baseline models
4. Visualization of results

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from drl.trading_env import create_trading_environment
    from drl.a2c_agent import A2CAgent
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)


class Backtester:
    """
    Backtesting system for DRL trading strategies
    
    Proposal requirement: Comprehensive backtesting with multiple performance metrics
    """
    
    def __init__(self, config: Optional[Dict] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize backtester
        
        Args:
            config: Configuration dictionary
            logger: Logger instance
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
        
        # Results storage
        self.backtest_results = {}
        self.performance_metrics = {}
        self.comparison_results = {}
        
        self.logger.info("Backtester initialized")
    
    def run_backtest(self,
                    model_path: str,
                    price_features: pd.DataFrame,
                    sentiment_features: pd.DataFrame,
                    initial_balance: float = 10000) -> Dict[str, Any]:
        """
        Run backtest on trained model
        
        Args:
            model_path: Path to trained model
            price_features: Price and technical indicators
            sentiment_features: Sentiment features
            initial_balance: Initial capital
            
        Returns:
            Backtest results dictionary
        """
        self.logger.info(f"Running backtest with model: {model_path}")
        
        try:
            # Create environment
            env_config = {
                'initial_balance': initial_balance,
                'window_size': self.config.get('window_size', 10),
                'transaction_cost': self.config.get('transaction_cost', 0.001),
                'max_position_pct': self.config.get('max_position_pct', 0.05),
                'max_daily_trades': self.config.get('max_daily_trades', 10)
            }
            
            env = create_trading_vironment(
                price_features=price_features,
                sentiment_features=sentiment_features,
                config=env_config,
                logger=self.logger
            )
            
            # Create agent and load model
            state_dim = env.observation_space.shape[0]
            agent = A2CAgent(state_dim=state_dim, logger=self.logger)
            agent.load_checkpoint(model_path)
            
            # Run backtest
            agent.model.eval()
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                # Select action (deterministic for backtesting)
                action, _, _ = agent.select_action(state, deterministic=True)
                
                # Take action
                state, reward, done, info = env.step(action)
                total_reward += reward
            
            # Get portfolio history
            portfolio_history = env.get_portfolio_history()
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(portfolio_history, initial_balance)
            
            # Build results
            results = {
                'total_reward': total_reward,
                'final_net_worth': info.get('net_worth', initial_balance),
                'total_trades': info.get('total_trades', 0),
                'total_transaction_cost': info.get('total_transaction_cost', 0),
                'portfolio_history': portfolio_history.to_dict() if not portfolio_history.empty else {},
                'performance_metrics': metrics,
                'environment_info': {
                    'initial_balance': initial_balance,
                    'data_points': len(price_features),
                    'time_period': f"{price_features.index.min()} to {price_features.index.max()}"
                }
            }
            
            # Store results
            self.backtest_results = results
            
            self.logger.info(f"Backtest completed: Total Reward={total_reward:.4f}, Final Net Worth=${results['final_net_worth']:.2f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def _calculate_performance_metrics(self, 
                                     portfolio_history: pd.DataFrame,
                                     initial_balance: float) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics
        
        Proposal requirement: Multiple performance metrics for strategy evaluation
        """
        if portfolio_history.empty:
            return {}
        
        metrics = {}
        
        # Basic metrics
        net_worth_series = portfolio_history['net_worth']
        returns_series = net_worth_series.pct_change().fillna(0)
        
        # 1. Return metrics
        total_return = (net_worth_series.iloc[-1] - initial_balance) / initial_balance * 100
        metrics['total_return_pct'] = total_return
        
        # Annualized return (assuming daily data)
        if len(portfolio_history) > 1:
            days = len(portfolio_history)
            annualized_return = ((1 + total_return/100) ** (252/days) - 1) * 100
            metrics['annualized_return_pct'] = annualized_return
        
        # 2. Risk metrics
        volatility = returns_series.std() * np.sqrt(252) * 100
        metrics['annualized_volatility_pct'] = volatility
        
        # Maximum drawdown
        cumulative_returns = (1 + returns_series).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        metrics['max_drawdown_pct'] = max_drawdown
        
        # 3. Risk-adjusted returns
        if volatility > 0:
            sharpe_ratio = (annualized_return / volatility) if 'annualized_return_pct' in metrics else 0
            metrics['sharpe_ratio'] = sharpe_ratio
        
        # Sortino ratio (downside risk only)
        downside_returns = returns_series[returns_series < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else 0
        
        if downside_volatility > 0:
            sortino_ratio = (annualized_return / downside_volatility) if 'annualized_return_pct' in metrics else 0
            metrics['sortino_ratio'] = sortino_ratio
        
        # 4. Trade statistics
        if 'today_trades' in portfolio_history.columns:
            total_trades = portfolio_history['today_trades'].sum()
            metrics['total_trades'] = total_trades
            
            if total_trades > 0:
                avg_trade_return = total_return / total_trades
                metrics['avg_trade_return_pct'] = avg_trade_return
        
        # 5. Win rate and profit factor
        # These would require individual trade data, which we don't have in portfolio history
        # For simplicity, we'll use daily returns
        winning_days = (returns_series > 0).sum()
        total_days = len(returns_series)
        
        if total_days > 0:
            win_rate = winning_days / total_days * 100
            metrics['win_rate_pct'] = win_rate
            
            # Profit factor (total gains / total losses)
            total_gains = returns_series[returns_series > 0].sum()
            total_losses = abs(returns_series[returns_series < 0].sum())
            
            if total_losses > 0:
                profit_factor = total_gains / total_losses
                metrics['profit_factor'] = profit_factor
        
        # 6. Calmar ratio (return / max drawdown)
        if abs(max_drawdown) > 0:
            calmar_ratio = annualized_return / abs(max_drawdown) if 'annualized_return_pct' in metrics else 0
            metrics['calmar_ratio'] = calmar_ratio
        
        return metrics
    
    def compare_with_baselines(self,
                              drl_results: Dict[str, Any],
                              baseline_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare DRL strategy with baseline strategies
        
        Proposal requirement: Comparison with baseline models
        
        Args:
            drl_results: DRL strategy results
            baseline_results: Dictionary of baseline strategies and their results
            
        Returns:
            Comparison results
        """
        self.logger.info("Comparing with baseline strategies")
        
        comparison = {
            'drl_strategy': drl_results,
            'baselines': baseline_results,
            'comparison_metrics': {}
        }
        
        # Extract key metrics for comparison
        drl_metrics = drl_results.get('performance_metrics', {})
        
        for baseline_name, baseline_result in baseline_results.items():
            baseline_metrics = baseline_result.get('performance_metrics', {})
            
            # Calculate relative performance
            comparison_metrics = {}
            
            for metric in ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']:
                if metric in drl_metrics and metric in baseline_metrics:
                    drl_value = drl_metrics[metric]
                    baseline_value = baseline_metrics[metric]
                    
                    if metric == 'max_drawdown_pct':
                        # Lower drawdown is better
                        relative = baseline_value - drl_value  # Positive means DRL is better
                    else:
                        # Higher is better
                        relative = drl_value - baseline_value  # Positive means DRL is better
                    
                    comparison_metrics[f'{metric}_diff'] = relative
                    comparison_metrics[f'{metric}_drl'] = drl_value
                    comparison_metrics[f'{metric}_baseline'] = baseline_value
            
            comparison['comparison_metrics'][baseline_name] = comparison_metrics
        
        self.comparison_results = comparison
        return comparison
    
    def run_statistical_tests(self,
                             drl_returns: pd.Series,
                             baseline_returns: pd.Series,
                             benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """
        Run statistical tests for strategy significance
        
        Proposal requirement: Statistical tests for strategy significance
        
        Args:
            drl_returns: DRL strategy returns
            baseline_returns: Baseline strategy returns
            benchmark_returns: Benchmark returns (e.g., market index)
            
        Returns:
            Statistical test results
        """
        self.logger.info("Running statistical tests")
        
        test_results = {}
        
        try:
            # 1. Test for normality (Jarque-Bera test)
            from scipy import stats
            
            # Jarque-Bera test
            jb_drl, jb_p_drl = stats.jarque_bera(drl_returns)
            jb_baseline, jb_p_baseline = stats.jarque_bera(baseline_returns)
            
            test_results['normality_tests'] = {
                'drl': {'statistic': jb_drl, 'p_value': jb_p_drl, 'normal': jb_p_drl > 0.05},
                'baseline': {'statistic': jb_baseline, 'p_value': jb_p_baseline, 'normal': jb_p_baseline > 0.05}
            }
            
            # 2. Paired t-test (if returns are paired and normal)
            if jb_p_drl > 0.05 and jb_p_baseline > 0.05:
                t_stat, p_value = stats.ttest_rel(drl_returns, baseline_returns)
                test_results['paired_t_test'] = {
                    'statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            else:
                # Use Wilcoxon signed-rank test (non-parametric)
                w_stat, p_value = stats.wilcoxon(drl_returns, baseline_returns)
                test_results['wilcoxon_test'] = {
                    'statistic': w_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            
            # 3. Test for autocorrelation (Ljung-Box test)
            from statsmodels.stats.diagnostic import acorr_ljungbox
            
            lb_drl = acorr_ljungbox(drl_returns, lags=[10], return_df=True)
            lb_baseline = acorr_ljungbox(baseline_returns, lags=[10], return_df=True)
            
            test_results['autocorrelation_tests'] = {
                'drl': {
                    'statistic': float(lb_drl['lb_stat'].iloc[0]),
                    'p_value': float(lb_drl['lb_pvalue'].iloc[0]),
                    'autocorrelated': float(lb_drl['lb_pvalue'].iloc[0]) < 0.05
                },
                'baseline': {
                    'statistic': float(lb_baseline['lb_stat'].iloc[0]),
                    'p_value': float(lb_baseline['lb_pvalue'].iloc[0]),
                    'autocorrelated': float(lb_baseline['lb_pvalue'].iloc[0]) < 0.05
                }
            }
            
            # 4. Information ratio (relative to benchmark)
            if benchmark_returns is not None:
                tracking_error = (drl_returns - benchmark_returns).std() * np.sqrt(252)
                excess_return = (drl_returns.mean() - benchmark_returns.mean()) * 252
                
                if tracking_error > 0:
                    information_ratio = excess_return / tracking_error
                    test_results['information_ratio'] = information_ratio
            
            # 5. Bootstrap test for Sharpe ratio difference
            n_bootstrap = 1000
            sharpe_differences = []
            
            for _ in range(n_bootstrap):
                # Resample returns
                drl_sample = np.random.choice(drl_returns, size=len(drl_returns), replace=True)
                baseline_sample = np.random.choice(baseline_returns, size=len(baseline_returns), replace=True)
                
                # Calculate Sharpe ratios
                sharpe_drl = drl_sample.mean() / drl_sample.std() * np.sqrt(252) if drl_sample.std() > 0 else 0
                sharpe_baseline = baseline_sample.mean() / baseline_sample.std() * np.sqrt(252) if baseline_sample.std() > 0 else 0
                
                sharpe_differences.append(sharpe_drl - sharpe_baseline)
            
            # Calculate confidence interval
            ci_lower = np.percentile(sharpe_differences, 2.5)
            ci_upper = np.percentile(sharpe_differences, 97.5)
            
            test_results['bootstrap_sharpe'] = {
                'mean_difference': np.mean(sharpe_differences),
                'ci_95_lower': ci_lower,
                'ci_95_upper': ci_upper,
                'significant_zero': not (ci_lower <= 0 <= ci_upper)
            }
            
        except Exception as e:
            self.logger.warning(f"Statistical tests failed: {e}")
            test_results['error'] = str(e)
        
        return test_results
    
    def visualize_results(self, 
                         results: Dict[str, Any],
                         output_dir: str = "backtest_results"):
        """
        Visualize backtest results
        
        Proposal requirement: Visualization of results
        
        Args:
            results: Backtest results
            output_dir: Output directory for plots
        """
        self.logger.info("Visualizing backtest results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # 1. Equity curve
            if 'portfolio_history' in results and results['portfolio_history']:
                portfolio_df = pd.DataFrame(results['portfolio_history'])
                
                plt.figure(figsize=(12, 8))
                
                # Equity curve
                plt.subplot(2, 2, 1)
                plt.plot(portfolio_df.index, portfolio_df['net_worth'])
                plt.title('Equity Curve')
                plt.xlabel('Time')
                plt.ylabel('Net Worth ($)')
                plt.grid(True)
                
                # Drawdown curve
                if 'net_worth' in portfolio_df.columns:
                    cumulative_max = portfolio_df['net_worth'].expanding().max()
                    drawdown = (portfolio_df['net_worth'] - cumulative_max) / cumulative_max * 100
                    
                    plt.subplot(2, 2, 2)
                    plt.fill_between(portfolio_df.index, drawdown, 0, color='red', alpha=0.3)
                    plt.title('Drawdown')
                    plt.xlabel('Time')
                    plt.ylabel('Drawdown (%)')
                    plt.grid(True)
                
                # Daily returns distribution
                if 'net_worth' in portfolio_df.columns:
                    returns = portfolio_df['net_worth'].pct_change().dropna()
                    
                    plt.subplot(2, 2, 3)
                    plt.hist(returns, bins=50, edgecolor='black', alpha=0.7)
                    plt.title('Daily Returns Distribution')
                    plt.xlabel('Return')
                    plt.ylabel('Frequency')
                    plt.grid(True)
                
                # Trade count (if available)
                if 'today_trades' in portfolio_df.columns:
                    plt.subplot(2, 2, 4)
                    cumulative_trades = portfolio_df['today_trades'].cumsum()
                    plt.plot(portfolio_df.index, cumulative_trades)
                    plt.title('Cumulative Trades')
                    plt.xlabel('Time')
                    plt.ylabel('Number of Trades')
                    plt.grid(True)
                
                plt.tight_layout()
                equity_curve_path = os.path.join(output_dir, 'equity_curve.png')
                plt.savefig(equity_curve_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Equity curve saved: {equity_curve_path}")
            
            # 2. Performance metrics bar chart
            if 'performance_metrics' in results:
                metrics = results['performance_metrics']
                
                # Select key metrics for visualization
                key_metrics = {
                    'Total Return (%)': metrics.get('total_return_pct', 0),
                    'Annualized Return (%)': metrics.get('annualized_return_pct', 0),
                    'Max Drawdown (%)': metrics.get('max_drawdown_pct', 0),
                    'Sharpe Ratio': metrics.get('sharpe_ratio', 0),
                    'Win Rate (%)': metrics.get('win_rate_pct', 0)
                }
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(range(len(key_metrics)), list(key_metrics.values()))
                plt.xticks(range(len(key_metrics)), list(key_metrics.keys()), rotation=45)
                plt.title('Performance Metrics')
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, key_metrics.values()):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{value:.2f}', ha='center', va='bottom')
                
                plt.tight_layout()
                metrics_path = os.path.join(output_dir, 'performance_metrics.png')
                plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
                plt.close()
                
                self.logger.info(f"Performance metrics saved: {metrics_path}")
            
            # 3. Comparison visualization (if available)
            if self.comparison_results:
                self._visualize_comparison(output_dir)
                
        except Exception as e:
            self.logger.error(f"Error visualizing results: {e}")
    
    def _visualize_comparison(self, output_dir: str):
        """Visualize comparison results"""
        try:
            comparison = self.comparison_results
            
            # Create comparison bar chart
            plt.figure(figsize=(12, 6))
            
            # Extract comparison data
            strategies = ['DRL'] + list(comparison['baselines'].keys())
            
            # Prepare metrics for comparison
            metrics_to_compare = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct']
            metric_labels = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)']
            
            n_metrics = len(metrics_to_compare)
            n_strategies = len(strategies)
            
            for i, (metric, label) in enumerate(zip(metrics_to_compare, metric_labels)):
                values = []
                
                # DRL value
                drl_value = comparison['drl_strategy']['performance_metrics'].get(metric, 0)
                values.append(drl_value)
                
                # Baseline values
                for baseline_name in comparison['baselines'].keys():
                    baseline_value = comparison['baselines'][baseline_name]['performance_metrics'].get(metric, 0)
                    values.append(baseline_value)
                
                # Create subplot
                plt.subplot(1, n_metrics, i + 1)
                bars = plt.bar(range(n_strategies), values)
                plt.xticks(range(n_strategies), strategies, rotation=45)
                plt.title(label)
                plt.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                            f'{value:.2f}', ha='center', va='bottom')
            
            plt.tight_layout()
            comparison_path = os.path.join(output_dir, 'strategy_comparison.png')
            plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Strategy comparison saved: {comparison_path}")
            
        except Exception as e:
            self.logger.error(f"Error visualizing comparison: {e}")
    
    def generate_report(self, 
                       results: Dict[str, Any],
                       output_path: str = "backtest_report.json") -> Dict[str, Any]:
        """
        Generate comprehensive backtest report
        
        Args:
            results: Backtest results
            output_path: Path to save report
            
        Returns:
            Complete report dictionary
        """
        self.logger.info(f"Generating backtest report: {output_path}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'backtest_results': results,
            'performance_metrics': results.get('performance_metrics', {}),
            'comparison_results': self.comparison_results,
            'summary': self._generate_summary(results)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Backtest report saved: {output_path}")
        
        return report
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of backtest results"""
        metrics = results.get('performance_metrics', {})
        
        summary = {
            'total_return': f"{metrics.get('total_return_pct', 0):.2f}%",
            'annualized_return': f"{metrics.get('annualized_return_pct', 0):.2f}%",
            'max_drawdown': f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            'sharpe_ratio': f"{metrics.get('sharpe_ratio', 0):.2f}",
            'total_trades': results.get('total_trades', 0),
            'final_net_worth': f"${results.get('final_net_worth', 0):.2f}",
            'total_reward': f"{results.get('total_reward', 0):.4f}"
        }
        
        return summary


# Quick demo
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("Backtesting System - Quick Demo")
    print("="*60)
    
    # Create sample data for demo
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }).set_index('Date')
    
    sentiment_data = pd.DataFrame({
        'Date': dates,
        'sentiment_score': np.random.uniform(-1, 1, 100)
    }).set_index('Date')
    
    print(f"Sample data created: price shape={price_data.shape}, sentiment shape={sentiment_data.shape}")
    
    # Create backtester
    backtester = Backtester(logger=logger)
    
    # Note: In a real scenario, we would have a trained model
    # For demo, we'll create mock results
    print("\n1. Running backtest (mock)...")
    
    mock_results = {
        'total_reward': 15.32,
        'final_net_worth': 11250.50,
        'total_trades': 25,
        'total_transaction_cost': 125.75,
        'portfolio_history': pd.DataFrame({
            'net_worth': np.random.uniform(10000, 12000, 100),
            'today_trades': np.random.randint(0, 2, 100)
        }).to_dict(),
        'performance_metrics': {
            'total_return_pct': 12.5,
            'annualized_return_pct': 15.2,
            'max_drawdown_pct': -8.3,
            'sharpe_ratio': 1.25,
            'win_rate_pct': 58.7,
            'profit_factor': 1.8
        }
    }
    
    # Calculate performance metrics from mock portfolio history
    if mock_results['portfolio_history']:
        portfolio_df = pd.DataFrame(mock_results['portfolio_history'])
        metrics = backtester._calculate_performance_metrics(portfolio_df, 10000)
        mock_results['performance_metrics'] = metrics
    
    print(f"   Backtest completed: Total Return={mock_results['performance_metrics'].get('total_return_pct', 0):.2f}%")
    
    # Test comparison with baselines
    print("\n2. Comparing with baseline strategies...")
    
    baseline_results = {
        'buy_and_hold': {
            'performance_metrics': {
                'total_return_pct': 8.2,
                'annualized_return_pct': 10.1,
                'max_drawdown_pct': -12.5,
                'sharpe_ratio': 0.85,
                'win_rate_pct': 52.3
            }
        },
        'moving_average': {
            'performance_metrics': {
                'total_return_pct': 6.7,
                'annualized_return_pct': 8.3,
                'max_drawdown_pct': -10.2,
                'sharpe_ratio': 0.72,
                'win_rate_pct': 54.8
            }
        }
    }
    
    comparison = backtester.compare_with_baselines(mock_results, baseline_results)
    print(f"   Comparison completed with {len(baseline_results)} baselines")
    
    # Test statistical tests
    print("\n3. Running statistical tests...")
    
    # Create sample returns
    drl_returns = np.random.normal(0.001, 0.02, 100)
    baseline_returns = np.random.normal(0.0005, 0.025, 100)
    
    test_results = backtester.run_statistical_tests(drl_returns, baseline_returns)
    print(f"   Statistical tests completed: {len(test_results)} tests")
    
    # Visualize results
    print("\n4. Visualizing results...")
    backtester.visualize_results(mock_results, output_dir="demo_backtest_results")
    
    # Generate report
    print("\n5. Generating report...")
    report = backtester.generate_report(mock_results, "demo_backtest_report.json")
    
    print("\n" + "="*60)
    print("Backtesting demo completed!")
    print(f"   Results saved to: demo_backtest_results/")
    print(f"   Report saved to: demo_backtest_report.json")
    print("="*60)