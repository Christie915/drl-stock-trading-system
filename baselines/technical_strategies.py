"""
Baseline Trading Strategies

According to proposal requirements:
1. Buy-and-hold strategy
2. Moving average crossover strategy
3. Momentum-based strategy
4. Mean reversion strategy

These are used as baselines for comparison with DRL strategy.

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import logging


class BaselineStrategies:
    """
    Baseline trading strategies for comparison
    
    Proposal requirement: Baseline models for comparison with DRL strategy
    """
    
    def __init__(self, initial_balance: float = 10000, transaction_cost: float = 0.001):
        """
        Initialize baseline strategies
        
        Args:
            initial_balance: Initial capital
            transaction_cost: Transaction cost rate
        """
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Baseline strategies initialized")
    
    def buy_and_hold(self, 
                    price_data: pd.DataFrame,
                    price_column: str = 'Close') -> Dict[str, Any]:
        """
        Buy-and-hold strategy
        
        Buy at the beginning, hold until the end
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name for price
            
        Returns:
            Strategy results dictionary
        """
        self.logger.info("Running buy-and-hold strategy")
        
        if price_data.empty or price_column not in price_data.columns:
            self.logger.error(f"Invalid price data or missing column: {price_column}")
            return {}
        
        # Buy at first price
        first_price = price_data[price_column].iloc[0]
        shares_to_buy = int(self.initial_balance // (first_price * (1 + self.transaction_cost)))
        
        if shares_to_buy <= 0:
            self.logger.warning("Cannot buy any shares with initial balance")
            return {}
        
        # Calculate cost
        cost = shares_to_buy * first_price * (1 + self.transaction_cost)
        remaining_cash = self.initial_balance - cost
        
        # Hold until end
        final_price = price_data[price_column].iloc[-1]
        final_value = shares_to_buy * final_price + remaining_cash
        
        # Calculate returns
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100
        
        # Create portfolio history
        portfolio_history = []
        for i, price in enumerate(price_data[price_column]):
            net_worth = shares_to_buy * price + remaining_cash
            portfolio_history.append({
                'step': i,
                'net_worth': net_worth,
                'shares': shares_to_buy,
                'price': price,
                'today_trades': 1 if i == 0 else 0  # Only trade on first day
            })
        
        results = {
            'strategy_name': 'buy_and_hold',
            'initial_balance': self.initial_balance,
            'final_net_worth': final_value,
            'total_return_pct': total_return,
            'total_trades': 1,
            'shares_held': shares_to_buy,
            'portfolio_history': portfolio_history,
            'transaction_costs': cost - (shares_to_buy * first_price)  # Only initial transaction
        }
        
        self.logger.info(f"Buy-and-hold: Total Return={total_return:.2f}%, Final Value=${final_value:.2f}")
        
        return results
    
    def moving_average_crossover(self,
                               price_data: pd.DataFrame,
                               price_column: str = 'Close',
                               short_window: int = 10,
                               long_window: int = 50) -> Dict[str, Any]:
        """
        Moving average crossover strategy
        
        Buy when short MA crosses above long MA, sell when crosses below
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name for price
            short_window: Short moving average window
            long_window: Long moving average window
            
        Returns:
            Strategy results dictionary
        """
        self.logger.info(f"Running moving average crossover (short={short_window}, long={long_window})")
        
        if price_data.empty or price_column not in price_data.columns:
            self.logger.error(f"Invalid price data or missing column: {price_column}")
            return {}
        
        # Calculate moving averages
        short_ma = price_data[price_column].rolling(window=short_window).mean()
        long_ma = price_data[price_column].rolling(window=long_window).mean()
        
        # Generate signals
        signals = pd.DataFrame(index=price_data.index)
        signals['price'] = price_data[price_column]
        signals['short_ma'] = short_ma
        signals['long_ma'] = long_ma
        
        # Signal: 1 = buy, -1 = sell, 0 = hold
        signals['position'] = 0
        signals.loc[short_ma > long_ma, 'position'] = 1
        signals.loc[short_ma <= long_ma, 'position'] = -1
        
        # Shift to avoid look-ahead bias
        signals['signal'] = signals['position'].shift(1).fillna(0)
        
        # Initialize trading
        cash = self.initial_balance
        shares = 0
        total_trades = 0
        total_transaction_cost = 0
        
        portfolio_history = []
        
        for i in range(len(signals)):
            current_price = signals['price'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Execute trades based on signal
            if signal == 1 and shares == 0:  # Buy signal, no position
                # Calculate max shares to buy
                max_shares = int(cash // (current_price * (1 + self.transaction_cost)))
                
                if max_shares > 0:
                    # Calculate cost
                    cost = max_shares * current_price * (1 + self.transaction_cost)
                    transaction_cost_amount = max_shares * current_price * self.transaction_cost
                    
                    # Update portfolio
                    cash -= cost
                    shares = max_shares
                    total_trades += 1
                    total_transaction_cost += transaction_cost_amount
                    
            elif signal == -1 and shares > 0:  # Sell signal, has position
                # Calculate proceeds
                proceeds = shares * current_price * (1 - self.transaction_cost)
                transaction_cost_amount = shares * current_price * self.transaction_cost
                
                # Update portfolio
                cash += proceeds
                shares = 0
                total_trades += 1
                total_transaction_cost += transaction_cost_amount
            
            # Calculate net worth
            net_worth = cash + shares * current_price
            
            portfolio_history.append({
                'step': i,
                'net_worth': net_worth,
                'cash': cash,
                'shares': shares,
                'price': current_price,
                'signal': signal,
                'today_trades': 1 if (signal == 1 and shares == 0) or (signal == -1 and shares > 0) else 0
            })
        
        final_net_worth = portfolio_history[-1]['net_worth'] if portfolio_history else self.initial_balance
        total_return = (final_net_worth - self.initial_balance) / self.initial_balance * 100
        
        results = {
            'strategy_name': f'moving_average_crossover_{short_window}_{long_window}',
            'initial_balance': self.initial_balance,
            'final_net_worth': final_net_worth,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'shares_held': shares,
            'portfolio_history': portfolio_history,
            'transaction_costs': total_transaction_cost,
            'signals': signals[['price', 'short_ma', 'long_ma', 'signal']].to_dict()
        }
        
        self.logger.info(f"Moving Average Crossover: Total Return={total_return:.2f}%, Trades={total_trades}")
        
        return results
    
    def momentum_strategy(self,
                        price_data: pd.DataFrame,
                        price_column: str = 'Close',
                        momentum_window: int = 20,
                        threshold: float = 0.02) -> Dict[str, Any]:
        """
        Momentum strategy
        
        Buy when price momentum is positive, sell when negative
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name for price
            momentum_window: Window for momentum calculation
            threshold: Momentum threshold for trading
            
        Returns:
            Strategy results dictionary
        """
        self.logger.info(f"Running momentum strategy (window={momentum_window}, threshold={threshold})")
        
        if price_data.empty or price_column not in price_data.columns:
            self.logger.error(f"Invalid price data or missing column: {price_column}")
            return {}
        
        # Calculate momentum
        momentum = price_data[price_column].pct_change(periods=momentum_window)
        
        # Generate signals
        signals = pd.DataFrame(index=price_data.index)
        signals['price'] = price_data[price_column]
        signals['momentum'] = momentum
        
        # Signal: 1 = buy (momentum > threshold), -1 = sell (momentum < -threshold), 0 = hold
        signals['signal'] = 0
        signals.loc[momentum > threshold, 'signal'] = 1
        signals.loc[momentum < -threshold, 'signal'] = -1
        
        # Shift to avoid look-ahead bias
        signals['signal'] = signals['signal'].shift(1).fillna(0)
        
        # Initialize trading
        cash = self.initial_balance
        shares = 0
        total_trades = 0
        total_transaction_cost = 0
        
        portfolio_history = []
        
        for i in range(len(signals)):
            current_price = signals['price'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Execute trades
            if signal == 1 and shares == 0:  # Buy
                max_shares = int(cash // (current_price * (1 + self.transaction_cost)))
                
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost)
                    transaction_cost_amount = max_shares * current_price * self.transaction_cost
                    
                    cash -= cost
                    shares = max_shares
                    total_trades += 1
                    total_transaction_cost += transaction_cost_amount
                    
            elif signal == -1 and shares > 0:  # Sell
                proceeds = shares * current_price * (1 - self.transaction_cost)
                transaction_cost_amount = shares * current_price * self.transaction_cost
                
                cash += proceeds
                shares = 0
                total_trades += 1
                total_transaction_cost += transaction_cost_amount
            
            # Calculate net worth
            net_worth = cash + shares * current_price
            
            portfolio_history.append({
                'step': i,
                'net_worth': net_worth,
                'cash': cash,
                'shares': shares,
                'price': current_price,
                'signal': signal,
                'today_trades': 1 if (signal == 1 and shares == 0) or (signal == -1 and shares > 0) else 0
            })
        
        final_net_worth = portfolio_history[-1]['net_worth'] if portfolio_history else self.initial_balance
        total_return = (final_net_worth - self.initial_balance) / self.initial_balance * 100
        
        results = {
            'strategy_name': f'momentum_{momentum_window}_{threshold}',
            'initial_balance': self.initial_balance,
            'final_net_worth': final_net_worth,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'shares_held': shares,
            'portfolio_history': portfolio_history,
            'transaction_costs': total_transaction_cost,
            'signals': signals[['price', 'momentum', 'signal']].to_dict()
        }
        
        self.logger.info(f"Momentum Strategy: Total Return={total_return:.2f}%, Trades={total_trades}")
        
        return results
    
    def mean_reversion(self,
                      price_data: pd.DataFrame,
                      price_column: str = 'Close',
                      window: int = 20,
                      std_dev: float = 2.0) -> Dict[str, Any]:
        """
        Mean reversion strategy (Bollinger Bands)
        
        Buy when price is below lower band, sell when above upper band
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name for price
            window: Window for moving average
            std_dev: Standard deviations for bands
            
        Returns:
            Strategy results dictionary
        """
        self.logger.info(f"Running mean reversion strategy (window={window}, std={std_dev})")
        
        if price_data.empty or price_column not in price_data.columns:
            self.logger.error(f"Invalid price data or missing column: {price_column}")
            return {}
        
        # Calculate Bollinger Bands
        middle_band = price_data[price_column].rolling(window=window).mean()
        std = price_data[price_column].rolling(window=window).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        # Generate signals
        signals = pd.DataFrame(index=price_data.index)
        signals['price'] = price_data[price_column]
        signals['middle_band'] = middle_band
        signals['upper_band'] = upper_band
        signals['lower_band'] = lower_band
        
        # Signal: 1 = buy (price < lower band), -1 = sell (price > upper band), 0 = hold
        signals['signal'] = 0
        signals.loc[signals['price'] < signals['lower_band'], 'signal'] = 1
        signals.loc[signals['price'] > signals['upper_band'], 'signal'] = -1
        
        # Shift to avoid look-ahead bias
        signals['signal'] = signals['signal'].shift(1).fillna(0)
        
        # Initialize trading
        cash = self.initial_balance
        shares = 0
        total_trades = 0
        total_transaction_cost = 0
        
        portfolio_history = []
        
        for i in range(len(signals)):
            current_price = signals['price'].iloc[i]
            signal = signals['signal'].iloc[i]
            
            # Execute trades
            if signal == 1 and shares == 0:  # Buy (oversold)
                max_shares = int(cash // (current_price * (1 + self.transaction_cost)))
                
                if max_shares > 0:
                    cost = max_shares * current_price * (1 + self.transaction_cost)
                    transaction_cost_amount = max_shares * current_price * self.transaction_cost
                    
                    cash -= cost
                    shares = max_shares
                    total_trades += 1
                    total_transaction_cost += transaction_cost_amount
                    
            elif signal == -1 and shares > 0:  # Sell (overbought)
                proceeds = shares * current_price * (1 - self.transaction_cost)
                transaction_cost_amount = shares * current_price * self.transaction_cost
                
                cash += proceeds
                shares = 0
                total_trades += 1
                total_transaction_cost += transaction_cost_amount
            
            # Calculate net worth
            net_worth = cash + shares * current_price
            
            portfolio_history.append({
                'step': i,
                'net_worth': net_worth,
                'cash': cash,
                'shares': shares,
                'price': current_price,
                'signal': signal,
                'today_trades': 1 if (signal == 1 and shares == 0) or (signal == -1 and shares > 0) else 0
            })
        
        final_net_worth = portfolio_history[-1]['net_worth'] if portfolio_history else self.initial_balance
        total_return = (final_net_worth - self.initial_balance) / self.initial_balance * 100
        
        results = {
            'strategy_name': f'mean_reversion_{window}_{std_dev}',
            'initial_balance': self.initial_balance,
            'final_net_worth': final_net_worth,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'shares_held': shares,
            'portfolio_history': portfolio_history,
            'transaction_costs': total_transaction_cost,
            'signals': signals[['price', 'middle_band', 'upper_band', 'lower_band', 'signal']].to_dict()
        }
        
        self.logger.info(f"Mean Reversion: Total Return={total_return:.2f}%, Trades={total_trades}")
        
        return results
    
    def random_strategy(self,
                       price_data: pd.DataFrame,
                       price_column: str = 'Close',
                       trade_probability: float = 0.1) -> Dict[str, Any]:
        """
        Random trading strategy
        
        Randomly buy or sell with given probability
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name for price
            trade_probability: Probability of trading on each day
            
        Returns:
            Strategy results dictionary
        """
        self.logger.info(f"Running random strategy (trade_probability={trade_probability})")
        
        if price_data.empty or price_column not in price_data.columns:
            self.logger.error(f"Invalid price data or missing column: {price_column}")
            return {}
        
        # Initialize
        cash = self.initial_balance
        shares = 0
        total_trades = 0
        total_transaction_cost = 0
        
        portfolio_history = []
        
        for i in range(len(price_data)):
            current_price = price_data[price_column].iloc[i]
            
            # Random decision
            if np.random.random() < trade_probability:
                # Random action: buy or sell
                action = np.random.choice(['buy', 'sell'])
                
                if action == 'buy' and cash > 0:
                    # Buy
                    max_shares = int(cash // (current_price * (1 + self.transaction_cost)))
                    
                    if max_shares > 0:
                        cost = max_shares * current_price * (1 + self.transaction_cost)
                        transaction_cost_amount = max_shares * current_price * self.transaction_cost
                        
                        cash -= cost
                        shares = max_shares
                        total_trades += 1
                        total_transaction_cost += transaction_cost_amount
                        
                elif action == 'sell' and shares > 0:
                    # Sell
                    proceeds = shares * current_price * (1 - self.transaction_cost)
                    transaction_cost_amount = shares * current_price * self.transaction_cost
                    
                    cash += proceeds
                    shares = 0
                    total_trades += 1
                    total_transaction_cost += transaction_cost_amount
            
            # Calculate net worth
            net_worth = cash + shares * current_price
            
            portfolio_history.append({
                'step': i,
                'net_worth': net_worth,
                'cash': cash,
                'shares': shares,
                'price': current_price,
                'today_trades': 1 if np.random.random() < trade_probability else 0
            })
        
        final_net_worth = portfolio_history[-1]['net_worth'] if portfolio_history else self.initial_balance
        total_return = (final_net_worth - self.initial_balance) / self.initial_balance * 100
        
        results = {
            'strategy_name': f'random_{trade_probability}',
            'initial_balance': self.initial_balance,
            'final_net_worth': final_net_worth,
            'total_return_pct': total_return,
            'total_trades': total_trades,
            'shares_held': shares,
            'portfolio_history': portfolio_history,
            'transaction_costs': total_transaction_cost
        }
        
        self.logger.info(f"Random Strategy: Total Return={total_return:.2f}%, Trades={total_trades}")
        
        return results
    
    def run_all_strategies(self,
                          price_data: pd.DataFrame,
                          price_column: str = 'Close') -> Dict[str, Dict[str, Any]]:
        """
        Run all baseline strategies
        
        Args:
            price_data: DataFrame with price data
            price_column: Column name for price
            
        Returns:
            Dictionary of all strategy results
        """
        self.logger.info("Running all baseline strategies")
        
        all_results = {}
        
        # 1. Buy and hold
        all_results['buy_and_hold'] = self.buy_and_hold(price_data, price_column)
        
        # 2. Moving average crossover
        all_results['ma_crossover_10_50'] = self.moving_average_crossover(
            price_data, price_column, short_window=10, long_window=50
        )
        
        all_results['ma_crossover_20_100'] = self.moving_average_crossover(
            price_data, price_column, short_window=20, long_window=100
        )
        
        # 3. Momentum strategy
        all_results['momentum_20_0.02'] = self.momentum_strategy(
            price_data, price_column, momentum_window=20, threshold=0.02
        )
        
        # 4. Mean reversion
        all_results['mean_reversion_20_2'] = self.mean_reversion(
            price_data, price_column, window=20, std_dev=2.0
        )
        
        # 5. Random strategy
        all_results['random_0.1'] = self.random_strategy(
            price_data, price_column, trade_probability=0.1
        )
        
        self.logger.info(f"All baseline strategies completed: {len(all_results)} strategies")
        
        return all_results


# Quick demo
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("Baseline Strategies - Quick Demo")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100)
    }).set_index('Date')
    
    print(f"Sample data created: shape={price_data.shape}")
    
    # Create baseline strategies
    strategies = BaselineStrategies(initial_balance=10000)
    
    # Run all strategies
    results = strategies.run_all_strategies(price_data, price_column='Close')
    
    print(f"\nBaseline strategy results:")
    print("-" * 60)
    
    for strategy_name, result in results.items():
        if result:
            total_return = result.get('total_return_pct', 0)
            trades = result.get('total_trades', 0)
            final_value = result.get('final_net_worth', 0)
            
            print(f"{strategy_name:20s}: Return={total_return:6.2f}%, Trades={trades:3d}, Final Value=${final_value:8.2f}")
    
    print("\n" + "="*60)
    print("Baseline strategies demo completed!")
    print("="*60)