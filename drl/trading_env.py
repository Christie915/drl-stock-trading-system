"""
DRL Trading Environment - Gym environment for stock trading

According to proposal requirements:
1. State space: Market state + Sentiment parameters + Portfolio state
2. Action space: Sell (0), Hold (1), Buy (2)
3. Reward function: R = α·portfolio_return - β·risk + γ·sentiment_alignment - δ·transaction_cost
4. MDP-compliant environment for DRL training

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
import gym
from gym import spaces
import logging
from collections import deque
import warnings
warnings.filterwarnings('ignore')


class StockTradingEnv(gym.Env):
    """
    Stock trading environment for DRL
    
    Proposal requirement: MDP environment with integrated sentiment features
    
    State space (s_t):
        s_t = [market_state, sentiment_parameters, portfolio_state]
    
    Action space (a_t):
        0: Sell all holdings
        1: Hold (no action)
        2: Buy (with available cash)
    
    Reward function:
        R(s_t, a_t, s_{t+1}) = α·portfolio_return - β·risk + γ·sentiment_alignment - δ·transaction_cost
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 price_features: pd.DataFrame,
                 sentiment_features: pd.DataFrame,
                 initial_balance: float = 10000,
                 window_size: int = 10,
                 transaction_cost: float = 0.001,
                 alpha: float = 1.0,      # Portfolio return weight
                 beta: float = 0.1,       # Risk penalty weight
                 gamma: float = 0.5,      # Sentiment alignment weight
                 delta: float = 0.01,     # Transaction cost weight
                 sentiment_threshold: float = 0.2,
                 max_position_pct: float = 0.05,
                 max_daily_trades: int = 10,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize trading environment
        
        Args:
            price_features: Price and technical indicators
            sentiment_features: Sentiment features
            initial_balance: Initial capital
            window_size: Historical window size
            transaction_cost: Transaction cost rate
            alpha: Return reward weight
            beta: Risk penalty weight
            gamma: Sentiment alignment reward weight
            delta: Transaction cost penalty weight
            sentiment_threshold: Threshold for sentiment alignment
            max_position_pct: Maximum position percentage per trade
            max_daily_trades: Maximum trades per day
            logger: Logger instance
        """
        super(StockTradingEnv, self).__init__()
        
        # Validate inputs
        if len(price_features) <= window_size:
            raise ValueError(f"Price data length ({len(price_features)}) must be > window size ({window_size})")
        
        if len(sentiment_features) <= window_size:
            raise ValueError(f"Sentiment data length ({len(sentiment_features)}) must be > window size ({window_size})")
        
        # Store data
        self.price_features = price_features.copy()
        self.sentiment_features = sentiment_features.copy()
        
        # Align indices
        common_idx = self.price_features.index.intersection(self.sentiment_features.index)
        if len(common_idx) == 0:
            raise ValueError("Price and sentiment features have no common time indices")
        
        self.price_features = self.price_features.loc[common_idx]
        self.sentiment_features = self.sentiment_features.loc[common_idx]
        
        # Parameters
        self.initial_balance = initial_balance
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sentiment_threshold = sentiment_threshold
        self.max_position_pct = max_position_pct
        self.max_daily_trades = max_daily_trades
        
        # Logger
        self.logger = logger or logging.getLogger(__name__)
        
        # Extract price column (for portfolio valuation)
        self.price_column = self._identify_price_column()
        
        # Action space: 0=sell, 1=hold, 2=buy
        self.action_space = spaces.Discrete(3)
        
        # State space dimension
        self.state_dim = self._calculate_state_dimension()
        
        # Observation space (continuous)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        
        # Initialize (will be set in reset())
        self.current_step = None
        self.balance = None
        self.shares_held = None
        self.net_worth = None
        self.prev_net_worth = None
        self.today_trades = None
        self.total_trades = None
        self.total_transaction_cost = None
        self.state_history = None
        self.portfolio_history = None
        self.debug_printed = False  # For debug logging
        
        self.logger.info(f"Trading environment initialized:")
        self.logger.info(f"  State dimension: {self.state_dim}")
        self.logger.info(f"  Data points: {len(self.price_features)}")
        self.logger.info(f"  Window size: {window_size}")
        self.logger.info(f"  Reward weights: α={alpha}, β={beta}, γ={gamma}, δ={delta}")
        self.logger.info(f"  DEBUG: Price features: {len(self.price_features.columns)}")
        self.logger.info(f"  DEBUG: Sentiment features: {len(self.sentiment_features.columns)}")
        self.logger.info(f"  DEBUG: Total features per timestep: {len(self.price_features.columns) + len(self.sentiment_features.columns)}")
    
    def _identify_price_column(self) -> str:
        """Identify the price column in features"""
        price_candidates = ['Close', 'close', 'Price', 'price', 'last']
        
        for col in price_candidates:
            if col in self.price_features.columns:
                return col
        
        # If no price column found, use the first column
        self.logger.warning(f"No price column found, using first column: {self.price_features.columns[0]}")
        return self.price_features.columns[0]
    
    def _calculate_state_dimension(self) -> int:
        """Calculate state space dimension"""
        # Historical features (price + sentiment)
        n_price_features = len(self.price_features.columns)
        n_sentiment_features = len(self.sentiment_features.columns)
        n_historical_features = (n_price_features + n_sentiment_features) * self.window_size
        
        # Portfolio state features
        n_portfolio_features = 5  # balance, shares, net_worth, position_pct, trade_count
        
        total_dim = n_historical_features + n_portfolio_features
        
        # Debug logging
        if self.logger:
            self.logger.debug(f"DEBUG _calculate_state_dimension:")
            self.logger.debug(f"  n_price_features: {n_price_features}")
            self.logger.debug(f"  n_sentiment_features: {n_sentiment_features}")
            self.logger.debug(f"  window_size: {self.window_size}")
            self.logger.debug(f"  n_historical_features: {n_historical_features}")
            self.logger.debug(f"  n_portfolio_features: {n_portfolio_features}")
            self.logger.debug(f"  total_dim: {total_dim}")
        
        return total_dim
    
    def reset(self) -> np.ndarray:
        """
        Reset environment to initial state
        
        Returns:
            Initial state vector
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        self.prev_net_worth = self.initial_balance
        self.today_trades = 0
        self.total_trades = 0
        self.total_transaction_cost = 0.0
        
        # Initialize state history
        self.state_history = deque(maxlen=self.window_size)
        
        # Fill history with initial window
        for i in range(self.window_size):
            # Combine price and sentiment features
            price_row = self.price_features.iloc[i]
            sentiment_row = self.sentiment_features.iloc[i]
            
            combined_state = np.concatenate([
                price_row.values,
                sentiment_row.values
            ])
            
            self.state_history.append(combined_state)
        
        # Initialize portfolio history
        self.portfolio_history = []
        self._record_portfolio_state()
        
        self.logger.debug(f"Environment reset: step={self.current_step}, balance=${self.balance:.2f}")
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation
        
        Returns:
            State vector: historical features + portfolio state
        """
        # Flatten historical window
        historical_state = np.array(self.state_history).flatten()
        
        # DEBUG: Print dimensions
        if hasattr(self, 'debug_printed') and not self.debug_printed:
            self.debug_printed = True
            if self.logger:
                self.logger.info(f"DEBUG: state_history length: {len(self.state_history)}")
                if len(self.state_history) > 0:
                    self.logger.info(f"DEBUG: state_history[0] shape: {self.state_history[0].shape}")
                    self.logger.info(f"DEBUG: Combined state length: {len(self.state_history[0])}")
                self.logger.info(f"DEBUG: Flattened historical state shape: {historical_state.shape}")
        
        # Current price
        current_price = self._get_current_price()
        
        # Portfolio state (normalized)
        balance_norm = self.balance / self.initial_balance
        shares_norm = self.shares_held * current_price / self.initial_balance if current_price > 0 else 0
        net_worth_norm = self.net_worth / self.initial_balance
        position_pct = self.shares_held * current_price / self.net_worth if self.net_worth > 0 else 0
        trade_count_norm = self.today_trades / self.max_daily_trades
        
        portfolio_state = np.array([
            balance_norm,
            shares_norm,
            net_worth_norm,
            position_pct,
            trade_count_norm
        ], dtype=np.float32)
        
        # Combine
        observation = np.concatenate([historical_state, portfolio_state])
        
        # DEBUG: Print final observation dimension
        if hasattr(self, 'debug_printed') and self.debug_printed:
            self.logger.info(f"DEBUG: Final observation dimension: {len(observation)}")
        
        return observation
    
    def _get_current_price(self) -> float:
        """Get current price"""
        return float(self.price_features.iloc[self.current_step][self.price_column])
    
    def _get_current_sentiment(self) -> float:
        """Get current sentiment score"""
        # Look for sentiment score column
        sentiment_cols = [col for col in self.sentiment_features.columns 
                         if 'sentiment' in col.lower() and 'score' in col.lower()]
        
        if sentiment_cols:
            return float(self.sentiment_features.iloc[self.current_step][sentiment_cols[0]])
        
        # If no sentiment score, use first column
        return float(self.sentiment_features.iloc[self.current_step].iloc[0])
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment
        
        Args:
            action: Action to take (0=sell, 1=hold, 2=buy)
            
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is done
            info: Additional information
        """
        # Validate action
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}. Must be 0, 1, or 2.")
        
        # Get current state
        current_price = self._get_current_price()
        current_sentiment = self._get_current_sentiment()
        
        # Execute trade
        trade_info = self._execute_trade(action, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= len(self.price_features) - 1
        
        # Update portfolio value
        if not done:
            next_price = self._get_current_price()  # Note: current_step has been incremented
            self.net_worth = self.balance + self.shares_held * next_price
        else:
            # Use last price for final valuation
            self.net_worth = self.balance + self.shares_held * current_price
        
        # Calculate portfolio return
        if abs(self.prev_net_worth) > 1e-8:
            portfolio_return = (self.net_worth - self.prev_net_worth) / abs(self.prev_net_worth)
        else:
            portfolio_return = 0.0
        
        # Calculate risk (using volatility if available)
        risk = self._calculate_risk()
        
        # Calculate sentiment alignment
        sentiment_alignment = self._calculate_sentiment_alignment(action, current_sentiment, trade_info['executed'])
        
        # Calculate reward
        reward = self._calculate_reward(
            portfolio_return=portfolio_return,
            risk=risk,
            sentiment_alignment=sentiment_alignment,
            transaction_cost=trade_info['transaction_cost'],
            trade_executed=trade_info['executed']
        )
        
        # Update history
        self.prev_net_worth = self.net_worth
        
        # Add new state to history
        if not done:
            price_row = self.price_features.iloc[self.current_step]
            sentiment_row = self.sentiment_features.iloc[self.current_step]
            
            combined_state = np.concatenate([
                price_row.values,
                sentiment_row.values
            ])
            
            self.state_history.append(combined_state)
        
        # Record portfolio state
        self._record_portfolio_state()
        
        # Build info dictionary
        info = self._build_info_dict(
            action=action,
            current_price=current_price,
            current_sentiment=current_sentiment,
            portfolio_return=portfolio_return,
            risk=risk,
            sentiment_alignment=sentiment_alignment,
            trade_info=trade_info,
            reward=reward
        )
        
        # Get new observation
        observation = self._get_observation()
        
        # Logging
        self.logger.debug(
            f"Step {self.current_step-1}: Action={['Sell','Hold','Buy'][action]}, "
            f"Price=${current_price:.2f}, Net Worth=${self.net_worth:.2f}, "
            f"Return={portfolio_return:.4f}, Reward={reward:.4f}"
        )
        
        return observation, reward, done, info
    
    def _execute_trade(self, action: int, current_price: float) -> Dict[str, Any]:
        """
        Execute trade based on action
        
        Args:
            action: Action to execute
            current_price: Current price
            
        Returns:
            Trade information dictionary
        """
        trade_info = {
            'action': action,
            'price': current_price,
            'shares_traded': 0,
            'amount': 0.0,
            'transaction_cost': 0.0,
            'executed': False
        }
        
        if action == 0:  # Sell
            if self.shares_held > 0:
                # Calculate trade
                trade_amount = self.shares_held * current_price
                transaction_cost = trade_amount * self.transaction_cost
                proceeds = trade_amount - transaction_cost
                
                # Update account
                self.balance += proceeds
                trade_info['shares_traded'] = self.shares_held
                trade_info['amount'] = -trade_amount
                trade_info['transaction_cost'] = transaction_cost
                trade_info['executed'] = True
                
                self.shares_held = 0
                self.today_trades += 1
                self.total_trades += 1
                self.total_transaction_cost += transaction_cost
                
                self.logger.debug(
                    f"Sell {trade_info['shares_traded']} shares @ ${current_price:.2f}, "
                    f"proceeds=${proceeds:.2f}, cost=${transaction_cost:.2f}"
                )
        
        elif action == 2:  # Buy
            if self.balance > 0 and current_price > 0:
                # Check daily trade limit
                if self.today_trades >= self.max_daily_trades:
                    self.logger.debug("Daily trade limit reached, skipping buy")
                    return trade_info
                
                # Calculate max trade amount
                max_trade_amount = self.max_position_pct * self.net_worth
                available_amount = min(self.balance, max_trade_amount)
                
                if available_amount > 0:
                    # Calculate shares to buy
                    shares_to_buy = int(available_amount // (current_price * (1 + self.transaction_cost)))
                    
                    if shares_to_buy > 0:
                        # Calculate cost
                        trade_amount = shares_to_buy * current_price
                        transaction_cost = trade_amount * self.transaction_cost
                        total_cost = trade_amount + transaction_cost
                        
                        # Update account
                        self.balance -= total_cost
                        self.shares_held += shares_to_buy
                        
                        trade_info['shares_traded'] = shares_to_buy
                        trade_info['amount'] = trade_amount
                        trade_info['transaction_cost'] = transaction_cost
                        trade_info['executed'] = True
                        
                        self.today_trades += 1
                        self.total_trades += 1
                        self.total_transaction_cost += transaction_cost
                        
                        self.logger.debug(
                            f"Buy {shares_to_buy} shares @ ${current_price:.2f}, "
                            f"cost=${total_cost:.2f}, fee=${transaction_cost:.2f}"
                        )
        
        # action == 1: Hold, no execution
        
        return trade_info
    
    def _calculate_risk(self) -> float:
        """Calculate risk metric"""
        # Try to get volatility from features
        volatility_cols = [col for col in self.price_features.columns 
                          if 'volatility' in col.lower() or 'std' in col.lower()]
        
        if volatility_cols and self.current_step < len(self.price_features):
            return float(self.price_features.iloc[self.current_step][volatility_cols[0]])
        
        # Default risk (price volatility over recent window)
        if self.current_step >= 10:
            recent_prices = self.price_features.iloc[self.current_step-10:self.current_step][self.price_column]
            if len(recent_prices) > 1:
                return recent_prices.pct_change().std()
        
        return 0.01  # Default risk
    
    def _calculate_sentiment_alignment(self, 
                                      action: int, 
                                      sentiment: float,
                                      trade_executed: bool) -> float:
        """
        Calculate sentiment alignment reward
        
        Proposal requirement: γ·𝟙{align}(s_t, a_t) - reward for sentiment-aligned trades
        """
        if not trade_executed:
            return 0.0
        
        # Check alignment
        if action == 2 and sentiment > self.sentiment_threshold:  # Buy on positive sentiment
            return 1.0
        elif action == 0 and sentiment < -self.sentiment_threshold:  # Sell on negative sentiment
            return 1.0
        else:
            return 0.0
    
    def _calculate_reward(self,
                         portfolio_return: float,
                         risk: float,
                         sentiment_alignment: float,
                         transaction_cost: float,
                         trade_executed: bool) -> float:
        """
        Calculate reward according to proposal formula
        
        R = α·portfolio_return - β·risk + γ·sentiment_alignment - δ·transaction_cost
        """
        # Return reward
        return_reward = self.alpha * portfolio_return
        
        # Risk penalty
        risk_penalty = self.beta * risk
        
        # Sentiment alignment reward
        alignment_reward = self.gamma * sentiment_alignment
        
        # Transaction cost penalty (only if trade executed)
        cost_penalty = self.delta * transaction_cost / self.initial_balance if trade_executed else 0
        
        # Total reward
        reward = return_reward - risk_penalty + alignment_reward - cost_penalty
        
        # Clip reward to reasonable range
        reward = np.clip(reward, -1.0, 1.0)
        
        return reward
    
    def _build_info_dict(self,
                        action: int,
                        current_price: float,
                        current_sentiment: float,
                        portfolio_return: float,
                        risk: float,
                        sentiment_alignment: float,
                        trade_info: Dict[str, Any],
                        reward: float) -> Dict[str, Any]:
        """Build information dictionary for step"""
        info = {
            'step': self.current_step - 1,  # Adjust for step increment
            'action': action,
            'action_name': ['Sell', 'Hold', 'Buy'][action],
            'price': current_price,
            'sentiment': current_sentiment,
            'balance': self.balance,
            'shares': self.shares_held,
            'net_worth': self.net_worth,
            'portfolio_return': portfolio_return,
            'risk': risk,
            'sentiment_alignment': sentiment_alignment,
            'trade_executed': trade_info['executed'],
            'shares_traded': trade_info['shares_traded'],
            'trade_amount': trade_info['amount'],
            'transaction_cost': trade_info['transaction_cost'],
            'reward': reward,
            'today_trades': self.today_trades,
            'total_trades': self.total_trades,
            'total_transaction_cost': self.total_transaction_cost,
            'position_pct': (self.shares_held * current_price) / self.net_worth if self.net_worth > 0 else 0,
            'reward_components': {
                'return_reward': self.alpha * portfolio_return,
                'risk_penalty': self.beta * risk,
                'alignment_reward': self.gamma * sentiment_alignment,
                'cost_penalty': self.delta * trade_info['transaction_cost'] / self.initial_balance if trade_info['executed'] else 0
            }
        }
        
        return info
    
    def _record_portfolio_state(self):
        """Record current portfolio state to history"""
        state = {
            'step': self.current_step,
            'balance': self.balance,
            'shares': self.shares_held,
            'net_worth': self.net_worth,
            'price': self._get_current_price(),
            'today_trades': self.today_trades
        }
        
        if self.portfolio_history is not None:
            self.portfolio_history.append(state)
    
    def render(self, mode: str = 'human'):
        """
        Render environment state
        
        Args:
            mode: Render mode ('human' or 'ansi')
        """
        if mode == 'human':
            print(f"\n=== Step {self.current_step} ===")
            print(f"Net Worth: ${self.net_worth:.2f} (Initial: ${self.initial_balance:.2f})")
            print(f"Return: {(self.net_worth - self.initial_balance) / self.initial_balance * 100:.2f}%")
            print(f"Cash: ${self.balance:.2f}")
            print(f"Shares: {self.shares_held}")
            print(f"Today's Trades: {self.today_trades}/{self.max_daily_trades}")
            print(f"Total Trades: {self.total_trades}")
            print(f"Total Transaction Cost: ${self.total_transaction_cost:.2f}")
        elif mode == 'ansi':
            return (f"Step {self.current_step}: "
                   f"Net Worth=${self.net_worth:.2f}, "
                   f"Cash=${self.balance:.2f}, "
                   f"Shares={self.shares_held}, "
                   f"Trades={self.today_trades}")
    
    def get_portfolio_history(self) -> pd.DataFrame:
        """
        Get portfolio history
        
        Returns:
            DataFrame with portfolio history
        """
        if self.portfolio_history:
            return pd.DataFrame(self.portfolio_history)
        return pd.DataFrame()
    
    def close(self):
        """Close environment"""
        self.logger.info("Trading environment closed")
        if self.portfolio_history:
            self.portfolio_history.clear()
        if self.state_history:
            self.state_history.clear()


# Factory function for creating environments
def create_trading_environment(
    price_features: pd.DataFrame,
    sentiment_features: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None
) -> StockTradingEnv:
    """
    Factory function for creating trading environments
    
    Args:
        price_features: Price and technical indicators
        sentiment_features: Sentiment features
        config: Environment configuration
        logger: Logger instance
        
    Returns:
        Configured trading environment
    """
    default_config = {
        'initial_balance': 10000,
        'window_size': 10,
        'transaction_cost': 0.001,
        'alpha': 1.0,
        'beta': 0.1,
        'gamma': 0.5,
        'delta': 0.01,
        'sentiment_threshold': 0.2,
        'max_position_pct': 0.05,
        'max_daily_trades': 10
    }
    
    if config:
        default_config.update(config)
    
    env = StockTradingEnv(
        price_features=price_features,
        sentiment_features=sentiment_features,
        initial_balance=default_config['initial_balance'],
        window_size=default_config['window_size'],
        transaction_cost=default_config['transaction_cost'],
        alpha=default_config['alpha'],
        beta=default_config['beta'],
        gamma=default_config['gamma'],
        delta=default_config['delta'],
        sentiment_threshold=default_config['sentiment_threshold'],
        max_position_pct=default_config['max_position_pct'],
        max_daily_trades=default_config['max_daily_trades'],
        logger=logger
    )
    
    return env


# Quick demo
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("DRL Trading Environment - Quick Demo")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    # Price features
    price_data = pd.DataFrame({
        'Date': dates,
        'Open': np.random.uniform(100, 200, 100),
        'High': np.random.uniform(100, 200, 100),
        'Low': np.random.uniform(100, 200, 100),
        'Close': np.random.uniform(100, 200, 100),
        'Volume': np.random.randint(1000000, 10000000, 100),
        'Volatility': np.random.uniform(0.01, 0.05, 100)
    }).set_index('Date')
    
    # Sentiment features
    sentiment_data = pd.DataFrame({
        'Date': dates,
        'sentiment_score': np.random.uniform(-1, 1, 100),
        'sentiment_intensity': np.random.uniform(0, 1, 100),
        'sentiment_volatility': np.random.uniform(0, 0.1, 100)
    }).set_index('Date')
    
    print(f"Price data shape: {price_data.shape}")
    print(f"Sentiment data shape: {sentiment_data.shape}")
    
    # Create environment
    env = create_trading_environment(price_data, sentiment_data, logger=logger)
    
    # Test reset
    print("\n1. Resetting environment...")
    state = env.reset()
    print(f"   State dimension: {state.shape[0]}")
    print(f"   Initial balance: ${env.balance:.2f}")
    print(f"   Initial net worth: ${env.net_worth:.2f}")
    
    # Test a few steps
    print("\n2. Testing environment steps...")
    actions = [1, 2, 1, 0, 1]  # Hold, Buy, Hold, Sell, Hold
    
    for i, action in enumerate(actions):
        state, reward, done, info = env.step(action)
        
        print(f"\n   Step {i}: Action={info['action_name']}")
        print(f"     Price: ${info['price']:.2f}")
        print(f"     Net Worth: ${info['net_worth']:.2f}")
        print(f"     Return: {info['portfolio_return']:.4f}")
        print(f"     Reward: {reward:.4f}")
        print(f"     Trades today: {info['today_trades']}")
        
        if info['trade_executed']:
            print(f"     Trade: {info['shares_traded']} shares, Cost: ${info['transaction_cost']:.2f}")
        
        if done:
            print("     Episode done!")
            break
    
    # Test portfolio history
    print("\n3. Portfolio history...")
    portfolio_history = env.get_portfolio_history()
    print(f"   History length: {len(portfolio_history)}")
    if len(portfolio_history) > 0:
        print(f"   Final net worth: ${portfolio_history['net_worth'].iloc[-1]:.2f}")
        print(f"   Max net worth: ${portfolio_history['net_worth'].max():.2f}")
        print(f"   Min net worth: ${portfolio_history['net_worth'].min():.2f}")
    
    # Render final state
    print("\n4. Final environment state:")
    env.render(mode='human')
    
    print("\nTrading environment demo completed!")