"""
A2C Agent - Advantage Actor-Critic algorithm for stock trading

According to proposal requirements:
1. Actor-Critic architecture with multilayer perceptron networks
2. Experience replay mechanism with prioritized sampling
3. Gradient clipping for stable training
4. Advantage normalization
5. Entropy regularization for exploration

Author: Big Dog (Electronic Brother)
Date: 2026-03-12
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')


class TradingNetwork(nn.Module):
    """
    Neural network for A2C algorithm
    
    Proposal requirement: Multilayer perceptron networks for value function
    approximation and policy representation
    """
    
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int = 3,
                 hidden_dim: int = 256,  # 增加隐藏层维度
                 num_layers: int = 3,    # 增加层数
                 dropout_rate: float = 0.2):  # 增加dropout率
        """
        Initialize network
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of hidden layers
        """
        super(TradingNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Shared feature extractor
        self.shared_layers = nn.ModuleList()
        
        # Input normalization layer (for numerical stability)
        # Use LayerNorm instead of BatchNorm for stable behavior with any batch size
        self.input_norm = nn.LayerNorm(state_dim) if state_dim > 1 else nn.Identity()
        
        # Input layer
        self.shared_layers.append(nn.Linear(state_dim, hidden_dim))
        self.shared_layers.append(nn.LayerNorm(hidden_dim))
        self.shared_layers.append(nn.ReLU())
        self.shared_layers.append(nn.Dropout(dropout_rate))
        
        # Additional hidden layers
        for _ in range(num_layers - 1):
            self.shared_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.shared_layers.append(nn.LayerNorm(hidden_dim))
            self.shared_layers.append(nn.ReLU())
            self.shared_layers.append(nn.Dropout(dropout_rate))
        
        # Actor (policy) head
        self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic (value) head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for layer in self.shared_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.constant_(self.actor.bias, 0.0)
        
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.constant_(self.critic.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            log_probs: Log probabilities for each action
            value: State value estimate
        """
        # Check for NaN/inf in input
        if torch.isnan(x).any() or torch.isinf(x).any():
            # Replace NaN/inf with 0
            x = torch.where(torch.isnan(x) | torch.isinf(x), torch.zeros_like(x), x)
        
        # Input normalization (LayerNorm works with any batch size)
        x = self.input_norm(x)
        
        # Shared feature extraction
        features = x
        for layer in self.shared_layers:
            features = layer(features)
            # Add gradient clipping at each layer for numerical stability
            if isinstance(layer, nn.Linear):
                # Clip features to prevent explosion
                features = torch.clamp(features, -10.0, 10.0)
        
        # Actor output (policy)
        action_logits = self.actor(features)
        
        # Add small epsilon to logits for numerical stability
        action_logits = action_logits - action_logits.max(dim=-1, keepdim=True)[0]  # Shift for stability
        log_probs = F.log_softmax(action_logits, dim=-1)
        
        # Critic output (value)
        value = self.critic(features)
        
        # Check for NaN in outputs
        if torch.isnan(log_probs).any() or torch.isnan(value).any():
            # Return safe default values
            log_probs = torch.zeros_like(log_probs)
            value = torch.zeros_like(value)
        
        return log_probs, value
    
    def get_action(self, 
                  state: torch.Tensor,
                  deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get action from policy
        
        Args:
            state: State tensor
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            value: State value estimate
        """
        log_probs, value = self.forward(state)
        
        # Ensure log_probs is valid (not NaN)
        if torch.isnan(log_probs).any():
            # Use uniform distribution as fallback
            log_probs = torch.log(torch.ones_like(log_probs) / self.action_dim)
        
        if deterministic:
            # Select action with highest probability
            action = torch.argmax(log_probs, dim=-1).item()
            log_prob = log_probs[0, action]
        else:
            # Sample action from distribution
            probs = torch.exp(log_probs)
            
            # Add small epsilon to avoid zero probabilities
            probs = probs + 1e-8
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize
            
            # Create distribution with safety check
            try:
                action_dist = torch.distributions.Categorical(probs)
                action = action_dist.sample().item()
                log_prob = log_probs[0, action]
            except:
                # Fallback to uniform distribution
                action = torch.randint(0, self.action_dim, (1,)).item()
                log_prob = torch.log(torch.tensor(1.0 / self.action_dim))
        
        return action, log_prob, value


class ExperienceReplay:
    """
    Experience replay buffer
    
    Proposal requirement: Experience replay mechanism with prioritized sampling
    """
    
    def __init__(self, 
                 capacity: int = 10000,
                 alpha: float = 0.6,
                 beta: float = 0.4,
                 beta_increment: float = 0.001):
        """
        Initialize replay buffer
        
        Args:
            capacity: Buffer capacity
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent
            beta_increment: Beta increment per sample
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
        self.position = 0
        self.logger = logging.getLogger(__name__)
    
    def push(self, 
             state: np.ndarray,
             action: int,
             reward: float,
             next_state: np.ndarray,
             done: bool,
             td_error: Optional[float] = None):
        """
        Push experience to buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
            td_error: TD error for priority (if None, use max priority)
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            
            # Set priority
            if td_error is not None:
                priority = (abs(td_error) + 1e-6) ** self.alpha
            else:
                priority = 1.0  # Max priority for new experiences
            
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = experience
            
            # Update priority
            if td_error is not None:
                priority = (abs(td_error) + 1e-6) ** self.alpha
            else:
                priority = 1.0
            
            self.priorities[self.position] = priority
            self.position = (self.position + 1) % self.capacity
    
    def sample(self, 
               batch_size: int = 64) -> Tuple[List, np.ndarray, np.ndarray]:
        """
        Sample batch from buffer
        
        Args:
            batch_size: Batch size
            
        Returns:
            batch: Sampled experiences
            indices: Indices of sampled experiences
            weights: Importance sampling weights
        """
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get experiences
        batch = [self.buffer[idx] for idx in indices]
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights = weights / weights.max()  # Normalize
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return batch, indices, weights
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """
        Update priorities for sampled experiences
        
        Args:
            indices: Indices to update
            td_errors: TD errors for priority calculation
        """
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 1e-6) ** self.alpha
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """Get current buffer size"""
        return len(self.buffer)


class A2CAgent:
    """
    A2C (Advantage Actor-Critic) Agent
    
    Proposal requirement: A2C algorithm for stable policy optimization
    in continuous action spaces
    """
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int = 3,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 entropy_coef: float = 0.01,
                 value_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_replay: bool = True,
                 replay_capacity: int = 10000,
                 hidden_dim: int = 256,
                 num_layers: int = 3,
                 dropout_rate: float = 0.2,
                 weight_decay: float = 1e-5,  # L2正则化
                 logger: Optional[logging.Logger] = None):
        """
        Initialize A2C agent
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            learning_rate: Learning rate
            gamma: Discount factor
            entropy_coef: Entropy coefficient for exploration
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm for clipping
            use_replay: Whether to use experience replay
            replay_capacity: Replay buffer capacity
            logger: Logger instance
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Network
        self.model = TradingNetwork(
            state_dim, 
            action_dim, 
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Optimizer with L2 regularization
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=10
        )
        
        # Experience replay
        self.use_replay = use_replay
        if use_replay:
            self.replay_buffer = ExperienceReplay(capacity=replay_capacity)
        
        # Training state
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        
        # Logger
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"A2C Agent initialized:")
        self.logger.info(f"  State dimension: {state_dim}")
        self.logger.info(f"  Action dimension: {action_dim}")
        self.logger.info(f"  Network: {hidden_dim}x{num_layers} (dropout={dropout_rate})")
        self.logger.info(f"  Learning rate: {learning_rate} (weight_decay={weight_decay})")
        self.logger.info(f"  Discount factor: {gamma}")
        self.logger.info(f"  Entropy coefficient: {entropy_coef}")
        self.logger.info(f"  Device: {self.device}")
        if use_replay:
            self.logger.info(f"  Using experience replay (capacity: {replay_capacity})")
    
    def select_action(self, 
                     state: np.ndarray,
                     deterministic: bool = False) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Select action given state
        
        Args:
            state: Current state
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            value: State value estimate
        """
        # Ensure state is float32 (handle object dtype)
        if state.dtype == np.object_ or state.dtype == np.dtype('O'):
            # Convert object type to float32
            state = state.astype(np.float32)
        elif state.dtype != np.float32:
            # Convert to float32 if not already
            state = state.astype(np.float32)
        
        # Convert to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Get action from network
        with torch.no_grad():
            action, log_prob, value = self.model.get_action(state_tensor, deterministic)
        
        return action, log_prob, value
    
    def store_transition(self,
                        state: np.ndarray,
                        action: int,
                        reward: float,
                        log_prob: torch.Tensor,
                        value: torch.Tensor,
                        done: bool):
        """
        Store transition for training
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: State value estimate
            done: Whether episode is done
        """
        # Ensure state is float32 before storing
        if state.dtype == np.object_ or state.dtype == np.dtype('O'):
            state = state.astype(np.float32)
        elif state.dtype != np.float32:
            state = state.astype(np.float32)
            
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def _compute_advantages(self,
                          rewards: List[float],
                          values: List[torch.Tensor],
                          dones: List[bool],
                          next_value: torch.Tensor = None) -> torch.Tensor:
        """
        Compute advantages using GAE (Generalized Advantage Estimation)
        
        Args:
            rewards: List of rewards
            values: List of value estimates
            dones: List of done flags
            next_value: Value of next state (for terminal states)
            
        Returns:
            advantages: Computed advantages
        """
        advantages = []
        returns = []
        
        # Convert to tensors
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.cat(values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute returns and advantages
        R = next_value if next_value is not None else 0
        
        for t in reversed(range(len(rewards))):
            # If episode ended, reset return
            if dones[t]:
                R = 0
            
            # Compute return
            R = rewards[t] + self.gamma * R
            returns.insert(0, R)
            
            # Compute advantage
            if t == len(rewards) - 1 and next_value is not None:
                delta = rewards[t] + self.gamma * next_value - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t+1] - values[t] if t < len(rewards)-1 else 0
            
            advantages.insert(0, delta)
        
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Normalize advantages (proposal requirement: advantage normalization)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self, next_value: torch.Tensor = None):
        """
        Update network parameters
        
        Args:
            next_value: Value of next state (for terminal states)
        """
        if len(self.states) == 0:
            self.logger.warning("No transitions to update")
            return
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(
            self.rewards, self.values, self.dones, next_value
        )
        
        # Convert stored data to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.LongTensor(self.actions).to(self.device)
        log_probs = torch.stack(self.log_probs).to(self.device)
        old_values = torch.cat(self.values).to(self.device)
        
        # Forward pass to get current policy and values
        current_log_probs, current_values = self.model(states)
        
        # Select action log probabilities
        action_log_probs = current_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Policy loss (actor)
        policy_loss = -(action_log_probs * advantages).mean()
        
        # Value loss (critic)
        value_loss = F.mse_loss(current_values.squeeze(), returns)
        
        # Entropy loss (for exploration)
        entropy = -(current_log_probs * torch.exp(current_log_probs)).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping (proposal requirement: gradient clipping)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        
        # Clear buffers
        self._clear_buffers()
        
        # Logging
        self.logger.debug(
            f"Update: Policy Loss={policy_loss.item():.4f}, "
            f"Value Loss={value_loss.item():.4f}, "
            f"Entropy={entropy.item():.4f}, "
            f"Total Loss={total_loss.item():.4f}"
        )
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
    
    def update_from_replay(self, batch_size: int = 64):
        """
        Update from experience replay
        
        Args:
            batch_size: Batch size for replay sampling
        """
        if not self.use_replay or len(self.replay_buffer) < batch_size:
            return
        
        # Sample from replay buffer
        batch, indices, weights = self.replay_buffer.sample(batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Get current policy and values
        current_log_probs, current_values = self.model(states)
        _, next_values = self.model(next_states)
        
        # Compute advantages
        td_targets = rewards + self.gamma * next_values.squeeze() * (1 - dones)
        advantages = td_targets - current_values.squeeze()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Select action log probabilities
        action_log_probs = current_log_probs.gather(1, actions.unsqueeze(1)).squeeze()
        
        # Policy loss (with importance sampling weights)
        policy_loss = -(action_log_probs * advantages * weights).mean()
        
        # Value loss
        value_loss = F.mse_loss(current_values.squeeze(), td_targets, reduction='none')
        value_loss = (value_loss * weights).mean()
        
        # Entropy loss
        entropy = -(current_log_probs * torch.exp(current_log_probs)).sum(dim=1).mean()
        entropy_loss = -self.entropy_coef * entropy
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        
        # Update parameters
        self.optimizer.step()
        
        # Update priorities
        td_errors = (td_targets - current_values.squeeze()).detach().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors)
        
        return {
            'replay_policy_loss': policy_loss.item(),
            'replay_value_loss': value_loss.item(),
            'replay_entropy': entropy.item(),
            'replay_total_loss': total_loss.item()
        }
    
    def _clear_buffers(self):
        """Clear transition buffers"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.dones.clear()
    
    def save_checkpoint(self, filepath: str):
        """
        Save agent checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'entropy_coef': self.entropy_coef,
            'value_coef': self.value_coef
        }
        
        torch.save(checkpoint, filepath)
        self.logger.info(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load agent checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {filepath}")
    
    def update_learning_rate(self, validation_reward: float):
        """
        Update learning rate based on validation performance
        
        Args:
            validation_reward: Validation reward for scheduler
        """
        self.scheduler.step(validation_reward)
        
        current_lr = self.optimizer.param_groups[0]['lr']
        self.logger.info(f"Learning rate updated: {current_lr:.6f}")


# Quick demo
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    print("="*60)
    print("A2C Agent - Quick Demo")
    print("="*60)
    
    # Create agent
    state_dim = 163  # Example state dimension
    agent = A2CAgent(state_dim=state_dim, logger=logger)
    
    print(f"Agent created:")
    print(f"  State dimension: {state_dim}")
    print(f"  Action dimension: {agent.action_dim}")
    print(f"  Network parameters: {sum(p.numel() for p in agent.model.parameters()):,}")
    
    # Test action selection
    print("\n1. Testing action selection...")
    sample_state = np.random.randn(state_dim)
    
    # Stochastic action
    action, log_prob, value = agent.select_action(sample_state, deterministic=False)
    print(f"   Stochastic action: {action} (log_prob: {log_prob.item():.4f}, value: {value.item():.4f})")
    
    # Deterministic action
    action, log_prob, value = agent.select_action(sample_state, deterministic=True)
    print(f"   Deterministic action: {action} (log_prob: {log_prob.item():.4f}, value: {value.item():.4f})")
    
    # Test storing transitions
    print("\n2. Testing transition storage...")
    for i in range(5):
        state = np.random.randn(state_dim)
        action = np.random.randint(0, 3)
        reward = np.random.uniform(-1, 1)
        next_state = np.random.randn(state_dim)
        done = i == 4  # Last transition is terminal
        
        # Get action from network
        _, log_prob, value = agent.select_action(state)
        
        # Store transition
        agent.store_transition(state, action, reward, log_prob, value, done)
        
        print(f"   Transition {i}: action={action}, reward={reward:.4f}, done={done}")
    
    # Test update
    print("\n3. Testing network update...")
    next_value = torch.tensor([0.5])  # Example next value
    losses = agent.update(next_value)
    
    print(f"   Update losses:")
    print(f"     Policy loss: {losses['policy_loss']:.6f}")
    print(f"     Value loss: {losses['value_loss']:.6f}")
    print(f"     Entropy: {losses['entropy']:.6f}")
    print(f"     Total loss: {losses['total_loss']:.6f}")
    
    # Test experience replay
    print("\n4. Testing experience replay...")
    if agent.use_replay:
        # Add some experiences to replay buffer
        for i in range(10):
            state = np.random.randn(state_dim)
            action = np.random.randint(0, 3)
            reward = np.random.uniform(-1, 1)
            next_state = np.random.randn(state_dim)
            done = i == 9
            
            # Store in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        print(f"   Replay buffer size: {len(agent.replay_buffer)}")
        
        # Update from replay
        replay_losses = agent.update_from_replay(batch_size=8)
        print(f"   Replay update losses:")
        print(f"     Policy loss: {replay_losses['replay_policy_loss']:.6f}")
        print(f"     Value loss: {replay_losses['replay_value_loss']:.6f}")
    
    # Test checkpoint
    print("\n5. Testing checkpoint...")
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
        checkpoint_path = tmp.name
    
    # Save checkpoint
    agent.save_checkpoint(checkpoint_path)
    print(f"   Checkpoint saved: {checkpoint_path}")
    
    # Load checkpoint
    agent2 = A2CAgent(state_dim=state_dim, logger=logger)
    agent2.load_checkpoint(checkpoint_path)
    print(f"   Checkpoint loaded successfully")
    
    # Clean up
    os.unlink(checkpoint_path)
    
    print("\nA2C Agent demo completed!")