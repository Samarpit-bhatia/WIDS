"""
Task 3: Deep Q-Learning for MountainCar
WiDS Kalman Filtered Trend Trader - Assignment 3

This script implements Deep Q-Network (DQN) for the MountainCar environment.
The car must learn to build momentum by moving back and forth to reach the goal.

Key DQN Components:
1. Neural network for function approximation
2. Experience replay buffer
3. Target network for stable learning
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not found. Installing...")
    print("Run: pip install torch")
    TORCH_AVAILABLE = False
    exit(1)

print("="*60)
print("Task 3: Deep Q-Learning on MountainCar")
print("="*60)

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")

# -----------------------------
# Neural Network Architecture
# -----------------------------
class DQN(nn.Module):
    """
    Deep Q-Network: Neural network that approximates Q(s, a)

    Input: State (position, velocity)
    Output: Q-values for each action
    """
    def __init__(self, state_dim, action_dim, hidden_size=128):
        super(DQN, self).__init__()

        # Simple feedforward network
        # Tried different architectures:
        # - Too shallow (1 layer): couldn't learn
        # - Too deep (4+ layers): overfitting
        # - 2 hidden layers with 128 units works well
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )

    def forward(self, state):
        return self.network(state)


# -----------------------------
# Experience Replay Buffer
# -----------------------------
class ReplayBuffer:
    """
    Stores transitions (s, a, r, s', done) for experience replay.
    Breaks temporal correlations in training data.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# -----------------------------
# DQN Agent
# -----------------------------
class DQNAgent:
    """
    DQN Agent that learns to play MountainCar
    """
    def __init__(self, state_dim, action_dim, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Main network (updated every step)
        self.policy_net = DQN(state_dim, action_dim).to(device)

        # Target network (updated periodically for stability)
        self.target_net = DQN(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Hyperparameters
        self.gamma = 0.99          # discount factor (long-term planning important)
        self.epsilon = 1.0         # exploration rate
        self.epsilon_min = 0.01    # minimum exploration
        self.epsilon_decay = 0.995 # decay rate (slower than FrozenLake)
        self.batch_size = 64       # replay batch size
        self.target_update = 10    # update target network every N episodes

        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, explore=True):
        """Epsilon-greedy action selection"""
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def train_step(self):
        """Train the network on a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # Sample random batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Current Q-values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (using target network)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # Compute loss and update
        loss = self.criterion(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """Copy weights from policy network to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


# -----------------------------
# Environment Setup
# -----------------------------
env = gym.make('MountainCar-v0')
state_dim = env.observation_space.shape[0]  # 2 (position, velocity)
action_dim = env.action_space.n              # 3 (left, no push, right)

print(f"\nEnvironment Details:")
print(f"  State space: {state_dim} (position, velocity)")
print(f"  Action space: {action_dim} (0=left, 1=nothing, 2=right)")
print(f"  Goal: Reach position >= 0.5")
print(f"  Challenge: Engine too weak to climb directly")

# -----------------------------
# Training Parameters
# -----------------------------
num_episodes = 500     # DQN typically needs fewer episodes than tabular
max_steps = 200        # episode limit

print(f"\nTraining Parameters:")
print(f"  Number of episodes: {num_episodes}")
print(f"  Max steps per episode: {max_steps}")
print(f"  Batch size: 64")
print(f"  Learning rate: 0.001")
print(f"  Replay buffer size: 10000")

# -----------------------------
# Initialize Agent
# -----------------------------
agent = DQNAgent(state_dim, action_dim, learning_rate=0.001)

# -----------------------------
# Training Loop
# -----------------------------
print("\n" + "="*60)
print("Starting Training...")
print("="*60)
print("Note: Learning delayed rewards takes time!")
print("Expect ~100-200 episodes before seeing improvement.\n")

episode_rewards = []
episode_lengths = []
losses = []
success_window = deque(maxlen=100)
epsilon_history = []

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_loss = []

    for step in range(max_steps):
        # Select and perform action
        action = agent.select_action(state, explore=True)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Reward shaping: encourage progress toward goal
        # Original reward is -1 per step, which is very sparse
        # Add small bonus for moving right (toward goal)
        shaped_reward = reward
        if next_state[0] > state[0]:  # moving right (toward goal)
            shaped_reward += 0.1
        if next_state[0] > 0.5:  # reached goal!
            shaped_reward += 100

        # Store transition
        agent.replay_buffer.push(state, action, shaped_reward, next_state, done)

        # Train the network
        loss = agent.train_step()
        if loss > 0:
            episode_loss.append(loss)

        state = next_state
        episode_reward += reward

        if done:
            break

    # Update target network periodically
    if episode % agent.target_update == 0:
        agent.update_target_network()

    # Decay exploration
    agent.decay_epsilon()

    # Track metrics
    episode_rewards.append(episode_reward)
    episode_lengths.append(step + 1)
    success_window.append(1 if episode_reward > -200 else 0)  # solved if < 200 steps
    epsilon_history.append(agent.epsilon)
    avg_loss = np.mean(episode_loss) if episode_loss else 0
    losses.append(avg_loss)

    # Print progress
    if (episode + 1) % 50 == 0:
        avg_reward = np.mean(episode_rewards[-50:])
        avg_length = np.mean(episode_lengths[-50:])
        success_rate = np.mean(success_window)
        print(f"Episode {episode + 1}/{num_episodes} - "
              f"Avg Reward: {avg_reward:.1f} - "
              f"Avg Length: {avg_length:.1f} - "
              f"Success Rate: {success_rate:.3f} - "
              f"Epsilon: {agent.epsilon:.4f} - "
              f"Loss: {avg_loss:.4f}")

# -----------------------------
# Final Evaluation
# -----------------------------
print("\n" + "="*60)
print("Training Complete! Running final evaluation...")
print("="*60)

test_rewards = []
test_lengths = []
num_test_episodes = 20

for episode in range(num_test_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = agent.select_action(state, explore=False)  # no exploration
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state
        episode_reward += reward

        if done:
            break

    test_rewards.append(episode_reward)
    test_lengths.append(step + 1)

avg_test_reward = np.mean(test_rewards)
avg_test_length = np.mean(test_lengths)

print(f"\nFinal Performance (20 test episodes):")
print(f"  Average Reward: {avg_test_reward:.2f}")
print(f"  Average Episode Length: {avg_test_length:.1f} steps")
print(f"  Success Rate: {np.sum(np.array(test_rewards) > -200) / len(test_rewards):.3f}")
print(f"  Best Episode: {np.max(test_rewards):.0f} reward")

# -----------------------------
# Analysis
# -----------------------------
print("\n" + "-"*60)
print("Analysis:")
print("-"*60)
print("1. Continuous state space handled by neural network")
print("2. Experience replay stabilizes learning")
print("3. Target network prevents feedback loops")
print("4. Reward shaping helps with sparse rewards")
print("5. Learning is slower than tabular Q-learning but generalizes better")
print("\nWhy DQN works better than tabular methods:")
print("- Neural network generalizes across similar states")
print("- No need for discretization (which loses information)")
print("- Can handle high-dimensional state spaces")
print("-"*60)

# -----------------------------
# Visualization
# -----------------------------
print("\nGenerating plots...")

fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Episode rewards over time
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(episode_rewards, alpha=0.6, linewidth=0.8, color='blue', label='Episode Reward')
# Smooth with moving average
window = 50
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
ax1.plot(range(window-1, len(episode_rewards)), smoothed,
         linewidth=2.5, color='red', label=f'{window}-Episode Moving Average')
ax1.axhline(y=-200, color='green', linestyle='--', linewidth=2,
            label='Solved Threshold (-200)')
ax1.set_xlabel('Episode', fontsize=11)
ax1.set_ylabel('Total Reward', fontsize=11)
ax1.set_title('Training Progress: Episode Rewards', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# 2. Episode lengths
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(episode_lengths, alpha=0.6, linewidth=0.8, color='purple')
smoothed_length = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
ax2.plot(range(window-1, len(episode_lengths)), smoothed_length,
         linewidth=2.5, color='darkred', label=f'{window}-Episode MA')
ax2.set_xlabel('Episode', fontsize=11)
ax2.set_ylabel('Episode Length (steps)', fontsize=11)
ax2.set_title('Episode Length Over Time', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. Training loss
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(losses, alpha=0.6, linewidth=0.8, color='orange')
smoothed_loss = np.convolve(losses, np.ones(window)/window, mode='valid')
ax3.plot(range(window-1, len(losses)), smoothed_loss,
         linewidth=2.5, color='red', label=f'{window}-Episode MA')
ax3.set_xlabel('Episode', fontsize=11)
ax3.set_ylabel('Loss', fontsize=11)
ax3.set_title('Training Loss', fontsize=13, fontweight='bold')
ax3.legend(loc='best', fontsize=10)
ax3.grid(True, alpha=0.3)

# 4. Epsilon decay
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(epsilon_history, linewidth=2, color='green')
ax4.set_xlabel('Episode', fontsize=11)
ax4.set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
ax4.set_title('Exploration Rate Decay', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 5. Test performance comparison
ax5 = fig.add_subplot(gs[2, 1])
test_data = [test_rewards]
ax5.boxplot(test_data, labels=['DQN'])
ax5.axhline(y=-200, color='green', linestyle='--', linewidth=2,
            label='Solved Threshold')
ax5.set_ylabel('Episode Reward', fontsize=11)
ax5.set_title('Final Test Performance Distribution', fontsize=13, fontweight='bold')
ax5.legend(loc='best', fontsize=10)
ax5.grid(True, alpha=0.3, axis='y')

plt.savefig('assignment-3/task3_dqn_results.png', dpi=300, bbox_inches='tight')
print("Saved: task3_dqn_results.png")
plt.show()

# Save model
torch.save(agent.policy_net.state_dict(), 'assignment-3/task3_dqn_model.pth')
print("Saved: task3_dqn_model.pth")

print("\n" + "="*60)
print("Task 3 Complete!")
print("="*60)

env.close()
