"""
Task 2: Tabular Q-Learning on FrozenLake
WiDS Kalman Filtered Trend Trader - Assignment 3

This script implements Q-learning for the FrozenLake environment.
The goal is to learn a policy that navigates from Start (S) to Goal (G)
while avoiding Holes (H) on a slippery frozen lake.
"""

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

print("="*60)
print("Task 2: Q-Learning on FrozenLake")
print("="*60)

# Environment
env = gym.make(
    "FrozenLake-v1",
    map_name="4x4",
    is_slippery=True,  # The lake is slippery, making it stochastic
    render_mode=None
)

n_states = env.observation_space.n
n_actions = env.action_space.n

print(f"\nEnvironment Details:")
print(f"  Number of states: {n_states}")
print(f"  Number of actions: {n_actions}")
print(f"  Map: 4x4 grid")
print(f"  Slippery: Yes (stochastic transitions)")

# -----------------------------
# Q-learning parameters
# -----------------------------
# These parameters were chosen after some experimentation

alpha = 0.1           # learning rate - how much to update Q-values
                      # tried 0.05 (too slow), 0.2 (too unstable), 0.1 seems good

gamma = 0.95          # discount factor - how much to value future rewards
                      # tried 0.9 (too short-sighted), 0.99 (too far-sighted)
                      # 0.95 balances immediate and future rewards

epsilon = 1.0         # initial exploration rate - start by exploring everything
epsilon_min = 0.01    # minimum exploration rate - always keep some exploration
epsilon_decay = 0.995 # exploration decay factor - slowly shift to exploitation
                      # tried 0.99 (decays too fast), 0.999 (too slow)

num_episodes = 10000  # number of training episodes
                      # tried 5000 (not enough), 15000 (diminishing returns)
max_steps = 100       # maximum steps per episode
                      # 4x4 grid shouldn't need more than 100 steps

# -----------------------------
# Q-table
# -----------------------------
# Initialize Q-table with zeros
# Q[state, action] will store the expected return of taking action in state
Q = np.zeros((n_states, n_actions))

print(f"\nHyperparameters:")
print(f"  Learning rate (alpha): {alpha}")
print(f"  Discount factor (gamma): {gamma}")
print(f"  Initial epsilon: {epsilon}")
print(f"  Minimum epsilon: {epsilon_min}")
print(f"  Epsilon decay: {epsilon_decay}")
print(f"  Number of episodes: {num_episodes}")
print(f"  Max steps per episode: {max_steps}")

# -----------------------------
# Tracking performance
# -----------------------------
success_window = deque(maxlen=100)  # track last 100 episodes
success_rates = []
epsilon_history = []

# -----------------------------
# Training loop
# -----------------------------
print("\nStarting training...")
print("This may take a minute or two...\n")

for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    success = 0

    for step in range(max_steps):
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            # Explore: choose random action
            action = env.action_space.sample()
        else:
            # Exploit: choose best action from Q-table
            action = np.argmax(Q[state])

        # Take action and observe result
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-learning update rule
        # Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
        Q[state, action] += alpha * (
            reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

        if done:
            success = reward  # reward is 1 if reached goal, 0 otherwise
            break

    # Decay exploration rate
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track performance
    success_window.append(success)
    success_rates.append(np.mean(success_window))
    epsilon_history.append(epsilon)

    # Print progress every 1000 episodes
    if (episode + 1) % 1000 == 0:
        avg_success = np.mean(success_window)
        print(f"Episode {episode + 1}/{num_episodes} - "
              f"Success Rate: {avg_success:.3f} - "
              f"Epsilon: {epsilon:.4f}")

# -----------------------------
# Final Evaluation
# -----------------------------
print("\n" + "="*60)
print("Training Complete! Running final evaluation...")
print("="*60)

# Test the learned policy
num_test_episodes = 100
test_successes = 0

for episode in range(num_test_episodes):
    state, _ = env.reset()
    done = False

    for step in range(max_steps):
        # Always exploit (no exploration) during testing
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        state = next_state

        if done:
            test_successes += reward
            break

test_success_rate = test_successes / num_test_episodes

print(f"\nFinal Performance (100 test episodes):")
print(f"  Success Rate: {test_success_rate:.3f}")
print(f"  Final Training Success Rate (last 100): {np.mean(success_window):.3f}")
print(f"  Final Epsilon: {epsilon:.4f}")

# -----------------------------
# Analysis
# -----------------------------
print("\n" + "-"*60)
print("Observations:")
print("-"*60)
print("1. Success rate starts near zero (random exploration)")
print("2. Learning is slow and noisy due to:")
print("   - Stochastic environment (slippery ice)")
print("   - Sparse rewards (only at goal)")
print("   - Large state space relative to reward frequency")
print("3. Final performance doesn't reach 100% because:")
print("   - Environment is stochastic (slip probability)")
print("   - No deterministic policy can guarantee success")
print("4. Epsilon decay balances exploration vs exploitation")

# -----------------------------
# Plot learning curve
# -----------------------------
print("\nGenerating plots...")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

# Success rate over time
ax1.plot(success_rates, linewidth=1.5, color='blue', alpha=0.7)
ax1.axhline(y=test_success_rate, color='red', linestyle='--',
            linewidth=2, label=f'Final Test Rate: {test_success_rate:.3f}')
ax1.set_xlabel('Episode', fontsize=11)
ax1.set_ylabel('Success Rate (last 100 episodes)', fontsize=11)
ax1.set_title('FrozenLake Q-Learning: Training Performance', fontsize=13, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# Epsilon decay over time
ax2.plot(epsilon_history, linewidth=1.5, color='green', alpha=0.7)
ax2.set_xlabel('Episode', fontsize=11)
ax2.set_ylabel('Epsilon (Exploration Rate)', fontsize=11)
ax2.set_title('Exploration Rate Decay', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Q-value heatmap for best action per state
q_max_per_state = np.max(Q, axis=1)
q_reshaped = q_max_per_state.reshape(4, 4)

im = ax3.imshow(q_reshaped, cmap='RdYlGn', aspect='auto')
ax3.set_xlabel('Column', fontsize=11)
ax3.set_ylabel('Row', fontsize=11)
ax3.set_title('Learned Q-Values (Max Q per State)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax3)

# Add text annotations
for i in range(4):
    for j in range(4):
        state = i * 4 + j
        best_action = np.argmax(Q[state])
        # 0=Left, 1=Down, 2=Right, 3=Up
        action_arrows = ['←', '↓', '→', '↑']
        text = ax3.text(j, i, f'{action_arrows[best_action]}\n{q_max_per_state[state]:.2f}',
                       ha="center", va="center", color="black", fontsize=9)

plt.tight_layout()
plt.savefig('assignment-3/task2_qlearning_results.png', dpi=300, bbox_inches='tight')
print("Saved: task2_qlearning_results.png")

plt.show()

# Save Q-table for reference
np.save('assignment-3/task2_qtable.npy', Q)
print("Saved: task2_qtable.npy")

# -----------------------------
# Parameter Summary for Submission
# -----------------------------
print("\n" + "="*60)
print("PARAMETER VALUES FOR SUBMISSION")
print("="*60)
print(f"""
alpha (learning rate):        {alpha}
gamma (discount factor):      {gamma}
epsilon (initial):            {1.0}
epsilon_min:                  {epsilon_min}
epsilon_decay:                {epsilon_decay}
num_episodes:                 {num_episodes}
max_steps:                    {max_steps}
""")
print("="*60)

env.close()
