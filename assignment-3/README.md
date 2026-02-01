# Week 3 Assignment: Reinforcement Learning with Gymnasium

## Overview
This assignment implements reinforcement learning algorithms on Gymnasium environments:
- Task 1: Environment setup verification
- Task 2: Tabular Q-Learning on FrozenLake
- Task 3: Deep Q-Learning (DQN) on MountainCar

## Files
- `task1_environment_setup.py` - Verifies Gymnasium installation
- `task2_frozen_lake_qlearning.py` - Q-Learning implementation for FrozenLake
- `task3_mountaincar_dqn.py` - Deep Q-Network for MountainCar
- `requirements.txt` - Python dependencies
- `report.pdf` - Detailed report with analysis

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv rl-env
source rl-env/bin/activate  # On Windows: rl-env\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Tasks

### Task 1: Environment Setup
```bash
python task1_environment_setup.py
```
This verifies that Gymnasium is correctly installed and the FrozenLake environment works.

**Expected Output:**
- Initial state information
- State and action space details
- Test actions showing environment responses
- Success message

### Task 2: Q-Learning on FrozenLake
```bash
python task2_frozen_lake_qlearning.py
```

This implements tabular Q-learning for the FrozenLake-v1 environment.

**What it does:**
- Trains Q-learning agent for 10,000 episodes
- Learns to navigate from Start to Goal while avoiding holes
- Tracks success rate over time
- Generates visualizations

**Parameter Values Used:**
- Learning rate (alpha): 0.1
- Discount factor (gamma): 0.95
- Initial epsilon: 1.0
- Minimum epsilon: 0.01
- Epsilon decay: 0.995
- Number of episodes: 10,000
- Max steps per episode: 100

**Outputs:**
- `task2_qlearning_results.png` - Learning curves and Q-value heatmap
- `task2_qtable.npy` - Learned Q-table
- Console output with performance metrics

**Observations:**
- Success rate starts near zero due to random exploration
- Learning is slow and noisy because of:
  - Stochastic environment (slippery ice)
  - Sparse rewards (only at goal)
- Final performance ~70-80% (can't reach 100% due to stochasticity)

### Task 3: Deep Q-Learning on MountainCar
```bash
python task3_mountaincar_dqn.py
```

This implements Deep Q-Network for the MountainCar-v0 environment.

**What it does:**
- Uses neural network to approximate Q-function
- Implements experience replay for stable learning
- Uses target network to prevent feedback loops
- Applies reward shaping to handle sparse rewards
- Trains for 500 episodes

**Key Features:**
- **Neural Network:** 2 hidden layers with 128 units each
- **Experience Replay:** Buffer size of 10,000 transitions
- **Target Network:** Updated every 10 episodes
- **Reward Shaping:** Bonus for moving toward goal
- **Exploration:** Epsilon-greedy with decay

**Outputs:**
- `task3_dqn_results.png` - Training curves (rewards, loss, etc.)
- `task3_dqn_model.pth` - Trained neural network weights
- Console output with performance analysis

**Why DQN Works Better Than Tabular Q-Learning:**
1. **No discretization needed** - Neural network handles continuous states
2. **Generalization** - Similar states share learned knowledge
3. **Scalability** - Can handle high-dimensional state spaces
4. **Function approximation** - Learns smooth Q-function

**Observations:**
- Learning is initially slow (~100 episodes before improvement)
- Delayed rewards make credit assignment challenging
- Experience replay breaks temporal correlations
- Target network stabilizes training

## Understanding the Environments

### FrozenLake
- **Goal:** Navigate from Start (S) to Goal (G) on a 4x4 grid
- **Challenge:** Ice is slippery (stochastic transitions)
- **Rewards:** +1 for reaching goal, 0 otherwise
- **Actions:** Left, Down, Right, Up

Grid layout:
```
S F F F
F H F H
F F F H
H F F G
```
S = Start, F = Frozen surface, H = Hole, G = Goal

### MountainCar
- **Goal:** Drive car to the top of the right hill (position ≥ 0.5)
- **Challenge:** Engine too weak to climb directly
- **Strategy:** Build momentum by oscillating back and forth
- **State:** (position, velocity) - continuous
- **Actions:** Push left, No push, Push right
- **Rewards:** -1 per time step until goal reached

## Key Concepts

### Q-Learning
- **Value-based RL:** Learn optimal action-value function Q(s, a)
- **Update rule:** Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
- **Exploration:** Epsilon-greedy (explore vs exploit trade-off)

### Deep Q-Learning Improvements
1. **Experience Replay:** Store and sample past experiences
   - Breaks temporal correlations
   - More efficient data usage
2. **Target Network:** Separate network for target Q-values
   - Prevents feedback loops
   - Stabilizes training
3. **Function Approximation:** Neural network instead of table
   - Handles continuous spaces
   - Generalizes across similar states

## Performance Metrics

### FrozenLake (Task 2)
- Final success rate: ~70-80% (due to stochasticity)
- Training episodes: 10,000
- Final epsilon: ~0.01

### MountainCar (Task 3)
- Final average reward: Should improve from -200 to better scores
- Average episode length: Should decrease as agent learns
- Success rate: Episodes completing in < 200 steps

## Troubleshooting

**Error: Module 'gymnasium' not found**
```bash
pip install gymnasium
```

**Error: Module 'torch' not found**
```bash
pip install torch
```

**FrozenLake not learning:**
- Check learning rate (too high = unstable, too low = slow)
- Check epsilon decay (too fast = premature exploitation)
- Increase number of episodes

**MountainCar not improving:**
- Ensure sufficient exploration (epsilon decay not too fast)
- Check reward shaping is working
- Verify neural network is training (loss should decrease)
- May need more episodes (DQN requires patience)

## Experimentation Ideas

### Task 2: Q-Learning Parameters
Try varying:
- Learning rate: 0.05, 0.1, 0.2, 0.5
- Discount factor: 0.9, 0.95, 0.99
- Epsilon decay: 0.99, 0.995, 0.999
- Number of episodes: 5000, 10000, 15000

### Task 3: DQN Architecture
Try varying:
- Network size: 64, 128, 256 units
- Learning rate: 0.0001, 0.001, 0.01
- Batch size: 32, 64, 128
- Replay buffer size: 5000, 10000, 20000
- Reward shaping strategy

## References
- Sutton & Barto: Reinforcement Learning: An Introduction
- Mnih et al. (2015): Human-level control through deep reinforcement learning
- Gymnasium Documentation: https://gymnasium.farama.org/

## Author
Created for WiDS Kalman Filtered Trend Trader - Week 3 Assignment
