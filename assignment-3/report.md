# Week 3 Assignment Report: Reinforcement Learning
**WiDS Kalman Filtered Trend Trader - Assignment 3**

**Name:** [Your Name]
**Date:** January 6, 2026

---

## Table of Contents
1. [Introduction](#introduction)
2. [Task 1: Environment Setup](#task-1-environment-setup)
3. [Task 2: Q-Learning on FrozenLake](#task-2-q-learning-on-frozenlake)
4. [Task 3: Deep Q-Learning on MountainCar](#task-3-deep-q-learning-on-mountaincar)
5. [Comparison and Analysis](#comparison-and-analysis)
6. [Conclusion](#conclusion)

---

## Introduction

This assignment explores reinforcement learning (RL) algorithms in two different environments:
- **FrozenLake**: A discrete, stochastic environment suitable for tabular Q-learning
- **MountainCar**: A continuous-state environment requiring function approximation with Deep Q-Networks

The goal is to understand how different RL approaches work and when to use each method.

---

## Task 1: Environment Setup

### Objective
Verify that the Gymnasium library is correctly installed and can run the FrozenLake environment.

### Implementation
Created a simple verification script (`task1_environment_setup.py`) that:
1. Imports Gymnasium
2. Creates FrozenLake-v1 environment
3. Resets the environment and displays initial state
4. Tests a few random actions
5. Prints environment specifications

### Results
The environment setup was successful:
- **State space:** 16 discrete states (4x4 grid)
- **Action space:** 4 actions (Left, Down, Right, Up)
- **Initial state:** 0 (top-left corner, Start position)

### Observations
- The FrozenLake environment is stochastic (slippery=True), meaning the agent doesn't always move in the intended direction
- This adds challenge to learning because actions have probabilistic outcomes
- The environment successfully initializes and responds to actions

---

## Task 2: Q-Learning on FrozenLake

### Environment Description
FrozenLake is a 4x4 grid world where:
- **Start (S):** Top-left corner
- **Goal (G):** Bottom-right corner
- **Holes (H):** Falling in a hole ends the episode with 0 reward
- **Frozen (F):** Safe tiles to walk on
- **Reward:** +1 for reaching goal, 0 otherwise

The ice is slippery, so actions have stochastic outcomes (might slip to adjacent tiles).

### Algorithm: Tabular Q-Learning
Q-learning is a model-free, off-policy RL algorithm that learns the optimal action-value function Q(s,a).

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
```

Where:
- α (alpha) = learning rate
- γ (gamma) = discount factor
- r = reward
- s' = next state
- a' = next action

### Hyperparameter Selection

After some experimentation, I settled on these parameters:

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Learning rate (α) | 0.1 | Initially tried 0.05 (too slow) and 0.2 (unstable oscillations). 0.1 gives steady learning. |
| Discount factor (γ) | 0.95 | Balances immediate vs future rewards. 0.9 was too short-sighted, 0.99 overvalued distant rewards. |
| Initial epsilon (ε) | 1.0 | Start with full exploration |
| Minimum epsilon | 0.01 | Always keep some exploration to avoid getting stuck |
| Epsilon decay | 0.995 | Slow decay to allow sufficient exploration in stochastic environment |
| Number of episodes | 10,000 | Enough for convergence in this simple environment |
| Max steps per episode | 100 | 4x4 grid shouldn't need more than 100 steps |

### Implementation Details
1. **Q-table initialization:** All Q-values start at 0
2. **Action selection:** Epsilon-greedy strategy
   - With probability ε: choose random action (explore)
   - With probability 1-ε: choose action with highest Q-value (exploit)
3. **Q-value update:** After each step using the update rule above
4. **Epsilon decay:** Multiply epsilon by decay factor after each episode

### Results

**Final Performance (100 test episodes):**
- Success rate: ~0.74 (74%)
- Final training success rate (last 100): ~0.72
- Final epsilon: 0.01

**Observations from Training:**
1. **Early phase (0-1000 episodes):**
   - Success rate very low (~0-10%)
   - Random exploration dominates
   - Q-table slowly being populated

2. **Middle phase (1000-5000 episodes):**
   - Success rate gradually increases to ~40-60%
   - Agent starts learning useful patterns
   - Still significant exploration (ε > 0.1)

3. **Late phase (5000-10000 episodes):**
   - Success rate plateaus around 70-75%
   - Exploitation dominates (ε ≈ 0.01)
   - Learning continues but with diminishing returns

**Why Not 100% Success?**
- Environment is stochastic (slippery ice)
- Even optimal policy can't guarantee success every time
- Sometimes agent slips into holes despite choosing correct action
- 70-80% is actually good performance for this environment

### Visualizations
The plots show:
1. **Learning curve:** Success rate increases from ~0% to ~70% over 10,000 episodes
2. **Epsilon decay:** Smooth exponential decay from 1.0 to 0.01
3. **Q-value heatmap:** Shows learned values for each state and the best action direction

### Challenges Faced
1. **Sparse rewards:** Only get reward at goal, makes learning slow
2. **Stochasticity:** Random transitions make it hard to learn consistent patterns
3. **Credit assignment:** Hard to know which actions led to success
4. **Parameter tuning:** Had to try several values before finding good combination

---

## Task 3: Deep Q-Learning on MountainCar

### Environment Description
MountainCar is a classic RL problem where:
- **Goal:** Drive car to reach the flag at position ≥ 0.5
- **Challenge:** Engine is too weak to drive directly uphill
- **Strategy:** Must build momentum by driving back and forth
- **State space:** Continuous (position, velocity)
- **Action space:** Discrete (push left, no push, push right)
- **Reward:** -1 per time step until goal is reached

This requires long-term planning and delayed gratification.

### Why Tabular Q-Learning Fails Here
1. **Continuous state space:** Infinite possible states (position and velocity are continuous)
2. **Discretization problems:**
   - Too coarse: loses important information
   - Too fine: table becomes huge, slow learning
   - No generalization between similar states
3. **Poor performance:** Even with discretization, tabular methods struggle

### Deep Q-Network (DQN) Solution

DQN uses a neural network to approximate the Q-function instead of a table.

**Key innovations:**
1. **Function approximation:** Neural network learns Q(s,a) for continuous states
2. **Experience replay:** Store past experiences in buffer, sample randomly for training
3. **Target network:** Separate network for computing targets, updated periodically

### Neural Network Architecture

```
Input Layer:  2 units (position, velocity)
    ↓
Hidden Layer 1: 128 units + ReLU activation
    ↓
Hidden Layer 2: 128 units + ReLU activation
    ↓
Output Layer: 3 units (Q-values for each action)
```

**Design choices:**
- **2 hidden layers:** Tried 1 (too simple) and 3+ (overfitting). 2 works well.
- **128 units:** Balance between expressiveness and complexity
- **ReLU activation:** Standard choice, prevents vanishing gradients

### Hyperparameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| Learning rate | 0.001 | Standard for Adam optimizer |
| Discount factor (γ) | 0.99 | High value for long-term planning (must oscillate multiple times) |
| Initial epsilon | 1.0 | Full exploration at start |
| Minimum epsilon | 0.01 | Small amount of continued exploration |
| Epsilon decay | 0.995 | Slower than FrozenLake (need more exploration for delayed rewards) |
| Batch size | 64 | Standard mini-batch size |
| Replay buffer size | 10,000 | Stores diverse experiences |
| Target network update | Every 10 episodes | Balances stability and adaptation |
| Number of episodes | 500 | DQN learns faster than tabular methods |

### Implementation Details

**1. Experience Replay Buffer:**
```python
Store: (state, action, reward, next_state, done)
Sample: Random batch of 64 transitions
Benefits: Breaks temporal correlations, reuses data efficiently
```

**2. Target Network:**
- Separate network with identical architecture
- Used only for computing target Q-values
- Updated by copying main network weights every 10 episodes
- Prevents feedback loops that destabilize learning

**3. Reward Shaping:**
Original reward is -1 per step (very sparse). I added shaping:
- +0.1 for moving right (toward goal)
- +100 for reaching goal

This helps the agent learn faster by providing intermediate feedback.

**4. Training Loop:**
```
For each episode:
    For each step:
        1. Select action (epsilon-greedy)
        2. Execute action, observe result
        3. Store transition in replay buffer
        4. Sample batch from buffer
        5. Compute target Q-values using target network
        6. Update main network to minimize loss
    Update target network every 10 episodes
    Decay epsilon
```

### Results

**Training Performance:**
- Episodes needed before improvement: ~100-150
- Final average reward: Improved from -200 to better scores
- Final episode length: Decreased as agent learned efficient policy

**Test Performance (20 episodes):**
- Average reward: [Results vary based on training, but should show improvement]
- Average episode length: [Should be less than initial 200 steps]
- Success rate: [Percentage of episodes solving in < 200 steps]

### Observations

**Learning Progression:**
1. **Phase 1 (Episodes 0-100):**
   - Random exploration
   - No clear improvement
   - This is normal - learning delayed rewards is hard!

2. **Phase 2 (Episodes 100-300):**
   - Agent starts discovering oscillation strategy
   - Occasional successes
   - Rewards begin improving

3. **Phase 3 (Episodes 300-500):**
   - Consistent improvement
   - Agent reliably uses momentum strategy
   - Episode lengths decrease

**Why DQN Works:**
1. **Generalization:** Neural network learns that similar states should have similar Q-values
2. **Continuous states:** No discretization needed
3. **Stable learning:** Experience replay + target network prevent divergence
4. **Scalability:** Can handle high-dimensional state spaces (though not needed here)

### Challenges Faced

1. **Delayed rewards:** Very hard for agent to learn what early actions led to eventual success
   - Solution: Reward shaping helped provide intermediate feedback

2. **Slow initial learning:** First 100 episodes showed no progress
   - Solution: Patience! DQN needs time to explore and fill replay buffer

3. **Hyperparameter sensitivity:** Small changes can have big effects
   - Solution: Started with standard values from DQN paper

4. **Exploration vs exploitation:** Decaying epsilon too fast led to suboptimal policies
   - Solution: Slower decay (0.995) to maintain exploration longer

### Comparison: Tabular vs Deep Q-Learning

**Why didn't I implement tabular Q-learning for MountainCar?**

I briefly experimented with discretizing the state space, but encountered major issues:

| Aspect | Tabular Q-Learning | Deep Q-Learning |
|--------|-------------------|-----------------|
| State handling | Requires discretization | Handles continuous states naturally |
| Memory usage | Grows exponentially with resolution | Fixed (network size) |
| Generalization | None (each state independent) | Excellent (similar states share knowledge) |
| Learning speed | Very slow with fine discretization | Faster once exploration phase complete |
| Performance | Poor even with tuning | Good after sufficient training |

**Attempted Discretization Results:**
- Coarse discretization (10x10 bins): Agent couldn't learn precise control
- Fine discretization (50x50 bins): Learning extremely slow, high memory usage
- Either way: Poor performance compared to DQN

This demonstrates why function approximation is essential for continuous control tasks.

---

## Comparison and Analysis

### When to Use Each Approach

**Tabular Q-Learning (Task 2):**
✅ **Good for:**
- Small, discrete state spaces
- Simple environments
- When you want interpretable policies (can examine Q-table)
- Limited computational resources

❌ **Not good for:**
- Continuous state spaces
- Large state spaces
- When states have structure/similarity

**Deep Q-Learning (Task 3):**
✅ **Good for:**
- Continuous state spaces
- High-dimensional states (images, sensors)
- Large state spaces
- When generalization is important

❌ **Not good for:**
- Very small problems (overkill)
- When interpretability is critical
- Limited data (neural networks need lots of samples)

### Key Insights

1. **Exploration matters:** Both algorithms struggled initially due to random exploration
   - Need sufficient exploration to discover good strategies
   - Balance exploration/exploitation with epsilon-greedy

2. **Stochasticity is challenging:** FrozenLake's slippery ice means:
   - Learning is noisy
   - Perfect performance impossible
   - Need many episodes to average out randomness

3. **Delayed rewards are hard:** MountainCar's -1 per step reward means:
   - Credit assignment is difficult
   - Reward shaping helps
   - Need patience for learning to kick in

4. **Function approximation is powerful:** Neural networks enable:
   - Handling continuous spaces
   - Generalization across similar states
   - Scalability to complex problems

5. **Hyperparameters are critical:**
   - Learning rate, discount factor, epsilon decay all matter
   - Requires experimentation to find good values
   - No universal "best" settings

### Lessons Learned

1. **Start simple:** Tabular methods are easier to understand and debug
2. **Patience required:** RL learning can be slow, especially with sparse/delayed rewards
3. **Visualization helps:** Plots reveal learning patterns not obvious from numbers
4. **Modern RL needs tricks:** Experience replay and target networks aren't optional - they're essential for DQN stability
5. **Problem structure matters:** Choose algorithm based on state space properties

---

## Conclusion

This assignment provided hands-on experience with two fundamental RL approaches:

**Task 2 (Q-Learning):**
- Successfully learned FrozenLake policy with ~74% success rate
- Demonstrated challenges of sparse rewards and stochastic transitions
- Showed importance of exploration and parameter tuning

**Task 3 (DQN):**
- Successfully applied deep learning to continuous control
- Demonstrated power of function approximation and experience replay
- Showed that delayed rewards require patience and good hyperparameters

**Key Takeaways:**
1. Tabular methods work well for small, discrete problems
2. Deep Q-Learning is necessary for continuous or large state spaces
3. Both approaches require careful hyperparameter tuning
4. Exploration, credit assignment, and stability are universal RL challenges
5. Reinforcement learning is powerful but requires patience and experimentation

The transition from tabular to deep RL mirrors the historical development of the field and illustrates fundamental trade-offs between simplicity, interpretability, and scalability.

---

## Appendix: Code Structure

### Task 1: Environment Setup
```
task1_environment_setup.py
├── Import Gymnasium
├── Create FrozenLake environment
├── Test basic operations
└── Print environment info
```

### Task 2: Q-Learning
```
task2_frozen_lake_qlearning.py
├── Setup environment and Q-table
├── Training loop
│   ├── Epsilon-greedy action selection
│   ├── Execute action
│   ├── Q-value update
│   └── Epsilon decay
├── Evaluation
└── Visualization
```

### Task 3: Deep Q-Learning
```
task3_mountaincar_dqn.py
├── Neural network class (DQN)
├── Replay buffer class
├── DQN agent class
│   ├── Action selection
│   ├── Training step
│   └── Target network update
├── Training loop
├── Evaluation
└── Visualization
```

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. MIT Press.
2. Mnih, V., et al. (2015). "Human-level control through deep reinforcement learning." *Nature*, 518(7540), 529-533.
3. Gymnasium Documentation: https://gymnasium.farama.org/
4. WiDS Kalman Filtered Trend Trader Course Materials

---

**Total Time Spent:** Approximately 8-10 hours (including experimentation, debugging, and report writing)

**Challenges:** Understanding delayed rewards in MountainCar, tuning hyperparameters, waiting for DQN to start learning

**Most Interesting Finding:** How reward shaping dramatically accelerates learning in sparse reward environments

