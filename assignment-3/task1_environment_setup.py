"""
Task 1: Environment Setup
WiDS Kalman Filtered Trend Trader - Assignment 3

This script verifies that the Gymnasium environment is correctly set up
and that the FrozenLake environment can be initialized properly.
"""

import gymnasium as gym

print("="*60)
print("Task 1: Environment Setup Verification")
print("="*60)

try:
    # Create the FrozenLake environment
    print("\nStep 1: Creating FrozenLake-v1 environment...")
    env = gym.make("FrozenLake-v1")

    # Reset the environment to get initial state
    print("Step 2: Resetting environment...")
    state, info = env.reset()

    # Print environment information
    print("\n" + "-"*60)
    print("Environment Information:")
    print("-"*60)
    print(f"Initial state: {state}")
    print(f"State space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Number of states: {env.observation_space.n}")
    print(f"Number of actions: {env.action_space.n}")
    print("-"*60)

    # Test a few random actions
    print("\nStep 3: Testing random actions...")
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print(f"  Action {i+1}: {action} -> Next State: {next_state}, Reward: {reward}, Done: {terminated or truncated}")

        if terminated or truncated:
            print("  Episode ended, resetting...")
            state, info = env.reset()

    # Close the environment
    env.close()

    print("\n" + "="*60)
    print("SUCCESS! Environment is correctly set up.")
    print("="*60)
    print("\nYou can now proceed to Task 2 (Q-Learning).")

except Exception as e:
    print("\n" + "="*60)
    print("ERROR: Environment setup failed!")
    print("="*60)
    print(f"\nError message: {str(e)}")
    print("\nPlease ensure you have installed gymnasium:")
    print("  pip install gymnasium")
    print("\nIf the problem persists, contact the course instructors.")
