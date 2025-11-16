"""
Test script for Mini Metro RL Environment
Tests the basic functionality of the RL environment
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from rl_environment import MiniMetroRLEnv, make_env


def test_basic_functionality():
    """Test basic environment functionality"""
    print("=== Testing Basic Functionality ===")
    
    # Test environment creation
    env = MiniMetroRLEnv()
    print(f"Environment created successfully")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Environment reset successfully")
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial observation range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test steps
    total_reward = 0
    for step in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"  Step {step}: Action={action}, Reward={reward:.3f}, "
              f"Action={info['action_type']}, Success={info['action_success']}")
        
        if terminated or truncated:
            print(f"  Episode terminated at step {step}")
            break
    
    print(f"Steps executed successfully. Total reward: {total_reward:.3f}")
    env.close()


def test_actions():
    """Test all types of actions"""
    print("\n=== Testing Actions ===")
    
    env = MiniMetroRLEnv()
    obs, info = env.reset()
    
    # Test build path action
    print("Testing build path actions...")
    action = 0  # Build path between stations 0 and 1
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Build path (0,1): Success={info['action_success']}, Reward={reward:.3f}")
    
    # Test extend path action
    print("Testing extend path actions...")
    action = env.extend_action_start  # Extend path 0 with station 2
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"  Extend path 0 with station 0: Success={info['action_success']}, Reward={reward:.3f}")
    
    action = env.extend_action_start + 2  # Extend path 0 with station 2
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Extend path 0 with station 2: Success={info['action_success']}, Reward={reward:.3f}")
    
    # Test another build path action
    action = 4  # Build path between stations 1 and 3
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Build path (1,3): Success={info['action_success']}, Reward={reward:.3f}")
    
    # Test remove path action
    action = env.remove_action_start  # Remove path 0
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Remove path 0: Success={info['action_success']}, Reward={reward:.3f}")
    
    # Test no-op action
    action = env.noop_action  # No operation
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"No-op: Success={info['action_success']}, Reward={reward:.3f}")
    
    env.close()


def test_configurations():
    """Test different environment configurations"""
    print("\n=== Testing Configurations ===")
    
    configs = ['easy', 'default', 'hard']
    
    for config_name in configs:
        print(f"Testing {config_name} configuration...")
        env = make_env(config_name)
        obs, info = env.reset()
        
        # Run a few steps
        for _ in range(3):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                break
        
        print(f"{config_name.capitalize()} config working. "
              f"Score: {info['score']}, Passengers: {info['passengers']}")
        env.close()


def test_state_consistency():
    """Test state vector consistency"""
    print("\n=== Testing State Consistency ===")
    
    env = MiniMetroRLEnv()
    obs, info = env.reset()
    
    print(f"State vector shape: {obs.shape}")
    print(f"State vector dtype: {obs.dtype}")
    
    # Check if state is normalized
    print(f"State range: [{obs.min():.3f}, {obs.max():.3f}]")
    
    # Test multiple steps and check state consistency
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Check for NaN or inf values
        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
            print(f"Invalid values in state at step {i}")
            break
        
        # Check state bounds
        if obs.min() < 0 or obs.max() > 1:
            print(f"State out of bounds at step {i}: [{obs.min():.3f}, {obs.max():.3f}]")
    
    print(f"State consistency check completed")
    env.close()


if __name__ == "__main__":
    print("Mini Metro RL Environment Test Suite")
    print("=" * 50)
    
    try:
        test_basic_functionality()
        test_actions()
        test_configurations()
        test_state_consistency()
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
