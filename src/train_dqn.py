"""
Deep Q-Network (DQN) Training Script for Mini Metro RL Environment
"""

import sys
import os

# Add the src directory to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import torch
from datetime import datetime
import json
import numpy as np

from rl_environment import make_env


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def setup_training():
    """Setup training environment and directories"""
    # Create directories for models and logs
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)
    
    print("Training directories created")


def train_dqn(config='default', total_timesteps=50000, learning_rate=1e-4):
    """
    Train a DQN agent on Mini Metro environment
    
    Args:
        config: Environment configuration ('easy', 'default', 'hard')
        total_timesteps: Total training timesteps
        learning_rate: Learning rate for the optimizer
    """
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    print(f"Starting DQN training...")
    print(f"Timestamp: {timestamp}")
    print(f"Config: {config}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Learning rate: {learning_rate}")
    
    # Create environment and get config details
    env = make_env(config)
    
    # Log environment configuration
    env_config = {
        'config_name': config,
        'num_stations': env.num_stations,
        'num_paths': env.num_paths,
        'num_metros': env.num_metros,
        'num_station_types': env.num_station_types,
        'step_interval': env.step_interval,
        'max_episode_steps': env.max_episode_steps,
        'max_station_passengers': env.max_station_passengers,
        'max_total_passengers': env.max_total_passengers,
        'action_space_size': env.action_space.n,
        'state_space_size': env.observation_space.shape[0]
    }
    
    print(f"Environment Configuration:")
    for key, value in env_config.items():
        print(f"   {key}: {value}")
    
    # Create timestamped directories
    log_dir = f"logs/{config}_{timestamp}/"
    model_dir = f"models/{config}_{timestamp}/"
    tensorboard_dir = f"tensorboard_logs/{config}_{timestamp}/"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    
    # Save configuration to file
    config_file = os.path.join(log_dir, "training_config.json")
    
    # Prepare configuration data and convert numpy types
    config_data = {
        'timestamp': timestamp,
        'training_params': {
            'config': config,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate
        },
        'environment_config': env_config,
        'model_params': {
            'buffer_size': 10000,
            'learning_starts': 1000,
            'batch_size': 32,
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.2,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.05,
        }
    }
    
    # Convert numpy types to JSON-serializable types
    config_data = convert_numpy_types(config_data)
    
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print(f"Configuration saved to: {config_file}")
    
    # Wrap environment with Monitor using timestamped log directory
    env = Monitor(env, log_dir)
    
    # Check environment compatibility
    print("Checking environment...")
    check_env(env)
    print("Environment check passed")
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.2,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        tensorboard_log=tensorboard_dir,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        seed=42
    )
    
    print("DQN model created")
    print(f"Logs will be saved to: {log_dir}")
    print(f"Models will be saved to: {model_dir}")
    print(f"TensorBoard logs: {tensorboard_dir}")
    
    # Setup callbacks with timestamped directories
    eval_env = make_env(config)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=1800,        # Evaluate every ~20 episodes (for 90-step episodes)
        deterministic=True,
        render=False,
        n_eval_episodes=5      # More episodes for reliable evaluation
    )
    
    # Train the model
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=4,
        progress_bar=True
    )
    
    # Save final model with timestamp
    final_model_path = os.path.join(model_dir, f"dqn_{config}_{total_timesteps}steps_final.zip")
    model.save(final_model_path)
    
    # Create training summary
    summary_file = os.path.join(log_dir, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Mini Metro DQN Training Summary\n")
        f.write(f"================================\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Environment: {env_config['num_stations']} stations, {env_config['num_paths']} paths, {env_config['num_metros']} metros\n")
        f.write(f"Action Space: {env_config['action_space_size']} actions\n")
        f.write(f"State Space: {env_config['state_space_size']} features\n")
        f.write(f"Final Model: {final_model_path}\n")
        f.write(f"Log Directory: {log_dir}\n")
        f.write(f"TensorBoard: tensorboard --logdir {tensorboard_dir}\n")
    
    print("Training completed successfully!")
    print(f"Final model saved: {final_model_path}")
    print(f"Training summary: {summary_file}")
    print(f"View training progress: tensorboard --logdir {tensorboard_dir}")
    
    return env


def evaluate_model(model_path="models", config='default', n_episodes=10, render_mode=None, seed=42):
    """
    Evaluate a trained model
    
    Args:
        model_path: Path to the saved model
        config: Environment configuration
        n_episodes: Number of episodes to evaluate
        seed: Base seed for random generation (each episode uses seed+episode_num)
    """
    import random
    print(f"Evaluating model: {model_path}")
    
    # Load model and environment
    model = DQN.load(model_path)
    env = make_env(config, render_mode=render_mode)
    
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        # Use different seed for each episode to get varied station layouts
        episode_seed = seed + episode if seed is not None else None
        if episode_seed is not None:
            np.random.seed(episode_seed)
            random.seed(episode_seed)
        
        obs, info = env.reset(seed=episode_seed)
        episode_reward = 0
        episode_length = 0
        
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Score={info['score']}, Length={episode_length}")
    
    # Print statistics
    print(f"\nEvaluation Results (n={n_episodes}):")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    return episode_rewards, episode_scores, episode_lengths


def test_random_baseline(config='default', n_episodes=10):
    """
    Test random action baseline for comparison
    
    Args:
        config: Environment configuration
        n_episodes: Number of episodes to test
    """
    print(f"Testing random baseline with config: {config}")
    
    env = make_env(config)
    episode_rewards = []
    episode_scores = []
    episode_lengths = []
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        
        while True:
            action = env.action_space.sample()  # Random action
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            
            if terminated or truncated:
                break
        
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Score={info['score']}, Length={episode_length}")
    
    print(f"\nRandom Baseline Results (n={n_episodes}):")
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    
    return episode_rewards, episode_scores, episode_lengths


if __name__ == "__main__":
    import argparse
    import time
    import random

    current_seed = int(time.time()) % 10000
    random.seed(current_seed)
    np.random.seed(current_seed)
    print(f"Using base seed: {current_seed} (each episode will use base_seed + episode_number)")
    
    parser = argparse.ArgumentParser(description='Train DQN on Mini Metro')
    parser.add_argument('--mode', choices=['train', 'eval', 'baseline'], 
                       default='train', help='Mode to run')
    parser.add_argument('--config', choices=['tiny', 'easy', 'default', 'hard', 'large'], 
                       default='default', help='Environment configuration')
    parser.add_argument('--timesteps', type=int, default=10000, 
                       help='Training timesteps')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Episodes for evaluation/baseline')
    parser.add_argument('--render', choices=['human', 'text', 'none'], 
                       default='human', help='Rendering mode for evaluation')
    parser.add_argument('--render-fps', type=int, default=10, 
                       help='FPS for human rendering (lower = slower)')
    
    args = parser.parse_args()
    
    print("Mini Metro DQN Training Script")
    print("=" * 50)
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    print(f"PyTorch version: {torch.__version__}")
    
    setup_training()
    
    if args.mode == 'train':
        env = train_dqn(
            config=args.config, 
            total_timesteps=args.timesteps, 
            learning_rate=args.learning_rate
        )
        env.close()
        
    elif args.mode == 'eval':
        # Find model if not specified
        model_path = args.model
        if model_path is None:
            import glob
            pattern = os.path.join("models", f"{args.config}_*", f"dqn_{args.config}_*steps_final.zip")
            model_files = glob.glob(pattern)
            if model_files:
                model_path = max(model_files, key=os.path.getctime)  # Most recent
                print(f"Using model: {model_path}")
            else:
                print(f"No trained model found for config '{args.config}'")
                print(f"Train a model first with: python train_dqn.py --mode train --config {args.config}")
                exit(1)
        
        render_mode = None if args.render == 'none' else args.render
        evaluate_model(
            model_path=model_path,
            config=args.config,
            n_episodes=args.episodes,
            render_mode=render_mode,
            seed=current_seed
        )
        
    elif args.mode == 'baseline':
        test_random_baseline(
            config=args.config,
            n_episodes=args.episodes
        )
    
    print("\nScript completed successfully!")
