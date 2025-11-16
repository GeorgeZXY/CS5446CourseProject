import sys
import os
import random
import argparse
import time
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch
from datetime import datetime
import json
import numpy as np

from rl_environment import make_env


class RenderCallback(BaseCallback):
    """
    Callback for rendering the environment during training
    Renders every N steps to visualize agent's progress
    """
    def __init__(self, render_freq=1000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.render_env = None
        
    def _on_training_start(self):
        # Create a separate environment for rendering
        config = 'default'  # You can make this configurable
        self.render_env = make_env(config, render_mode='human', render_fps=60)
        
    def _on_step(self):
        # Render periodically
        if self.n_calls % self.render_freq == 0 and self.render_env is not None:
            obs, _ = self.render_env.reset()
            done = False
            step_count = 0
            max_steps = 100  # Limit visualization steps
            
            while not done and step_count < max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.render_env.step(action)
                done = terminated or truncated
                step_count += 1
                
        return True
    
    def _on_training_end(self):
        if self.render_env is not None:
            self.render_env.close()


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
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("tensorboard_logs", exist_ok=True)


def make_vec_env(config='default', n_envs=4, seed=42):
    """
    Create vectorized environments
    """
    def make_env_fn(rank):
        def _init():
            env = make_env(config)
            env.reset(seed=seed + rank)
            return env
        return _init
    # Use SubprocVecEnv for true parallelism
    if n_envs > 1:
        vec_env = SubprocVecEnv([make_env_fn(i) for i in range(n_envs)])
    else:
        vec_env = DummyVecEnv([make_env_fn(0)])
    
    return vec_env


def train_ppo(
    config='default', 
    total_timesteps=100000, 
    learning_rate=3e-4,
    n_envs=4,
    custom_params=None,
    device='auto'
):
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    # Select suitble device for training
    if device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    # Environment configuration for logging purposes
    single_env = make_env(config)
    env_config = {
        'config_name': config,
        'num_stations': single_env.num_stations,
        'num_paths': single_env.num_paths,
        'num_metros': single_env.num_metros,
        'num_station_types': single_env.num_station_types,
        'step_interval': single_env.step_interval,
        'max_episode_steps': single_env.max_episode_steps,
        'max_station_passengers': single_env.max_station_passengers,
        'max_total_passengers': single_env.max_total_passengers,
        'action_space_size': single_env.action_space.n,
        'state_space_size': single_env.observation_space.shape[0]
    }
    single_env.close()
    # Make diretcories for logs and models
    log_dir = f"logs/ppo_{config}_{timestamp}/"
    model_dir = f"models/ppo_{config}_{timestamp}/"
    tensorboard_dir = f"tensorboard_logs/ppo_{config}_{timestamp}/"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    # PPO Hyperparameters
    ppo_params = {
        'n_steps': 2048,
        'batch_size': 64,
        'n_epochs': 10,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_range': 0.2,
        'clip_range_vf': None,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'max_grad_norm': 0.5,
        'use_sde': False,
        'sde_sample_freq': -1,
        'target_kl': None,
    }
    # Override with custom parameters if provided
    if custom_params:
        ppo_params.update(custom_params)
    # Save configuration to file
    config_file = os.path.join(log_dir, "training_config.json")
    config_data = {
        'timestamp': timestamp,
        'algorithm': 'PPO',
        'training_params': {
            'config': config,
            'total_timesteps': total_timesteps,
            'learning_rate': learning_rate,
            'n_envs': n_envs
        },
        'environment_config': env_config,
        'ppo_params': ppo_params
    }
    # Convert numpy types to JSON-serializable types for archiving
    config_data = convert_numpy_types(config_data)
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=4)
    # Create vectorized environments
    vec_env = make_vec_env(config, n_envs=n_envs)
    # Policy network architecture
    policy_kwargs = dict(
        net_arch=[
            dict(pi=[256, 256], vf=[256, 256])
        ]
    )
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=learning_rate,
        n_steps=ppo_params['n_steps'],
        batch_size=ppo_params['batch_size'],
        n_epochs=ppo_params['n_epochs'],
        gamma=ppo_params['gamma'],
        gae_lambda=ppo_params['gae_lambda'],
        clip_range=ppo_params['clip_range'],
        clip_range_vf=ppo_params['clip_range_vf'],
        ent_coef=ppo_params['ent_coef'],
        vf_coef=ppo_params['vf_coef'],
        max_grad_norm=ppo_params['max_grad_norm'],
        use_sde=ppo_params['use_sde'],
        sde_sample_freq=ppo_params['sde_sample_freq'],
        target_kl=ppo_params['target_kl'],
        tensorboard_log=tensorboard_dir,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=42,
        device=device
    )
    # Setup callbacks
    eval_env = make_env(config)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(5000 // n_envs, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
        verbose=1
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10000 // n_envs, 1),
        save_path=model_dir,
        name_prefix='ppo_checkpoint',
        save_replay_buffer=False,
        save_vecnormalize=True,
        verbose=1
    )
    # Combine callbacks
    callbacks = [eval_callback, checkpoint_callback]
    callback = CallbackList(callbacks)
    # Train the model
    print("Training Model with PPO...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True
    )
    # Save final model
    final_model_path = os.path.join(model_dir, f"ppo_{config}_{total_timesteps}steps_final.zip")
    model.save(final_model_path)
    # Training summary
    summary_file = os.path.join(log_dir, "training_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Mini Metro PPO Training Summary:\n")
        f.write(f"Algorithm: Proximal Policy Optimization (PPO)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Configuration: {config}\n")
        f.write(f"Total Timesteps: {total_timesteps}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Parallel Environments: {n_envs}\n")
        f.write(f"Environment: {env_config['num_stations']} stations, "
                f"{env_config['num_paths']} paths, {env_config['num_metros']} metros\n")
        f.write(f"Action Space: {env_config['action_space_size']} actions\n")
        f.write(f"State Space: {env_config['state_space_size']} features\n\n")
        f.write(f"PPO Hyperparameters:\n")
        for key, value in ppo_params.items():
            f.write(f"  {key}: {value}\n")
        f.write(f"Final Model: {final_model_path}\n")
    print("Training completed successfully!")
    # Cleanup
    vec_env.close()
    eval_env.close()
    
    return model


def evaluate_model(model_path, config='default', n_episodes=10, render_mode=None, seed=42):
    """
    Evaluate a trained PPO model
    """
    # Load model and environment
    model = PPO.load(model_path)
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
        # Explicitly render initial state
        if render_mode == 'human':
            env.render()
        while True:
            # Use deterministic policy for evaluation
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if render_mode == 'human':
                env.render()
            if terminated or truncated:
                break
        episode_rewards.append(episode_reward)
        episode_scores.append(info['score'])
        episode_lengths.append(episode_length)
    
    # Print statistics
    print(f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Score: {np.mean(episode_scores):.2f} ± {np.std(episode_scores):.2f}")
    print(f"Average Length: {np.mean(episode_lengths):.2f} ± {np.std(episode_lengths):.2f}")
    print(f"Best Score: {np.max(episode_scores)}")
    print(f"Worst Score: {np.min(episode_scores)}")
    
    env.close()
    
    return episode_rewards, episode_scores, episode_lengths


if __name__ == "__main__":
    # Set random seed based on current time
    current_seed = int(time.time()) % 10000
    random.seed(current_seed)
    np.random.seed(current_seed)
    
    # Argument parser for PPO training and evaluation
    parser = argparse.ArgumentParser(description='Train PPO Script')
    parser.add_argument('--mode', choices=['train', 'eval'], 
                       default='train', help='Mode to run')
    parser.add_argument('--timesteps', type=int, default=100000, 
                       help='Training timesteps')
    parser.add_argument('--lr', type=float, default=3e-4, 
                       help='Learning rate')
    parser.add_argument('--n-envs', type=int, default=4, 
                       help='Number of parallel environments')
    parser.add_argument('--model', type=str, default=None, 
                       help='Model path for evaluation')
    parser.add_argument('--episodes', type=int, default=10, 
                       help='Episodes for evaluation/baseline')
    parser.add_argument('--render', choices=['human', 'text', 'none'], 
                       default='none', help='Rendering mode for evaluation')
    parser.add_argument('--warmstart', type=str, default=None,
                       help='Path to model to warmstart from')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'mps', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--render-training', action='store_true',
                       help='Render game periodically during training')
    parser.add_argument('--render-freq', type=int, default=5000,
                       help='Render every N steps during training')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Number of steps per environment before update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Minibatch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                       help='Number of epochs per update')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                       help='Entropy coefficient')
    parser.add_argument('--clip-range', type=float, default=0.2,
                       help='PPO clipping range')
    
    args = parser.parse_args()
    setup_training()
    
    if args.mode == 'train':
        # Custom PPO parameters if provided
        custom_params = {}
        if args.n_steps != 2048:
            custom_params['n_steps'] = args.n_steps
        if args.batch_size != 64:
            custom_params['batch_size'] = args.batch_size
        if args.n_epochs != 10:
            custom_params['n_epochs'] = args.n_epochs
        if args.ent_coef != 0.01:
            custom_params['ent_coef'] = args.ent_coef
        if args.clip_range != 0.2:
            custom_params['clip_range'] = args.clip_range
        model = train_ppo(
            total_timesteps=args.timesteps, 
            learning_rate=args.lr,
            n_envs=args.n_envs,
            custom_params=custom_params if custom_params else None,
            device=args.device
        )
    elif args.mode == 'eval':
        # Find model if not specified
        model_path = args.model
        if model_path is None:
            
            pattern = os.path.join("models", f"ppo_{args.config}_*", "*.zip")
            model_files = glob.glob(pattern)
            if model_files:
                model_path = max(model_files, key=os.path.getctime)
            else:
                print(f"No available model found")
                exit(1)
        render_mode = None if args.render == 'none' else args.render
        evaluate_model(
            model_path=model_path,
            config=args.config,
            n_episodes=args.episodes,
            render_mode=render_mode,
            seed=current_seed
        )