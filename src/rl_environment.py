"""
Reinforcement Learning Environment for Mini Metro Game
Implements a Gym-compatible environment for training DQN agents
"""

import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any, Optional

from mediator import Mediator
from config import (
    rl_num_stations,
    rl_num_paths, 
    rl_num_metros,
    rl_num_station_types,
    rl_step_interval,
    rl_max_episode_steps,
    rl_max_station_passengers,
    rl_max_total_passengers,
    framerate
)


class MiniMetroRLEnv(gym.Env):
    """
    Mini Metro RL Environment for Deep Q-Learning with Dynamic Configuration
    
    Action Space:
        Discrete(N): Dynamic based on configuration
        - 0 to C(num_stations,2)-1: Build path between station pairs
        - C(num_stations,2) to C(num_stations,2)+(num_paths×num_stations)-1: Extend existing paths
        - Next num_paths actions: Remove paths
        - Last action: No operation
        
        Example for 5 stations, 2 paths: Discrete(23)
        - 0-9: Build paths (station pairs: (0,1), (0,2), (0,3), (0,4), (1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
        - 10-19: Extend paths (path 0 + station 0-4, path 1 + station 0-4)
        - 20-21: Remove path 0, Remove path 1
        - 22: No operation
    
    State Space:
        Box(M,): Dynamic based on configuration
        - Station features (num_stations × 8 features)
        - Path features (num_paths × 5 features) 
        - Global features (8 features)
        
        Example for 5 stations, 2 paths: Box(60,)
        - Station features: 5 × 8 = 40
        - Path features: 2 × 5 = 10
        - Global features: 8
        - Total: 60 features
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        
        # Load configuration
        self.config = config or {}
        self.num_stations = self.config.get('num_stations', rl_num_stations)
        self.num_paths = self.config.get('num_paths', rl_num_paths)
        self.num_metros = self.config.get('num_metros', rl_num_metros)
        self.num_station_types = self.config.get('num_station_types', rl_num_station_types)
        self.step_interval = self.config.get('step_interval', rl_step_interval)
        self.max_episode_steps = self.config.get('max_episode_steps', rl_max_episode_steps)
        self.max_station_passengers = self.config.get('max_station_passengers', rl_max_station_passengers)
        self.max_total_passengers = self.config.get('max_total_passengers', rl_max_total_passengers)
        
        # Rendering configuration
        self.render_mode = self.config.get('render_mode', None)  # None, 'human', 'rgb_array'
        self.render_fps = self.config.get('render_fps', 30)  # 30 FPS for smoother rendering on macOS
        self.screen = None
        self.clock = None
        
        # Dynamic action mapping for building paths
        self.build_actions = []
        for i in range(self.num_stations):
            for j in range(i + 1, self.num_stations):
                self.build_actions.append((i, j))
        
        # Dynamic action mapping for extending paths
        self.extend_actions = []
        for path_id in range(self.num_paths):
            for station_id in range(self.num_stations):
                self.extend_actions.append((path_id, station_id))
        
        # Dynamic action space calculation
        num_build_actions = len(self.build_actions)    # All possible station pairs
        num_extend_actions = len(self.extend_actions)  # All path-station combinations
        num_remove_actions = self.num_paths            # Remove each possible path
        num_noop_actions = 1                           # No operation
        total_actions = num_build_actions + num_extend_actions + num_remove_actions + num_noop_actions
        
        # Action space: Dynamic based on configuration
        self.action_space = gym.spaces.Discrete(total_actions)
        
        # Dynamic state space calculation
        station_features = self.num_stations * 8     # 8 features per station
        path_features = self.num_paths * 5           # 5 features per path
        global_features = 8                          # Fixed global features
        total_state_size = station_features + path_features + global_features
        
        # State space: Dynamic based on configuration
        self.observation_space = gym.spaces.Box(
            low=0.0, high=100.0, shape=(total_state_size,), dtype=np.float32
        )
        
        # Store action space boundaries for easier action parsing
        self.build_action_start = 0
        self.build_action_end = num_build_actions
        self.extend_action_start = num_build_actions
        self.extend_action_end = num_build_actions + num_extend_actions
        self.remove_action_start = num_build_actions + num_extend_actions
        self.remove_action_end = num_build_actions + num_extend_actions + num_remove_actions
        self.noop_action = total_actions - 1
        
        # Environment state
        self.mediator = None
        self.step_count = 0
        self.game_step_counter = 0
        self.last_score = 0
        
        # Print configuration info
        self._print_config_info()
    
    def _print_config_info(self):
        """Print dynamic configuration information"""
        print(f"   Mini Metro RL Environment Configuration:")
        print(f"   Stations: {self.num_stations}, Paths: {self.num_paths}, Metros: {self.num_metros}")
        print(f"   Station Types: {self.num_station_types} (shapes: {self.num_station_types} different)")
        print(f"   Action space: Discrete({self.action_space.n})")
        print(f"     - Build actions: {len(self.build_actions)} (station pairs)")
        print(f"     - Extend actions: {len(self.extend_actions)} (path-station combinations)")
        print(f"     - Remove actions: {self.num_paths} (one per path)")
        print(f"     - No-op actions: 1")
        print(f"   State space: Box({self.observation_space.shape[0]},)")
        print(f"     - Station features: {self.num_stations} × 8 = {self.num_stations * 8}")
        print(f"     - Path features: {self.num_paths} × 5 = {self.num_paths * 5}")
        print(f"     - Global features: 8")
        print(f"   Episode length: {self.max_episode_steps} steps")
        print()
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Create new mediator with RL config
        self.mediator = Mediator()
        self.mediator.set_rl_config(
            num_stations=self.num_stations,
            num_paths=self.num_paths,
            num_metros=self.num_metros,
            num_station_types=self.num_station_types
        )
        
        # Reset environment state
        self.step_count = 0
        self.game_step_counter = 0
        self.last_score = 0
        
        # Get initial state
        initial_state = self.mediator.get_rl_state_vector()
        
        return initial_state, {}
    
    def step(self, action: int):
        """Execute one environment step"""
        # Execute action
        action_success = self._execute_action(action)
        
        # Run game simulation for RL step interval
        for _ in range(self.step_interval):
            self.mediator.increment_time(1000 // framerate)  # ~33ms per frame
            self.game_step_counter += 1
        
        # Get new state
        state = self.mediator.get_rl_state_vector()
        
        # Calculate reward
        reward = self._calculate_reward(action, action_success)
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.step_count >= self.max_episode_steps
        
        # Update step count
        self.step_count += 1
        
        # Auto-render if render mode is set
        if self.render_mode is not None:
            self.render(self.render_mode)
        
        # Info for debugging/logging
        info = {
            'score': self.mediator.score,
            'passengers': len(self.mediator.passengers),
            'paths': len(self.mediator.paths),
            'action_success': action_success,
            'action_type': self._get_action_type(action),
            'step_count': self.step_count,
            'game_time_minutes': self.mediator.time_ms / (60 * 1000)
        }
        
        return state, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> bool:
        """Execute the given action and return success status"""
        if self.build_action_start <= action < self.build_action_end:
            # Build path actions
            build_idx = action - self.build_action_start
            if build_idx < len(self.build_actions):
                station1, station2 = self.build_actions[build_idx]
                return self.mediator.rl_build_path([station1, station2])
            return False
        elif self.extend_action_start <= action < self.extend_action_end:
            # Extend path actions
            extend_idx = action - self.extend_action_start
            if extend_idx < len(self.extend_actions):
                path_id, station_id = self.extend_actions[extend_idx]
                return self.mediator.rl_extend_path(path_id, station_id)
            return False
        elif self.remove_action_start <= action < self.remove_action_end:
            # Remove path actions
            path_id = action - self.remove_action_start
            return self.mediator.rl_remove_path(path_id)
        elif action == self.noop_action:
            # No-op action
            return True
        else:
            # Invalid action
            return False
    
    def _calculate_reward(self, action: int, action_success: bool) -> float:
        """
        Calculate reward for the current step 
        """
        # Score Reward
        raw_score_change = self.mediator.score - self.last_score
        score_reward = raw_score_change * 0.5
        self.last_score = self.mediator.score
        # Time penalty to encourage efficiency
        time_penalty = -0.01
        # Action-specific costs and bonuses
        action_cost = 0.0
        if self.build_action_start <= action < self.build_action_end and action_success:
            # Successful path build
            action_cost = -0.1
        elif self.extend_action_start <= action < self.extend_action_end and action_success:
            # Successful path extension (Cheaper than building new path)
            action_cost = -0.05
        elif self.remove_action_start <= action < self.remove_action_end and action_success:
            # Successful path removal
            action_cost = -0.05
        elif not action_success and action != self.noop_action:
            # Invalid action penalty
            action_cost = -0.5
        # Connectivity and completeness bonuses
        connectivity_bonus = self.mediator._get_connectivity_score() * 0.3
        completeness_score = self.mediator.get_network_completeness_score()
        completeness_bonus = completeness_score * 0.6
        # Passenger delivery bonus
        delivery_bonus = score_reward * 0.5 if score_reward > 0 else 0.0
        # Station overcrowding penalty
        overcrowd_penalty = 0.0
        for station in self.mediator.stations:
            if len(station.passengers) > 10:
                overcrowd_penalty -= (len(station.passengers) - 10) * 0.005
        # Passenger waiting penalty
        waiting_penalty = 0.0
        for passenger in self.mediator.passengers:
            wait_seconds = passenger.waiting_time / 1000.0
            if wait_seconds > 30:
                excess_time = wait_seconds - 30
                waiting_penalty -= excess_time * 0.001
        # Path utilization bonus
        path_utilization_ratio = len(self.mediator.paths) / self.num_paths if self.num_paths > 0 else 0.0
        path_utilization_bonus = path_utilization_ratio * 0.4
        # Total reward : Sum of all components
        total_reward = (score_reward + time_penalty + action_cost + 
                       connectivity_bonus + delivery_bonus + overcrowd_penalty +
                       completeness_bonus + waiting_penalty + path_utilization_bonus)
        return float(total_reward)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate due to failure conditions"""
        # Overcrowding at any station
        max_station_passengers = max(len(s.passengers) for s in self.mediator.stations)
        if max_station_passengers > self.max_station_passengers:
            return True
        
        # System overload
        if len(self.mediator.passengers) > self.max_total_passengers:
            return True
        
        return False
    
    def _get_action_type(self, action: int) -> str:
        """Get human-readable action type for logging"""
        if self.build_action_start <= action < self.build_action_end:
            build_idx = action - self.build_action_start
            if build_idx < len(self.build_actions):
                return f"build_path_{self.build_actions[build_idx]}"
            else:
                return "invalid_build"
        elif self.extend_action_start <= action < self.extend_action_end:
            extend_idx = action - self.extend_action_start
            if extend_idx < len(self.extend_actions):
                return f"extend_path_{self.extend_actions[extend_idx]}"
            else:
                return "invalid_extend"
        elif self.remove_action_start <= action < self.remove_action_end:
            path_id = action - self.remove_action_start
            return f"remove_path_{path_id}"
        elif action == self.noop_action:
            return "no_op"
        else:
            return "invalid_action"
    
    def render(self, mode: str = 'human'):
        """Render the environment for visualization"""
        if mode == 'human':
            # Initialize pygame if not already done
            if self.screen is None:
                import pygame
                pygame.init()
                pygame.font.init()  # Explicitly initialize font module
                from config import screen_width, screen_height
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                pygame.display.set_caption("Mini Metro RL Evaluation")
                self.clock = pygame.time.Clock()
                print(f"Pygame window initialized: {screen_width}x{screen_height}")
            
            # Import pygame for this method scope
            import pygame
            
            # Process pygame events to keep window responsive
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return
            
            # Clear screen
            self.screen.fill((255, 255, 255))  # White background
            
            # Render the game state
            if self.mediator:
                self.mediator.render(self.screen)
            
            # Add RL info overlay
            font = pygame.font.Font(None, 36)
            info_text = f"Step: {self.step_count}/{self.max_episode_steps} | Score: {self.mediator.score}"
            text_surface = font.render(info_text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10))
            
            # Update display
            pygame.display.flip()
            self.clock.tick(self.render_fps)
            
        elif mode == 'text':
            # Print current state info
            print(f"Step: {self.step_count}/{self.max_episode_steps} | "
                  f"Score: {self.mediator.score} | "
                  f"Passengers: {len(self.mediator.passengers)} | "
                  f"Paths: {len(self.mediator.paths)}/{self.num_paths} | "
                  f"Metros: {len(self.mediator.metros)}/{self.num_metros}")
        
        elif mode == 'rgb_array':
            # Return numpy array of screen for video recording
            if self.screen is None:
                import pygame
                pygame.init()
                from config import screen_width, screen_height
                self.screen = pygame.display.set_mode((screen_width, screen_height))
                self.clock = pygame.time.Clock()
            
            # Import pygame for this method scope
            import pygame
            
            self.screen.fill((255, 255, 255))
            if self.mediator:
                self.mediator.render(self.screen)
            
            import numpy as np
            return np.transpose(pygame.surfarray.array3d(self.screen), axes=(1, 0, 2))
    
    def close(self):
        """Clean up environment"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None
            self.clock = None
        if self.mediator:
            self.mediator = None


class MiniMetroRLConfig:
    """Configuration class for the RL environment with dynamic sizing"""
    
    DEFAULT = {
        'num_stations': 5,
        'num_paths': 2,
        'num_metros': 2,
        'num_station_types': 3,  # RECT, CIRCLE, TRIANGLE
        'step_interval': 60,
        'max_episode_steps': 90,
        'max_station_passengers': 20,
        'max_total_passengers': 100
    }
    # State space: 5×8 + 2×5 + 8 = 58 features
    # Action space: C(5,2) + (2×5) + 2 + 1 = 10 + 10 + 2 + 1 = 23 actions
    
    EASY = {
        'num_stations': 4,
        'num_paths': 2,
        'num_metros': 2,
        'num_station_types': 2,  # RECT, CIRCLE only
        'step_interval': 90,
        'max_episode_steps': 90,
        'max_station_passengers': 25,
        'max_total_passengers': 120
    }
    # State space: 4×8 + 2×5 + 8 = 50 features
    # Action space: C(4,2) + (2×4) + 2 + 1 = 6 + 8 + 2 + 1 = 17 actions
    
    HARD = {
        'num_stations': 6,
        'num_paths': 3,
        'num_metros': 3,
        'num_station_types': 4,  # All 4 types: RECT, CIRCLE, TRIANGLE, CROSS
        'step_interval': 45,
        'max_episode_steps': 300,
        'max_station_passengers': 15,
        'max_total_passengers': 80
    }
    # State space: 6×8 + 3×5 + 8 = 73 features
    # Action space: C(6,2) + (3×6) + 3 + 1 = 15 + 18 + 3 + 1 = 37 actions
    
    TINY = {
        'num_stations': 3,
        'num_paths': 1,
        'num_metros': 1,
        'num_station_types': 2,  # RECT, CIRCLE only
        'step_interval': 90,
        'max_episode_steps': 60,
        'max_station_passengers': 30,
        'max_total_passengers': 150
    }
    # State space: 3×8 + 1×5 + 8 = 37 features
    # Action space: C(3,2) + (1×3) + 1 + 1 = 3 + 3 + 1 + 1 = 8 actions
    
    LARGE = {
        'num_stations': 8,
        'num_paths': 4,
        'num_metros': 4,
        'num_station_types': 4,  # All 4 types for maximum complexity
        'step_interval': 30,
        'max_episode_steps': 600,
        'max_station_passengers': 12,
        'max_total_passengers': 60
    }
    # State space: 8×8 + 4×5 + 8 = 96 features
    # Action space: C(8,2) + (4×8) + 4 + 1 = 28 + 32 + 4 + 1 = 65 actions


def make_env(config: str = 'default', render_mode: str = None, render_fps: int = 30) -> MiniMetroRLEnv:
    """
    Factory function to create environment with predefined configs
    
    Args:
        config: Configuration preset ('tiny', 'easy', 'default', 'hard', 'large')
        render_mode: Rendering mode ('human', 'text', 'rgb_array', or None for no rendering)
        render_fps: FPS for human rendering (default 30 for smooth rendering on macOS)
    """
    configs = {
        'tiny': MiniMetroRLConfig.TINY,
        'easy': MiniMetroRLConfig.EASY,
        'default': MiniMetroRLConfig.DEFAULT,
        'hard': MiniMetroRLConfig.HARD,
        'large': MiniMetroRLConfig.LARGE
    }
    
    if config not in configs:
        raise ValueError(f"Unknown config: {config}. Available: {list(configs.keys())}")
    
    # Add rendering configuration
    env_config = configs[config].copy()
    env_config['render_mode'] = render_mode
    env_config['render_fps'] = render_fps
    
    return MiniMetroRLEnv(env_config)


if __name__ == "__main__":
    # Test the environment
    env = make_env('default')
    
    print("Testing MiniMetroRLEnv...")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test random episode
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {step}: Action={action}, Reward={reward:.3f}, "
              f"Score={info['score']}, Passengers={info['passengers']}")
        
        if terminated or truncated:
            break
    
    print(f"Episode finished. Total reward: {total_reward:.3f}")
    env.close()
