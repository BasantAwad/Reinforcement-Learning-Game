"""
Jungle Dash - Custom Pygame Gymnasium Environment

A 2D grid-based game where an agent navigates through a jungle,
collecting rewards while avoiding obstacles to reach the treasure.

Game Elements:
- Agent (Pink Monster): The player controlled by RL algorithms
- Obstacles (Rocks): Block movement
- Rewards (Coins): Give positive reinforcement (+10)
- Goal (Treasure): Ends the episode (+100)
- Traps (Pits): Give penalty (-50) and end episode

Actions: 0=Up, 1=Down, 2=Left, 3=Right
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import math


class JungleDashEnv(gym.Env):
    """
    Jungle Dash Environment - A grid-world game where the agent must navigate
    through a jungle, collecting rewards while avoiding obstacles and traps.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    # Grid cell types
    EMPTY, AGENT, OBSTACLE, REWARD, GOAL, TRAP = 0, 1, 2, 3, 4, 5
    
    # Actions
    ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2, 3
    
    # Colors for jungle platformer theme
    COLORS = {
        'sky_top': (135, 206, 235),
        'sky_bottom': (200, 235, 255),
        'sun': (255, 236, 139),
        'sun_glow': (255, 250, 200),
        'mountain_back': (60, 120, 80),
        'mountain_mid': (80, 160, 100),
        'mountain_front': (100, 180, 120),
        'grass_top': (34, 180, 34),
        'platform': (101, 67, 33),
        'water': (64, 164, 223),
        'water_light': (100, 200, 255),
        'agent': (255, 105, 180),
        'obstacle': (139, 90, 43),
        'obstacle_dark': (100, 60, 30),
        'reward': (255, 215, 0),
        'reward_shine': (255, 250, 150),
        'trap': (60, 40, 20),
    }
    
    def __init__(self, render_mode=None, grid_size=8, num_obstacles=6, num_rewards=4, num_traps=2):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_rewards = num_rewards
        self.num_traps = num_traps
        self.render_mode = render_mode
        self.max_steps = grid_size * grid_size * 2
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        # Pygame setup
        self.cell_size = 64
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None
        self.pygame_initialized = False
        self.sprites = {}
        self.sprites_loaded = False
        
        # Asset directory
        self.asset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "1 Pink_Monster"
        )
        
        # Initialize state
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.rewards_collected = 0
        self.steps = 0
        self.reward_positions = []
        self.trap_positions = []
        self.obstacle_positions = []
    
    def _load_sprites(self):
        """Load sprite images for game elements."""
        if self.sprites_loaded:
            return
        
        try:
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
            
            sprite_size = (self.cell_size - 8, self.cell_size - 8)
            
            # Load Pink Monster sprite
            agent_path = os.path.join(self.asset_dir, "Pink_Monster.png")
            if os.path.exists(agent_path):
                self.sprites['agent'] = pygame.transform.scale(
                    pygame.image.load(agent_path), sprite_size
                )
            
            # Load Rock sprites
            for i in [1, 2]:
                rock_path = os.path.join(self.asset_dir, f"Rock{i}.png")
                if os.path.exists(rock_path):
                    self.sprites[f'rock{i}'] = pygame.transform.scale(
                        pygame.image.load(rock_path), sprite_size
                    )
            
            self.sprites_loaded = True
        except Exception as e:
            print(f"Warning: Could not load sprites: {e}")
            self.sprites_loaded = True
    
    def _get_obs(self):
        """Convert agent position to single integer observation."""
        return self.agent_pos[0] * self.grid_size + self.agent_pos[1]
    
    def _get_info(self):
        """Return additional info about the current state."""
        return {
            "rewards_collected": self.rewards_collected,
            "steps": self.steps,
            "agent_pos": self.agent_pos,
            "goal_pos": self.goal_pos,
        }
    
    def _is_valid_placement(self, pos):
        """Check if a position is valid for placing an object."""
        return (pos != [0, 0] and 
                pos != self.goal_pos and 
                self.grid[pos[0], pos[1]] == self.EMPTY)
    
    def _place_random_objects(self, obj_type, count, position_list):
        """Place objects randomly on the grid."""
        placed = 0
        while placed < count:
            pos = [self.np_random.integers(0, self.grid_size), 
                   self.np_random.integers(0, self.grid_size)]
            if self._is_valid_placement(pos):
                self.grid[pos[0], pos[1]] = obj_type
                position_list.append(tuple(pos))
                placed += 1
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place agent and goal
        self.agent_pos = [0, 0]
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        
        # Place obstacles, rewards, and traps
        self.obstacle_positions = []
        self._place_random_objects(self.OBSTACLE, self.num_obstacles, self.obstacle_positions)
        
        self.reward_positions = []
        self._place_random_objects(self.REWARD, self.num_rewards, self.reward_positions)
        
        self.trap_positions = []
        self._place_random_objects(self.TRAP, self.num_traps, self.trap_positions)
        
        self.rewards_collected = 0
        self.steps = 0
        
        # Build transition probability matrix for Dynamic Programming
        self._build_transition_matrix()
        
        return self._get_obs(), self._get_info()
    
    def _build_transition_matrix(self):
        """
        Build the transition probability matrix P for Dynamic Programming.
        P[s][a] = [(probability, next_state, reward, done)]
        
        Optimized reward structure:
        - Obstacle collision: -1 (small penalty, don't waste moves)
        - Coin collection: +20 (encourage exploration and collection)
        - Goal reached: +100 + (coins * 20) (big reward, bonus for coins)
        - Trap fallen: -20 (moderate penalty)
        - Regular movement: 0 (no penalty for exploring)
        """
        n_states = self.grid_size * self.grid_size
        self.P = {}
        
        for state in range(n_states):
            row, col = state // self.grid_size, state % self.grid_size
            self.P[state] = {}
            
            for action in range(4):
                new_row, new_col = row, col
                
                # Calculate new position
                if action == self.ACTION_UP:
                    new_row = max(0, row - 1)
                elif action == self.ACTION_DOWN:
                    new_row = min(self.grid_size - 1, row + 1)
                elif action == self.ACTION_LEFT:
                    new_col = max(0, col - 1)
                elif action == self.ACTION_RIGHT:
                    new_col = min(self.grid_size - 1, col + 1)
                
                # Determine reward and next state
                cell_type = self.grid[new_row, new_col]
                
                if cell_type == self.OBSTACLE:
                    next_state, reward, done = state, -1.0, False
                elif cell_type == self.GOAL:
                    next_state, reward, done = new_row * self.grid_size + new_col, 100.0, True
                elif cell_type == self.TRAP:
                    next_state, reward, done = new_row * self.grid_size + new_col, -20.0, True
                elif cell_type == self.REWARD:
                    next_state, reward, done = new_row * self.grid_size + new_col, 20.0, False
                else:
                    next_state, reward, done = new_row * self.grid_size + new_col, 0.0, False
                
                self.P[state][action] = [(1.0, next_state, reward, done)]
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Calculate new position
        new_pos = self.agent_pos.copy()
        if action == self.ACTION_UP:
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == self.ACTION_DOWN:
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == self.ACTION_LEFT:
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == self.ACTION_RIGHT:
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        
        # Check collision and calculate reward
        if self.grid[new_pos[0], new_pos[1]] == self.OBSTACLE:
            return self._get_obs(), -1, False, False, self._get_info()
        
        # Move agent
        self.agent_pos = new_pos
        cell_type = self.grid[new_pos[0], new_pos[1]]
        
        # Process cell type with optimized rewards
        if cell_type == self.REWARD:
            self.rewards_collected += 1
            self.grid[new_pos[0], new_pos[1]] = self.EMPTY
            reward, terminated = 20, False
        elif cell_type == self.GOAL:
            reward = 100 + (self.rewards_collected * 20)
            terminated = True
        elif cell_type == self.TRAP:
            reward, terminated = -20, True
        else:
            reward, terminated = 0, False
        
        # Check max steps
        truncated = self.steps >= self.max_steps
        if truncated and not terminated:
            reward = -5
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render the current state with jungle platformer visuals."""
        if self.render_mode is None:
            return None
        
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
        
        self._load_sprites()
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Jungle Dash - Pink Monster Adventure")
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        
        # Draw sky gradient
        for y in range(self.window_size):
            ratio = y / self.window_size
            color = tuple(int(self.COLORS['sky_top'][i] * (1 - ratio) + 
                             self.COLORS['sky_bottom'][i] * ratio) for i in range(3))
            pygame.draw.line(canvas, color, (0, y), (self.window_size, y))
        
        # Draw sun
        self._draw_sun(canvas)
        
        # Draw mountains
        self._draw_mountains(canvas)
        
        # Draw water
        self._draw_water(canvas)
        
        # Draw platforms
        self._draw_platforms(canvas)
        
        # Draw grid cells
        self._draw_grid_cells(canvas)
        
        # Draw agent
        self._draw_agent(canvas)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    
    def _draw_sun(self, canvas):
        """Draw the sun with glow and rays."""
        sun_x, sun_y = self.window_size - 70, 60
        pygame.draw.circle(canvas, self.COLORS['sun_glow'], (sun_x, sun_y), 45)
        pygame.draw.circle(canvas, self.COLORS['sun'], (sun_x, sun_y), 35)
        
        for angle in range(0, 360, 30):
            rad = math.radians(angle)
            x1 = sun_x + int(40 * math.cos(rad))
            y1 = sun_y + int(40 * math.sin(rad))
            x2 = sun_x + int(55 * math.cos(rad))
            y2 = sun_y + int(55 * math.sin(rad))
            pygame.draw.line(canvas, self.COLORS['sun'], (x1, y1), (x2, y2), 3)
    
    def _draw_mountains(self, canvas):
        """Draw layered mountain background."""
        mountain_base_y = self.window_size // 2 + 40
        
        # Back mountains
        points = [(0, mountain_base_y)]
        for x in range(0, self.window_size + 50, 80):
            points.append((x, mountain_base_y - 80 - (x % 60)))
        points.append((self.window_size, mountain_base_y))
        pygame.draw.polygon(canvas, self.COLORS['mountain_back'], points)
        
        # Mid mountains
        points = [(0, mountain_base_y + 20)]
        for x in range(0, self.window_size + 40, 60):
            points.append((x + 30, mountain_base_y - 40 - (x % 50)))
        points.append((self.window_size, mountain_base_y + 20))
        pygame.draw.polygon(canvas, self.COLORS['mountain_mid'], points)
        
        # Front mountains
        points = [(0, mountain_base_y + 40)]
        for x in range(0, self.window_size + 30, 50):
            points.append((x + 20, mountain_base_y + 10 - (x % 40)))
        points.append((self.window_size, mountain_base_y + 40))
        pygame.draw.polygon(canvas, self.COLORS['mountain_front'], points)
    
    def _draw_water(self, canvas):
        """Draw water at the bottom with ripples."""
        water_y = self.window_size - self.cell_size
        pygame.draw.rect(canvas, self.COLORS['water'], (0, water_y, self.window_size, self.cell_size))
        
        for i in range(4):
            ripple_y = water_y + 10 + i * 15
            pygame.draw.line(canvas, self.COLORS['water_light'], 
                           (20 + i * 30, ripple_y), (80 + i * 30, ripple_y), 2)
    
    def _draw_platforms(self, canvas):
        """Draw grass platforms."""
        ground_y = self.window_size - self.cell_size - 20
        
        # Left platform
        pygame.draw.rect(canvas, self.COLORS['platform'], (0, ground_y, 180, 40))
        pygame.draw.rect(canvas, self.COLORS['grass_top'], (0, ground_y - 8, 180, 12))
        
        # Right platform
        pygame.draw.rect(canvas, self.COLORS['platform'], (self.window_size - 180, ground_y, 180, 40))
        pygame.draw.rect(canvas, self.COLORS['grass_top'], (self.window_size - 180, ground_y - 8, 180, 12))
    
    def _draw_grid_cells(self, canvas):
        """Draw all grid cells (obstacles, rewards, goal, traps)."""
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = self.grid[row, col]
                cell_x, cell_y = col * self.cell_size, row * self.cell_size
                
                # Draw cell border for empty cells (except water row)
                if row < self.grid_size - 1:
                    pygame.draw.rect(canvas, (255, 255, 255, 30), 
                                   (cell_x + 2, cell_y + 2, self.cell_size - 4, self.cell_size - 4), 1)
                
                if cell == self.OBSTACLE:
                    self._draw_obstacle(canvas, cell_x, cell_y, row, col)
                elif cell == self.REWARD:
                    self._draw_reward(canvas, cell_x, cell_y)
                elif cell == self.GOAL:
                    self._draw_goal(canvas, cell_x, cell_y)
                elif cell == self.TRAP:
                    self._draw_trap(canvas, cell_x, cell_y)
    
    def _draw_obstacle(self, canvas, x, y, row, col):
        """Draw obstacle (rock)."""
        x, y = x + 4, y + 4
        block_size = self.cell_size - 8
        
        # Use rock sprite if available
        rock_key = f'rock{(row + col) % 2 + 1}'
        if rock_key in self.sprites:
            canvas.blit(self.sprites[rock_key], (x, y))
        elif 'rock1' in self.sprites:
            canvas.blit(self.sprites['rock1'], (x, y))
        else:
            # Fallback: draw brick block
            pygame.draw.rect(canvas, self.COLORS['obstacle'], (x, y, block_size, block_size), border_radius=4)
            # Brick pattern
            for i, third in enumerate([block_size//3, 2*block_size//3]):
                pygame.draw.line(canvas, self.COLORS['obstacle_dark'], (x, y + third), (x + block_size, y + third), 2)
            pygame.draw.line(canvas, self.COLORS['obstacle_dark'], (x + block_size//2, y), (x + block_size//2, y + block_size//3), 2)
            pygame.draw.line(canvas, self.COLORS['obstacle_dark'], (x + block_size//4, y + block_size//3), (x + block_size//4, y + 2*block_size//3), 2)
    
    def _draw_reward(self, canvas, x, y):
        """Draw reward coin."""
        center_x, center_y = x + self.cell_size // 2, y + self.cell_size // 2
        pygame.draw.circle(canvas, self.COLORS['reward_shine'], (center_x, center_y), 22)
        pygame.draw.circle(canvas, self.COLORS['reward'], (center_x, center_y), 18)
        pygame.draw.circle(canvas, (255, 240, 100), (center_x - 4, center_y - 4), 6)
        # Sparkle
        pygame.draw.line(canvas, (255, 255, 255), (center_x, center_y - 24), (center_x, center_y - 18), 2)
        pygame.draw.line(canvas, (255, 255, 255), (center_x - 16, center_y - 16), (center_x - 12, center_y - 12), 2)
    
    def _draw_goal(self, canvas, x, y):
        """Draw treasure chest goal."""
        chest_x, chest_y = x + 8, y + 12
        # Chest body
        pygame.draw.rect(canvas, (139, 69, 19), (chest_x, chest_y + 15, 48, 35), border_radius=4)
        pygame.draw.rect(canvas, (100, 50, 15), (chest_x + 2, chest_y + 20, 44, 25), border_radius=2)
        # Chest lid
        pygame.draw.rect(canvas, (160, 82, 45), (chest_x - 2, chest_y, 52, 18), border_radius=6)
        # Gold coins inside
        for dx, dy, r in [(24, 6, 10), (14, 10, 7), (34, 10, 7)]:
            pygame.draw.circle(canvas, self.COLORS['reward'], (chest_x + dx, chest_y + dy), r)
        # Sparkle
        pygame.draw.line(canvas, (255, 255, 255), (chest_x + 24, chest_y - 8), (chest_x + 24, chest_y - 2), 2)
        pygame.draw.line(canvas, (255, 255, 255), (chest_x + 20, chest_y - 5), (chest_x + 28, chest_y - 5), 2)
    
    def _draw_trap(self, canvas, x, y):
        """Draw spiky pit trap."""
        trap_x, trap_y = x + 6, y + 10
        trap_w, trap_h = self.cell_size - 12, self.cell_size - 16
        # Dark pit
        pygame.draw.ellipse(canvas, self.COLORS['trap'], (trap_x, trap_y + 10, trap_w, trap_h - 10))
        pygame.draw.ellipse(canvas, (30, 20, 10), (trap_x + 8, trap_y + 18, trap_w - 16, trap_h - 24))
        # Spikes
        for i in range(5):
            sx = trap_x + 6 + i * 10
            pygame.draw.polygon(canvas, (100, 100, 100), 
                              [(sx, trap_y + 20), (sx + 5, trap_y + 8), (sx + 10, trap_y + 20)])
    
    def _draw_agent(self, canvas):
        """Draw agent (Pink Monster)."""
        agent_x = self.agent_pos[1] * self.cell_size + 4
        agent_y = self.agent_pos[0] * self.cell_size + 4
        
        if 'agent' in self.sprites:
            canvas.blit(self.sprites['agent'], (agent_x, agent_y))
        else:
            # Fallback: draw cute pink monster
            center_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
            center_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
            # Body
            pygame.draw.circle(canvas, self.COLORS['agent'], (center_x, center_y), 24)
            pygame.draw.circle(canvas, (255, 150, 200), (center_x, center_y), 20)
            # Eyes
            for eye_x in [-8, 8]:
                pygame.draw.circle(canvas, (255, 255, 255), (center_x + eye_x, center_y - 6), 8)
                pupil_x = -6 if eye_x == -8 else 10
                pygame.draw.circle(canvas, (0, 0, 0), (center_x + pupil_x, center_y - 6), 4)
                shine_x = -5 if eye_x == -8 else 11
                pygame.draw.circle(canvas, (255, 255, 255), (center_x + shine_x, center_y - 8), 2)
            # Mouth
            pygame.draw.arc(canvas, (0, 0, 0), (center_x - 10, center_y, 20, 12), 3.14, 0, 2)
    
    def close(self):
        """Clean up pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.pygame_initialized = False
            self.sprites_loaded = False


def register_jungle_dash():
    """Register JungleDash with gymnasium."""
    from gymnasium.envs.registration import register
    
    try:
        register(
            id='JungleDash-v0',
            entry_point='backend.envs.jungle_dash:JungleDashEnv',
            max_episode_steps=200,
        )
    except Exception:
        pass  # Already registered


# Auto-register when module is imported
register_jungle_dash()
