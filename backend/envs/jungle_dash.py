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

Actions:
- 0: Up
- 1: Down  
- 2: Left
- 3: Right

This satisfies the project requirement:
"Design a 2D game using Python libraries (e.g., Pygame or custom implementations)"
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os


class JungleDashEnv(gym.Env):
    """
    Jungle Dash Environment
    
    A grid-world game where the agent must navigate through a jungle,
    collecting rewards while avoiding obstacles and traps,
    ultimately reaching the treasure (goal).
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}
    
    # Grid cell types
    EMPTY = 0
    AGENT = 1
    OBSTACLE = 2  # Rocks
    REWARD = 3    # Coins
    GOAL = 4      # Treasure
    TRAP = 5      # Pit
    
    # Action mappings
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    
    def __init__(self, render_mode=None, grid_size=8, num_obstacles=6, num_rewards=4, num_traps=2):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.num_rewards = num_rewards
        self.num_traps = num_traps
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        self.observation_space = spaces.Discrete(grid_size * grid_size)
        
        # Pygame setup
        self.cell_size = 64
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None
        self.font = None
        self.pygame_initialized = False
        
        # Sprite images (loaded on first render)
        self.sprites = {}
        self.sprites_loaded = False
        
        # Asset paths - relative to project root
        self.asset_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "1 Pink_Monster"
        )
        
        # Colors for jungle platformer theme
        self.colors = {
            # Sky
            'sky_top': (135, 206, 235),       # Light sky blue
            'sky_bottom': (200, 235, 255),    # Pale blue/white at horizon
            'sun': (255, 236, 139),           # Warm yellow
            'sun_glow': (255, 250, 200),      # Lighter glow
            # Mountains
            'mountain_back': (60, 120, 80),   # Dark green
            'mountain_mid': (80, 160, 100),   # Medium green
            'mountain_front': (100, 180, 120),# Light green
            # Platforms & ground
            'grass_top': (34, 180, 34),       # Bright grass green
            'grass_dark': (25, 140, 25),      # Dark grass
            'platform': (101, 67, 33),        # Brown platform
            'platform_dark': (70, 45, 20),    # Dark brown
            # Water
            'water': (64, 164, 223),          # Blue water
            'water_light': (100, 200, 255),   # Light water ripple
            # Game elements
            'agent': (255, 105, 180),         # Pink
            'obstacle': (139, 90, 43),        # Brown brick
            'obstacle_dark': (100, 60, 30),   # Dark brick
            'reward': (255, 215, 0),          # Gold
            'reward_shine': (255, 250, 150),  # Gold shine
            'goal': (0, 255, 255),            # Cyan
            'trap': (60, 40, 20),             # Dark pit
            'text': (255, 255, 255),          # White
        }
        
        # Initialize state
        self.grid = None
        self.agent_pos = None
        self.goal_pos = None
        self.rewards_collected = 0
        self.steps = 0
        self.max_steps = grid_size * grid_size * 2
        
        # Keep track of positions
        self.reward_positions = []
        self.trap_positions = []
        self.obstacle_positions = []
    
    def _load_sprites(self):
        """Load sprite images for game elements."""
        if self.sprites_loaded:
            return
        
        try:
            # Initialize pygame if not done
            if not self.pygame_initialized:
                pygame.init()
                self.pygame_initialized = True
            
            sprite_size = (self.cell_size - 8, self.cell_size - 8)
            
            # Load Pink Monster sprite for agent
            agent_path = os.path.join(self.asset_dir, "Pink_Monster.png")
            if os.path.exists(agent_path):
                agent_img = pygame.image.load(agent_path)
                self.sprites['agent'] = pygame.transform.scale(agent_img, sprite_size)
                print(f"Loaded agent sprite: {agent_path}")
            
            # Load Rock sprites for obstacles
            rock1_path = os.path.join(self.asset_dir, "Rock1.png")
            rock2_path = os.path.join(self.asset_dir, "Rock2.png")
            
            if os.path.exists(rock1_path):
                rock1_img = pygame.image.load(rock1_path)
                self.sprites['rock1'] = pygame.transform.scale(rock1_img, sprite_size)
                print(f"Loaded rock1 sprite: {rock1_path}")
            
            if os.path.exists(rock2_path):
                rock2_img = pygame.image.load(rock2_path)
                self.sprites['rock2'] = pygame.transform.scale(rock2_img, sprite_size)
                print(f"Loaded rock2 sprite: {rock2_path}")
            
            self.sprites_loaded = True
            
        except Exception as e:
            print(f"Warning: Could not load sprites: {e}")
            self.sprites_loaded = True  # Don't try again
    
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
    
    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Initialize empty grid
        self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        
        # Place agent in top-left corner
        self.agent_pos = [0, 0]
        
        # Place goal in bottom-right corner
        self.goal_pos = [self.grid_size - 1, self.grid_size - 1]
        self.grid[self.goal_pos[0], self.goal_pos[1]] = self.GOAL
        
        # Place obstacles randomly (avoid agent start and goal)
        self.obstacle_positions = []
        placed = 0
        while placed < self.num_obstacles:
            pos = [self.np_random.integers(0, self.grid_size), 
                   self.np_random.integers(0, self.grid_size)]
            if self._is_valid_placement(pos):
                self.grid[pos[0], pos[1]] = self.OBSTACLE
                self.obstacle_positions.append(tuple(pos))
                placed += 1
        
        # Place rewards randomly
        self.reward_positions = []
        placed = 0
        while placed < self.num_rewards:
            pos = [self.np_random.integers(0, self.grid_size), 
                   self.np_random.integers(0, self.grid_size)]
            if self._is_valid_placement(pos):
                self.grid[pos[0], pos[1]] = self.REWARD
                self.reward_positions.append(tuple(pos))
                placed += 1
        
        # Place traps randomly
        self.trap_positions = []
        placed = 0
        while placed < self.num_traps:
            pos = [self.np_random.integers(0, self.grid_size), 
                   self.np_random.integers(0, self.grid_size)]
            if self._is_valid_placement(pos):
                self.grid[pos[0], pos[1]] = self.TRAP
                self.trap_positions.append(tuple(pos))
                placed += 1
        
        self.rewards_collected = 0
        self.steps = 0
        
        # Build transition probability matrix P for Dynamic Programming
        self._build_transition_matrix()
        
        return self._get_obs(), self._get_info()
    
    def _build_transition_matrix(self):
        """
        Build the transition probability matrix P for Dynamic Programming.
        P[s][a] = [(probability, next_state, reward, done), ...]
        
        Since Jungle Dash is deterministic, each action has probability 1.0
        for a single outcome.
        """
        n_states = self.grid_size * self.grid_size
        n_actions = 4
        
        # Initialize P dictionary
        self.P = {}
        
        for state in range(n_states):
            self.P[state] = {}
            row = state // self.grid_size
            col = state % self.grid_size
            
            for action in range(n_actions):
                # Calculate new position
                new_row, new_col = row, col
                
                if action == self.ACTION_UP:
                    new_row = max(0, row - 1)
                elif action == self.ACTION_DOWN:
                    new_row = min(self.grid_size - 1, row + 1)
                elif action == self.ACTION_LEFT:
                    new_col = max(0, col - 1)
                elif action == self.ACTION_RIGHT:
                    new_col = min(self.grid_size - 1, col + 1)
                
                # Check cell type at new position
                cell_type = self.grid[new_row, new_col]
                
                if cell_type == self.OBSTACLE:
                    # Blocked by obstacle, stay in place
                    next_state = state
                    reward = -5.0  # Penalty for hitting obstacle
                    done = False
                elif cell_type == self.GOAL:
                    next_state = new_row * self.grid_size + new_col
                    reward = 100.0
                    done = True
                elif cell_type == self.TRAP:
                    next_state = new_row * self.grid_size + new_col
                    reward = -50.0
                    done = True
                elif cell_type == self.REWARD:
                    # For DP, treat rewards as regular cells (can't track collection)
                    next_state = new_row * self.grid_size + new_col
                    reward = 10.0
                    done = False
                else:
                    # Empty cell
                    next_state = new_row * self.grid_size + new_col
                    reward = 0.0  # No penalty for regular movement
                    done = False
                
                # P[s][a] = [(prob, next_state, reward, done)]
                self.P[state][action] = [(1.0, next_state, reward, done)]
    
    def _is_valid_placement(self, pos):
        """Check if a position is valid for placing an object."""
        if pos == [0, 0]:
            return False
        if pos == self.goal_pos:
            return False
        if self.grid[pos[0], pos[1]] != self.EMPTY:
            return False
        return True
    
    def step(self, action):
        """Execute one step in the environment."""
        self.steps += 1
        
        # Calculate new position based on action
        new_pos = self.agent_pos.copy()
        if action == self.ACTION_UP:
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == self.ACTION_DOWN:
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        elif action == self.ACTION_LEFT:
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == self.ACTION_RIGHT:
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        
        # Check if movement is blocked by obstacle
        if self.grid[new_pos[0], new_pos[1]] == self.OBSTACLE:
            reward = -5  # Penalty for hitting obstacle
            terminated = False
            truncated = False
        else:
            self.agent_pos = new_pos
            cell_type = self.grid[new_pos[0], new_pos[1]]
            
            if cell_type == self.REWARD:
                reward = 10
                self.rewards_collected += 1
                self.grid[new_pos[0], new_pos[1]] = self.EMPTY
                terminated = False
                truncated = False
            elif cell_type == self.GOAL:
                reward = 100 + (self.rewards_collected * 10)
                terminated = True
                truncated = False
            elif cell_type == self.TRAP:
                reward = -50
                terminated = True
                truncated = False
            else:
                reward = 0  # No penalty for regular movement
                terminated = False
                truncated = False
        
        if self.steps >= self.max_steps and not terminated:
            truncated = True
            reward = -10
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render the current state of the environment with jungle platformer visuals."""
        if self.render_mode is None:
            return None
        
        # Initialize pygame if needed
        if not self.pygame_initialized:
            pygame.init()
            self.pygame_initialized = True
        
        # Load sprites on first render
        self._load_sprites()
        
        if self.window is None and self.render_mode == "human":
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Jungle Dash - Pink Monster Adventure")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create surface
        canvas = pygame.Surface((self.window_size, self.window_size))
        
        # === DRAW SKY GRADIENT BACKGROUND ===
        for y in range(self.window_size):
            # Interpolate between sky_top and sky_bottom
            ratio = y / self.window_size
            r = int(self.colors['sky_top'][0] * (1 - ratio) + self.colors['sky_bottom'][0] * ratio)
            g = int(self.colors['sky_top'][1] * (1 - ratio) + self.colors['sky_bottom'][1] * ratio)
            b = int(self.colors['sky_top'][2] * (1 - ratio) + self.colors['sky_bottom'][2] * ratio)
            pygame.draw.line(canvas, (r, g, b), (0, y), (self.window_size, y))
        
        # === DRAW SUN ===
        sun_x = self.window_size - 70
        sun_y = 60
        # Sun glow
        pygame.draw.circle(canvas, self.colors['sun_glow'], (sun_x, sun_y), 45)
        # Sun core
        pygame.draw.circle(canvas, self.colors['sun'], (sun_x, sun_y), 35)
        # Sun rays
        for angle in range(0, 360, 30):
            import math
            rad = math.radians(angle)
            x1 = sun_x + int(40 * math.cos(rad))
            y1 = sun_y + int(40 * math.sin(rad))
            x2 = sun_x + int(55 * math.cos(rad))
            y2 = sun_y + int(55 * math.sin(rad))
            pygame.draw.line(canvas, self.colors['sun'], (x1, y1), (x2, y2), 3)
        
        # === DRAW MOUNTAINS (back to front) ===
        mountain_base_y = self.window_size // 2 + 40
        
        # Back mountains (darkest)
        points_back = [(0, mountain_base_y)]
        for x in range(0, self.window_size + 50, 80):
            peak_height = mountain_base_y - 80 - (x % 60)
            points_back.append((x, peak_height))
        points_back.append((self.window_size, mountain_base_y))
        pygame.draw.polygon(canvas, self.colors['mountain_back'], points_back)
        
        # Mid mountains
        points_mid = [(0, mountain_base_y + 20)]
        for x in range(0, self.window_size + 40, 60):
            peak_height = mountain_base_y - 40 - (x % 50)
            points_mid.append((x + 30, peak_height))
        points_mid.append((self.window_size, mountain_base_y + 20))
        pygame.draw.polygon(canvas, self.colors['mountain_mid'], points_mid)
        
        # Front mountains (lightest)
        points_front = [(0, mountain_base_y + 40)]
        for x in range(0, self.window_size + 30, 50):
            peak_height = mountain_base_y + 10 - (x % 40)
            points_front.append((x + 20, peak_height))
        points_front.append((self.window_size, mountain_base_y + 40))
        pygame.draw.polygon(canvas, self.colors['mountain_front'], points_front)
        
        # === DRAW WATER AT BOTTOM ===
        water_height = self.cell_size  # Bottom row becomes water
        water_y = self.window_size - water_height
        pygame.draw.rect(canvas, self.colors['water'], (0, water_y, self.window_size, water_height))
        # Water ripples
        for i in range(4):
            ripple_y = water_y + 10 + i * 15
            pygame.draw.line(canvas, self.colors['water_light'], (20 + i * 30, ripple_y), 
                           (80 + i * 30, ripple_y), 2)
        
        # === DRAW GRASS PLATFORMS (ground layer) ===
        # Draw a ground platform covering the bottom portion
        ground_y = self.window_size - water_height - 20
        # Left ground platform
        pygame.draw.rect(canvas, self.colors['platform'], (0, ground_y, 180, 40))
        pygame.draw.rect(canvas, self.colors['grass_top'], (0, ground_y - 8, 180, 12))
        # Right ground platform
        pygame.draw.rect(canvas, self.colors['platform'], (self.window_size - 180, ground_y, 180, 40))
        pygame.draw.rect(canvas, self.colors['grass_top'], (self.window_size - 180, ground_y - 8, 180, 12))
        
        # === DRAW GRID CELLS AS FLOATING PLATFORMS ===
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = self.grid[row, col]
                cell_x = col * self.cell_size
                cell_y = row * self.cell_size
                
                # Draw subtle platform for empty cells (except water row)
                if row < self.grid_size - 1:  # Not water row
                    # Subtle cell indicator
                    pygame.draw.rect(canvas, (255, 255, 255, 30), 
                                   (cell_x + 2, cell_y + 2, self.cell_size - 4, self.cell_size - 4), 1)
                
                if cell == self.OBSTACLE:
                    # Draw brick/stone obstacle block
                    x = cell_x + 4
                    y = cell_y + 4
                    block_w = self.cell_size - 8
                    block_h = self.cell_size - 8
                    
                    # Use rock sprite if available
                    rock_idx = (row + col) % 2
                    rock_key = f'rock{rock_idx + 1}'
                    if rock_key in self.sprites:
                        canvas.blit(self.sprites[rock_key], (x, y))
                    elif 'rock1' in self.sprites:
                        canvas.blit(self.sprites['rock1'], (x, y))
                    else:
                        # Draw brick block
                        pygame.draw.rect(canvas, self.colors['obstacle'], (x, y, block_w, block_h), border_radius=4)
                        # Brick pattern
                        pygame.draw.line(canvas, self.colors['obstacle_dark'], (x, y + block_h//3), (x + block_w, y + block_h//3), 2)
                        pygame.draw.line(canvas, self.colors['obstacle_dark'], (x, y + 2*block_h//3), (x + block_w, y + 2*block_h//3), 2)
                        pygame.draw.line(canvas, self.colors['obstacle_dark'], (x + block_w//2, y), (x + block_w//2, y + block_h//3), 2)
                        pygame.draw.line(canvas, self.colors['obstacle_dark'], (x + block_w//4, y + block_h//3), (x + block_w//4, y + 2*block_h//3), 2)
                        pygame.draw.line(canvas, self.colors['obstacle_dark'], (x + 3*block_w//4, y + block_h//3), (x + 3*block_w//4, y + 2*block_h//3), 2)
                        
                elif cell == self.REWARD:
                    # Draw shiny coin
                    center_x = cell_x + self.cell_size // 2
                    center_y = cell_y + self.cell_size // 2
                    # Outer glow
                    pygame.draw.circle(canvas, self.colors['reward_shine'], (center_x, center_y), 22)
                    # Main coin
                    pygame.draw.circle(canvas, self.colors['reward'], (center_x, center_y), 18)
                    # Inner shine
                    pygame.draw.circle(canvas, (255, 240, 100), (center_x - 4, center_y - 4), 6)
                    # Star/sparkle effect
                    pygame.draw.line(canvas, (255, 255, 255), (center_x, center_y - 24), (center_x, center_y - 18), 2)
                    pygame.draw.line(canvas, (255, 255, 255), (center_x - 16, center_y - 16), (center_x - 12, center_y - 12), 2)
                        
                elif cell == self.GOAL:
                    # Draw treasure chest
                    chest_x = cell_x + 8
                    chest_y = cell_y + 12
                    # Chest body (brown)
                    pygame.draw.rect(canvas, (139, 69, 19), (chest_x, chest_y + 15, 48, 35), border_radius=4)
                    pygame.draw.rect(canvas, (100, 50, 15), (chest_x + 2, chest_y + 20, 44, 25), border_radius=2)
                    # Chest lid (open, showing gold)
                    pygame.draw.rect(canvas, (160, 82, 45), (chest_x - 2, chest_y, 52, 18), border_radius=6)
                    # Gold coins inside
                    pygame.draw.circle(canvas, self.colors['reward'], (chest_x + 24, chest_y + 6), 10)
                    pygame.draw.circle(canvas, self.colors['reward'], (chest_x + 14, chest_y + 10), 7)
                    pygame.draw.circle(canvas, self.colors['reward'], (chest_x + 34, chest_y + 10), 7)
                    # Sparkle
                    pygame.draw.line(canvas, (255, 255, 255), (chest_x + 24, chest_y - 8), (chest_x + 24, chest_y - 2), 2)
                    pygame.draw.line(canvas, (255, 255, 255), (chest_x + 20, chest_y - 5), (chest_x + 28, chest_y - 5), 2)
                    
                elif cell == self.TRAP:
                    # Draw spiky pit trap
                    trap_x = cell_x + 6
                    trap_y = cell_y + 10
                    trap_w = self.cell_size - 12
                    trap_h = self.cell_size - 16
                    # Dark pit
                    pygame.draw.ellipse(canvas, self.colors['trap'], (trap_x, trap_y + 10, trap_w, trap_h - 10))
                    pygame.draw.ellipse(canvas, (30, 20, 10), (trap_x + 8, trap_y + 18, trap_w - 16, trap_h - 24))
                    # Spikes
                    spike_color = (100, 100, 100)
                    for i in range(5):
                        sx = trap_x + 6 + i * 10
                        pygame.draw.polygon(canvas, spike_color, [(sx, trap_y + 20), (sx + 5, trap_y + 8), (sx + 10, trap_y + 20)])
        
        # === DRAW AGENT (Pink Monster) ===
        agent_x = self.agent_pos[1] * self.cell_size + 4
        agent_y = self.agent_pos[0] * self.cell_size + 4
        
        if 'agent' in self.sprites:
            canvas.blit(self.sprites['agent'], (agent_x, agent_y))
        else:
            # Fallback: Draw cute pink monster
            center_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
            center_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
            # Body
            pygame.draw.circle(canvas, self.colors['agent'], (center_x, center_y), 24)
            pygame.draw.circle(canvas, (255, 150, 200), (center_x, center_y), 20)
            # Eyes
            pygame.draw.circle(canvas, (255, 255, 255), (center_x - 8, center_y - 6), 8)
            pygame.draw.circle(canvas, (255, 255, 255), (center_x + 8, center_y - 6), 8)
            pygame.draw.circle(canvas, (0, 0, 0), (center_x - 6, center_y - 6), 4)
            pygame.draw.circle(canvas, (0, 0, 0), (center_x + 10, center_y - 6), 4)
            # Eye shine
            pygame.draw.circle(canvas, (255, 255, 255), (center_x - 5, center_y - 8), 2)
            pygame.draw.circle(canvas, (255, 255, 255), (center_x + 11, center_y - 8), 2)
            # Mouth (smile)
            pygame.draw.arc(canvas, (0, 0, 0), (center_x - 10, center_y, 20, 12), 3.14, 0, 2)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        # Return RGB array for rgb_array mode
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )
    
    def close(self):
        """Clean up pygame resources."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.pygame_initialized = False
            self.sprites_loaded = False


# Register the environment with gymnasium
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

