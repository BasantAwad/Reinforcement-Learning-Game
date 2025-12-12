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
        
        # Colors (fallback when sprites not available)
        self.colors = {
            'background': (34, 139, 34),      # Forest Green
            'grid_line': (25, 100, 25),       # Darker green
            'agent': (255, 105, 180),         # Pink
            'obstacle': (101, 67, 33),        # Brown
            'reward': (255, 215, 0),          # Gold
            'goal': (0, 255, 255),            # Cyan
            'trap': (139, 69, 19),            # Saddle brown
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
                    reward = -1.0
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
                    reward = -0.1
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
            reward = -1
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
                reward = -0.1
                terminated = False
                truncated = False
        
        if self.steps >= self.max_steps and not terminated:
            truncated = True
            reward = -10
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render the current state of the environment."""
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
        canvas.fill(self.colors['background'])
        
        # Draw grid lines
        for i in range(self.grid_size + 1):
            pygame.draw.line(
                canvas,
                self.colors['grid_line'],
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size),
                2
            )
            pygame.draw.line(
                canvas,
                self.colors['grid_line'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size),
                2
            )
        
        # Draw cells
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                cell = self.grid[row, col]
                x = col * self.cell_size + 4
                y = row * self.cell_size + 4
                
                if cell == self.OBSTACLE:
                    # Use rock sprite if available
                    rock_idx = (row + col) % 2  # Alternate between rock1 and rock2
                    rock_key = f'rock{rock_idx + 1}'
                    if rock_key in self.sprites:
                        canvas.blit(self.sprites[rock_key], (x, y))
                    elif 'rock1' in self.sprites:
                        canvas.blit(self.sprites['rock1'], (x, y))
                    else:
                        # Fallback to drawing shapes
                        pygame.draw.rect(
                            canvas,
                            self.colors['obstacle'],
                            (x, y, self.cell_size - 8, self.cell_size - 8),
                            border_radius=8
                        )
                        
                elif cell == self.REWARD:
                    # Draw coin/reward
                    center_x = col * self.cell_size + self.cell_size // 2
                    center_y = row * self.cell_size + self.cell_size // 2
                    pygame.draw.circle(canvas, self.colors['reward'], (center_x, center_y), 18)
                    pygame.draw.circle(canvas, (255, 180, 0), (center_x, center_y), 12)
                    # Dollar sign
                    if self.font is None:
                        try:
                            self.font = pygame.font.Font(None, 24)
                        except:
                            pass
                    if self.font:
                        text = self.font.render("$", True, (139, 90, 0))
                        text_rect = text.get_rect(center=(center_x, center_y))
                        canvas.blit(text, text_rect)
                        
                elif cell == self.GOAL:
                    # Draw treasure chest
                    chest_x = col * self.cell_size + 8
                    chest_y = row * self.cell_size + 16
                    # Chest body
                    pygame.draw.rect(canvas, (139, 69, 19), (chest_x, chest_y + 10, 48, 30), border_radius=4)
                    # Chest lid
                    pygame.draw.rect(canvas, (160, 82, 45), (chest_x - 2, chest_y, 52, 18), border_radius=6)
                    # Gold inside
                    pygame.draw.circle(canvas, self.colors['reward'], (chest_x + 24, chest_y + 8), 8)
                    pygame.draw.circle(canvas, self.colors['reward'], (chest_x + 16, chest_y + 12), 6)
                    pygame.draw.circle(canvas, self.colors['reward'], (chest_x + 32, chest_y + 12), 6)
                    
                elif cell == self.TRAP:
                    # Draw pit/hole
                    center_x = col * self.cell_size + self.cell_size // 2
                    center_y = row * self.cell_size + self.cell_size // 2
                    pygame.draw.ellipse(canvas, (60, 40, 20), 
                        (col * self.cell_size + 8, row * self.cell_size + 12, 48, 40))
                    pygame.draw.ellipse(canvas, (30, 20, 10), 
                        (col * self.cell_size + 16, row * self.cell_size + 20, 32, 24))
        
        # Draw agent (Pink Monster)
        agent_x = self.agent_pos[1] * self.cell_size + 4
        agent_y = self.agent_pos[0] * self.cell_size + 4
        
        if 'agent' in self.sprites:
            canvas.blit(self.sprites['agent'], (agent_x, agent_y))
        else:
            # Fallback: Draw pink circle with face
            center_x = self.agent_pos[1] * self.cell_size + self.cell_size // 2
            center_y = self.agent_pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(canvas, self.colors['agent'], (center_x, center_y), 24)
            # Eyes
            pygame.draw.circle(canvas, (255, 255, 255), (center_x - 8, center_y - 6), 6)
            pygame.draw.circle(canvas, (255, 255, 255), (center_x + 8, center_y - 6), 6)
            pygame.draw.circle(canvas, (0, 0, 0), (center_x - 8, center_y - 6), 3)
            pygame.draw.circle(canvas, (0, 0, 0), (center_x + 8, center_y - 6), 3)
            # Mouth
            pygame.draw.arc(canvas, (0, 0, 0), (center_x - 10, center_y - 4, 20, 16), 3.14, 0, 2)
        
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

