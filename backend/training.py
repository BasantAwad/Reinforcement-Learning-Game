import gymnasium as gym
import numpy as np
import base64
import cv2
import json
import asyncio
import ale_py # Register Atari Envs
from .agents.dqn import DQNAgent
from .agents.policy_gradient import PolicyGradientAgent
from .agents.tabular import TabularQLearningAgent, SARSAAgent, DynamicProgrammingAgent
from .envs.wrappers import ResizeAndGrayscale, FrameStack
from .envs.jungle_dash import JungleDashEnv, register_jungle_dash

# Register custom environments
register_jungle_dash()

# Action name mappings for different games
ACTION_NAMES = {
    "Taxi": {0: "South", 1: "North", 2: "East", 3: "West", 4: "Pickup", 5: "Dropoff"},
    "FrozenLake": {0: "Left", 1: "Down", 2: "Right", 3: "Up"},
    "Blackjack": {0: "Stand", 1: "Hit"},
    "JungleDash": {0: "Up", 1: "Down", 2: "Left", 3: "Right"},
    "MsPacman": {0: "NOOP", 1: "Up", 2: "Right", 3: "Left", 4: "Down"},
    "KungFuMaster": {0: "NOOP", 1: "Up", 2: "Right", 3: "Left", 4: "Down", 5: "Fire"},
    "MiniWorld-Maze": {0: "Left", 1: "Right", 2: "Forward"}
}

class BlackjackObsWrapper(gym.ObservationWrapper):
    """Convert Blackjack tuple observation to integer for tabular agents."""
    def __init__(self, env):
        super().__init__(env)
        # State space: player_sum (4-21=18 vals) x dealer_card (1-10=10 vals) x usable_ace (2 vals)
        self.observation_space = gym.spaces.Discrete(18 * 10 * 2)
    
    def observation(self, obs):
        player_sum, dealer_card, usable_ace = obs
        # Map player_sum 4-21 to 0-17, dealer_card 1-10 to 0-9, usable_ace 0-1
        return (player_sum - 4) * 20 + (dealer_card - 1) * 2 + int(usable_ace)

class TrainingManager:
    def __init__(self):
        self.env = None
        self.agent = None
        self.running = False
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.total_steps = 0
        self.total_penalties = 0
        self.current_game_id = None
        self.episode_count = 0

    def get_action_name(self, action: int) -> str:
        """Get human-readable action name for current game."""
        if self.current_game_id in ACTION_NAMES:
            return ACTION_NAMES[self.current_game_id].get(action, f"Action {action}")
        return f"Action {action}"

    async def send_log(self, websocket, message: str, log_type: str = "action"):
        """Send log message to frontend."""
        await websocket.send_json({
            "type": "log",
            "data": {
                "message": message,
                "logType": log_type,
                "step": self.total_steps
            }
        })

    async def start_training(self, game_id: str, algo_id: str, websocket):
        self.running = True
        self.current_game_id = game_id
        self.total_penalties = 0
        self.episode_count = 0
        
        try:
            # Create Env
            print(f"Attempting to create game: {game_id}")
            env_id = None
            render_mode = "rgb_array"
            
            # Set environment ID based on game selection
            if game_id == "MiniWorld-Maze":
                try:
                    import miniworld
                except ImportError as e:
                    await websocket.send_json({"type": "error", "message": f"MiniWorld Module Missing: {e}"})
                    raise e
                env_id = "MiniWorld-Maze-v0"
            elif game_id == "MsPacman":
                env_id = "ALE/MsPacman-v5"
            elif game_id == "KungFuMaster":
                env_id = "ALE/KungFuMaster-v5"
            elif game_id == "Taxi":
                env_id = "Taxi-v3"
            elif game_id == "Blackjack":
                env_id = "Blackjack-v1"
            elif game_id == "FrozenLake":
                env_id = "FrozenLake-v1"
            elif game_id == "JungleDash":
                env_id = "JungleDash-v0"
                
            try:
                # Create environment based on game type
                if game_id == "MiniWorld-Maze":
                    self.env = gym.make("MiniWorld-Maze-v0", render_mode="rgb_array")
                elif game_id in ["MsPacman", "KungFuMaster"]:
                    self.env = gym.make(env_id, render_mode=render_mode, frameskip=4)
                elif game_id == "FrozenLake":
                    self.env = gym.make(env_id, render_mode=render_mode, is_slippery=False)
                elif game_id == "Blackjack":
                    self.env = gym.make(env_id, render_mode=render_mode)
                    self.env = BlackjackObsWrapper(self.env)
                elif game_id == "Taxi":
                    self.env = gym.make(env_id, render_mode=render_mode)
                elif game_id == "JungleDash":
                    self.env = JungleDashEnv(render_mode=render_mode)
                else:
                    self.env = gym.make(env_id, render_mode=render_mode)
                
                print(f"Env Created Successfully: {env_id}")
                await self.send_log(websocket, f"Environment {game_id} initialized", "info")
            except Exception as e:
                err_msg = f"Environment {env_id} creation failed: {str(e)}"
                print(err_msg)
                await websocket.send_json({"type": "error", "message": err_msg})
                print(f"Fallback to FrozenLake.")
                self.env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
                self.current_game_id = "FrozenLake"
            
            # Apply wrappers for visual environments
            if game_id == "MiniWorld-Maze":
                self.env = ResizeAndGrayscale(self.env)
            elif game_id in ["MsPacman", "KungFuMaster"]:
                self.env = ResizeAndGrayscale(self.env)
                self.env = FrameStack(self.env, 4)
                
            print("Resetting Env...")
            obs, _ = self.env.reset()
            print("Env Reset Configured.")
            
            # Determine input_shape and is_visual
            is_visual = False
            if game_id in ["MsPacman", "KungFuMaster"]:
                input_shape = (4, 84, 84)
                is_visual = True
            elif game_id == "MiniWorld-Maze":
                input_shape = (1, 84, 84)
                is_visual = True
            elif game_id in ["Taxi", "FrozenLake", "Blackjack", "JungleDash"]:
                input_shape = self.env.observation_space.shape if hasattr(self.env.observation_space, 'shape') else ()
                is_visual = False
            else:
                input_shape = self.env.observation_space.shape if hasattr(self.env.observation_space, 'shape') else ()
                is_visual = False

            self.init_agent(algo_id, self.env.action_space.n, input_shape, is_visual)
            print("Agent Initialized.")
            await self.send_log(websocket, f"Agent ({algo_id}) initialized and ready", "info")
            
        except Exception as e:
            print(f"Setup Error Crashing: {e}")
            import traceback
            traceback.print_exc()
            try:
                await websocket.send_json({"type": "error", "message": f"Startup Error: {str(e)}"})
            except:
                pass
            return

        print("Starting Training Loop...")
        await self.send_log(websocket, "Training started!", "info")
        
        while self.running:
            # Frame broadcast
            try:
                frame = self.env.render()
                if frame is None:
                    print("Render returned None")
                else:
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    await websocket.send_json({
                        "type": "frame",
                        "data": frame_b64
                    })
            except Exception as e:
                print(f"Render/Send Error: {e}")
                break

            # Agent Action
            try:
                action = self.agent.act(obs)
                action_name = self.get_action_name(action)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Log agent action
                await self.send_log(websocket, f"Agent: {action_name}", "action")
                
                # Track penalties (negative rewards)
                if reward < 0:
                    self.total_penalties += abs(reward)
                    await self.send_log(websocket, f"Penalty received: {reward}", "penalty")
                elif reward > 0:
                    await self.send_log(websocket, f"Reward received: +{reward}", "reward")

                self.current_episode_reward += reward
                self.total_steps += 1

                # Train
                self.agent.step(obs, action, reward, next_obs, done)
                obs = next_obs

                # Send Stats (every 10 steps to reduce traffic)
                if self.total_steps % 10 == 0:
                    await websocket.send_json({
                        "type": "stats",
                        "data": {
                            "reward": self.current_episode_reward,
                            "steps": self.total_steps,
                            "penalties": self.total_penalties,
                            "episode": self.episode_count,
                            "done": done
                        }
                    })

                if done:
                    self.episode_count += 1
                    if terminated and reward > 0:
                        await self.send_log(websocket, f"Episode {self.episode_count} WON! Reward: {self.current_episode_reward:.2f}", "success")
                    elif terminated:
                        await self.send_log(websocket, f"Episode {self.episode_count} LOST. Reward: {self.current_episode_reward:.2f}", "failure")
                    else:
                        await self.send_log(websocket, f"Episode {self.episode_count} ended (truncated). Reward: {self.current_episode_reward:.2f}", "info")
                    
                    print(f"Episode Done. Reward: {self.current_episode_reward}")
                    self.episode_rewards.append(self.current_episode_reward)
                    self.current_episode_reward = 0
                    obs, _ = self.env.reset()
                    await self.send_log(websocket, "Environment reset - new episode starting", "info")
                
                await asyncio.sleep(0.01)  # Slightly longer sleep for readability
            except Exception as e:
                print(f"Error in training loop: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
                break

    def init_agent(self, algo_id, n_actions, input_shape, is_visual):
        action_space = self.env.action_space
        if algo_id == "DQN":
            self.agent = DQNAgent(input_shape, action_space)
        elif algo_id == "PG":
            self.agent = PolicyGradientAgent(input_shape, action_space)
        elif algo_id == "Q-Learning":
            if hasattr(self.env.observation_space, 'n'):
                self.agent = TabularQLearningAgent(action_space, state_space_size=self.env.observation_space.n)
            else:
                print("WARNING: Q-Learning selected for Visual Box space. Switching to DQN.")
                self.agent = DQNAgent(input_shape, action_space)
        elif algo_id == "DP":
            if hasattr(self.env.observation_space, 'n'):
                self.agent = DynamicProgrammingAgent(self.env)
                self.agent.value_iteration()
            else:
                raise ValueError("Dynamic Programming requires discrete state space. Visual envs not supported.")
        elif algo_id == "SARSA":
            if hasattr(self.env.observation_space, 'n'):
                # Use proper SARSAAgent with on-policy updates
                self.agent = SARSAAgent(action_space, state_space_size=self.env.observation_space.n)
            else:
                print("WARNING: SARSA selected for Visual Box space. Switching to DQN.")
                self.agent = DQNAgent(input_shape, action_space)

    def stop(self):
        self.running = False
        if self.env:
            self.env.close()
