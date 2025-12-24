import numpy as np
import base64
import cv2
import asyncio
from .agents.q_learning import TabularQLearningAgent
from .agents.sarsa import SARSAAgent
from .agents.dynamic_programming import DynamicProgrammingAgent
from .agents.reinforce import REINFORCEAgent
from .envs.jungle_dash import JungleDashEnv, register_jungle_dash

# Register custom environments
register_jungle_dash()

# Action name mappings for different games
ACTION_NAMES = {
    "JungleDash": {0: "Up", 1: "Down", 2: "Left", 3: "Right"},
}

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
            render_mode = "rgb_array"
            
            try:
                # Create JungleDash environment
                if game_id == "JungleDash":
                    self.env = JungleDashEnv(render_mode=render_mode)
                    print(f"Env Created Successfully: {game_id}")
                    await self.send_log(websocket, f"Environment {game_id} initialized", "info")
                else:
                    error_msg = f"Unknown game: {game_id}. Only JungleDash is supported."
                    print(error_msg)
                    await websocket.send_json({"type": "error", "message": error_msg})
                    return
            except Exception as e:
                err_msg = f"Environment {game_id} creation failed: {str(e)}"
                print(err_msg)
                await websocket.send_json({"type": "error", "message": err_msg})
                return
                
            print("Resetting Env...")
            obs, _ = self.env.reset()
            print("Env Reset Configured.")

            self.init_agent(algo_id, self.env.action_space.n)
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
                    
                    # Determine win/loss status
                    if terminated and reward > 0:
                        status = "WON"
                        log_type = "success"
                        message = f"🏆 Episode {self.episode_count} VICTORY! Collected {info['rewards_collected']}/{self.env.num_rewards} coins! Total reward: {self.current_episode_reward:.2f}"
                    elif terminated:
                        status = "LOST"
                        log_type = "failure"
                        # Check if hit trap by inspecting the environment
                        cell_type = self.env.grid[self.env.agent_pos[0], self.env.agent_pos[1]]
                        if cell_type == self.env.TRAP:
                            message = f"💥 Episode {self.episode_count} LOST! Fell into trap. Collected {info['rewards_collected']}/{self.env.num_rewards} coins. Total reward: {self.current_episode_reward:.2f}"
                        else:
                            message = f"❌ Episode {self.episode_count} LOST. Collected {info['rewards_collected']}/{self.env.num_rewards} coins. Total reward: {self.current_episode_reward:.2f}"
                    elif truncated:
                        status = "TIMEOUT"
                        log_type = "failure"
                        message = f"⏱️ Episode {self.episode_count} TIMEOUT! Ran out of time. Collected {info['rewards_collected']}/{self.env.num_rewards} coins. Total reward: {self.current_episode_reward:.2f}"
                    else:
                        status = "ENDED"
                        log_type = "info"
                        message = f"Episode {self.episode_count} ended. Reward: {self.current_episode_reward:.2f}"
                    
                    # Log the episode result
                    await self.send_log(websocket, message, log_type)
                    print(f"Episode {self.episode_count} {status}. Reward: {self.current_episode_reward}")
                    
                    # Pause for 2 seconds to show win/loss screen
                    for _ in range(20):  # 20 frames at 0.1s each = 2 seconds
                        try:
                            # Keep rendering the win/loss screen
                            frame = self.env.render()
                            if frame is not None:
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
                            await asyncio.sleep(0.1)  # 100ms per frame
                        except Exception as e:
                            print(f"Error showing win/loss screen: {e}")
                            break
                    
                    # Save episode reward and reset
                    self.episode_rewards.append(self.current_episode_reward)
                    self.current_episode_reward = 0
                    obs, _ = self.env.reset()
                    
                    # Log restart
                    await self.send_log(websocket, f"🔄 Starting Episode {self.episode_count + 1}...", "info")
                
                await asyncio.sleep(0.01)  # Slightly longer sleep for readability
            except Exception as e:
                print(f"Error in training loop: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
                break

    def init_agent(self, algo_id, n_actions):
        """Initialize agent based on selected algorithm (tabular only)."""
        action_space = self.env.action_space
        state_space_size = self.env.observation_space.n
        
        if algo_id == "Q-Learning":
            self.agent = TabularQLearningAgent(action_space, state_space_size=state_space_size)
        elif algo_id == "SARSA":
            self.agent = SARSAAgent(action_space, state_space_size=state_space_size)
        elif algo_id == "DP":
            self.agent = DynamicProgrammingAgent(self.env)
            self.agent.value_iteration()
        elif algo_id == "REINFORCE":
            self.agent = REINFORCEAgent(action_space, state_space_size=state_space_size)
        else:
            # Default to Q-Learning
            self.agent = TabularQLearningAgent(action_space, state_space_size=state_space_size)

    def stop(self):
        self.running = False
        if self.env:
            self.env.close()
