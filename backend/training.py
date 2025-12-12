import gymnasium as gym
import numpy as np
import base64
import cv2
import json
import asyncio
import ale_py # Register Atari Envs
from .agents.dqn import DQNAgent
from .agents.policy_gradient import PolicyGradientAgent
from .agents.tabular import TabularQLearningAgent, DynamicProgrammingAgent
from .envs.wrappers import ResizeAndGrayscale, FrameStack

class TrainingManager:
    def __init__(self):
        self.env = None
        self.agent = None
        self.running = False
        self.episode_rewards = []
        self.current_episode_reward = 0
        self.total_steps = 0

    async def start_training(self, game_id: str, algo_id: str, websocket):
        self.running = True
        
        # Initialize Environment
        # Environment selection logic handled in try-block below

        
        try:
            # Create Env
            print(f"Attempting to create game: {game_id}") # DEBUG
            env_id = None
            if game_id == "MiniWorld-Maze":
                try:
                    # Explicit checking for user feedback
                    import miniworld
                except ImportError as e:
                    await websocket.send_json({"type": "error", "message": f"MiniWorld Module Missing: {e}"})
                    raise e
                
                env_id = "MiniWorld-Maze-v0"

            if game_id == "MsPacman":
                env_id = "ALE/MsPacman-v5"
            elif game_id == "KungFuMaster":
                env_id = "ALE/KungFuMaster-v5"
            else: # Valid fallthrough if we moved MiniWorld logic up, but let's keep it safe
                pass 
                
            try:
                render_mode = "rgb_array"
                # If MiniWorld, use the ID we just validated
                if game_id == "MiniWorld-Maze":
                     self.env = gym.make("MiniWorld-Maze-v0", render_mode="rgb_array")
                elif game_id in ["MsPacman", "KungFuMaster"]:
                     self.env = gym.make(env_id, render_mode=render_mode, frameskip=4)
                else:
                     self.env = gym.make(env_id, render_mode=render_mode)
                
                print(f"Env Created Successfully: {env_id}")
            except Exception as e:
                err_msg = f"Environment {env_id} creation failed: {str(e)}"
                print(err_msg)
                await websocket.send_json({"type": "error", "message": err_msg})
                # Fallback
                print(f"Fallback to FrozenLake.")
                self.env = gym.make("FrozenLake-v1", render_mode="rgb_array", is_slippery=False)
            
            # Wrapper logic...

            
            if game_id == "MiniWorld-Maze":
                # MiniWorld: Apply grayscale resize (outputs 1x84x84)
                self.env = ResizeAndGrayscale(self.env)
            
            elif game_id in ["MsPacman", "KungFuMaster"]:
                # Atari: Apply grayscale resize and frame stacking (outputs 4x84x84)
                self.env = ResizeAndGrayscale(self.env)
                self.env = FrameStack(self.env, 4)
            print("Resetting Env...")
            obs, _ = self.env.reset()
            print("Env Reset Configured.")
            
            # Determine input_shape and is_visual based on the actual environment created
            if "FrozenLake" in str(self.env.spec.id) or "CartPole" in str(self.env.spec.id):
                input_shape = self.env.observation_space.shape
                is_visual = False
            elif game_id in ["MsPacman", "KungFuMaster"]:
                input_shape = (4, 84, 84) # After FrameStack
                is_visual = True
            elif game_id == "MiniWorld-Maze":
                input_shape = (1, 84, 84) # After ResizeAndGrayscale
                is_visual = True
            else: # Default for other cases, e.g., CartPole
                input_shape = self.env.observation_space.shape
                is_visual = False

            self.init_agent(algo_id, self.env.action_space.n, input_shape, is_visual)
            print("Agent Initialized.")
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
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) # Correct colors
                    
                    # Ensure uint8
                    if frame.dtype != np.uint8:
                        frame = (frame * 255).astype(np.uint8)

                    # Encode
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buffer).decode('utf-8')
                    
                    # Send
                    await websocket.send_json({
                        "type": "frame",
                        "data": frame_b64
                    })
                    # print("Frame sent") # Comment out to reduce spam if working
            except Exception as e:
                print(f"Render/Send Error: {e}")
                break

            # Agent Action
            try:
                action = self.agent.act(obs)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                self.current_episode_reward += reward
                self.total_steps += 1

                # Train
                self.agent.step(obs, action, reward, next_obs, done)
                
                obs = next_obs

                # Send Stats
                if self.total_steps % 10 == 0: # Reduce WS traffic
                    await websocket.send_json({
                        "type": "stats",
                        "data": {
                            "reward": self.current_episode_reward,
                            "total_steps": self.total_steps,
                            "done": done
                        }
                    })

                if done:
                    print(f"Episode Done. Reward: {self.current_episode_reward}")
                    self.episode_rewards.append(self.current_episode_reward)
                    self.current_episode_reward = 0
                    obs, _ = self.env.reset()
                
                await asyncio.sleep(0.001) # Small sleep to yield
            except Exception as e:
                print(f"Error in training loop: {e}")
                import traceback
                traceback.print_exc()
                self.running = False
                break



    def init_agent(self, algo_id, n_actions, input_shape, is_visual):
        action_space = self.env.action_space
        if algo_id == "DQN":
            if not is_visual:
                 pass
            self.agent = DQNAgent(input_shape, action_space)
        elif algo_id == "PG":
            self.agent = PolicyGradientAgent(input_shape, action_space)
        elif algo_id == "Q-Learning":
            if hasattr(self.env.observation_space, 'n'):
                 self.agent = TabularQLearningAgent(action_space, state_space_size=self.env.observation_space.n)
            else:
                 # Fallback for visual: Use DQN or Error?
                 print("WARNING: Q-Learning selected for Visual Box space. Switching to DQN.")
                 self.agent = DQNAgent(input_shape, action_space) # Auto-switch
        elif algo_id == "DP":
             if hasattr(self.env.observation_space, 'n'):
                  self.agent = DynamicProgrammingAgent(self.env)
                  self.agent.value_iteration()
             else:
                  raise ValueError("Dynamic Programming requires discrete state space (frozenlake). MiniWorld is visual.")
        elif algo_id == "SARSA":
             if hasattr(self.env.observation_space, 'n'):
                self.agent = TabularQLearningAgent(action_space, state_space_size=self.env.observation_space.n)
             else:
                 print("WARNING: SARSA selected for Visual Box space. Switching to DQN.")
                 self.agent = DQNAgent(input_shape, action_space)
        
        # Re-verify imports
        # Line 10: from .agents.tabular import TabularQLearningAgent, DynamicProgrammingAgent
        # I need to verify if SARSA is in there.

    def stop(self):
        self.running = False
        if self.env:
            self.env.close()
