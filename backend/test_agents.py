"""
Quick test script to verify all algorithms work correctly
"""
import sys
import os

# Add parent directory to path for imports
backend_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(backend_dir)
sys.path.insert(0, parent_dir)

from backend.agents.dynamic_programming import DynamicProgrammingAgent
from backend.agents.q_learning import TabularQLearningAgent
from backend.agents.sarsa import SARSAAgent
from backend.agents.reinforce import REINFORCEAgent
from backend.envs.jungle_dash import JungleDashEnv

def test_dp():
    """Test Dynamic Programming agent"""
    print("\n=== Testing Dynamic Programming ===")
    env = JungleDashEnv()
    agent = DynamicProgrammingAgent(env)
    iterations = agent.value_iteration()
    print(f"✓ DP converged in {iterations} iterations")
    
    # Test acting
    obs, _ = env.reset()
    action = agent.act(obs)
    print(f"✓ DP can select actions (state {obs} → action {action})")

def test_q_learning():
    """Test Q-Learning agent"""
    print("\n=== Testing Q-Learning ===")
    env = JungleDashEnv()
    agent = TabularQLearningAgent(env.action_space, env.observation_space.n)
    
    # Run a few steps
    obs, _ = env.reset()
    for _ in range(10):
        action = agent.act(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            break
    print(f"✓ Q-Learning runs successfully")

def test_sarsa():
    """Test SARSA agent"""
    print("\n=== Testing SARSA ===")
    env = JungleDashEnv()
    agent = SARSAAgent(env.action_space, env.observation_space.n)
    
    # Run a few steps
    obs, _ = env.reset()
    for _ in range(10):
        action = agent.act(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            break
    print(f"✓ SARSA runs successfully")

def test_reinforce():
    """Test REINFORCE agent"""
    print("\n=== Testing REINFORCE ===")
    env = JungleDashEnv()
    agent = REINFORCEAgent(env.action_space, env.observation_space.n)
    
    # Run one episode
    obs, _ = env.reset()
    for _ in range(50):
        action = agent.act(obs)
        next_obs, reward, done, truncated, _ = env.step(action)
        agent.step(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            break
    print(f"✓ REINFORCE runs successfully")

if __name__ == "__main__":
    print("=" * 50)
    print("Testing All RL Algorithms")
    print("=" * 50)
    
    try:
        test_dp()
        test_q_learning()
        test_sarsa()
        test_reinforce()
        
        print("\n" + "=" * 50)
        print("✅ ALL TESTS PASSED!")
        print("=" * 50)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
