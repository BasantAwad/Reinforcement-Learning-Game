"""
Report generation for RL algorithms comparison.
"""

from typing import Dict, List
from .metrics import metrics_tracker
from datetime import datetime


def generate_markdown_report(comparison_data: Dict, output_file: str = "RL_ANALYSIS_REPORT.md"):
    """
    Generate a comprehensive markdown report for LLM analysis.
    
    Args:
        comparison_data: Comparison data from metrics_tracker
        output_file: Output filename for the markdown report
    """
    
    report = []
    
    # Header
    report.append("# Reinforcement Learning Algorithms Comparison Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("---\n\n")
    
    # 1. Environment Overview
    report.append("## 1. Game Environment Overview\n\n")
    report.append("### JungleDash Environment\n\n")
    report.append("**Description:**\n")
    report.append("JungleDash is a grid-based navigation game where an agent (Pink Monster) must navigate through a jungle environment to reach a treasure while collecting coins and avoiding obstacles and traps.\n\n")
    
    report.append("**Environment Specifications:**\n")
    report.append("- **Grid Size:** 8x8 (64 states)\n")
    report.append("- **Action Space:** Discrete(4) - Up, Down, Left, Right\n")
    report.append("- **Observation Space:** Discrete(64) - Agent position on grid\n")
    report.append("- **Max Steps per Episode:** 128 (grid_size² × 2)\n\n")
    
    report.append("**Environment Elements:**\n")
    report.append("- **Agent (Pink Monster):** Starting position (0, 0)\n")
    report.append("- **Goal (Treasure):** Located at (7, 7), +100 reward\n")
    report.append("- **Rewards (Coins):** 4 coins, +20 reward each\n")
    report.append("- **Obstacles (Rocks):** 6 obstacles, -1 penalty for collision\n")
    report.append("- **Traps (Pits):** 2 traps, -20 penalty and episode termination\n\n")
    
    report.append("**Reward Structure:**\n")
    report.append("- Reaching goal: +100 points\n")
    report.append("- Collecting coin: +20 points\n")
    report.append("- Hitting obstacle: -1 point (no movement)\n")
    report.append("- Falling into trap: -20 points (episode ends)\n")
    report.append("- Timeout (max steps): -5 points\n")
    report.append("- Normal movement: 0 points\n\n")
    
    report.append("**Environment Modes:**\n")
    report.append("1. **Static Mode:**\n")
    report.append("   - Fixed layout (seed=42) - same obstacle/reward positions every episode\n")
    report.append("   - Rewards persist on map (can be collected multiple times)\n")
    report.append("   - Goal always gives exactly 100 points\n")
    report.append("   - Ideal for fair algorithm comparison\n\n")
    
    report.append("2. **Dynamic Mode:**\n")
    report.append("   - Random layout - different positions each episode\n")
    report.append("   - Rewards disappear after collection\n")
    report.append("   - Goal gives 100 + (collected_coins × 20) bonus\n")
    report.append("   - Tests generalization capability\n\n")
    
    report.append("---\n\n")
    
    # 2. Algorithms
    report.append("## 2. Reinforcement Learning Algorithms\n\n")
    
    report.append("### 2.1 Dynamic Programming (Value Iteration)\n\n")
    report.append("**Type:** Model-Based, Offline Planning\n\n")
    report.append("**Algorithm:**\n")
    report.append("```\n")
    report.append("Initialize V(s) = 0 for all states\n")
    report.append("Repeat until convergence:\n")
    report.append("    For each state s:\n")
    report.append("        V(s) = max_a Σ p(s',r|s,a)[r + γV(s')]\n")
    report.append("Extract policy: π(s) = argmax_a Σ p(s',r|s,a)[r + γV(s')]\n")
    report.append("```\n\n")
    report.append("**Key Properties:**\n")
    report.append("- Requires complete knowledge of environment dynamics (transition probabilities)\n")
    report.append("- Computes optimal policy before any interaction\n")
    report.append("- Guaranteed convergence to optimal policy\n")
    report.append("- Deterministic policy with epsilon-greedy exploration (ε=0.05) to prevent getting stuck\n\n")
    report.append("**Hyperparameters:**\n")
    report.append("- Discount factor (γ): 0.99\n")
    report.append("- Convergence threshold (θ): 1e-8\n")
    report.append("- Exploration rate (ε): 0.05\n\n")
    
    report.append("### 2.2 Q-Learning\n\n")
    report.append("**Type:** Model-Free, Off-Policy, Temporal Difference\n\n")
    report.append("**Algorithm:**\n")
    report.append("```\n")
    report.append("Initialize Q(s,a) arbitrarily\n")
    report.append("For each episode:\n")
    report.append("    For each step:\n")
    report.append("        Choose action a using ε-greedy policy\n")
    report.append("        Take action a, observe r, s'\n")
    report.append("        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]\n")
    report.append("        s ← s'\n")
    report.append("```\n\n")
    report.append("**Key Properties:**\n")
    report.append("- Learns optimal policy while following ε-greedy exploration policy\n")
    report.append("- Off-policy: learns from max Q-value regardless of action taken\n")
    report.append("- No environment model required\n")
    report.append("- Converges to optimal Q* with probability 1\n\n")
    report.append("**Hyperparameters:**\n")
    report.append("- Learning rate (α): 0.1\n")
    report.append("- Discount factor (γ): 0.99\n")
    report.append("- Exploration rate (ε): 0.1\n\n")
    
    report.append("### 2.3 SARSA\n\n")
    report.append("**Type:** Model-Free, On-Policy, Temporal Difference\n\n")
    report.append("**Algorithm:**\n")
    report.append("```\n")
    report.append("Initialize Q(s,a) arbitrarily\n")
    report.append("For each episode:\n")
    report.append("    Initialize s, choose a using ε-greedy\n")
    report.append("    For each step:\n")
    report.append("        Take action a, observe r, s'\n")
    report.append("        Choose a' using ε-greedy from s'\n")
    report.append("        Q(s,a) ← Q(s,a) + α[r + γQ(s',a') - Q(s,a)]\n")
    report.append("        s ← s', a ← a'\n")
    report.append("```\n\n")
    report.append("**Key Properties:**\n")
    report.append("- On-policy: learns value of policy being followed\n")
    report.append("- Uses actual next action (a') chosen by ε-greedy policy\n")
    report.append("- More conservative than Q-Learning\n")
    report.append("- Name: State-Action-Reward-State-Action\n\n")
    report.append("**Hyperparameters:**\n")
    report.append("- Learning rate (α): 0.1\n")
    report.append("- Discount factor (γ): 0.99\n")
    report.append("- Exploration rate (ε): 0.1\n\n")
    
    report.append("### 2.4 REINFORCE\n\n")
    report.append("**Type:** Model-Free, Policy Gradient, Monte Carlo\n\n")
    report.append("**Algorithm:**\n")
    report.append("```\n")
    report.append("Initialize policy parameters θ\n")
    report.append("For each episode:\n")
    report.append("    Generate episode using π(a|s,θ)\n")
    report.append("    For each step t in episode:\n")
    report.append("        Gt ← return from step t\n")
    report.append("        θ ← θ + α·Gt·∇log π(at|st,θ)\n")
    report.append("```\n\n")
    report.append("**Key Properties:**\n")
    report.append("- Policy-based method (not value-based)\n")
    report.append("- Learns stochastic policy directly\n")
    report.append("- Monte Carlo: updates after complete episodes\n")
    report.append("- Can handle high-dimensional/continuous action spaces\n\n")
    report.append("**Hyperparameters:**\n")
    report.append("- Learning rate (α): varies by implementation\n")
    report.append("- Discount factor (γ): 0.99\n\n")
    
    report.append("---\n\n")
    
    # 3. Experimental Results
    report.append("## 3. Experimental Results and Analysis\n\n")
    
    if comparison_data and 'algorithms' in comparison_data:
        for algo_name, runs in comparison_data['algorithms'].items():
            report.append(f"### 3.{list(comparison_data['algorithms'].keys()).index(algo_name) + 1} {algo_name} Performance\n\n")
            
            for run in runs:
                summary = run['summary']
                env_mode = run['env_mode']
                
                report.append(f"**Environment Mode:** {env_mode.upper()}\n\n")
                report.append("**Performance Metrics:**\n\n")
                report.append(f"| Metric | Value |\n")
                report.append(f"|--------|-------|\n")
                report.append(f"| Total Episodes | {summary['total_episodes']} |\n")
                report.append(f"| Average Reward | {summary['avg_reward']:.2f} ± {summary['std_reward']:.2f} |\n")
                report.append(f"| Final 10 Avg Reward | {summary['final_avg_reward']:.2f} |\n")
                report.append(f"| Max Reward | {summary['max_reward']:.2f} |\n")
                report.append(f"| Min Reward | {summary['min_reward']:.2f} |\n")
                report.append(f"| Average Steps | {summary['avg_steps']:.2f} ± {summary['std_steps']:.2f} |\n")
                report.append(f"| Final 10 Avg Steps | {summary['final_avg_steps']:.2f} |\n")
                report.append(f"| Success Rate | {summary['success_rate']:.2f}% |\n")
                report.append(f"| Final 10 Success Rate | {summary['final_success_rate']:.2f}% |\n")
                report.append(f"| Total Penalties | {summary['total_penalties']:.2f} |\n")
                report.append(f"| Average Penalties/Episode | {summary['avg_penalties']:.2f} |\n\n")
        
        # Best Performer
        if 'best_performer' in comparison_data and comparison_data['best_performer']:
            report.append("### Overall Best Performer\n\n")
            best = comparison_data['best_performer']
            report.append(f"**Algorithm:** {best['algorithm']}\n\n")
            report.append(f"**Final Average Reward:** {best['final_avg_reward']:.2f}\n\n")
    else:
        report.append("*No experimental data available yet. Run algorithms to collect metrics.*\n\n")
    
    report.append("---\n\n")
    
    # 4. Challenges and Solutions
    report.append("## 4. Challenges Faced and Solutions Implemented\n\n")
    
    report.append("### Challenge 1: Unfair Algorithm Comparison in Dynamic Environments\n\n")
    report.append("**Problem:**\n")
    report.append("- Initial implementation had a dynamic environment where rewards disappeared after collection\n")
    report.append("- This caused inconsistent evaluation across algorithms\n")
    report.append("- Dynamic Programming computed policy based on initial state but environment changed during execution\n")
    report.append("- Different episodes had different reward distributions, making performance comparison unreliable\n\n")
    
    report.append("**Solution:**\n")
    report.append("- Implemented toggleable environment modes: Static and Dynamic\n")
    report.append("- Static mode uses fixed seed (42) for consistent layout across episodes\n")
    report.append("- Rewards persist in static mode, maintaining environment consistency\n")
    report.append("- Goal reward is fixed at 100 in static mode (no dynamic bonus)\n")
    report.append("- All algorithms now tested on identical task for fair comparison\n\n")
    
    report.append("**Impact:**\n")
    report.append("- Fair and reproducible algorithm comparison\n")
    report.append("- DP algorithm works correctly with consistent transition model\n")
    report.append("- Performance metrics are directly comparable\n\n")
    
    report.append("### Challenge 2: DP Agent Getting Stuck on Obstacles\n\n")
    report.append("**Problem:**\n")
    report.append("- Dynamic Programming computes deterministic policy offline\n")
    report.append("- If policy leads agent to obstacle, agent stays in same state\n")
    report.append("- Policy never updates (computed once), creating infinite loops\n\n")
    
    report.append("**Solution:**\n")
    report.append("- Added epsilon-greedy exploration (ε=0.05) to DP agent\n")
    report.append("- Agent occasionally takes random actions to escape bad situations\n")
    report.append("- Maintains mostly optimal behavior while preventing stuck states\n\n")
    
    report.append("### Challenge 3: Real-time Performance Monitoring\n\n")
    report.append("**Problem:**\n")
    report.append("- Needed to track exploration vs exploitation behavior\n")
    report.append("- Required comprehensive metrics for analysis and comparison\n\n")
    
    report.append("**Solution:**\n")
    report.append("- Implemented epsilon tracking displayed in real-time GUI\n")
    report.append("- Built comprehensive metrics aggregation system\n")
    report.append("- Tracks: rewards, steps, success rate, penalties, exploration rate\n")
    report.append("- Automated report generation for analysis\n\n")
    
    report.append("---\n\n")
    
    # 5. Raw Data
    report.append("## 5. Raw Experimental Data\n\n")
    report.append("```json\n")
    import json
    report.append(json.dumps(comparison_data, indent=2))
    report.append("\n```\n\n")
    
    report.append("---\n\n")
    report.append("**End of Report**\n")
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(report)
    
    print(f"Report generated: {output_file}")
    return output_file
