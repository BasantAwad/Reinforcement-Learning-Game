"""
Metrics tracking and analysis for RL agents comparison.
"""

import numpy as np
from typing import Dict, List, Any
import json
from datetime import datetime


class MetricsTracker:
    """Track and aggregate metrics for RL agent performance analysis."""
    
    def __init__(self):
        self.runs = {}  # Store metrics for each algorithm run
        self.current_run = None
        
    def start_run(self, algo_id: str, env_mode: str, config: Dict[str, Any]):
        """Start tracking a new algorithm run."""
        run_id = f"{algo_id}_{env_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_run = {
            'run_id': run_id,
            'algorithm': algo_id,
            'env_mode': env_mode,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'episodes': [],
            'episode_rewards': [],
            'episode_steps': [],
            'episode_success': [],
            'penalties_per_episode': [],
            'epsilon_history': [],
        }
        
        self.runs[run_id] = self.current_run
        return run_id
    
    def log_episode(self, episode_num: int, reward: float, steps: int, 
                   success: bool, penalties: float, epsilon: float = None):
        """Log metrics for a completed episode."""
        if not self.current_run:
            return
            
        self.current_run['episodes'].append(episode_num)
        self.current_run['episode_rewards'].append(reward)
        self.current_run['episode_steps'].append(steps)
        self.current_run['episode_success'].append(success)
        self.current_run['penalties_per_episode'].append(penalties)
        
        if epsilon is not None:
            self.current_run['epsilon_history'].append(epsilon)
    
    def end_run(self):
        """Finalize the current run and compute summary statistics."""
        if not self.current_run:
            return None
            
        self.current_run['end_time'] = datetime.now().isoformat()
        
        # Compute summary statistics
        rewards = np.array(self.current_run['episode_rewards'])
        steps = np.array(self.current_run['episode_steps'])
        success = np.array(self.current_run['episode_success'])
        
        self.current_run['summary'] = {
            'total_episodes': len(rewards),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards)),
            'final_avg_reward': float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards)),
            
            'avg_steps': float(np.mean(steps)),
            'std_steps': float(np.std(steps)),
            'final_avg_steps': float(np.mean(steps[-10:])) if len(steps) >= 10 else float(np.mean(steps)),
            
            'success_rate': float(np.mean(success) * 100),
            'final_success_rate': float(np.mean(success[-10:]) * 100) if len(success) >= 10 else float(np.mean(success) * 100),
            
            'total_penalties': float(np.sum(self.current_run['penalties_per_episode'])),
            'avg_penalties': float(np.mean(self.current_run['penalties_per_episode'])),
        }
        
        run_summary = self.current_run.copy()
        self.current_run = None
        return run_summary
    
    def get_comparison(self, algo_ids: List[str] = None) -> Dict[str, Any]:
        """Get comparison of all tracked runs or specific algorithms."""
        if algo_ids:
            filtered_runs = {k: v for k, v in self.runs.items() 
                           if v['algorithm'] in algo_ids}
        else:
            filtered_runs = self.runs
            
        comparison = {
            'algorithms': {},
            'best_performer': None,
            'comparison_date': datetime.now().isoformat(),
        }
        
        best_reward = float('-inf')
        
        for run_id, run_data in filtered_runs.items():
            algo = run_data['algorithm']
            
            if 'summary' not in run_data:
                continue
                
            if algo not in comparison['algorithms']:
                comparison['algorithms'][algo] = []
                
            comparison['algorithms'][algo].append({
                'run_id': run_id,
                'env_mode': run_data['env_mode'],
                'summary': run_data['summary']
            })
            
            # Track best performer
            if run_data['summary']['final_avg_reward'] > best_reward:
                best_reward = run_data['summary']['final_avg_reward']
                comparison['best_performer'] = {
                    'algorithm': algo,
                    'run_id': run_id,
                    'final_avg_reward': best_reward
                }
        
        return comparison
    
    def export_to_json(self, filepath: str):
        """Export all tracked runs to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.runs, f, indent=2)
    
    def export_run_to_json(self, run_id: str, filepath: str):
        """Export specific run to JSON file."""
        if run_id in self.runs:
            with open(filepath, 'w') as f:
                json.dump(self.runs[run_id], f, indent=2)


# Global metrics tracker instance
metrics_tracker = MetricsTracker()
