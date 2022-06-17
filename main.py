import torch
import os
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment
# REMEMBER apt-get install -y xvfb python-opengl ffmpeg

def main():
    device = torch.device('cuda')
    

    os.chdir('./rl-agents/scripts')
    env_config = 'configs/HighwayEnv/env_obs_attention.json'
    agent_config = 'configs/HighwayEnv/agents/DQNAgent/ego_attention1Head.json'

    env = load_environment(env_config)
    agent = load_agent(agent_config, env)
    
    evaluation = Evaluation(env, agent, num_episodes=10, display_env=False, display_agent=False)
    print(f"Ready to train {agent} on {env}")
    evaluation.train()
    
if __name__ == '__main__':
    main()