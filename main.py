import torch
import os
from rl_agents.trainer.evaluation import Evaluation
from rl_agents.agents.common.factory import load_agent, load_environment
import shap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from termcolor import colored
from scipy.special import kl_div
from scipy.special import softmax
from math import modf
# REMEMBER apt-get install -y xvfb python-opengl ffmpeg

def removeAction (action_log, state_log):
    state_log_abss = np.zeros(shape=(action_log.shape[0],8,7), dtype = 'float')
    for fr in range (action_log.shape[0]):
        for veh in range(0,8):
            for ft in range(1,7):
                if (veh == 0):
                    state_log_abss[fr][veh,ft] = state_log[fr][veh,ft]
                else:
                    state_log_abss[fr][veh,ft] = state_log[fr][veh,ft] + state_log[fr][0,ft]

    for fr in range (action_log.shape[0]):
        x_ego = (state_log_abss[fr][0,1])
        y_ego = (state_log_abss[fr][0,2])

        ego = [x_ego,y_ego]

        if (ego[1] > 0.6):
            lane = "low"
            
        elif ((ego[1] < 0.6) and (ego[1] > 0.15)):
            lane = "mid"
            
        else:
            lane = "top"
            
        if ((lane == "top") and (action_log[fr] == 0)):
            action_log[fr] = 1
            
        if ((lane == "bot") and (action_log[fr] == 2)):
            action_log[fr] = 1
        
    return action_log

def test(env, agent):
    # Test a trained model without metrics, no video recording
    num_failed_episodes = 0
    num_episodes = 100
    for ep in range (num_episodes):
        env.configure({"offscreen_rendering": True})
        evaluation = Evaluation(env, agent, num_episodes=1, training=False, display_env=False, display_agent=False, recover= "/dgx/home/userexternal/llazzaro/Explainable_RL/rl-agents/scripts/out/HighwayEnv/DQNAgent/saved_models/latest.tar")
        state, action, img = evaluation.test()

        state_log = np.array(state)

        action_log = np.array(action)

        action_log = removeAction(action_log, state_log)

        num_of_frame = action_log.shape[0]
        print('EPISODE %i' %ep)
        print(f'Number of frames: {num_of_frame}')
        if num_of_frame < 80: # Hard coded!!!!
            print("Episode failed, skipped")
            num_failed_episodes = num_failed_episodes + 1
    
    print(f'RATE OF FAILED EPISODES: {num_failed_episodes/num_episodes*100}%')

def main():
    device = torch.device('cuda')
    

    os.chdir('./rl-agents/scripts')
    env_config = 'configs/HighwayEnv/env_obs_attention.json'
    agent_config = 'configs/HighwayEnv/agents/DQNAgent/ego_attention1Head.json'

    env = load_environment(env_config)
    agent = load_agent(agent_config, env)
    
    evaluation = Evaluation(env, agent, num_episodes=1500, display_env=False, display_agent=False, recover=False)
    print(f"Ready to train {agent} on {env}")
    evaluation.train()
    test(env, agent)
    
if __name__ == '__main__':
    main()
