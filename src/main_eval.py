import gym
import gym_pas
# We are now using from gym_pas.envs.pas_env import PaSEnv in envs/__init__.py
# If we want to change back to toy environment, uncomment # from gym_pas.envs.pas_env_toy import PaSEnv.

import random
import os
import sys
import yaml
import numpy as np
from collections import deque
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from storage_multi_env import RolloutStorage

from modeling.model_no import Policy

from ppo import PPO
from arguments import get_args
from parallelEnv import parallelEnv

from torch.utils.tensorboard import SummaryWriter
import time
import math
import copy


# def get_config_and_result_folder(yaml_config, create=False):
#     yaml_filename = yaml_config+'.yaml'
#     with open(os.path.join('yaml_configurations', yaml_filename), 'r') as file:
#         config = yaml.load(file, Loader=yaml.FullLoader) 
#         # print ("Successfully loaded %s" % yaml_filename)
#     result_folder_name = os.path.join('results_paper', yaml_config)
#     if create:
#         try:
#             os.mkdir(result_folder_name)
#         except OSError:
#             print ("Creation of the directory %s failed" % result_folder_name)
#         else:
#             print ("Successfully created the directory %s " % result_folder_name)
#     return config, result_folder_name

def create_pas_env_config(config):
    with open('pas_env_config/config.yaml', 'w') as file:
        yaml.dump(config, file)
        yaml_config = 't'+str(config['start_time'])+'_lw'+str(config['car_lw'][0])+str(config['car_lw'][1])
        print ("Successfully created config.yaml from %s for pas_env." % yaml_config)
    return


def main_eval(device):
    time_limit = 300 #1000
    time_limit -= 2 # if we want to get at most (1000, 2) for each episode, we need time_limit = 1000 - 2 = 998
    num_steps = time_limit+1 
    # total_training_epochs = 20
    # intialization
    args = get_args()
    dataset_filepath = '/home/hcalab/Desktop/MYJ/Class/CS598/pas/human_experiment_data/CSV/S0'
   
    env_name = 'pas-v0'
    env = gym.make(env_name)
    all_files = env.datapath(dataset_filepath)
    total_batch = len(all_files)
    print(total_batch)

    nenvs = 8
    test_flag = False
    envs = parallelEnv(test_flag, env_name, nenvs, seed=1234)

    ecar_policy = torch.load('checkpoint/policy_epoch_23000.pt') # 'checkpoint/policy_epoch_16500.pt'

    episode_count_all_epochs = 0

    states_all_episodes = []
    infos_all_episodes = []
    collision_all_episodes = []
    # for epoch in range(total_training_epochs):

    for nth_episode in range(total_batch):
        print('episode_count_all_epochs: ', episode_count_all_epochs)
        rollouts = RolloutStorage()
        state = envs.reset(time_limit=time_limit) #(nenv,5) 
        done = False

        states_episode = [state] # have the next state after reaching the end. (1000, 5)
        mask_episode = [] # (999, 5) # last must have been invalid.
        

        for i in range(num_steps):
            # state, action, action_log_prob, value, reward, mask should all be tensors with a shape of (1, ...)
            #    0         1         2        3         4
            # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
            # state_ego_ped_tensor = torch.tensor(state[:, 1:3]) # (nenv, 2) # (y_ego, vel) # this is toy setting.
            ### project setting ###
            # we use relative position of pedestrian as input to the network.
            state_ego_ped_tensor = torch.tensor(state).to(device) # (nenv, 5) # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
            state_ego_ped_tensor[:, 3:] = state_ego_ped_tensor[:, 3:] - state_ego_ped_tensor[:, :2] # ['x_ego', 'y_ego', 'velocity','x_ped_rel', 'y_ped_rel']
            ### project setting ###

            with torch.no_grad():
                model_value_preds, action, model_action_log_probs= ecar_policy.act(
                    state_ego_ped_tensor) # model_value_preds (nenv, 1), action (nenv)
                state_ego_ped_tensor = state_ego_ped_tensor.unsqueeze(0) # (1, nenv, 2)
                model_value_preds = model_value_preds.unsqueeze(0) # (1, nenv, 1)
                model_action_log_probs_tensor = model_action_log_probs.unsqueeze(0).unsqueeze(2) # (1, batch, 1) where batch = nenv here

            next_state, reward, done, infos, collision_flag = envs.step(action.cpu().numpy()) # action is 0-4 now
            '''
            These are the shape info that help me debug.
            # next_state (nenv, 5)
            # reward (nenv,)
            # done (nenv,)
            # infos (None, None, None, None)
            '''
            state = np.copy(next_state)
            states_episode.append(state)
            if done.sum() == nenvs: # done is all one, finish the episode.
                print(collision_flag)
                break

        
        states_episode = np.stack(states_episode)
        states_all_episodes.append(states_episode)
        infos_all_episodes.append(infos)
        collision_all_episodes.append(collision_flag)

        # if len(states_all_episodes) % 200 == 0:
            # render_data_all_episodes = [states_all_episodes, infos_all_episodes, collision_all_episodes]
            # filename = 'rendering_results_eval/test_collision_location_render_'+str(len(states_all_episodes))+'.pt'
            # torch.save(render_data_all_episodes, filename)
            # print(filename+' is saved.')
        
        episode_count_all_epochs += 1

    # if episode_count_all_epochs % 1250 == 0:
    render_data_all_episodes = [states_all_episodes, infos_all_episodes, collision_all_episodes]
    filename = 'result/final.pt'
    torch.save(render_data_all_episodes, filename)
    print(filename+' is saved.')
    print('evaluation done.')
    return


def render_policy_results(device, logging, anime=True):

    # _, policy_folder = 'result' # get_config_and_result_folder(policy_yaml_config)
    render_filename = os.path.join('result', 'final.pt')
    render_data_all_episodes = torch.load(render_filename)
    
    states_all_episodes, infos_all_episodes, collision_all_episodes = render_data_all_episodes
    # 0 means ongoing, 1 means collision, 2. means timeout, 3 means success.
    collision_results = []
    for collision_episode in collision_all_episodes:
        collision_results += list(collision_episode)
    
    collision_results = np.array(collision_results) # (num_tests,)
    num_tests = len(collision_results)
    num_collision_front = sum(collision_results==1)
    num_collision_side = sum(collision_results==1.5)
    num_collision = num_collision_front + num_collision_side
    num_timeout = sum(collision_results==2)
    num_success = sum(collision_results==3)

    print()
    logging.info('number of samples: {}'.format(num_tests))
    logging.info('time-out rate: {0:.4f}'.format(num_timeout/num_tests))
    logging.info('total collision rate: {0:.4f}'.format(num_collision/num_tests))
    logging.info('front collision rate: {0:.4f}'.format(num_collision_front/num_tests))
    logging.info('side collision rate: {0:.4f}'.format(num_collision_side/num_tests))
    logging.info('success rate: {0:.4f}'.format(num_success/num_tests))
    print()

    if anime:
        args = get_args()
        env_name = 'pas-v0' 
        env = gym.make(env_name)
        render_count = np.array([0,0,0,0])
        render_max = 10

        for eps_idx, (states_episode, infos_episode, collision_episode) \
            in enumerate(zip(states_all_episodes, infos_all_episodes, collision_all_episodes)):
            for env_idx, collision_flag in enumerate(collision_episode):
                if np.random.random()<0.5: # Render random episodes
                    if collision_flag==1: # collision
                        if render_count[0] <render_max:
                            output_file = 'result/epi_'+str(eps_idx)+'_env_'+str(env_idx)+'_collision.gif'
                            render_count[0]+=1
                        else:
                            output_file = None
                    elif collision_flag==1.5: # side collision
                        if render_count[1] <render_max:
                            output_file = 'result/epi_'+str(eps_idx)+'_env_'+str(env_idx)+'_SIDEcollision.gif'
                            render_count[1]+=1
                        else:
                            output_file = None
                    elif collision_flag==2.: # timeout
                        if render_count[2] <render_max:                    
                            output_file = 'result/epi_'+str(eps_idx)+'_env_'+str(env_idx)+'_Timeout.gif'
                            render_count[2]+=1
                        else:
                            output_file = None
                    elif collision_flag==3: # reached goal
                        if render_count[3] <render_max:
                            output_file = 'result/epi_'+str(eps_idx)+'_env_'+str(env_idx)+'_ReachedGoal.gif'
                            render_count[3]+=1
                        else:
                            output_file = None

                    if output_file is not None:
                        num_steps = infos_episode[env_idx]
                        states = states_episode[:num_steps, env_idx]
                        env.anime(output_file, saved_data=states, episode_index=eps_idx, env_index=env_idx,interval=10)




if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    resume = False
    os.makedirs('result')
    log_file = os.path.join('result', 'eval.log')
    mode = 'a' if resume else 'w'
    file_handler = logging.FileHandler(log_file, mode=mode)
    stdout_handler = logging.StreamHandler(sys.stdout)
    level = logging.INFO
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
                        format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")\


    main_eval(device)
    render_policy_results(device, logging)
    
    # policy_yaml_config = 't0_lw64'
    # policy_yaml_config = 't0_lw42'
    # policy_yaml_config = 't50_lw64' # 't50_lw42'# 't0_lw42'
    # policy_yaml_config = 't50_lw42'
    # policy_yaml_config = 't50_lw42'
    # policy_yaml_config = 't0_lw42'
    # policy_yaml_config = #'t50_lw64'
    # test_yaml_configs = ['t0_lw42', 't0_lw64', 't50_lw42', 't50_lw64']

    # policy_yaml_config = 'src/pas_env_config/config.yaml'
    # test_yaml_configs = ['lw64']

    # for test_yaml_config in test_yaml_configs:
    #     main_eval_v02(policy_yaml_config, test_yaml_config, device)

    # for test_yaml_config in test_yaml_configs:
    #     render_policy_results(policy_yaml_config, test_yaml_config, device)

    # anime = True
    # anime = False
    # render_results_v02(anime)
    # 6000 break