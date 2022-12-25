import gym
import gym_pas
# We are now using from gym_pas.envs.pas_env import PaSEnv in envs/__init__.py
# If we want to change back to toy environment, uncomment # from gym_pas.envs.pas_env_toy import PaSEnv.

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from storage_multi_env import RolloutStorage

# We select the model here. 'no' means no sharing weights between actor and critic.
# from modeling.model_toy_no import Policy
from modeling.model_no import Policy
import os
import yaml

from ppo import PPO
from arguments import get_args
from parallelEnv import parallelEnv

from torch.utils.tensorboard import SummaryWriter
import time
import math
import copy


def get_config_and_result_folder(yaml_config, create=False):
    yaml_filename = yaml_config+'.yaml'
    with open(os.path.join('src/yaml_configurations', yaml_filename), 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader) 
        # print ("Successfully loaded %s" % yaml_filename)
    result_folder_name = os.path.join('results_paper', yaml_config)
    if create:
        try:
            os.mkdir(result_folder_name)
        except OSError:
            print ("Creation of the directory %s failed" % result_folder_name)
        else:
            print ("Successfully created the directory %s " % result_folder_name)
    return config, result_folder_name

def create_pas_env_config(config):
    with open('src/pas_env_config/config.yaml', 'w') as file:
        yaml.dump(config, file)
        yaml_config = 't'+str(config['start_time'])+'_lw'+str(config['car_lw'][0])+str(config['car_lw'][1])
        print ("Successfully created config.yaml from %s for pas_env." % yaml_config)
    return




def main_automated(policy_yaml_config):


    policy_config, policy_folder = get_config_and_result_folder(policy_yaml_config)
    create_pas_env_config(policy_config)

    print(policy_config)


    test_flag = False
    env_name = 'pas-v0'
    nenvs = 8
    envs = parallelEnv(test_flag, env_name, nenvs, seed=1234)

    train_policy(envs, nenvs, policy_folder)


def train_policy(envs, nenvs, policy_folder):

    time_limit = 1000
    time_limit -= 2 # if we want to get at most (1000, 2) for each episode, we need time_limit = 1000 - 2 = 998
    num_steps = time_limit+1 
    total_training_epochs = 20

    # intialization
    args = get_args()

    writer = SummaryWriter(policy_folder) 

    # env here is just used for us to get the number of total batches.
    # We are using envs to simulate.
    # files are shuffled inside pas_env.py. in the function reset().
    dataset_filepath = '/home/hcalab/Desktop/MYJ/Class/CS598/pas/human_experiment_data/CSV/S0'
    # dataset_filepath = '/Users/huang/iCloud Drive (Archive)/Documents/Courses@Illinois/2020_spring/ECE586RL/pas/human_experiment_data/CSV/S0'
    # dataset_filepath = '/Users/huang/iCloud Drive (Archive)/Documents/Research@Illinois/people_as_sensors/pas/human_experiment_data/CSV/S0'
    env_name = 'pas-v0'
    env = gym.make(env_name)
    all_files = env.datapath(dataset_filepath)
    total_batch = len(all_files)
    

    # Multi-Environment
    # We are using envs to simulate. 


    ecar_policy = Policy(action_shape=5)
    agent = PPO(
        ecar_policy,
        args.clip_param,
        args.ppo_epoch,
        32, # change the mini batch size here. The default mini batch size is 32.
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)


    episode_count_all_epochs = 0

    # states_all_episodes = []
    # infos_all_episodes = []
    for epoch in range(total_training_epochs):

        for nth_episode in range(total_batch):
            rollouts = RolloutStorage()
            state = envs.reset() #(nenv,5) 
            done = False

            # states_episode = [state] # have the next state after reaching the end. (1000, 5)
            mask_episode = [] # (999, 5) # last must have been invalid.

            for i in range(num_steps):
                # state, action, action_log_prob, value, reward, mask should all be tensors with a shape of (1, ...)
                
                #    0         1         2        3         4
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
                # state_ego_ped_tensor = torch.tensor(state[:, 1:3]) # (nenv, 2) # (y_ego, vel) # this is toy setting.

                ### project setting ###
                # we use relative position of pedestrian as input to the network.
                state_ego_ped_tensor = torch.tensor(state) # (nenv, 5) # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
                state_ego_ped_tensor[:, 3:] = state_ego_ped_tensor[:, 3:] - state_ego_ped_tensor[:, :2] # ['x_ego', 'y_ego', 'velocity','x_ped_rel', 'y_ped_rel']
                ### project setting ###

                with torch.no_grad():
                    model_value_preds, action, model_action_log_probs= ecar_policy.act(
                        state_ego_ped_tensor) # model_value_preds (nenv, 1), action (nenv)
                    
                    state_ego_ped_tensor = state_ego_ped_tensor.unsqueeze(0) # (1, nenv, 2)
                    model_value_preds = model_value_preds.unsqueeze(0) # (1, nenv, 1)
                    model_action_log_probs_tensor = model_action_log_probs.unsqueeze(0).unsqueeze(2) # (1, batch, 1) where batch = nenv here


                next_state, reward, done, infos, collision_flag = envs.step(action.numpy()) # action is 0-4 now
                '''
                These are the shape info that help me debug.
                # next_state (nenv, 5)
                # reward (nenv,)
                # done (nenv,)
                # infos (None, None, None, None)
                '''
               

                mask = 1. - done

                action_tensor = action.unsqueeze(0).unsqueeze(2) # (1, nenv, 1)
                reward_tensor = torch.tensor([reward]).unsqueeze(2) # (1,nenv,1)
                mask_tensor = torch.tensor([mask]).unsqueeze(2) # (1, nenv, 1)

                rollouts.append(state_ego_ped_tensor, action_tensor, model_action_log_probs_tensor, model_value_preds, reward_tensor, mask_tensor, infos)
                state = np.copy(next_state)

                mask_episode.append(mask) # not used yet
                # states_episode.append(state)


                if done.sum() == nenvs: # done is all one, finish the episode.
                    rollouts.append_next_state(torch.tensor(state).unsqueeze(0)) # (1, nenv, 5)
                    break

            # states_episode = np.stack(states_episode)
            # states_all_episodes.append(states_episode)
            # infos_all_episodes.append(infos)
            # render_data_all_episodes = [states_all_episodes, infos_all_episodes, collision_flag]

            # if len(states_all_episodes) % 200 == 0:
            #     filename = policy_folder+'/render_'+str(len(states_all_episodes))+'.pt'
            #     torch.save(render_data_all_episodes, filename)
            #     print(filename+' is saved.')
            

            rollouts.valid_data_extraction()
            episode_return, episode_length = rollouts.tensorboard_results_valid()

            with torch.no_grad():
                next_state = rollouts.next_state # a list of (1, state_space) with len nenvs.
                next_state = torch.cat(next_state, dim=0) # (nenvs, state_space)
                next_value = ecar_policy.get_value(next_state) # (nenvs, 1)

            rollouts.compute_gae_valid(next_value)
            rollouts.list2tensor_valid()

            value_loss, action_loss, dist_entropy = agent.update(rollouts)
            writer.add_scalar('value_loss', value_loss, episode_count_all_epochs)
            writer.add_scalar('action_loss', action_loss, episode_count_all_epochs)
            writer.add_scalar('dist_entropy', dist_entropy, episode_count_all_epochs)
            writer.add_scalar('accumulative reward', episode_return, episode_count_all_epochs)
            writer.add_scalar('episode length', episode_length, episode_count_all_epochs)

            episode_count_all_epochs += 1

            if episode_count_all_epochs % 500 == 0:
                torch.save(ecar_policy, policy_folder+'/policy_'+str(episode_count_all_epochs)+'.pt')

            if episode_count_all_epochs % 4000 == 0:
                torch.save(ecar_policy, policy_folder+'/final_policy.pt')
                print(policy_folder+'/final_policy.pt'+' is saved.')
                writer.close()
                return
    
    writer.close()
    return


if __name__ == "__main__":
    """
    Choose the function you want to run.
    """
    # policy_yaml_configs = ['t0_lw42', 't50_lw42', 't50_lw64']
    policy_yaml_configs = ['t50_lw64', 't50_lw42']
    for policy_yaml_config in policy_yaml_configs:
        main_automated(policy_yaml_config) 
    # policy_yaml_config = 't0_lw32'
    # main_automated(policy_yaml_config)

    # main()
    # render_results()

    # main_evaluation()
