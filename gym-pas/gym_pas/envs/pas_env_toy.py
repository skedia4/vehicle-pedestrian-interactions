import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import glob
import itertools
import csv
import torch
import pdb
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import random

class PaSEnv(gym.Env):
    def __init__(self):
        super(PaSEnv, self).__init__()
        self.test_flag = False
        
        self.df = None 
        self.timestep = None
        self.obs = None
        self.time_limit = None
        self.reward = 0 
        self.done= 0.
        self.action_space = spaces.Discrete(5) # yjm may need to change according to what we have now
        self.observation_space = spaces.Box(low=-1, high=30, dtype=np.float64, shape=(128,5))
        self.infos = (-999)
        
        self.all_files = self.datapath()
        self.num_files = len(self.all_files)
        self.file_index = 0
        random.shuffle(self.all_files)
        self.filepath = self.all_files[self.file_index]


    def datapath(self, path=None):
        """
        get the data files.
        inputs:
            - path # None or the string of path. None means the default path in machine rl.
        outputs:
            - all_files # list of files.
        """
        # PaS dataset 
        ## subject_id = 1~4 : Training / subject id = 5 : Testing
        subject_id = 5 # [1-5]
        trial_id = 8 # [1-8]  
        sample_id = 36  # [1-36]

        if path is None:
            # original setting in rl
            path = '/home/hcalab/Desktop/MYJ/Class/CS598/pas/human_experiment_data/CSV/S0'
            # path = '/home/rl/Documents/PasPPO/human_experiment_data/CSV/S0'
            # path = '/home/zhe/Documents/ECE586RL/pas/human_experiment_data/CSV/S0'

        all_files = []
        
        if self.test_flag: 
            all_files = glob.glob(path+ str(subject_id)+'/*/*.csv') #5

        else:
            for i in range(1,subject_id,1):  #1~4
                all_files.append(glob.glob(path+ str(i)+'/*/*.csv'))  
            all_files = list(itertools.chain.from_iterable(all_files))        
        return all_files

    def dataloader(self, filepath):
        """
        load the data for one episode.
        inputs:
            - filepath
                # dataframe in the file includes
                # {'Time': 0, 'x_ego': 1, 'y_ego': 2, 'velocity': 3, 
                # 'acceleration': 4, 'action': 5, 'x_ped': 6, 'y_ped': 7}

        outputs:
            - observation_data 
                # np array. (time_step, 5)
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
        """
        with open(filepath) as f:
            dataframe = np.array(np.loadtxt(f, delimiter=",", skiprows=1))
        observation_data = np.concatenate([dataframe[:,1:4],dataframe[:,6:]],axis=1)
        return observation_data

    def step(self, action):
        """
        move one step in env.
        inputs:
            - action
                # integer that represents the action. 
                # possible options are 1, 2, 3, 4, 5.
                # 1 => "acceleration = +2"
                # 2 => "acceleration = +1"
                # 3 => "acceleration = -1"
                # 4 => "acceleration = -2"
                # 5 => "acceleration =  0"

        outputs:
            - self.obs
                # np. (5, )
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
            - reward
                # float scalar.
            - self.done
                # float scalar. 0 or 1.
            - self.infos
                # (-999)  (When reaches done=1, it can give the index when self.done becomes 1.)


        """
        # reverse the order of reward and obs in step function.
        reward = self.reward_func_toy(action, time_limit=self.time_limit)
        self.obs = self._next_observation_toy(action)
        # infos = None 
        self.timestep = self.timestep + 1
        # return self.obs, reward, self.done, infos
        return self.obs, reward, self.done, self.infos

    def reset(self, init_vel=[-3,2], time_limit=998):
        """
        reset env.
        inputs:
            - init_vel
                # [min_init_vel, max_init_vel]
                # in default, min_init_vel is -6 m/s and max_init_vel is -1 m/s
                # [1, 2]
            
            - time_limit
                # check out explanation in reward_func_toy.
                # Bascially finally we get (1000, 5) episode for time_limit=998.
        
        intermediate:
            - self.df 
                # np. (time_step, 5)
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
        
        outputs:
            - self.obs
                # np. (5, )
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
            
        """

        self.time_limit = time_limit

        self.file_index += 1
        if self.file_index == self.num_files:
            random.shuffle(self.all_files)
            self.file_index = 0
        self.filepath = self.all_files[self.file_index]

        self.df = self.dataloader(self.filepath) #Loading one episode
        self.df[:, 0] = 12. # fix x_ego. May change it when we use ecar instead of human car.
        self.timestep = 0  

        self.obs = self.df[self.timestep,:]# observation in first timestep of the episode
        self.obs[2] = np.random.uniform(init_vel[0], init_vel[0],1) # toy
        self.done = 0.
        
        return self.obs


    def anime(self, saved_data=None, episode_index=0, interval=30):
        """
        render the episode.
        inputs:
            - saved_data
                # None or np.
                # None means we are using the self.df for rendering.
                # np array means we are rendering an episode saved earlier. (time_step, 5)
                # ['x_ego', 'y_ego','velocity','x_ped', 'y_ped']
            - episode_index
                # integer.
                # The episode index where we will be using for the title of animation.
            - interval
                # integer. Same as freq. default is 30Hz, which is 0.0333 sec.
        """

        if saved_data is None:
            df = self.df
        else:
            df = saved_data

        self.fig, self.ax = plt.subplots()

        ped_pos_hist = df[:, 3:]
        ecar_pos_hist = df[:, :2]

        ped_traj, = self.ax.plot([], [], 'c.', ms=10)
        ecar_traj, = self.ax.plot([], [], 'k.', ms=20)

        patches = [ped_traj, ecar_traj]

        def init():
            self.ax.set_xlim([4, 17])
            self.ax.set_ylim([7, 42])
            self.ax.set_title('Episode '+str(episode_index))
            return patches


        def update(ts):
            ped_traj.set_data(ped_pos_hist[ts, 0], ped_pos_hist[ts, 1])
            ecar_traj.set_data(ecar_pos_hist[ts, 0], ecar_pos_hist[ts, 1])

            return patches

        ani = FuncAnimation(self.fig, update, frames=df.shape[0],
                            init_func=init, interval=interval, repeat=True)
        plt.show()


    def _next_observation_toy(self, action, freq=5.):
        """
        render the episode.
        inputs:
            - action
                # integer that represents the action. 
                # possible options are 0, 1, 2, 3, 4.
                # 0 => "acceleration = +2"
                # 1 => "acceleration = +1"
                # 2 => "acceleration = -1"
                # 3 => "acceleration = -2"
                # 4 => "acceleration =  0"
            - freq
                # float scalar.
                # 1./time_step. default should be 30 (correspond to 0.0333 sec)
            
        outputs:
            - next_obs
                # np. (5, )
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
        """
        y_ego = self.obs[1]
        dy_ego = self.obs[2]


        if action == 0:  
            delta_y_prime = 2.
            y_ego_prime = dy_ego + float(delta_y_prime/freq)
        elif action == 1:  
            delta_y_prime = 1.  
            y_ego_prime = dy_ego + float(delta_y_prime/freq)      
        elif action == 2:    
            delta_y_prime = -1.      
            y_ego_prime = dy_ego + float(delta_y_prime/freq)             
        elif action == 3: 
            delta_y_prime = -2.        
            y_ego_prime = dy_ego + float(delta_y_prime/freq)
        elif action == 4: 
            delta_y_prime = 0.              
            y_ego_prime = dy_ego


        y_ego_new = y_ego + float(y_ego_prime/freq) # updating the position of the ego car. **considering the length of a single timestep!
        
        # if the ego car is not done with the episode within the length of the data, we're keeping the pedestrian in the last position of the data for now.
        if self.timestep >= self.df.shape[0]-1:
            self.next_obs = self.df[-1,:]    
            self.next_obs[1] = y_ego_new 
            self.next_obs[2] = y_ego_prime
            self.next_obs[3], self.next_obs[4] = self._ped_behavior(freq)
        else:
            self.next_obs = self.df[self.timestep+1,:]        
            self.next_obs[1] = y_ego_new 
            self.next_obs[2] = y_ego_prime
            self.next_obs[3], self.next_obs[4] = self._ped_behavior(freq)

        return self.next_obs

    def _ped_behavior(self, freq=5.):
        # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
        # pass
        #  start_of_cw = 15
        # end_of_cw = 7

        # self.obs
        # np. (5, )
        # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']

        x_ped_curr, y_ped_curr = self.obs[3], self.obs[4]
        vx_ped = -1.5 # constant speed 1.5 m/s

        if x_ped_curr < 6.:
            x_ped_next = x_ped_curr
        else:
            x_ped_next = x_ped_curr + vx_ped/freq
        y_ped_next = y_ped_curr

        return x_ped_next, y_ped_next



    def reward_func_toy(self, action, time_limit=998, zero_reward=True):
        """
        inputs:
            - action
                # integer that represents the action. 
                # possible options are 0, 1, 2, 3, 4.
                # 0 => "acceleration = +2"
                # 1 => "acceleration = +1"
                # 2 => "acceleration = -1"
                # 3 => "acceleration = -2"
                # 4 => "acceleration =  0"
                # We are supposed to compute R(s, a), but currently action is not used in reward function.
            - time_limit
                # integer. the max number of time steps an episode can take. default is 999.
                # the step function is called at maximum 1000.
                # The initial state is not created by step.
                # when the reward function is called, it is the current time step. time_step+=1 is not called.
                # when time_limit = 998, the states in the episode are already 999, and then will include next state.
                # So finally we will get (1000, 5) in the episodes.
            - zero_reward # False
                # True -> use 0 for else condition.
                # False -> Use the attraction-to-goal reward. Think about positive shift.
        outputs:
            - next_obs
                # np. (5, )
                # ['x_ego', 'y_ego', 'velocity','x_ped', 'y_ped']
        """

        ego_w = 4. #width of the ego car
        ego_l = 6. #length of the ego car
       
        x_ego = self.obs[0] # x position of our vehicle
        y_ego = self.obs[1] # y_ego is y position of our vehicle
        dy_ego = self.obs[2] # dy_ego is the y velocity of our vehicle
        x_p = self.obs[3] # x_p is the x position of the pedestrian
        y_p = self.obs[4] # y_p is the y position of the pedestrian   

        if self.done == 1.: # it is already done in t-1, so it is invalid in t.
            reward = -999. # invalid number
            return reward

        if self.timestep == time_limit:  # toy
            reward = -2.# -0.5
            self.done = 1.
            self.infos = (self.timestep) # keep the record of time step when done becomes 1
        
        elif((y_ego-ego_l/2) > 33.): # reward for the task completion! (Safely passing the crosswalk and reach the goal state)    
            reward = 2.0# 0.5
            self.done = 1.
            self.infos = (self.timestep) # keep the record of time step when done becomes 1
        else:
            if zero_reward:
                reward = 0.
            else:
                # may do up-shift.
                goal_dist = 33.0 - (y_ego-ego_l/2) #distance from the goal
                # reward = -0.001* goal_dist - 0.05 *(np.log(0.03*goal_dist**2+0.00001)+1)
                reward = ((-0.001* goal_dist - 0.05 *(np.log(0.03*goal_dist**2+0.00001)+1))+1.)/30.
        # if action == 0:
        #     reward += 2.
        return reward        


    




