import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import pdb


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage:
    # now we assume we only implement 1 environment. Maybe it can naturally extend to multi environment cases.
    # For example, if num_envs = 3, use list = [[],[],[]]
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.action_log_probs = []
        self.values = [] # value prediction during simulation (model.act)
        self.rewards = []
        self.masks = [] # mask is 1 - done.

        self.returns = []
        self.advantages = None # it never becomes a list. Just becomes a tensor in list2tensor()
        self.infos = None

    def append(self, state, action, action_log_prob, value, reward, mask, infos):
        # state, action, action_log_prob, value, reward, mask should all be tensors with a shape of (1, ...)
        
        # infos: a tuple of scalars with length nenvs. Has the index of the state that reached done = 1 for the first time.
        
        # even if it is scalar, like reward, it should be (1,) so eventually we can cat them.


        # state = env_states[t:t+1] # (1, num_envs, state_space)
        # action = env_actions[t:t+1] # (1, num_envs, action_space)
        # reward = env_rewards[t:t+1] # (1, num_envs, 1)
        # mask = env_mask[t:t+1] # (1, num_envs) # set this as (1, num_envs, 1)
        # model_value_pred = model_value_preds[t:t+1] # (1, num_envs, 1) # based on neural networks
        # model_action_log_prob = model_action_log_probs[t:t+1] # (1, num_envs, action_space)
        # rollouts.append(state, action, model_action_log_prob, model_value_pred, reward, mask)


        self.states.append(state) 
        self.actions.append(action)
        self.action_log_probs.append(action_log_prob)
        self.values.append(value) # a list of values (1, num_envs, 1) with len num_steps
        self.rewards.append(reward) # a list of rewards (1, num_envs, 1) with len num_steps
        self.masks.append(mask) # a list of masks (1, num_envs, 1) with len num_steps
        self.infos = infos

    def append_next_state(self, next_state):
        self.states.append(next_state) # next_state: (1, num_envs, state_space)



    def compute_gae(self, next_value, gamma=0.99, tau=0.95):
        values = self.values + [next_value] # not update to self.values to keep everything same len.
        gae = 0
        returns = []
        for step in reversed(range(len(self.rewards))):
            delta = self.rewards[step] + gamma * values[step + 1] * self.masks[step] - values[step]
            gae = delta + gamma * tau * self.masks[step] * gae
            returns.insert(0, gae + values[step]) # a list of (1, num_envs, 1)
        
        self.returns = returns # same len as rewards, which is num_steps.
        


    def valid_data_extraction(self):
        """
        intermediate:
            - self.states # tensor. size: (num_steps, num_envs, state_space) # we can actually just stack the pedestrian history with the car's current position as the state space.
            - self.actions # tensor. size: (num_steps, num_envs, action_space)
            - self.action_log_probs # tensor. size: (num_steps, num_envs, action_space)
            - self.values # tensor. size: (num_steps, num_envs, 1)
            - self.rewards # tensor. size: (num_steps, num_envs, 1)
            - self.masks # tensor. size: (num_steps, num_envs, 1)

        outputs:
            - 

        """
        self.states = torch.cat(self.states, dim=0) # when states becomes complicated, use list itself. e.g. T = [L[i] for i in Idx] for mini batch
        self.actions = torch.cat(self.actions, dim=0)
        self.action_log_probs = torch.cat(self.action_log_probs, dim=0)
        self.values = torch.cat(self.values, dim=0) # value prediction during simulation (model.act)
        self.rewards = torch.cat(self.rewards, dim=0)
        self.masks = torch.cat(self.masks, dim=0) # mask is 1 - done.

        self.states_valid = []
        self.actions_valid = []
        self.action_log_probs_valid = []
        self.values_valid = []
        self.rewards_valid = []
        self.next_state = []
        self.masks_valid = []

        for env_i in range(len(self.infos)): # nenvs
            done_time_step = self.infos[env_i]
            # (done_time_step+1, state_space) etc
            self.states_valid.append(self.states[:done_time_step+1, env_i].clone()) # 0, ..., done_time_step
            self.actions_valid.append(self.actions[:done_time_step+1, env_i].clone()) # 0, ..., done_time_step
            self.action_log_probs_valid.append(self.action_log_probs[:done_time_step+1, env_i].clone())
            self.values_valid.append(self.values[:done_time_step+1, env_i].clone())
            self.rewards_valid.append(self.rewards[:done_time_step+1, env_i].clone())
            self.next_state.append(self.states[done_time_step+1:done_time_step+2, env_i].clone()) # a list of (1, state_space) with len nenvs.
            self.masks_valid.append(self.masks[:done_time_step+1, env_i].clone())# last is 0., before is always 1.

    def compute_gae_valid(self, next_value, gamma=0.99, tau=0.95):
        # next_value (nenvs, 1)
        self.returns = [] # a list of (done_time_step+1, 1) with len nenvs
        self.advantages = [] # a list of (done_time_step+1, 1) with len nenvs
        for env_i in range(len(self.infos)): # nenvs
            values = torch.cat((self.values_valid[env_i].clone(), next_value[env_i:env_i+1].clone()), dim=0) # (done_time_step+2, 1)
            gae = 0.
            returns_i = []
            rewards = self.rewards_valid[env_i].clone() # (done_time_step+1, 1)
            masks = self.masks_valid[env_i].clone() # (done_time_step+1, 1)

            for step in reversed(range(rewards.shape[0])): # done_time_step, ..., 0
                delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step] # first step is done_time_step
                gae = delta + gamma * tau * masks[step] * gae 
                returns_i.insert(0, gae + values[step]) # a list of (1,) with len done_time_step+1
            
            returns_i = torch.cat(returns_i, dim=0).unsqueeze(1) # (done_time_step+1, 1)
            advantages_i = returns_i - self.values_valid[env_i].clone() # (done_time_step+1, 1)
            self.returns.append(returns_i)
            self.advantages.append(advantages_i)
    
    def tensorboard_results_valid(self):
        episode_return = []
        for rewards in self.rewards_valid:
            episode_return.append(rewards.sum())
        episode_return = (sum(episode_return)/len(episode_return)).item()
        episode_length = sum(self.infos)/len(self.infos) # 0, 1, ..., infos[env_i]

        return episode_return, episode_length

        
    def list2tensor_valid(self, device):

        self.states_valid = torch.cat(self.states_valid, dim=0).to(device)
        self.actions_valid = torch.cat(self.actions_valid, dim=0).to(device)
        self.action_log_probs_valid = torch.cat(self.action_log_probs_valid, dim=0).to(device)
        self.values_valid = torch.cat(self.values_valid, dim=0).to(device)
        self.rewards_valid = torch.cat(self.rewards_valid, dim=0).to(device)
        self.returns_valid = torch.cat(self.returns, dim=0).to(device)
        self.advantages_valid = torch.cat(self.advantages, dim=0).to(device)
        self.masks_valid = torch.cat(self.masks_valid, dim=0).to(device)


    def list2tensor(self, device):
        """
        outputs:
            - self.states # tensor. size: (num_steps*num_envs, state_space) # we can actually just stack the pedestrian history with the car's current position as the state space.
            - self.actions # tensor. size: (num_steps*num_envs, action_space)
            - self.action_log_probs # tensor. size: (num_steps*num_envs, action_space)
            - self.values # tensor. size: (num_steps*num_envs, 1)
            - self.rewards # tensor. size: (num_steps*num_envs, 1)
            - self.masks # tensor. size: (num_steps*num_envs, 1)

            - self.returns # tensor. size: (num_steps*num_envs, 1)
            - self.advantages # tensor. size: (num_steps*num_envs, 1)
        """
        self.states = torch.cat(self.states, dim=0).to(device) # when states becomes complicated, use list itself. e.g. T = [L[i] for i in Idx] for mini batch
        self.actions = torch.cat(self.actions, dim=0).to(device)
        self.action_log_probs = torch.cat(self.action_log_probs, dim=0).to(device)
        self.values = torch.cat(self.values, dim=0).to(device) # value prediction during simulation (model.act)
        self.rewards = torch.cat(self.rewards, dim=0).to(device)
        self.masks = torch.cat(self.masks, dim=0).to(device) # mask is 1 - done.

        self.returns = torch.cat(self.returns, dim=0).to(device)
        self.advantages = self.returns - self.values.to(device)

        # flatten
        self.states = self._flatten_time_env(self.states)
        self.actions = self._flatten_time_env(self.actions)
        self.action_log_probs = self._flatten_time_env(self.action_log_probs)
        self.values = self._flatten_time_env(self.values) # value prediction during simulation (model.act)
        self.rewards = self._flatten_time_env(self.rewards)
        self.masks = self._flatten_time_env(self.masks) # mask is 1 - done.
        self.returns = self._flatten_time_env(self.returns)
        self.advantages = self._flatten_time_env(self.advantages)
        return self.states, self.actions, self.action_log_probs, self.values, self.rewards, self.masks, self.returns, self.advantages
    
    def _flatten_time_env(self, _tensor):
        """
        inputs:
            - _tensor # tensor. size: (num_steps, num_envs, ...)
        outputs:
            - _flattened_tensor # tensor. size: (num_steps*num_envs, ...)
        """

        return _tensor.reshape(torch.prod(torch.tensor(_tensor.size()[:2])), *_tensor.size()[2:])


    
    def iter(self, mini_batch_size):
        data_size = self.values.size(0) # num_steps*num_envs
        # sampler return indices with mini_batch_size without replacement
        sampler = BatchSampler(
            SubsetRandomSampler(range(data_size)),
            mini_batch_size,
            drop_last=True) # can turn drop_last off if wanted
        for indices in sampler:
            # b_states, b_actions, b_values, b_returns, b_masks, b_action_log_probs, b_advantages, b_rewards = \
            yield self.states_valid[indices], self.actions_valid[indices], self.values_valid[indices], \
                self.returns_valid[indices], self.masks_valid[indices], self.action_log_probs_valid[indices], \
                self.advantages_valid[indices], self.rewards_valid[indices]



