import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

# share weights.
class Policy(nn.Module):

    def __init__(self, action_shape=3):
        super(Policy, self).__init__()
        self.actor_critic = ActorCritic(action_shape=action_shape, dropout=0., rnn_model='gru')
    
    def act(self, x):
        """
        inputs:
            - x # input data. dict.
                - ['ecar']
                    # Current ego car position.
                    # tensor. size: (batch, 2)
                - ['ped'] 
                    # History of pedestrian positions. 
                    # The last one in seq is current pedestrian position.
                    # tensor. size: (batch, seq, 2)
        outputs:
            - state_val
            - action
            - action_log_probs
        """
        # self, inputs, rnn_hxs, masks, deterministic=False
        action_probs, state_val = self.actor_critic(x) # (batch, action_shape), (batch, 1)
        m = Categorical(action_probs)
        action = m.sample() # (batch,) with numbers corrsponding to the action that belongs to [0, 1, ..., action_shape-1]
        print('act: ', action_probs)
        action_log_probs = m.log_prob(action) # (batch,)
        # dist_entropy = m.entropy().mean() # scalar in tensor. size: (,)

        return state_val, action, action_log_probs

    def get_value(self, x):
        """
        inputs:
            - x # input data. dict.
                - ['ecar']
                    # Current ego car position.
                    # tensor. size: (batch, 2)
                - ['ped'] 
                    # History of pedestrian positions. 
                    # The last one in seq is current pedestrian position.
                    # tensor. size: (batch, seq, 2)
        outputs:
            - state_val

        """
        _, state_val = self.actor_critic(x) # (batch, 1)
        return state_val
    
    def evaluate_actions(self, x, action):
        """
        inputs:
            - x # input data. dict.
                - ['ecar']
                    # Current ego car position.
                    # tensor. size: (batch, 2)
                - ['ped'] 
                    # History of pedestrian positions. 
                    # The last one in seq is current pedestrian position.
                    # tensor. size: (batch, seq, 2)
            - action 
                # action that is to be evaluated.
                # tensor. size: (batch)

        outputs:
            - state_val

        """
        # self, inputs, rnn_hxs, masks, deterministic=False
        action_probs, state_val = self.actor_critic(x) # (batch, action_shape), (batch, 1)
        m = Categorical(action_probs)

        action_log_probs = m.log_prob(action) # (batch,)
        dist_entropy = m.entropy().mean() # scalar in tensor. size: (,)
        return state_val, action, dist_entropy


class ActorCritic(nn.Module):
    def __init__(self, action_shape=3, hidden_size=32, mlp_size=128, num_layers=1, 
        dropout=0., rnn_model='gru'):
        # action shape is the number of action options
        super(ActorCritic, self).__init__()

        self.hidden_size = hidden_size
        if rnn_model == 'gru':
            self.rnn = nn.GRU(
                    input_size=2, # pedestrian's representation
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                    # bidirectional=bidirectional,
                    )
        elif rnn_model == 'lstm':
            self.rnn = nn.LSTM(
                    input_size=2, # pedestrian's representation
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                    # bidirectional=bidirectional,
                    )
        else:
            print('Wrong rnn setting.')
            raise NotImplementedError

        spatial_embedding_mlp_dims = [2, mlp_size, hidden_size]
        self.spatial_embedding = self.make_mlp(spatial_embedding_mlp_dims, dropout=dropout)


        # actor_mlp_dims = [3*hidden_size, mlp_size, action_shape]
        actor_mlp_dims = [hidden_size, mlp_size, action_shape] # toy
        self.actor = self.make_mlp(actor_mlp_dims, dropout=dropout)
        self.actor_softmax = nn.Softmax(dim=1)

        # critic_mlp_dims = [3*hidden_size, mlp_size, mlp_size]
        critic_mlp_dims = [hidden_size, mlp_size, mlp_size] # toy
        self.critic = self.make_mlp(critic_mlp_dims, dropout=dropout)
        self.critic_linear = nn.Linear(mlp_size, 1)
    
    def forward(self, x):
        """
        inputs:
            - x # input data. dict.
                - ['ecar']
                    # Current ego car position.
                    # tensor. size: (batch, 2)
                - ['ped'] 
                    # History of pedestrian positions. 
                    # The last one in seq is current pedestrian position.
                    # tensor. size: (batch, seq, 2)
        outputs:
            - action_probs 
                # probability of actions in the stochastic policy.
                # tensor. size: (batch, action_shape)
            - state_val
                # values of state from critic.
                # tensor. size: (batch, 1)
        """
        # ecar_pos_curr, ped_pos_hist = x[:, -1].clone().float(), x[:, :-1].clone().float()
        # ped_pos_curr = ped_pos_hist[:, -1].clone().float() # (batch, 2)

        # ecar_pos_curr, ped_pos_hist = x['ecar'], x['ped']

        ecar_pos_curr = x.float()
        ecar_pos_curr_ebd = self.spatial_embedding(ecar_pos_curr) # (batch, hidden_size)
        # ped_pos_curr = ped_pos_hist[:, -1].clone() # (batch, 2)
        # ped_pos_curr_ebd = self.spatial_embedding(ped_pos_curr) # (batch, hidden_size)

        # out_rnn, _ = self.rnn(ped_pos_hist) # (batch, seq, hidden_size)
        # ped_hist_ebd = out_rnn[:, -1] # (batch, hidden_size)
        
        x_ebd = ecar_pos_curr_ebd # torch.cat((ecar_pos_curr_ebd, ped_pos_curr_ebd, ped_hist_ebd), dim=1) # (batch, 3*hidden_size)
        ## Actor and Critic based on results from NNBase...
        # print('during process:', x_ebd)
        action_probs = self.actor_softmax(self.actor(x_ebd)) # (batch, action_shape)
        state_val = self.critic_linear(self.critic(x_ebd)) # (batch, 1)
        
        return action_probs, state_val


    def make_mlp(self, dim_list, activation='relu', batch_norm=False, dropout=0):
        layers = []
        for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
            layers.append(nn.Linear(dim_in, dim_out))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim_out))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'leakyrelu':
                layers.append(nn.LeakyReLU())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
        return nn.Sequential(*layers)
