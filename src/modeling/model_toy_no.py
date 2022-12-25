import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Categorical

# no weights shared.

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
        print('action_prob', action_probs)
        action = m.sample() # (batch,) with numbers corrsponding to the action that belongs to [0, 1, ..., action_shape-1]
        
        # print('action_probs', action_probs)
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
        # value, action_log_probs, dist_entropy
        return state_val, action_log_probs, dist_entropy


class ActorCritic(nn.Module):
    # In this version parameters of Actor and Critic are completely split.
    def __init__(self, action_shape=3, hidden_size=32, mlp_size=128, num_layers=1, 
        dropout=0., rnn_model='gru'):
        # action shape is the number of action options
        super(ActorCritic, self).__init__()

        self.hidden_size = hidden_size
        if rnn_model == 'gru':
            self.rnn_actor = nn.GRU(
                    input_size=2, # pedestrian's representation
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                    # bidirectional=bidirectional,
                    )
            self.rnn_critic = nn.GRU(
                    input_size=2, # pedestrian's representation
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                    # bidirectional=bidirectional,
                    )
        elif rnn_model == 'lstm':
            self.rnn_actor = nn.LSTM(
                    input_size=2, # pedestrian's representation
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    batch_first=True,
                    dropout=dropout,
                    # bidirectional=bidirectional,
                    )
            self.rnn_critic = nn.LSTM(
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
        self.spatial_embedding_actor = self.make_mlp(spatial_embedding_mlp_dims, batch_norm=False, dropout=dropout)
        self.spatial_embedding_critic = self.make_mlp(spatial_embedding_mlp_dims, batch_norm=False, dropout=dropout)


        # actor_mlp_dims = [3*hidden_size, mlp_size, action_shape]
        actor_mlp_dims = [hidden_size, mlp_size, action_shape]
        self.actor = self.make_mlp(actor_mlp_dims, batch_norm=False, dropout=dropout)
        self.actor_softmax = nn.Softmax(dim=1)

        # critic_mlp_dims = [3*hidden_size, mlp_size, mlp_size]
        critic_mlp_dims = [hidden_size, mlp_size, mlp_size]
        self.critic = self.make_mlp(critic_mlp_dims, batch_norm=False, dropout=dropout)
        self.critic_linear = nn.Linear(mlp_size, 1)
    
    def forward(self, x):
        """
        inputs:
            - x # input data. tensor. size: (batch, 11, 2) # seq = 10 for now
            - x[:, -1, :] # ego car position at current time step # size: (batch, 2)
            - x[:, :-1, :] # pedestrian position history upto now # size: (batch, 10, 2)
            # torch tensor for each x[i]
            # [[x_ped_{T-9}, y_ped_{T-9}],
            # [x_ped_{T-8}, y_ped_{T-8}],
            # [x_ped_{T-7}, y_ped_{T-7}],
            # ...
            # [x_ped_{T}, y_ped_{T}], # current time step
            # [x_ego_{T}, y_ego_{T}]] # current time step


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
        # Data processing
        # - x[:, -1, :] # ego car position at current time step # size: (batch, 2)
        # - x[:, :-1, :] # pedestrian position history upto now # size: (batch, 10, 2) 
        # ecar_pos_curr, ped_pos_hist = x[:, -1].clone().float(), x[:, :-1].clone().float()

        ecar_pos_curr = x.float()
        # ped_pos_curr = ped_pos_hist[:, -1].clone().float() # (batch, 2)
        
        # Actor
        ecar_pos_curr_ebd_actor = self.spatial_embedding_actor(ecar_pos_curr) # (batch, hidden_size)
        #ped_pos_curr_ebd_actor = self.spatial_embedding_actor(ped_pos_curr) # (batch, hidden_size)
        #out_rnn_actor, _ = self.rnn_actor(ped_pos_hist) # (batch, seq, hidden_size)
        #ped_hist_ebd_actor = out_rnn_actor[:, -1] # (batch, hidden_size)
        #x_ebd_actor = torch.cat((ecar_pos_curr_ebd_actor, ped_pos_curr_ebd_actor, ped_hist_ebd_actor), dim=1) # (batch, 3*hidden_size)
        x_ebd_actor = ecar_pos_curr_ebd_actor
        action_probs = self.actor_softmax(self.actor(x_ebd_actor)) # (batch, action_shape)
        
        # Critic
        ecar_pos_curr_ebd_critic = self.spatial_embedding_critic(ecar_pos_curr) # (batch, hidden_size)
        # ped_pos_curr_ebd_critic = self.spatial_embedding_critic(ped_pos_curr) # (batch, hidden_size)
        # out_rnn_critic, _ = self.rnn_critic(ped_pos_hist) # (batch, seq, hidden_size)
        # ped_hist_ebd_critic = out_rnn_critic[:, -1] # (batch, hidden_size)
        # x_ebd_critic = torch.cat((ecar_pos_curr_ebd_critic, ped_pos_curr_ebd_critic, ped_hist_ebd_critic), dim=1) # (batch, 3*hidden_size)
        
        x_ebd_critic = ecar_pos_curr_ebd_critic
        state_val = self.critic_linear(self.critic(x_ebd_critic)) # (batch, 1)
        
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


def main():
    # policy = Policy(5)
    actor_critic = ActorCritic()
    x_test = torch.rand(1,2)
    x_test_2 = actor_critic.spatial_embedding_actor(x_test)
    # print(actor_critic.spatial_embedding_actor)

if __name__ == "__main__":
    main()
