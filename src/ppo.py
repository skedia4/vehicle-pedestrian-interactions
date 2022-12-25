import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from statistics import mean
from modeling.model_toy_no import Policy


class PPO():
    def __init__(self,
                 actor_critic,
                 clip_param,
                 ppo_epoch,
                 mini_batch_size, # originally this argument is num_mini_batch
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=False): # True at first

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.mini_batch_size = mini_batch_size

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)


    '''
    def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = model(state)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    '''


    def update(self, rollouts):

        value_loss_epoch, action_loss_epoch, dist_entropy_epoch = [], [], []


        for _ in range(self.ppo_epoch):
            
            for b_states, b_actions, b_values, b_returns, b_masks, b_action_log_probs, b_advantages, b_rewards in rollouts.iter(self.mini_batch_size):
            
                obs_batch, actions_batch, value_preds_batch, return_batch, masks_batch, \
                    old_action_log_probs_batch, adv_targ, rewards_batch = \
                    b_states.detach(), b_actions.detach(), b_values.detach(), b_returns.detach(), b_masks.detach(), b_action_log_probs.detach(), \
                    b_advantages.detach(), b_rewards.detach()

                # Reshape to do in a single forward pass for all steps
                
                # values, action_log_probs, dist_entropy, _ = self.actor_critic.evaluate_actions(
                #     obs_batch, masks_batch,
                #     actions_batch)

                # not consider mask by now
                # print(adv_targ.shape)
                # print('adv_targ_mean',adv_targ.mean(),'adv_targ_std',adv_targ.std())

                # print('The number of adv_targ: ', adv_targ.shape)
                if torch.isnan(adv_targ.std()):
                    print('fuck')
                    print('The number of adv_targ: ', adv_targ.shape)
                adv_targ = (adv_targ - adv_targ.mean()) / adv_targ.std()

                
                ####
                # ZHE DEBUG
                # rewards = torch.tensor(rewards).to(device)
                # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
                #return_batch = (return_batch - return_batch.mean()) / (return_batch.std())
                # rewards_batch = (rewards_batch - rewards_batch.mean()) / (rewards_batch.std()+1e-12)
                # adv_targ = rewards_batch - value_preds_batch
                # print('adv_targ_mean',adv_targ.mean(),'adv_targ_std',adv_targ.std())
                # adv_targ = (adv_targ - adv_targ.mean()) / adv_targ.std()
                ####

                
                values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(obs_batch,actions_batch)
                ratio = torch.exp(action_log_probs -
                                  old_action_log_probs_batch.detach())
                surr1 = ratio * adv_targ.detach()
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * adv_targ.detach()
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + \
                        (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses,
                                                 value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                
                (value_loss * self.value_loss_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                

                # ZHE DEBUG
                # print('love and peace')
                # (action_loss-dist_entropy * self.entropy_coef).backward()


                nn.utils.clip_grad_norm_(self.actor_critic.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch.append(value_loss.item())
                action_loss_epoch.append(action_loss.item())
                dist_entropy_epoch.append(dist_entropy.item())
        
        value_loss_epoch = mean(value_loss_epoch)
        action_loss_epoch = mean(action_loss_epoch)
        dist_entropy_epoch = mean(dist_entropy_epoch)

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
