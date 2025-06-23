import numpy as np
import matplotlib.pyplot as plt
import Configurations as config
import torch
from torch.optim import Adam
import torch.nn as nn


def normalise_actions(actions):
    diff = config.MAX_EMB_SIZE - config.MIN_EMB_SIZE
    return (actions - config.MIN_EMB_SIZE) / diff * 2 - config.MIN_EMB_SIZE


class Buffer:
    def __init__(self, target_actor, target_critic,
                 actor_model, critic_model, mode, dataset):
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        BUFFER_SIZE_USER = round(config.SAMPLING_RATIO_USER * dataset.n_users) * 100
        BUFFER_SIZE_ITEM = round(config.SAMPLING_RATIO_ITEM * dataset.n_items) * 100

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        buffer_sizes = {'user': BUFFER_SIZE_USER, 'item': BUFFER_SIZE_ITEM}
        self.buffer_size = buffer_sizes[mode]
        self.state_buffer = np.zeros((self.buffer_size, config.STATE_SIZE))
        self.action_buffer = np.zeros((self.buffer_size, 1))
        self.reward_buffer = np.zeros((self.buffer_size, 1))
        self.next_state_buffer = np.zeros((self.buffer_size, config.STATE_SIZE))
        self.target_actor = target_actor
        self.target_critic = target_critic
        self.actor_model = actor_model
        self.critic_model = critic_model

        # copy weights to the target networks
        self.update_target(
            [(self.target_actor, self.actor_model),
             (self.target_critic, self.critic_model)],
            1.0)

        # Learning rate for actor-critic models
        critic_lr = 0.003
        actor_lr = 0.001
        self.critic_optimizer = Adam(self.critic_model.parameters(), lr=critic_lr)
        self.actor_optimizer = Adam(self.actor_model.parameters(), lr=actor_lr)

        # Discount factor for future rewards
        self.gamma = 0.99

        self.criterion = nn.MSELoss()

        self.update_count = 0

    def record(self, obs_tuple):
        assert len(obs_tuple[0]) == len(obs_tuple[1]) == len(obs_tuple[3])
        # replacing old records
        index = self.buffer_counter % self.buffer_size
        self.state_buffer[index: index + len(obs_tuple[0])] = obs_tuple[0]
        self.action_buffer[index: index + len(obs_tuple[0])] = np.expand_dims(obs_tuple[1], axis=1)
        self.reward_buffer[index: index + len(obs_tuple[0])] = np.expand_dims(obs_tuple[2], axis=1)
        self.next_state_buffer[index: index + len(obs_tuple[0])] = obs_tuple[3]
        self.buffer_counter += len(obs_tuple[0])

    def update_target(self, models, tau):
        for target_model, model in models:
            target_weights = target_model.parameters()
            weights = model.parameters()
            for (a, b) in zip(target_weights, weights):
                a.data.copy_(b.data * tau + a.data * (1 - tau))

    def update(self, state_batch, action_batch, reward_batch, next_state_batch, mode):
        self.update_count += 1
        with torch.no_grad():
            # compute target actions
            target_actions = self.target_actor(next_state_batch)
            # add noise to the target actions
            target_actions += torch.normal(0, 2.0, size=(1,)).clamp(-5, 5).to(config.device)
            target_actions = target_actions.clamp(config.MIN_EMB_SIZE, config.MAX_EMB_SIZE)
            # normalise actions to [-1, 1]
            normalised_target_actions = normalise_actions(target_actions)
            # compute target critic values
            Q1, Q2 = self.target_critic(
                [next_state_batch, normalised_target_actions]
            )
            target_critic_val = torch.min(Q1, Q2)
            y = reward_batch + self.gamma * target_critic_val

        normalised_action_batch = normalise_actions(action_batch)
        critic_value_1, critic_value_2 = self.critic_model([state_batch, normalised_action_batch])
        critic_loss = self.criterion(critic_value_1, y) + self.criterion(critic_value_2, y)

        # backprop and gradient update
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic_model.parameters(), 0.01)
        self.critic_optimizer.step()

        if self.update_count % 2 == 0:
            # update the actor
            actions = self.actor_model(state_batch)
            normalised_actions = normalise_actions(actions)
            critic_value = self.critic_model.Q1([state_batch, normalised_actions])
            # this term will be minimised
            actor_loss = -critic_value.mean()

            # backprop and gradient update
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_model.parameters(), 0.01)
            self.actor_optimizer.step()

            self.update_target([(self.target_actor, self.actor_model)], config.TAU)

            print('actor loss = {:.4f}, critic loss = {:.4}'.format(actor_loss.item(), critic_loss.item()))

        print('actor loss = {:.4f}, critic loss = {:.4f}'.format(actor_loss.item(), critic_loss.item()))

        # soft updates
        self.update_target([(self.target_critic, self.critic_model)], config.TAU)

    # We compute the loss and update parameters
    def learn(self, mode, ep):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_size)
        print('record range = ', record_range)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, config.SAMPLE_SIZE)
        # Convert to tensors
        state_batch = torch.tensor(self.state_buffer[batch_indices], dtype=torch.float32).to(config.device)
        action_batch = torch.tensor(self.action_buffer[batch_indices], dtype=torch.float32).to(config.device)
        reward_batch = torch.tensor(self.reward_buffer[batch_indices], dtype=torch.float32).to(config.device)
        next_state_batch = torch.tensor(self.next_state_buffer[batch_indices], dtype=torch.float32).to(config.device)
        self.update(state_batch, action_batch, reward_batch, next_state_batch, mode)
