#!/usr/bin/env python
import random
import numpy as np
from tile_coding import tiles, IHT

class SARSAAgent(object):
    """docstring for SARSAAgent."""
    def __init__(self, env, weights, num_tilings=8,
                alpha=0.01, eps=0.2, gamma=0.98):
        super(SARSAAgent, self).__init__()
        self.env = env
        self.weights = weights
        self.eps = eps
        self.gamma = gamma
        self.alpha = alpha
        self.iht = IHT(len(weights))
        self.num_tilings=num_tilings
        self.max_x, self.max_v = tuple(env.observation_space.high)
        self.min_x, self.min_v = tuple(env.observation_space.low)
        self.episode_total_rewards = []
        self.num_episodes = None


    def active_tiles(self, s, a):
        x, v = s
        active_tiles = tiles(
            self.iht,
            self.num_tilings,
            [self.num_tilings * x / (self.max_x - self.min_x),
             self.num_tilings * v / (self.max_v - self.min_v)],
            [a])

        return active_tiles


    def state_action_values(self, s, a):
        return np.sum(self.weights[self.active_tiles(s, a)])


    def select_action(self, s, find_max=False):
        if random.random() < self.eps and not find_max:
            return random.randint(0, 2)
        return np.argmax([self.state_action_values(s, a) \
                            for a in range(self.env.action_space.n)])


    def train(self, num_episodes, print_episode_stats=True, save_weights=True):
        self.num_episodes = num_episodes

        for episode in range(num_episodes):
            episode_reward = 0
            s = self.env.reset()
            a = self.select_action(s)

            while True:
                s_prime, r, done, info = self.env.step(a)
                episode_reward += r
                self.env.render()

                if done:
                    self.weights[self.active_tiles(s, a)] += \
                    self.alpha * (r - self.state_action_values(s, a))
                    self.episode_total_rewards.append(episode_reward)
                    if print_episode_stats:
                        print('Episode: {}\nEpisode reward: {}\n'\
                        .format(episode + 1, episode_reward))
                        print(np.sum(self.weights))
                    break

                a_prime = self.select_action(s_prime)
                self.weights[self.active_tiles(s, a)] += \
                self.alpha * (
                r + self.gamma * self.state_action_values(s_prime, a_prime) -
                self.state_action_values(s, a)
                )
                s = s_prime
                a = a_prime

        self.env.close()
        if save_weights:
            np.save('weights.npy', self.weights)
