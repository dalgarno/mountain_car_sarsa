#!/usr/bin/env python
import gym
from tile_coding import IHT, tiles
import random
import numpy as np

NUM_EPISODES = 1
NUM_TILINGS = 8
EPSILON = 0.1
ALPHA = 0.01
GAMMA = 0.9
MAX_STATES = 4096
MAX_STEPS = 2500

random.seed(0)
iht = IHT(MAX_STATES)

def active_tiles(s, a, env):
    x, x_dot = s
    max_x, max_x_dot = tuple(env.observation_space.high)
    min_x, min_x_dot = tuple(env.observation_space.low)

    return tiles(iht,NUM_TILINGS,[NUM_TILINGS*x/(max_x - min_x),
                        NUM_TILINGS*x_dot/(max_x_dot - min_x_dot)],[a])

def state_action_value(s, w, a, env):
    return sum(w[active_tiles(s, a, env)])

def select_action(s, w, env):
    action_list = [0, 1, 2]
    if random.random() < EPSILON:
        return random.choice(action_list)
    max_action_values = max(map(lambda a: state_action_value(s, w, a, env), action_list))
    return action_list.index(max_action_values)


def update_weights_terminal(r, a, w, s, env):
    return ALPHA * (r - state_action_value(s, w, a, env))

def update_weights_step(r, a, w, s, env):
    return ALPHA * (GAMMA * state_action_value(s, w, a, env)) - (state_action_value(s, w, a, env))

def main():

    w = np.zeros(MAX_STATES)
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = MAX_STEPS

    for episode in range(NUM_EPISODES):
        step = 0
        s = env.reset()
        a = select_action(s, w, env)

        while True:
            print('Step: {}'.format(step))
            s_prime, r, done, info = env.step(a)
            env.render()
            if done:
                w[active_tiles(s, a, env)] += update_weights_terminal(r, a, w, s, env)
                break

            a_prime = select_action(s, w, env)
            w[active_tiles(s, a, env)] += update_weights_step(r, a, w, s, env)
            s = s_prime
            a = a_prime
            print('a: {}'.format(a))
            step += 1


if __name__ == '__main__':
    main()
