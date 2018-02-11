#!/usr/bin/env python
import gym
from tile_coding import IHT, tiles
import random
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

NUM_EPISODES = 1
NUM_TILINGS = 8
EPSILON = 0.1
ALPHA = 0.01
GAMMA = 0.9
MAX_STATES = 4096
MAX_STEPS = 428

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
    action_values = list(map(lambda a: state_action_value(s, w, a, env), action_list))
    max_action = max(action_values)
    return action_values.index(max_action)


def update_weights_terminal(r, a, w, s, env):
    return ALPHA * (r - state_action_value(s, w, a, env))

def update_weights_step(r, a, w, s, env):
    return ALPHA * (r + GAMMA * state_action_value(s, w, a, env)) - (state_action_value(s, w, a, env))

def max_action_value(s, w, env):
    return max(list(map(lambda a: state_action_value(s, w, a, env), [0, 1, 2])))

def show_plot(w, env):
    max_x, max_x_dot = tuple(env.observation_space.high)
    min_x, min_x_dot = tuple(env.observation_space.low)
    xs = np.linspace(min_x, max_x, 100)
    ys = np.linspace(min_x_dot, max_x_dot, 100)

    zs = np.array([-max_action_value([x, y], w, env) for x in xs for y in ys]).reshape((100, 100))
    xs, ys = np.meshgrid(xs, ys)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    print(np.count_nonzero(zs))
    print('here')
    plt.show()


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
            # env.render()
            if done:
                w[active_tiles(s, a, env)] += update_weights_terminal(r, a, w, s, env)
                break

            a_prime = select_action(s_prime, w, env)
            w[active_tiles(s, a, env)] += update_weights_step(r, a, w, s, env)
            s = s_prime
            a = a_prime
            print('a: {}'.format(a))
            step += 1

    # show_plot(w, env)


if __name__ == '__main__':
    main()
