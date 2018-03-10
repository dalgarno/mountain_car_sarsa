#!/usr/bin/env python
import random
import numpy as np
import matplotlib as mpl
from sarsa_agent import SARSAAgent
mpl.use('TkAgg')
import gym
from gym import wrappers

from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


random.seed(0)

my_var = 42



def show_plot(agent):
    resolution = 200
    xs = np.linspace(agent.min_x, agent.max_x, resolution)
    ys = np.linspace(agent.min_v, agent.max_v, resolution)

    zs = np.array([-agent.state_action_values([x, y], 
            agent.select_action([x, y], find_max=True)) 
            for x in xs for y in ys]).reshape((resolution, resolution))
    xs, ys = np.meshgrid(xs, ys)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(xs, ys, zs, cmap=plt.cm.viridis,
                           linewidth=0.2, antialiased=False)

    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.text2D(0.05, 0.95, 'Episode {}'.format(
        agent.num_episodes), transform=ax.transAxes)

    plt.show()

def main():
    weights = np.zeros(4096)
    env = gym.make('MountainCar-v0')
    env = wrappers.Monitor(env, '/tmp/mountain_car', force=True)
    env._max_episode_steps = 1000
    agent = SARSAAgent(env, weights, eps=0.0)
    num_episodes = 500
    agent.train(num_episodes)

    show_plot(agent)

if __name__ == '__main__':
    main()
