#!/usr/bin/env python
import gym
from tile_coding import IHT, tiles

def main():
    env = gym.make('MountainCar-v0')
    env.reset()
    iht = IHT(4096)
    for _ in range(100):
        env.render()
        a = env.action_space.sample()
        obs, reward, done, infor = env.step(a) # take a random action
        x, x_dot = obs
        feature_vec = tiles(iht, 8, [8*x/(0.5+1.2), 8*x_dot/(0.07+0.07)], [-1, 0, 1])
        print(feature_vec)
        print(obs)

if __name__ == '__main__':
    main()
