import gym
from gym import spaces
import numpy as np
from typing import Union, Dict, AnyStr, Optional, Tuple
from metadrive.envs.marl_envs.marl_tintersection import MultiAgentTIntersectionEnv

class HighLevelControllerEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple(spaces.Discrete(2), spaces.Discrete(2)) # Ask Peide: Does this have to be a numpy array?
        self.observation_space =  spaces.Box(low = -100, high = 100, size = (12,3)) # needs to be size = (12,3) numpy array or maybe a tuple

        
    @property 
    def LowLevelControllerEnv(self, MultiAgentTIntersectionEnv):
        pass

    def step(self, action):
        # give high level action to MultiAgentTIntersectionEnv
        o,r,i,d = self.LowLevelControllerEnv.step(action)
        return o, r, i, d

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

if __name__ == "__main__":
    env = HighLevelControllerEnv
    print((env.observation_space).shape)