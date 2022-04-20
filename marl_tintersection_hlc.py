from concurrent.futures.process import _ThreadWakeup
import gym
from gym import spaces
import numpy as np
from typing import Union, Dict, AnyStr, Optional, Tuple
from metadrive.envs.marl_envs.marl_tintersection import MultiAgentTIntersectionEnv

class HighLevelControllerEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2))) # Ask Peide: Does this have to be a numpy array?
        self.observation_space = spaces.Box(low = -100, high = 100, shape= (12,3), dtype = np.float16) # needs to be size = (12,3) numpy array or maybe a tuple

        
    @property 
    def LowLevelControllerEnv(self, MultiAgentTIntersectionEnv):
        pass

    def step(self, action):
        # give high level action to MultiAgentTIntersectionEnv
        o, r, d_agents, _ = self.LowLevelControllerEnv.step(action)

        # d will have done information for each agent

        if (d_agents[AVs_index] == True).all(): #AVs exited 
            d = True
        else:
            d = False

        # we do not necessarily need info, it may be an empty dict
        i = {}

        return o, r, d, i

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

if __name__ == "__main__":
    env = HighLevelControllerEnv()
    print((env.observation_space).shape)