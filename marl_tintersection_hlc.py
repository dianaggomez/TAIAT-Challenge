import gym
from gym import spaces
import numpy as np
from typing import Union, Dict, AnyStr, Optional, Tuple
from metadrive.envs.marl_envs.marl_tintersection import MultiAgentTIntersectionEnv
from itertools import compress 

class HighLevelControllerEnv(gym.Env):
    def __init__(self,config=None):
        # self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2))) # Ask Peide: Does this have to be a numpy array?
        self.action_space = spaces.MultiBinary(2)
        self.observation_space = spaces.Box(low = -100, high = 100, shape= (12,3), dtype = np.float16) # needs to be size = (12,3) numpy array or maybe a tuple
        self.LowLevelControllerEnv = self.initialize_LowLevelControllerEnv()
        self.queue = ''
        self.time = 0
        self.AVs_done = False
        self.Humans_done = False
        self.AVs_time = 0
        self.Humans_time = 0
        self.queue = self.LowLevelControllerEnv.queue_config

    def initialize_LowLevelControllerEnv(self):
        config = {
            "horizon": 100000,
            "vehicle_config": {
                "vehicle_model": 's',
                "lidar": {
                    "num_lasers": 0, # the more dense, the more accurate but longer computation time
                    "num_others": 4,
                    "distance": 40
                },
                "show_lidar": False,
            },
            # "target_vehicle_configs": {
            #     "vehicle_model": "default",  "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 1), 
            # },
            #
            "use_render": False,
            "debug": False,
            "allow_respawn": False,
            "manual_control": True,
            "num_agents": 12,
            "delay_done": 0,
            "num_RL_agents" : 12 # no one has IDM
        }
        environment = MultiAgentTIntersectionEnv(config)
        environment.reset()
        vehicles = environment.vehicles
        # print("Vehicles!!!! ", environment.vehicles)
        environment.generate_vehicle_queue(vehicles)
        environment.assign_idm_policy()
        environment.bring_vehicles_to_front()
        return environment

    # # DONE FUNCTION FOR TIME DATA 
    # def done(self):
    #     self.time += self.LowLevelControllerEnv.time
    #     # print("DONE DICT: ", self.LowLevelControllerEnv._done)
    #     AV_index = self.LowLevelControllerEnv.AV_index
    #     Human_index = self.LowLevelControllerEnv.human_index
    #     # return np.array(self.LowLevelControllerEnv._done)[AV_index].all()
    #     if np.array(self.LowLevelControllerEnv._done)[AV_index].all() and not self.AVs_done:
    #         self.AVs_time = self.time
    #         self.AVs_done = True
    #     if np.array(self.LowLevelControllerEnv._done)[Human_index].all() and not self.Humans_done:
    #         self.Humans_time = self.time
    #         self.Humans_done = True
    #     return np.array(self.LowLevelControllerEnv._done).all()
    
    # DONE FUNCTION FOR TRAINING
    def done(self):
        # print("DONE DICT: ", self.LowLevelControllerEnv._done)
        AV_index = self.LowLevelControllerEnv.AV_index
        return  np.array(self.LowLevelControllerEnv._done)[AV_index].all()

    def step(self, action):
        action = (action[0], action[1]) # convert numpy array to tuple
        # give high level action to MultiAgentTIntersectionEnv
        o, r, d_agents, _ = self.LowLevelControllerEnv.step(action)
        # d will have done information for each agent
        d = self.done()
        # print("done ", d)
        # we do not necessarily need info, it may be an empty dict
        i = {}
        # get reward
        r = self.LowLevelControllerEnv.reward

        return o, r, d, i

    def reset(self):
        obs = self.LowLevelControllerEnv.reset()
        vehicles = self.LowLevelControllerEnv.vehicles
        # print("Vehicles!!!! ", environment.vehicles)
        self.LowLevelControllerEnv.generate_vehicle_queue(vehicles)
        self.LowLevelControllerEnv.assign_idm_policy()
        self.LowLevelControllerEnv.bring_vehicles_to_front()
        self.queue = str(list(self.LowLevelControllerEnv.left_queue) + list(self.LowLevelControllerEnv.right_queue))
        self.time = 0
        self.AVs_done = False
        self.Humans_done = False
        self.AVs_time = 0
        self.Humans_time = 0
        self.queue_config = self.LowLevelControllerEnv.queue_config
        return obs

    def render(self):
        self.LowLevelControllerEnv.render()      

    def close(self):
        self.LowLevelControllerEnv.close()

    def seed(self):
        pass

if __name__ == "__main__":
    env = HighLevelControllerEnv()
    # print("Inital Queue: ", env.LowLevelControllerEnv.visualize_queue())
    # print((env.observation_space).shape)
    # action = env.action_space.sample()
    # action = (1, 0)
    # print("Original High Level Action", action)
    # env.step(action)
    # # action = (1,0)
    # # env.step(action)
    # action = (0,1)
    # print("Original High Level Action", action)
    # env.step(action)
    # action = (0,1)
    # print("Original High Level Action", action)
    # env.step(action)
    done = False
    while done == False:
        action = env.action_space.sample()
        print("High Level Action ", action)
        o, r, done, i = env.step(action)  
        print("###########################################################")
        print("Obervation: ", o)
        print("Reward: ", r)
        print("Timesteps", env.LowLevelControllerEnv.time)
        print("###########################################################")

    env.reset()
    done = False
    while done == False:
        action = env.action_space.sample()
        print("High Level Action ", action)
        o, r, done, i = env.step(action)  
        print("###########################################################")
        print("Obervation: ", o)
        print("Reward: ", r)
        print("Timesteps", env.LowLevelControllerEnv.time)
        print("###########################################################")    