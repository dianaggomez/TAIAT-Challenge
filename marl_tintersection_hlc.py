import gym
from gym import spaces
import numpy as np
from typing import Union, Dict, AnyStr, Optional, Tuple
from metadrive.envs.marl_envs.marl_tintersection import MultiAgentTIntersectionEnv
from itertools import compress 

class HighLevelControllerEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(2))) # Ask Peide: Does this have to be a numpy array?
        self.observation_space = spaces.Box(low = -100, high = 100, shape= (12,3), dtype = np.float16) # needs to be size = (12,3) numpy array or maybe a tuple
        self.LowLevelControllerEnv = self.initialize_LowLevelControllerEnv()

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
            "use_render": True,
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
        return environment

    def done(self, d_agents):
        print("done of agents", d_agents)
        left_array = np.array(self.LowLevelControllerEnv.orig_left_queue)
        right_array = np.array(self.LowLevelControllerEnv.orig_right_queue)

        vehicle_order = self.LowLevelControllerEnv.vehicle_order # [0,1,2,3,4,5,11,10,9,8,7,6]
        all_vehicle_location = np.concatenate((list(left_array == 2), list(right_array == 2))) # [true, false ....]
        AVs_index = list(compress(vehicle_order, all_vehicle_location)) # get AV index from all vehciles
        print("Conditional ", np.array(list(d_agents.values()))[AVs_index].all())
        if (np.array(list(d_agents.values()))[AVs_index]).all():
            return True
        else:
            return False

    def step(self, action):
        # give high level action to MultiAgentTIntersectionEnv
        o, r, d_agents, _ = self.LowLevelControllerEnv.step(action)
        # d will have done information for each agent
        d = self.done(d_agents)
        print("done ", d)
        # we do not necessarily need info, it may be an empty dict
        i = {}
        # get reward
        r = self.LowLevelControllerEnv

        return o, r, d, i

    def reset(self):
        self.LowLevelControllerEnv.reset()

    def render(self):
        self.LowLevelControllerEnv.render()      

    def close(self):
        self.LowLevelControllerEnv.close()

    def seed(self):
        pass

if __name__ == "__main__":
    env = HighLevelControllerEnv()
    # print("Inital Queue: ", env.LowLevelControllerEnv.visualize_queue())
    # env.LowLevelControllerEnv.bring_vehicles_to_front()
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

