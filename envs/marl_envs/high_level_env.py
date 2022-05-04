import numpy as np
import gym

from metadrive.envs.marl_envs.marl_tintersection import MultiAgentTIntersectionEnv 


class high_level_env():
    def __init__(self):
        super(high_level_env, self).__init__()
        # Define a 3-D observation space
        self.high_level_observation_shape = (12, 1) #array.shape[row 12 to col 1]
        self.high_level_observation_space = gym.spaces.Box(x = np.zeros(self.observation_shape), 
                                                y = np.zeros(self.observation_shape),
                                                coalition = np.zeros(self.observation_shape),
                                                dtype = np.float16)

        # Define an action space: 4 high level actions we have (0,0) (0,1) (1,0) (1,1)
        self.high_level_action_space = gym.spaces.Discrete(4)

    def step(self, high_level_actions): 
        # r, d, i are missing for now
        o, r, d, i = super(MultiAgentTIntersectionEnv, self).step(high_level_actions)
        # Update information
        self.high_level_observation_space = o
        return o, r, d, i


    def get_high_level_action_space(self):
        return self.high_level_action_space
    
    def get_high_level_observation_space(self):
        return self.high_level_observation_space

    def _next_observation(self):
        return self.get_high_level_observation_space()

    def reset(self):
        # Reset the state of the environment to an initial state
        obs = self._next_observation()
        return obs

    def close(self):
        pass

if __name__ == "__main__":
    env = high_level_env()
    o = env.reset()

    for i in range(1, 100000):
        o, r, d, info = env.step((0,0))
    env.close()
