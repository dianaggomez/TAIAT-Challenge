import metadrive
import gym
import numpy as np

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

#environment
env = gym.make("MARLIntersectionHLC-v0")

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo2_hlc_reward_changed")

del model # remove to demonstrate saving and loading

model = PPO2.load("ppo2_hlc_reward_changed")

# Enjoy trained agent
obs = env.reset()
print("First Queue: ", env.queue)
time_step = 0.2

time_to_exit = {}
for i in range(2000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        # save the data dic: {queue: [AV_time, Human_time, max(AV_time, Human_time)]}
        queue = env.queue
        print("Queue: ",queue)
        time_to_exit[queue] = [env.AVs_time*time_step, env.Humans_time*time_step, max(env.AVs_time, env.Humans_time)*time_step]
        obs = env.reset()
print("number of queues recorded: ", len(time_to_exit))
np.save("time_data_6", time_to_exit)

env.reset()
done = False
action = (1,1)
while not done:
    o, r, done, i = env.step(action)
print("Baseline for One-and-One: ", max(env.AVs_time, env.Humans_time)*time_step, "sec")

