# TAIAT-Challenge

This project is based off the META-drive repo: https://github.com/decisionforce/metadrive

Place marl_tintersection.py under metadrive/metadrive/envs/marl_envs
Place spawn_manager.py under metadrive/metadrive/manager (replace the original)
Place state_obs.py under obs (replace the original)
To run the T-intersection env: python -m metadrive.envs.marl_envs.marl_tintersection 

## Structure 
HighLevelController is its own environment with the MetaDrive that we have been working on as a propoerty

## Checklist
* Change observation space:
  * obs = {agent# : [x,y]} with length 12
  * Q: Does coaltion need to be part of the observation or can it be part of the agent's name?
* wrap the env.step() to simulate a traffic light
  * actions : [0:stop, 1:go]
* Implement "controller" for human vehicles (they should know how to drive)
  * look into pid or idm 
* train PPO 

## Checkpoint locations
1. [60., 0.] (left)
2. [73.5, -13.499999999999996] (center)
3. [83.5, -3.5] (right)
