# TAIAT-Challenge

This project is based off the META-drive repo: https://github.com/decisionforce/metadrive

Place marl_tintersection.py under metadrive/metadrive/envs/marl_envs and place spawn_manager.py under metadrive/metadrive/manager (replace the original)
To run the T-intersection env: python -m metadrive.envs.marl_envs.marl_tintersection 


## Checklist
* Change observation space:
  *len(obs) = 12 
 * each space may be empty, coalition 1 vehicle, or coalition 2 vehicle
 * need position for each vehicle, may also need velocity
* wrap the env.step() to simulate a traffic light
 * actions : [0:stop, 1:go]
* Implement "controller" for human vehicles (they should know how to drive)
 * look into pid or idm 
* train PPO 
