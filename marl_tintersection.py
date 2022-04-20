import copy
from re import S
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.manager.map_manager import MapManager
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.t_intersection import TInterSection
from metadrive.component.map.pg_map import PGMap
from metadrive.component.road_network.road import Road
from metadrive.component.vehicle.base_vehicle import BaseVehicle
# from metadrive.envs.marl_envs.marl_inout_roundabout import LidarStateObservationMARound
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.envs.marl_envs.tinyinter import MixedIDMAgentManager 
from metadrive.envs import MetaDriveEnv
from metadrive.obs.observation_base import ObservationBase
from metadrive.utils import get_np_random, Config
from metadrive.component.vehicle.vehicle_type import SVehicle, XLVehicle
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.policy.idm_policy import WaymoIDMPolicy
import numpy as np
from collections import deque
import random

coalition = {"agent0": 0, "agent1": 0, "agent2": 1, "agent3": 0,"agent4": 1, "agent5": 0,"agent6": 0,"agent7": 1, "agent8": 1,"agent9": 1,"agent10": 0,"agent11": 1}

MATIntersectionConfig = dict(
    spawn_roads=[
        Road(FirstPGBlock.NODE_2, FirstPGBlock.NODE_3),
        # -Road(TInterSection.node(1, 0, 0), TInterSection.node(1, 0, 1)),
        -Road(TInterSection.node(1, 1, 0), TInterSection.node(1, 1, 1)),
        # -Road(TInterSection.node(1, 2, 0), TInterSection.node(1, 2, 1)),
    ],
    num_agents=12,
    num_RL_agents = 6,
    map_config=dict(exit_length=60, lane_num=1),
    top_down_camera_initial_x=80,
    top_down_camera_initial_y=0,
    top_down_camera_initial_z=120,
    # IDM_agent = True,
    # Whether to remove dead vehicles immediately
    ignore_delay_done=True,

    # The target speed of IDM agents, if any
    target_speed=10,
)


class MATIntersectionMap(PGMap):
    def _generate(self):
        length = self.config["exit_length"]

        parent_node_path, physics_world = self.engine.worldNP, self.engine.physics_world
        assert len(self.road_network.graph) == 0, "These Map is not empty, please create a new map to read config"

        # Build a first-block
        last_block = FirstPGBlock(
            self.road_network,
            self.config[self.LANE_WIDTH],
            self.config[self.LANE_NUM],
            parent_node_path,
            physics_world,
            length=length
        )
        self.blocks.append(last_block)

        # Build TIntersection
        TInterSection.EXIT_PART_LENGTH = length
        last_block = TInterSection(
            1, last_block.get_socket(index=0), self.road_network, random_seed=1, ignore_intersection_checking=False
        )

        # last_block.add_u_turn(True)
        last_block.construct_block(parent_node_path, physics_world)
        self.blocks.append(last_block)



class MATIntersectionSpawnManager(SpawnManager):
    def update_destination_for(self, agent_id, vehicle_config):
        end_roads = copy.deepcopy(self.engine.global_config["spawn_roads"])
        end_road = -self.np_random.choice(end_roads)  # Use negative road!
        # vehicle_config["destination_node"] = end_road.end_node
        vehicle_config["destination"] = TInterSection.node(1, 2, 1)
        return vehicle_config


class MATIntersectionMapManager(MapManager):
    def reset(self):
        config = self.engine.global_config
        if len(self.spawned_objects) == 0:
            _map = self.spawn_object(MATIntersectionMap, map_config=config["map_config"], random_seed=None)
        else:
            assert len(self.spawned_objects) == 1, "It is supposed to contain one map in this manager"
            _map = self.spawned_objects.values()[0]
        self.load_map(_map)
        self.current_map.spawn_roads = config["spawn_roads"]


class MultiAgentTIntersectionEnv(MultiAgentMetaDrive):
    STEERING_INCREMENT = 0.04
    STEERING_DECAY = 0.25

    THROTTLE_INCREMENT = 0.1
    THROTTLE_DECAY = 0.2

    BRAKE_INCREMENT = 0.5
    BRAKE_DECAY = 0.5

    def __init__(self):
        self.vehicle_order = [0,1,2,3,4,5,11,10,9,8,7,6]
        # steering and throttle for all agents 
        self.agents_steering = []
        self.agents_throttle = []
        self.left_queue, self.right_queue = self.generate_queue()
        self.left_vehicle_queue, self.right_vehicle_queue = self.generate_vehicle_queue()

    @staticmethod
    def default_config() -> Config:
        return MultiAgentMetaDrive.default_config().update(MATIntersectionConfig, allow_add_new_key=True)

    # def get_single_observation(self, vehicle_config: "Config") -> "ObservationBase":
    #     return LidarStateObservationMARound(vehicle_config)
    
    #################################################################################33
    def setup_engine(self):
        super(MultiAgentTIntersectionEnv, self).setup_engine()
        self.engine.update_manager("map_manager", MATIntersectionMapManager())
        self.engine.update_manager("spawn_manager", MATIntersectionSpawnManager())

    def _get_reset_return(self):
        org = super(MultiAgentTIntersectionEnv, self)._get_reset_return()
        if self.num_RL_agents == self.num_agents:
            return org

        return self.agent_manager.filter_RL_agents(org)
    #################################################################################33
    def generate_queue(self):
        queue = [1]*6 + [2]*6
        random.shuffle(queue)
        left_queue = deque(queue[:len(queue)//2])
        right_queue = deque(queue[len(queue)//2:])
        return left_queue, right_queue
    
    # generate the queue above but with vehicles objects 
    def generate_vehicle_queue(self):
        all_vehicles = list(self.vehicles.items()).reverse()
        left_vehicle_queue = deque([v for v in all_vehicles[6:]]) # 5 ~ 0 (front of queue at index 0)
        right_vehicle_queue = deque([v for v in all_vehicles[:6]]) # 11 ~ 6
        return left_vehicle_queue, right_vehicle_queue

    def process_high_level_action(self, high_level_action):
        # high-level actions: (0,0), (1,0), (0,1), (1,1)
        # human :1, coatilition: 2
        front_of_queue = (self.left_queue[0], self.right_queue[0])
        # Need to define front_of_queue
        if high_level_action == (0, 0):
            if front_of_queue == (2,2):
                pass
            elif front_of_queue == (1,2):
                high_level_action = (1,0)
                num_of_vehicles = np.array([1,0])
            elif front_of_queue == (2,1):
                high_level_action = (0,1)
                num_of_vehicles = np.array([0,1])
            else: # proceed to Social Rule
                high_level_action == (1,1)
                num_of_vehicles = np.array([1,1])
        elif high_level_action == (0, 1):
            if front_of_queue == (2,2):
                num_of_vehicles =  self.num_of_vehicles_to_exit(high_level_action)
            elif front_of_queue == (2,1):
                high_level_action = (0,1)
                num_of_vehicles = np.array([0,1])
            else:# proceed to Social Rule
                high_level_action = (1,1)
                num_of_vehicles = np.array([1,1])
        elif high_level_action == (1,0):
            if front_of_queue == (2,2):
                num_of_vehicles = self.num_of_vehicles_to_exit(high_level_action)
            elif front_of_queue == (1,2):
                high_level_action = (1,0)
                num_of_vehicles = np.array([1,0])
            else: #proceed to Social Rule
                high_level_action = (1,1)
                num_of_vehicles = np.array([1,1])
        elif high_level_action == (1,1):
            num_of_vehicles = np.array([1,1])

        return high_level_action, num_of_vehicles

    def num_of_vehicles_to_exit(self, high_level_action):
        # check the side of the queue with go, up to 3 vehicles may exit
        counter = 0
        if high_level_action == (0,1):
            if len(self.right_queue) > 0:
                for i in range(min(len(self.right_queue)),3):
                    if self.right_queue[i] == 2:
                        counter+=1
                    else:
                        break
            num_of_vehicles = np.rray([0,counter])
        elif high_level_action == (1,0):
            if len(self.left_queue) > 0:
                for i in range(min(len(self.left_queue)),3):
                    if self.left_queue[i] == 2:
                        counter+=1
                    else:
                        break
            num_of_vehicles = np.array([counter,0])

        return num_of_vehicles
        
    def process_input(self, low_level_action):
        steering = 0.
        throttle_brake = 0.
        # forward, turnLeft, turnRight
        if low_level_action == "forward":
            throttle_brake = 1.0
        elif low_level_action == 'turnLeft':
            steering = 1.0
        elif low_level_action == 'turnRight':
            steering = -1.0

        # call further process before 
        self.further_process(steering, throttle_brake)

        return np.array([self.steering, self.throttle_brake], dtype=np.float64)

    def further_process(self, steering, throttle_brake):
        if steering == 0.:
            if self.steering > 0.:
                self.steering -= self.STEERING_DECAY
                self.steering = max(0., self.steering)
            elif self.steering < 0.:
                self.steering += self.STEERING_DECAY
                self.steering = min(0., self.steering)
        if throttle_brake == 0.:
            if self.throttle_brake > 0.:
                self.throttle_brake -= self.THROTTLE_DECAY
                self.throttle_brake = max(self.throttle_brake, 0.)
            elif self.throttle_brake < 0.:
                self.throttle_brake += self.BRAKE_DECAY
                self.throttle_brake = min(0., self.throttle_brake)

        if steering > 0.:
            self.steering += self.STEERING_INCREMENT if self.steering > 0. else self.STEERING_DECAY
        elif steering < 0.:
            self.steering -= self.STEERING_INCREMENT if self.steering < 0. else self.STEERING_DECAY

        if throttle_brake > 0.:
            self.throttle_brake = max(self.throttle_brake, 0.)
            self.throttle_brake += self.THROTTLE_INCREMENT
        elif throttle_brake < 0.:
            self.throttle_brake = min(self.throttle_brake, 0.)
            self.throttle_brake -= self.BRAKE_INCREMENT

        rand = self.np_random.rand(2, 1) / 10000
        # self.throttle_brake += rand[0]
        self.steering += rand[1]

        self.throttle_brake = min(max(-1., self.throttle_brake), 1.)
        self.steering = min(max(-1., self.steering), 1.)

    def within_box_range(self, vehicle, side, check_turn_complete = False) -> bool:
        x,y = vehicle.position()
        # hyperparameter
        radius = 5 
        # location of checkpoints
        left = [60., 0.] 
        right = [83.5, -3.5] 
        top = [73.5, -13.5]

        if check_turn_complete:
            x_checkpoint, y_checkpoint = top
        elif side == "left": # vehicle on left side
            x_checkpoint, y_checkpoint = left
        else:
            x_checkpoint, y_checkpoint = right

        if np.abs(x - x_checkpoint) < radius and np.abs(y - y_checkpoint) < radius:
            return True
        else:
            return False

    def vehicle_take_action(self, vehicle, action):
        if vehicle.increment_steering: # an attribute from base vehicle class (line 147 of baseVehicle)
            vehicle._set_incremental_action(action)
        else:
            vehicle._set_action(action)

    def vehicle_is_turning(self, vehicle) -> bool:
        # check whether the vehicle has already passed 1st checkpoint
        # but are not yet at the 2nd checkpoint
        left = [60., 0.] 
        right = [83.5, -3.5] 
        top = [73.5, -13.5]

        x,y = vehicle.position()

        if ((left[0] <= x <= top[0]) or (top[0] <= x <= right[0])) and (left[1] <= y <= top[1]):
            return True

    def vehicle_exit_check(self, vehicle) -> bool:
        if self.vehicle_is_turning:
            # is turning means the vehicle is between 1st and 2nd checkpoint
            if self.within_box_range(vehicle, check_turn_complete=True):
                return True
            else:
                return False
        else:
            return False

    def take_step(self, high_level_action, exited, num_of_vehicles):
        vehicles_to_exit = np.abs(exited - num_of_vehicles)
        # need to pass through the vehicle that we are taking a step for
        vehicle_remaining_in_queues = np.abs(vehicles_to_exit - np.array([len(self.left_vehicle_queue),len(self.right_vehicle_queue)]))

        # first we check the high_level actions
        if high_level_action == (0,1):
            side = 'right'# only right queue 
            for i in range(vehicles_to_exit[1]):
                vehicle = self.right_vehicle_queue[i] # initialized in generate_vehicle_queue
                if self.within_box_range(vehicle, side) or self.vehicle_is_turning(vehicle):
                    # when a vehicle reaches 1st checkpoint or is in the process of turning
                    low_level_action = self.process_input('turnRight')
                    self.vehicle_take_action(vehicle, low_level_action)
                else: 
                    low_level_action = self.process_input('forward')
                    self.vehicle_take_action(vehicle, low_level_action)
            
            # rest of the vehicles should take a step 
            for i in range(vehicles_to_exit[1], vehicle_remaining_in_queues[1]):
                vehicle = self.right_vehicle_queue[i]
                low_level_action = self.process_input('forward')
                self.vehicle_take_action(vehicle, low_level_action)
        elif high_level_action == (1,0):
            side = 'left' # only left_queue moves
            for i in range(vehicles_to_exit[0]):
                vehicle = self.left_vehicle_queue[i]
                if self.within_box_range(vehicle, side) or self.vehicle_is_turning(vehicle):
                    low_level_action = self.process_input('turnLeft')
                    self.vehicle_take_action(vehicle, low_level_action)
                else: 
                    low_level_action = self.process_input('forward')
                    self.vehicle_take_action(vehicle, low_level_action)
            # rest of the vehicles should take a step forward
            for i in range(vehicles_to_exit[0], vehicle_remaining_in_queues[0]):
                vehicle = self.left_vehicle_queue[i]
                low_level_action = self.process_input('forward')
                self.vehicle_take_action(vehicle, low_level_action)
        else:
            # both move
            front_vehicles = [self.right_vehicle_queue[0], self.left_vehicle_queue[0]] # the two vehicles at front
            for i in range(2): 
                # right side should go first and then left, if they collide we will offset the initial step for the left side
                if i == 0:
                    side = "right"
                else:
                    side = "left"

                if (not self.within_box_range(front_vehicles[i]), side) and (not self.vehicle_is_turning(front_vehicles[i])):
                    low_level_action = self.process_input('forward')
                    self.vehicle_take_action(vehicle, low_level_action)
                elif i == 0: # right side
                    low_level_action = self.process_input("turnRight")
                    self.vehicle_take_action(vehicle, low_level_action)
                     # rest of the vehicles should take a step forward
                    for j in range(vehicles_to_exit[1], vehicle_remaining_in_queues[1]):
                        vehicle = self.right_vehicle_queue[j]
                        low_level_action = self.process_input('forward')
                        self.vehicle_take_action(vehicle, low_level_action)
                else:
                    low_level_action = self.process_input("turnLeft")
                    self.vehicle_take_action(vehicle, low_level_action)
                    # rest of the vehicles should take a step forward
                    for k in range(vehicles_to_exit[0], vehicle_remaining_in_queues[0]):
                        vehicle = self.left_vehicle_queue[k]
                        low_level_action = self.process_input('forward')
                        self.vehicle_take_action(vehicle, low_level_action)
           

        # then we check if the front of queue (both left and right) have completed turning
        # if so, remove them from the queues and increment num_vehicles_exited
        if self.vehicle_exit_check(self.left_vehicle_queue[0]):
            self.left_vehicle_queue.remove(self.left_vehicle_queue[0])
            exited += np.array([1,0])
        if self.vehicle_exit_check(self.right_vehicle_queue[0]):
            self.left_vehicle_queue.remove(self.right_vehicle_queue[0])
            exited += np.array([0,1])

        return exited

    def _get_reward(self, high_level_action, num_of_vehicles, done, fairness = False):
        # Reward
        if (done[human_index] == True).all(): # r = -2, if human drivers exit first
            r = -2
        elif (done[AV_index] == True).all(): # r = +1, if AVs exit first
            r = 1
        else: # r = -1 for each high level decision
            r = -1
        
        # Reward Shaping: Fairness Reward
        if fairness:
            N_SR = 6
            N = 12
            # v_pi: the no. of vehicles exiting from the AV coalition at each high_level_decision
            if high_level_action == (0,0):
                v_pi = 0.75 # to avoid 0
            elif high_level_action == (0,1):
                v_pi = num_of_vehicles[1]
            elif high_level_action == (1,0):
                v_pi = num_of_vehicles[0]
            else:
                # need to check if the vehicles that exited belong to the coaltion or human drivers
                pass
            r_f = (N_SR/N) * (1/v_pi)
        else:
            r_f = 0

        reward = r + r_f
        return reward

    def step(self, actions):
        exited = np.array([0,0])

        ########## High-Level Step Function (Discrete) #########
        high_level_action, num_of_vehicles = self.process_high_level_action(actions)

        ######## Low-Level Step Function (Continuous) ##########
        if high_level_action == (0,0):
            pass
        else:
            while exited != num_of_vehicles: 
                exited, actions = self.take_step(high_level_action, num_of_vehicles, exited)

                # WE COULD POTENTIALLY CONTINUES TO FEED OUR ACTIONS INTO THE .step() and ensure the evniroment updates/renders every st
                o, r, d, i = super(MultiAgentTIntersectionEnv, self).step(actions)

        # Update observation, reward, done, and info
         #### New Observation #####
        # Output as numpy array
        # If agent exists, or dies, output [0,0,coalition]
        obs = []
        for num in self.vehicle_order:
            if 'agent{n}'.format(n=num) in o:
                obs.append(o['agent{n}'.format(n=num)])
            else:
                obs.append([0,0, coalition['agent{n}'.format(n=num)]])
        o = np.array(obs)

        r = self._get_reward(high_level_action, num_of_vehicles, d)

        return o, r, d, i

    # def _preprocess_actions(self, actions):
    #     if self.num_RL_agents == self.num_agents:
    #         return super(MultiAgentTIntersectionEnv, self)._preprocess_actions(actions)

    #     actions = {v_id: actions[v_id] for v_id in self.vehicles.keys() if v_id in self.agent_manager.RL_agents}
    #     return actions

    def __init__(self, config=None):
        super(MultiAgentTIntersectionEnv, self).__init__(config=config)
        self.num_RL_agents = self.config["num_RL_agents"]
        if self.num_RL_agents == self.num_agents:  # Not using mixed traffic and only RL agents are running.
            pass
        else:
            self.agent_manager = MixedIDMAgentManager(
                init_observations=self._get_observations(),
                init_action_space=self._get_action_space(),
                num_RL_agents=self.num_RL_agents,
                ignore_delay_done=self.config["ignore_delay_done"],
                target_speed=self.config["target_speed"]
            )
    



def _draw():
    env = MultiAgentTIntersectionEnv()
    o = env.reset()
    from metadrive.utils.draw_top_down_map import draw_top_down_map
    import matplotlib.pyplot as plt

    plt.imshow(draw_top_down_map(env.current_map))
    plt.show()
    env.close()



def _expert():
    env = MultiAgentTIntersectionEnv(
        {
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 240,
                    "num_others": 4,
                    "distance": 50
                },
            },
            "use_AI_protector": True,
            "save_level": 1.,
            "debug_physics_world": True,

            # "use_render": True,
            "debug": True,
            "manual_control": True,
            "num_agents": 4,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        d.update({"total_r": total_r, "episode length": ep_s})
        # env.render(text=d)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis_debug_respawn():
    env = MultiAgentTIntersectionEnv(
        {
            "horizon": 100000,
            "vehicle_config": {
                "lidar": {
                    "num_lasers": 72,
                    "num_others": 0,
                    "distance": 40
                },
                "show_lidar": False,
            },
            "debug_physics_world": True,
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "num_agents": 40,
        }
    )
    o = env.reset()
    total_r = 0
    ep_s = 0
    for i in range(1, 100000):
        action = {k: [0.0, .0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(action)
        for r_ in r.values():
            total_r += r_
        ep_s += 1
        # d.update({"total_r": total_r, "episode length": ep_s})
        render_text = {
            "total_r": total_r,
            "episode length": ep_s,
            "cam_x": env.main_camera.camera_x,
            "cam_y": env.main_camera.camera_y,
            "cam_z": env.main_camera.top_down_camera_height
        }
        env.render(text=render_text)
        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _vis():
    # config = {"traffic_mode": "respawn", "map": "T", "traffic_density": 0.2,}
    # if True:
    #     config.update({"use_render": True, "manual_control": True})
    # env = MetaDriveEnv(config)
    # env.reset(force_seed=0)
    ######
    config = {
            "horizon": 100000,
            "vehicle_config": {
                "vehicle_model": 's',
                "lidar": {
                    "num_lasers": 240, # the more dense, the more accurate but longer computation time
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
            "num_RL_agents" : 0,
        }
    env = MultiAgentTIntersectionEnv(config)
    o = env.reset()
    print("===============================================")
    print("vehicle num", env.vehicles.values(), type(list(env.vehicles.values())[0]))
    print("===============================================")
    # print("RL agent num", len(o))
    #########
    total_r = 0
    ep_s = 0
    # print(env.engine.traffic_manager.get_policy())
    import numpy as np
    # actions = {k: [0.0, 0.0] for k in env.vehicles.keys()}
    # for k in range(12):
    #     actions['agent' + str(k)] = [0.0, 1.0] # we are advancing all the vehicles with max acceleration
    #     actions = env.action_space.sample()

    for i in range(1, 100000):
        # actions = {k: [0.0, 1.0] for k in env.vehicles.keys()}
        # TODO: add randomness
        # for key in actions.keys():
            # if np.random.uniform() < 0.2:
                # actions[key][1] *= -1
        # Vehicles will reach their goal and exit, therefore, we need to iterate over the k vehicles that are present
        actions = env.action_space.sample()
        print(actions)
        # actions = {k: [-0, 1.0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(actions) # the actions are a dictionary corresponding to each vehicles so observation will also be dictiary for each vehicle's observation
        # print("Observation: ", o)
        for r_ in r.values():
            total_r += r_
        # total_r += r
        ep_s += 1

        ####################### Extras that we can render ####################### 
        # d.update({"total_r": total_r, "episode length": ep_s})
        # render_text = {
        #     "total_r": total_r,
        #     "episode length": ep_s,
        #     "cam_x": env.main_camera.camera_x,
        #     "cam_y": env.main_camera.camera_y,
        #     "cam_z": env.main_camera.top_down_camera_height,
        #     "alive": len(env.vehicles)
        # }
        # env.render(text=render_text)
        # env.render(mode="top_down")
        #########################################################################

        if d["__all__"]:
            print(
                "Finish! Current step {}. Group Reward: {}. Average reward: {}".format(
                    i, total_r, total_r / env.agent_manager.next_agent_count
                )
            )
            env.reset()
            # break
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()


def _profile():
    import time
    env = MultiAgentTIntersectionEnv({"num_agents": 16})
    obs = env.reset()
    start = time.time()
    for s in range(10000):
        o, r, d, i = env.step(env.action_space.sample())

        # mask_ratio = env.engine.detector_mask.get_mask_ratio()
        # print("Mask ratio: ", mask_ratio)

        if all(d.values()):
            env.reset()
        if (s + 1) % 100 == 0:
            print(
                "Finish {}/10000 simulation steps. Time elapse: {:.4f}. Average FPS: {:.4f}".format(
                    s + 1,
                    time.time() - start, (s + 1) / (time.time() - start)
                )
            )
    print(f"(MATIntersection) Total Time Elapse: {time.time() - start}")


def _long_run():
    # Please refer to test_ma_TIntersection_reward_done_alignment()
    _out_of_road_penalty = 3
    env = MultiAgentTIntersectionEnv(
        {
            "num_agents": 32,
            "vehicle_config": {
                "lidar": {
                    "num_others": 8
                }
            },
            **dict(
                out_of_road_penalty=_out_of_road_penalty,
                crash_vehicle_penalty=1.333,
                crash_object_penalty=11,
                crash_vehicle_cost=13,
                crash_object_cost=17,
                out_of_road_cost=19,
            )
        }
    )
    try:
        obs = env.reset()
        assert env.observation_space.contains(obs)
        for step in range(10000):
            act = env.action_space.sample()
            o, r, d, i = env.step(act)
            if step == 0:
                assert not any(d.values())

            if any(d.values()):
                print("Current Done: {}\nReward: {}".format(d, r))
                for kkk, ddd in d.items():
                    if ddd and kkk != "__all__":
                        print("Info {}: {}\n".format(kkk, i[kkk]))
                print("\n")

            for kkk, rrr in r.items():
                if rrr == -_out_of_road_penalty:
                    assert d[kkk]

            if (step + 1) % 200 == 0:
                print(
                    "{}/{} Agents: {} {}\nO: {}\nR: {}\nD: {}\nI: {}\n\n".format(
                        step + 1, 10000, len(env.vehicles), list(env.vehicles.keys()),
                        {k: (oo.shape, oo.mean(), oo.min(), oo.max())
                         for k, oo in o.items()}, r, d, i
                    )
                )
            if d["__all__"]:
                print('Current step: ', step)
                break
    finally:
        env.close()


def show_map_and_traj():
    import matplotlib.pyplot as plt
    from metadrive.obs.top_down_renderer import draw_top_down_map, draw_top_down_trajectory
    import json
    import cv2
    import pygame
    env = MultiAgentTIntersectionEnv()
    env.reset()
    with open("metasvodist_inter_best.json", "r") as f:
        traj = json.load(f)
    m = draw_top_down_map(env.current_map, simple_draw=False, return_surface=True, reverse_color=True)
    m = draw_top_down_trajectory(
        m, traj, entry_differ_color=True, color_list=[(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    )
    ret = cv2.resize(pygame.surfarray.pixels_red(m), (512, 512), interpolation=cv2.INTER_LINEAR)
    #
    plt.imshow(ret)
    plt.show()
    pygame.image.save(m, "image.png")
    env.close()


if __name__ == "__main__":
    # _draw()
    _vis()
    # _vis_debug_respawn()
    # _profiwdle()
    # _long_run()
    # show_map_and_traj()
    # pygame_replay("parking", MultiAgentParkingLotEnv, False, other_traj="metasvodist_parking_best.json")
    # panda_replay(
    #     "parking",
    #     MultiAgentTIntersectionEnv,
    #     False,
    #     other_traj="metasvodist_inter.json",
    #     extra_config={
    #         "global_light": True
    #     }
    # )
    # pygame_replay()