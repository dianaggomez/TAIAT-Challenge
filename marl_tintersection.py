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
from itertools import compress 

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
        self.vehicle_order = [0,1,2,3,4,5,11,10,9,8,7,6]
        # steering and throttle for all agents 
        self.agents_steering = np.zeros((12,1)) # idx corresponds to agentID
        self.agents_throttle = np.zeros((12,1))
        self.left_queue, self.right_queue = self.generate_queue()
        # self.left_vehicle_queue, self.right_vehicle_queue = self.generate_vehicle_queue()
        self.left_vehicle_queue = deque([])
        self.right_vehicle_queue = deque([])
        self.action_offset = 5 # offset 5 steps
        self.exited_agentID = []
        self.reward = 0

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
        all_vehicles = [v for v in self.vehicles.values()]
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
                print("Imhere")
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
        else:
            high_level_action = (0,0)
            num_of_vehicles = np.array([0,0])

        return high_level_action, num_of_vehicles

    def num_of_vehicles_to_exit(self, high_level_action):
        # check the side of the queue with go, up to 3 vehicles may exit
        print(self.left_queue)
        print(self.right_queue)
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
        
    def process_input(self, low_level_action, agentID):
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
        self.further_process(steering, throttle_brake, agentID)

        # return np.array([self.agents_steering, self.agents_throttle[idx]], dtype=np.float64)

    def further_process(self, steering, throttle_brake, agentID):
        idx = agentID
        if steering == 0.:
            if self.agents_steering[idx] > 0.:
                self.agents_steering[idx] -= self.STEERING_DECAY
                self.agents_steering[idx] = max(0., self.agents_steering[idx])
            elif self.agents_steering[idx] < 0.:
                self.agents_steering[idx] += self.STEERING_DECAY
                self.agents_steering[idx] = min(0., self.agents_steering[idx])
        if throttle_brake == 0.:
            if self.agents_throttle[idx] > 0.:
                self.agents_throttle[idx] -= self.THROTTLE_DECAY
                self.agents_throttle[idx] = max(self.agents_throttle[idx], 0.)
            elif self.agents_throttle[idx] < 0.:
                self.agents_throttle[idx] += self.BRAKE_DECAY
                self.agents_throttle[idx] = min(0., self.agents_throttle[idx])

        if steering > 0.:
            self.agents_steering[idx] += self.STEERING_INCREMENT if self.agents_steering[idx] > 0. else self.STEERING_DECAY
        elif steering < 0.:
            self.agents_steering[idx] -= self.STEERING_INCREMENT if self.agents_steering[idx] < 0. else self.STEERING_DECAY

        if throttle_brake > 0.:
            self.agents_throttle[idx] = max(self.agents_throttle[idx], 0.)
            self.agents_throttle[idx] += self.THROTTLE_INCREMENT
        elif throttle_brake < 0.:
            self.agents_throttle[idx] = min(self.agents_throttle[idx], 0.)
            self.agents_throttle[idx] -= self.BRAKE_INCREMENT

        rand = self.np_random.rand(2, 1) / 10000
        # self.agents_throttle[idx] += rand[0]
        self.agents_steering[idx] += rand[1]

        self.agents_throttle[idx] = min(max(-1., self.agents_throttle[idx]), 1.)
        self.agents_steering[idx] = min(max(-1., self.agents_steering[idx]), 1.)

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

    def get_agentID(self, side, position):
        """
        param position: the position in the current queue, starting
            from the front of the queue (=0)
        EXAMPLE:
        If the length of lef_vehicle_queue is 4, then it has to be 
        agent 0 ~ 3 with 4 & 5 already completed the turn and
        poped out of the queue.
        Then if we want retrieve vehicle at position 2, that means 
        the second vehicle from the front of the quene (which is 
        currently agent 3), i.e. agent 2.
        """
        
        if side == 'left':
            queue_len = len(self.left_vehicle_queue) # this reflects how many vehicles are left in the queue
            idx = (queue_len - 1) - position
            return self.vehicle_order[idx]
        elif side == 'right':
            queue_len = len(self.right_vehicle_queue) # this reflects how many vehicles are left in the queue
            idx = 6 + (queue_len - 1) - position
            return self.vehicle_order[idx]
        else:
            print("Invalid side indicated.\n")
        
    def array2dict_action(self, actions):
        action_dict = {}
        for i in range(12):
            key = 'agent{n}'.format(n=i)
            action_dict[key] = actions[i,:]
        return action_dict
    
    def take_step(self, high_level_action, exited, num_of_vehicles):
        vehicles_to_exit = np.abs(exited - num_of_vehicles)
        # check whether we have enough vehicles left
        if vehicles_to_exit[0] > len(self.left_vehicle_queue):
            vehicles_to_exit[0] = len(self.left_vehicle_queue)
        if vehicles_to_exit[1] > len(self.right_vehicle_queue):
            vehicles_to_exit[1] = len(self.right_vehicle_queue)
        # vehicle_remaining_in_queues = np.abs(vehicles_to_exit - \
        #     np.array([len(self.left_vehicle_queue),len(self.right_vehicle_queue)]))

        # first we check the high_level actions
        ################################# RIGHT ONLY #################################
        if high_level_action == (0,1):
            side = 'right'# only right queue 
            # 1. assign actions for turning vehicles on this side
            for i in range(vehicles_to_exit[1]):
                vehicle = self.right_vehicle_queue[i] # initialized in generate_vehicle_queue
                agentID = self.get_agentID(side, i)
                if self.within_box_range(vehicle, side) or self.vehicle_is_turning(vehicle):
                    # when a vehicle reaches 1st checkpoint or is in the process of turning
                    low_level_action = self.process_input('turnRight', agentID)
                    # self.vehicle_take_action(vehicle, low_level_action)
                else: 
                    low_level_action = self.process_input('forward', agentID)
                    # self.vehicle_take_action(vehicle, low_level_action)
            
            # 2. assign actions for the rest of vehicles on this side
            front_of_remaining_vehicles = self.right_vehicle_queue[vehicles_to_exit[1]]
            for i in range(vehicles_to_exit[1],len(self.right_vehicle_queue)): # i represents position in the queue
                """
                NOTE: Vehicle that are turning would still be in the queue. 
                    They are only popped out when they finish turning. Thus 
                    we need to skip those at the front of the quene.
                EXAMPLE:
                Assume 2 vehicles are exiting and 4 vehicles are not, then 
                vehicles_to_exit = 2 and len(queue) = 6. The rest of 
                the vehicles start with queue[2] and ends with queue[5].
                """
                vehicle = self.right_vehicle_queue[i]
                agentID = self.get_agentID(side, i)
                # if the front of the remaing vehicles is already at the intersection
                if self.within_box_range(front_of_remaining_vehicles, side):
                    self.agents_steering[agentID] = 0.
                    self.agents_throttle[agentID] = 0.
                else:
                    low_level_action = self.process_input('forward', agentID)
                # self.vehicle_take_action(vehicle, low_level_action)
            
            # 3. assign actions for vehicles on the opposite side
            for i in range(len(self.left_vehicle_queue)):
                agentID = self.get_agentID(side='left', agentID=i)
                self.agents_steering[agentID] = 0.
                self.agents_throttle[agentID] = 0.
        
        ################################# LEFT ONLY #################################
        elif high_level_action == (1,0):
            side = 'left' # only left_queue moves
            # 1. assign actions for turning vehicles on this side
            for i in range(vehicles_to_exit[0]):
                vehicle = self.left_vehicle_queue[i]
                agentID = self.get_agentID(side, i)
                if self.within_box_range(vehicle, side) or self.vehicle_is_turning(vehicle):
                    low_level_action = self.process_input('turnLeft', agentID)
                    # self.vehicle_take_action(vehicle, low_level_action)
                else: 
                    low_level_action = self.process_input('forward', agentID)
                    # self.vehicle_take_action(vehicle, low_level_action)
                
            # 2. assign actions for the rest of vehicles on this side
            front_of_remaining_vehicles = self.left_vehicle_queue[vehicles_to_exit[0]]
            for i in range(vehicles_to_exit[0],len(self.left_vehicle_queue)): # i represents position in the queue
                vehicle = self.left_vehicle_queue[i]
                agentID = self.get_agentID(side, i)
                # if the front of the remaing vehicles is already at the intersection
                if self.within_box_range(front_of_remaining_vehicles, side):
                    self.agents_steering[agentID] = 0.
                    self.agents_throttle[agentID] = 0.
                else:
                    low_level_action = self.process_input('forward', agentID)
                # self.vehicle_take_action(vehicle, low_level_action)
                
            # 3. assign actions for vehicles on the opposite side
            for i in range(len(self.right_vehicle_queue)):
                agentID = self.get_agentID(side='right', agentID=i)
                self.agents_steering[agentID] = 0.
                self.agents_throttle[agentID] = 0.
                
        ################################# BOTH SIDES #################################
        else:
            # both move
            for i in range(2): 
                # right side should go first and then left, if they collide we will offset the initial step for the left side
                if i == 0:
                    side = "right"
                    vehicle_queue = self.right_vehicle_queue
                    front_vehicle = self.right_vehicle_queue[0]
                    front_agentID = self.get_agentID(side, 0)
                    turn_action = "turnRight"
                    delay = False # for offset
                else:
                    side = "left"
                    vehicle_queue = self.left_vehicle_queue
                    front_vehicle = self.left_vehicle_queue[0]
                    front_agentID = self.get_agentID(side, 0)
                    turn_action = "turnLeft"
                    delay = False
                
                # scenario 1: front vehicle is not at the intersection yet
                if (not self.within_box_range(front_vehicle), side) and (not self.vehicle_is_turning(front_vehicle)):
                    # all vehicles move forward
                    for i in range(len(vehicle_queue)):
                        agentID = self.get_agentID(side, i)
                        low_level_action = self.process_input('forward', agentID)
                    # self.vehicle_take_action(vehicle, low_level_action)
                
                # scenario 2: front vehicle (single) is ready to turn 
                else:
                    # Optional: Offset actions from the left side if delay is set to true
                    if delay and (self.action_offset != 0):
                        for i in range(len(vehicle_queue)):
                            vehicle = vehicle_queue[i]
                            agentID = self.get_agentID(side, i)
                            self.agents_steering[agentID] = 0.
                            self.agents_throttle[agentID] = 0.
                        self.action_offset -= 1
                        continue
                    # 1. front vehicle turns
                    low_level_action = self.process_input(turn_action, front_agentID)
                    # self.vehicle_take_action(vehicle, low_level_action)
                    # 2. rest of the vehicles moves forward
                    front_of_remaining_vehicles = vehicle_queue[vehicles_to_exit[i == 0]]
                    for i in range(vehicles_to_exit[i == 0],len(vehicle_queue)):
                        vehicle = vehicle_queue[i]
                        agentID = self.get_agentID(side, i)
                        # if the front of the remaing vehicles is already at the intersection
                        if self.within_box_range(front_of_remaining_vehicles, side):
                            self.agents_steering[agentID] = 0.
                            self.agents_throttle[agentID] = 0.
                        else:
                            low_level_action = self.process_input('forward', agentID)
           

        # then we check if the front of queue (both left and right) have completed turning
        # if so, remove them from the queues and increment num_vehicles_exited
        if self.vehicle_exit_check(self.left_vehicle_queue[0]):
            self.exited_agentID.append(self.get_agentID('left',0))
            self.left_vehicle_queue.remove(self.left_vehicle_queue[0])
            exited += np.array([1,0])
        if self.vehicle_exit_check(self.right_vehicle_queue[0]):
            self.exited_agentID.append(self.get_agentID('right',0))
            self.left_vehicle_queue.remove(self.right_vehicle_queue[0])
            exited += np.array([0,1])
            
        # check whether records are consistent
        assert len(self.exited_agentID) == exited.sum()
        # all exited vehicles move forward
        for ID in self.exited_agentID:
            self.process_input('forward', ID)
        
        current_actions = np.hstack((self.agents_steering, self.agents_throttle))
        assert current_actions.shape == (12,2)
        current_actions = self.array2dict_action(current_actions)
        
        return exited, current_actions

    def coalition_exit_check(self, done, coalition_index) -> bool:
        for idx in coalition_index:
            if done['agent{n}'.format(n=idx)] != True:
                return False
        return True

    def _get_reward(self, high_level_action, num_of_vehicles, done, fairness = False):
        all_vehicle_location = np.concatenate((np.array(self.left_queue) == 2, np.array(self.right_queue) == 2)) # [true, false ....]
        AV_index = list(compress(self.vehicle_order, all_vehicle_location)) # get AV index from all vehciles
        all_vehicle_location_false = np.concatenate((np.array(self.left_queue) == 1, np.array(self.left_queue) == 1))
        human_index = list(compress(self.vehicle_order, all_vehicle_location_false))

        # Reward
        if self.coalition_exit_check(done, human_index): # r = -2, if human drivers exit first
            r = -2
        elif self.coalition_exit_check(done, AV_index): # r = +1, if AVs exit first
            r = 1
        else: # r = -1 for each high level decision
            r = -1
        # if (done[human_agent] == True).all(): # r = -2, if human drivers exit first
        #     r = -2
        # elif (done[AV_agent] == True).all(): # r = +1, if AVs exit first
        #     r = 1
        # else: # r = -1 for each high level decision
        #     r = -1
        
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

        print("high_level_action: ", high_level_action)
        ######## Low-Level Step Function (Continuous) ##########
        if high_level_action == (0,0):
            actions =self.array2dict_action(np.zeros((12,2)))
            o, r, d, i = super(MultiAgentTIntersectionEnv, self).step(actions)
        else:
            while (exited != num_of_vehicles).all(): 
                print('exited', exited)
                print('exited', num_of_vehicles)
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

        self.reward = self._get_reward(high_level_action, num_of_vehicles, d)

        return o, r, d, i

    
    
    def reset(self):
        super(MultiAgentTIntersectionEnv, self).reset()
        # self.__init__()
    



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
    # print("===============================================")
    # print("vehicle num", env.vehicles.values(), type(list(env.vehicles.values())[0]))
    # print("===============================================")
    # env.generate_vehicle_queue()
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