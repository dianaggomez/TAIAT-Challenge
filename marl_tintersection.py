import copy
from metadrive.manager.spawn_manager import SpawnManager
from metadrive.manager.map_manager import MapManager
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.pgblock.t_intersection import TInterSection
from metadrive.component.map.pg_map import PGMap
from metadrive.component.road_network.road import Road
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

    def step(self, actions):
        o, r, d, i = super(MultiAgentTIntersectionEnv, self).step(actions)

        #### New Observation #####
        # Output as numpy array
        # If agent exists, or dies, output [0,0,coalition]
        obs = []
        order = [0,1,2,3,4,5,11,10,9,8,7,6]
        for num in order:
            if 'agent{n}'.format(n=num) in o:
                obs.append(o['agent{n}'.format(n=num)])
            else:
                obs.append([0,0, coalition['agent{n}'.format(n=num)]])
        o = np.array(obs)

        ########### INFO For RL Agent ###################
        # We can use this when identify the rewards

        # if self.num_RL_agents == self.num_agents:
        #     return o, r, d, i

        # original_done_dict = copy.deepcopy(d)
        # d = self.agent_manager.filter_RL_agents(d, original_done_dict=original_done_dict)
        # if "__all__" in d:
        #     d.pop("__all__")
        # # assert len(d) == self.agent_manager.num_RL_agents, d
        # d["__all__"] = all(d.values())
        # return (
        #     self.agent_manager.filter_RL_agents(o, original_done_dict=original_done_dict),
        #     self.agent_manager.filter_RL_agents(r, original_done_dict=original_done_dict),
        #     d,
        #     self.agent_manager.filter_RL_agents(i, original_done_dict=original_done_dict),
        # )
        return o, r, d, i
    def _preprocess_actions(self, actions):
        if self.num_RL_agents == self.num_agents:
            return super(MultiAgentTIntersectionEnv, self)._preprocess_actions(actions)

        actions = {v_id: actions[v_id] for v_id in self.vehicles.keys() if v_id in self.agent_manager.RL_agents}
        return actions

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
                    "num_lasers": 0, # the more dense, the more accurate but longer computation time
                    "num_others": 0,
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
            "num_RL_agents" : 6,
        }
    # config.update({'IDM_agent': True,})
    env = MultiAgentTIntersectionEnv(config)
    o = env.reset()
    # print("vehicle num", len(env.engine.traffic_manager.vehicles))
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
        # actions = {k: [-0, 1.0] for k in env.vehicles.keys()}
        o, r, d, info = env.step(actions) # the actions are a dictionary corresponding to each vehicles so observation will also be dictiary for each vehicle's observation
        print("Observation: ", o)
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