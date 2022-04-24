from metadrive.policy.idm_policy import IDMPolicy
import numpy as np

class IDM2(IDMPolicy):

    def __init__(self, control_object, random_seed, vehicles_all_obs):
        super(IDMPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.vehicles_all_obs = vehicles_all_obs     

    def find_surrounding(self):
        lat, lont = self.control_object.position
        object_id = self.control_object.id
        vehicle_in_range = set() 
        LIDAR_RADIUS = 40

        for vehicle in self.vehicles_all_obs:
             if object_id != vehicle.id:
                 x, y = vehicle.position
                 if np.abs(lat - x) < LIDAR_RADIUS and np.abs(lont - y) < LIDAR_RADIUS:
                     vehicle_in_range.add(vehicle)

        return vehicle_in_range