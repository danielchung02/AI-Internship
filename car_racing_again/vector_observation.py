import math
import gymnasium as gym
import numpy as np
from gymnasium import spaces

class VectorObservation(gym.ObservationWrapper):
    state_dim = 15
    lookahead = [5,10,20,40] 
    track_half_width = 40.0/6.0 
    max_steering_angle = 0.4 

    def __init__(self,env, speed_scale = 50.0, angular_velocity_scale = 5.0): 
        super().__init__(env) 
        self.speed_scale = speed_scale
        self.angular_velocity_scale = angular_velocity_scale 
        self.observation_space = spaces.Box(-1.0, 1.0, (self.state_dim,), dtype=np.float32)  # add
        

    def wrap_angle(self,angle):
        return (math.pi + angle) % (2*math.pi) -math.pi
 
    def observation(self, image): 
        base_env = self.env.unwrapped
        car = base_env.car
        track = base_env.track

        x_pos, y_pos = car.hull.position[0], car.hull.position[1]
        car_position = np.asarray([x_pos, y_pos], dtype = np.float32)
        car_angle = np.float32(car.hull.angle)
        
        # track[i]=(alpha, beta, x, y)

        distances_power=[]
        for point in track:
            distance = np.sum((car_position-[point[2], point[3]])**2)
            distances_power.append(distance)
        nearest = np.argmin(distances_power)
        _, track_angle, track_x, track_y = track[nearest]

        forward_axis = np.array([-math.sin(car_angle), math.cos(car_angle)], dtype = np.float32)
        lateral_axis = np.array([math.cos(car_angle), math.sin(car_angle)], dtype = np.float32)

        velocity = np.asarray(car.hull.linearVelocity, dtype = np.float32)

        heading_error = np.float32(track_angle-car_angle)
        heading_error = self.wrap_angle(heading_error)

        track_normal = np.array([math.cos(track_angle), math.sin(track_angle)], dtype = np.float32)
        lateral_offset = np.dot((car_position - np.array([track_x, track_y])), track_normal)

        steering = 0.5 * (float(car.wheels[0].joint.angle) + float(car.wheels[1].joint.angle))

        forward_vel = np.dot(forward_axis, velocity)/self.speed_scale
        lateral_vel = np.dot(lateral_axis, velocity)/self.speed_scale
        angular_vel = car.hull.angularVelocity/self.angular_velocity_scale
        steering /= self.max_steering_angle
        lateral_offset /= self.track_half_width

        values = [
            np.clip(forward_vel, -1.0, 1.0),
            np.clip(lateral_vel, -1.0, 1.0),
            np.clip(angular_vel, -1.0, 1.0),
            np.clip(steering, -1.0, 1.0),
            np.clip(lateral_offset, -1.0, 1.0),
            math.sin(heading_error),
            math.cos(heading_error)
        ]

        for i in self.lookahead:
            future_index = (nearest + i) % len(track)
            future_angle = np.float32(track[future_index][1])
            future_error = self.wrap_angle(future_angle - car_angle)
            values.extend([math.sin(future_error), math.cos(future_error)])

        state = np.array(values, dtype = np.float32)
        return state

def make_vector_env(render_mode=None, continuous = False):
    env = gym.make(
        "CarRacing-v3",
        continuous=continuous,
        domain_randomize=False,
        render_mode=render_mode
    )

    env = VectorObservation(env)

    return env



        









