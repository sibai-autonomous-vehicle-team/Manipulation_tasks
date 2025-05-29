import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from PIL import Image
import torch
import numpy as np
import cv2

class Maniskill:
    LOCK = None
    metadata = {}  
    def __init__(
            self,
            name = "UnitreeG1PlaceAppleInBowl-v1", # Preset environment name to 
            action_repeat = 1,
            size = (128, 128),
            seed = None,
            obs_mode = "rgb+depth" # changed obs_mode so that aan rbg image is returned in observations
    ):
        assert size[0] == size[1]
        self._action_repeat = action_repeat
        self._size = size
        self._obs_mode = obs_mode
        self._env = gym.make(name, obs_mode = obs_mode)
        # Wrapper for returning different obs values
        self._env = FlattenRGBDObservationWrapper(self._env)
        print(self._env.observation_space)
        if seed is not None:
            self._env.reset(seed=seed)
        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size
        obs, _ = self._env.reset()
        #Get the shape of the state
        state_size = obs['state'][0].shape[0]
        return gym.spaces.Dict(
            {
                "image" : gym.spaces.Box(0,255, img_shape + (3,), np.uint8),
                "vector": gym.spaces.Box(-np.inf, np.inf, shape = (state_size,), dtype = np.float32 )
            }
        )

    def transform_obs(self, observation):
        obs = {}
        #Convert rgb rgb array from a tensor to an rgb array
        rgb_array = observation['rgb'][0].cpu().numpy() 
        if rgb_array.max() <=1.0:
            rgb_array = (rgb_array*255).astype('uint8')
        #Resize rgb image
        rgb_array = cv2.resize(rgb_array, self._size)
        obs['image'] = rgb_array
        #Convert state vector from tensor to a numpy array
        state_data = observation['state'][0].cpu().numpy().astype(np.float32)
        obs['vector'] = state_data
        
        return obs
    
    @property  
    def action_space(self):
        return self._env.action_space
    
    def calculate_cost(self, collision_threshold = 1e-6):
            #Get objects from environment
            bowl = self._env.unwrapped.bowl      
            scene = self._env.unwrapped.scene    
            robot = self._env.unwrapped.agent.robot  
            #Find the correct hand link
            all_links = robot.get_links()
            right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
            #Calculate contact forces
            contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
            if contact_forces is not None and len(contact_forces) > 0:
                forces_magnitudes = torch.norm(contact_forces, dim = -1)
                total_force = torch.sum(forces_magnitudes).item()
                # Returns 0.8 if there is a collision
                if total_force > collision_threshold:
                    return 0.8
            return 0.0

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        total_cost = 0

        for _ in range(self._action_repeat):
            observation, cur_reward, terminated, truncated, info = self._env.step(action)
            reward += cur_reward
            cost = self.calculate_cost()
            total_cost +=cost

            if terminated or truncated:
                break

        obs_dict = self.transform_obs(observation)
        obs = {}
        obs['image'] = obs_dict['image']
        obs['vector'] = obs_dict['vector']
        obs['is_terminal'] = terminated or truncated
        obs['is_first'] = False
        info['cost'] = total_cost   
        done = terminated or truncated
        return obs, -total_cost, done, info
    
    ##Removed render function because rgb image is returned in obs with Maniskill

    def reset(self):
        observation, info = self._env.reset()
        obs_dict = self.transform_obs(observation)

        obs = {'is_terminal': False, 'is_first': True}
        obs['image'] = obs_dict['image']
        obs['vector'] = obs_dict['vector']
        info['cost'] = 0

        return obs
    
    def close(self):
        return self._env.close()

# Same changes just returns reward instead of cost in step function
class ManiskillEval:
    LOCK = None
    metadata = {}  
    def __init__(
            self,
            name = "UnitreeG1PlaceAppleInBowl-v1",
            action_repeat = 1,
            size = (128, 128),
            seed = None,
            obs_mode = "rgb+depth"
    ):
        assert size[0] == size[1]
        self._action_repeat = action_repeat
        self._size = size
        self._obs_mode = obs_mode
        self._env = gym.make(name, obs_mode = obs_mode)

        self._env = FlattenRGBDObservationWrapper(self._env)
        print(self._env.observation_space)
        if seed is not None:
            self._env.reset(seed=seed)
        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size
        obs, _ = self._env.reset()
        state_size = obs['state'][0].shape[0]
        return gym.spaces.Dict(
            {
                "image" : gym.spaces.Box(0,255, img_shape + (3,), np.uint8),
                "vector": gym.spaces.Box(-np.inf, np.inf, shape = (state_size,), dtype = np.float32 )
            }
        )

    def transform_obs(self, observation):
        obs = {}
        rgb_array = observation['rgb'][0].cpu().numpy()  
        if rgb_array.max() <=1.0:
            rgb_array = (rgb_array*255).astype('uint8')

        rgb_array = cv2.resize(rgb_array, self._size)
        obs['image'] = rgb_array

        state_data = observation['state'][0].cpu().numpy().astype(np.float32)
        obs['vector'] = state_data
        
        return obs
    
    @property  
    def action_space(self):
        return self._env.action_space
    
    def calculate_cost(self, collision_threshold = 1e-6):
        
            #Get objects from environment
            bowl = self._env.unwrapped.bowl      
            scene = self._env.unwrapped.scene    
            robot = self._env.unwrapped.agent.robot  
            #Find the correct hand link
            all_links = robot.get_links()
            right_hand_link = next((link for link in all_links if link.name == 'right_palm_link'), None)
            
            if right_hand_link is None:
                return 0.0
            
            #Calculate contact forces
            contact_forces = scene.get_pairwise_contact_forces(right_hand_link, bowl)
            if contact_forces is not None and len(contact_forces) > 0:
                forces_magnitudes = torch.norm(contact_forces, dim = -1)
                total_force = torch.sum(forces_magnitudes).item()
                # Returns 0.8 if there is a collision
                if total_force > collision_threshold:
                    return 0.8
            return 0.0

    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        total_cost = 0

        for _ in range(self._action_repeat):
            observation, cur_reward, terminated, truncated, info = self._env.step(action)
            reward += cur_reward
            cost = self.calculate_cost()
            total_cost +=cost

            if terminated or truncated:
                break

        obs_dict = self.transform_obs(observation)
        obs = {}
        obs['image'] = obs_dict['image']
        obs['vector'] = obs_dict['vector']
        obs['is_terminal'] = terminated or truncated
        obs['is_first'] = False
        info['cost'] = total_cost   
        done = terminated or truncated
        return obs, reward, done, info 
    
    def reset(self):
        observation, info = self._env.reset()
        obs_dict = self.transform_obs(observation)

        obs = {'is_terminal': False, 'is_first': True}
        obs['image'] = obs_dict['image']
        obs['vector'] = obs_dict['vector']
        info['cost'] = 0

        return obs
    
    def close(self):
        return self._env.close()