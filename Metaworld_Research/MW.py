import metaworld 
import gymnasium as gym
import numpy as np 
import torch

class MetaWorld:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name = "pick-place-wall-v2", # preset environment name to replicate experiment
        action_repeat = 1,
        size=(128,128),
        seed=None,
    ):
        assert size[0] == size[1]
        self._action_repeat = action_repeat
        self._size = size
        # Changed line 21-25 for adapt to Metaworld 
        self._benchmark = metaworld.MT1(env_name = name, seed = seed)
        env_cls = self._benchmark.train_classes[name]
        self._env = env_cls(render_mode="rgb_array")
        self._task = self._benchmark.train_tasks[0]
        self._env.set_task(self._task)
        self._env.camera_name = "camera3"

        print(f"Observation space: {self._env.observation_space}")
        # position of wall or obstacle in this environment
        # found in 
        self._wall_position = ([0.1, 0.75, 0.06])
        self._wall_half_size = ([0.12, 0.01, 0.06])  
        # collision threshold
        self._collision_threshold = 0.001     
                
        self._done = True
        self.reward_range = [-np.inf,np.inf]
    
    ## Wrote cost calculator method to determine cost if the end-effector position gets too close to the wall in this environment(applies only to 'pick-place-wall-v2')
    ## Cost set to 0.8 if end-effector too close
    def calculate_cost(self, observation):
        ee_pos = observation[:3]

        wall_pos = self._wall_position
        wall_half_size = self._wall_half_size
        closest_point = np.zeros(3)
        for i in range(3):
            closest_point[i] = max(wall_pos[i] - wall_half_size[i],
                                    min(ee_pos[i], wall_pos[i] + wall_half_size[i]))
        
        distance = np.linalg.norm(ee_pos - closest_point)

        if distance < self._collision_threshold:
            return 0.8
        
        return 0.0

    @property
    def observation_space(self):
        img_shape = self._size
        # added in next line to determine the correct shape of the vector
        vector_size = self._env.observation_space.shape[0]
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, img_shape + (3,), np.uint8),
                "vector": gym.spaces.Box(-np.inf, np.inf, shape=(vector_size,), dtype=np.float32)
            }
        )
    
    def transform_obs(self, observation):
        obs = {}
        obs["vector"] = observation
        return obs
    
    @property
    def action_space(self):
        return self._env.action_space
    
    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        for _ in range(self._action_repeat):
            observation, cur_reward, terminated, truncated, info = self._env.step(action)
            reward +=cur_reward
            if terminated or truncated:
                break

        img = self.render()

        obs = {}
        obs_dict = self.transform_obs(observation)
        obs["image"] = img
        obs["vector"] = obs_dict["vector"]
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = False

        cost = self.calculate_cost(observation)
        info["cost"] = cost

        done = terminated or truncated

        return obs, -cost, done, info
    def reset(self):
        observation, info = self._env.reset()
        obs_dict = self.transform_obs(observation)

        img = self.render()
        obs = {"is_terminal": False, "is_first": True}
        obs["image"] = img
        obs["vector"] = obs_dict["vector"]
        info["cost"] = 0
        return obs
    
    def render(self, *args, **kwargs):
        img = self._env.render()
        return img
    
    def close(self):
        return self._env.close()
## Most parts of eval such as initiating environment and calcuting cost are the same as main MetaWorld class 
class MetaWorldEval:
    LOCK = None
    metadata = {}

    def __init__(
        self,
        name = "pick-place-wall-v2",
        action_repeat = 1,
        size = (128,128),
        seed = None
    ):
        assert size[0] == size[1]
        self._action_repeat = action_repeat
        self._size = size
        self._benchmark = metaworld.MT1(env_name = name, seed = seed)
        env_cls = self._benchmark.train_classes[name]
        self._env = env_cls(render_mode="rgb_array")
        self._task = self._benchmark.train_tasks[0]
        self._env.set_task(self._task)
        self._env.camera_name = "camera3"
        #Wall parameters
        self._wall_position = np.array([0.1, 0.75, 0.06])
        self._wall_half_size = np.array([0.12, 0.01, 0.06])
        # collision threshold
        self._collision_threshold = 0.001 
        print(f"Evaluation observation space: {self._env.observation_space}")

        self._done = True
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size
        vector_size = self._env.observation_space.shape[0]
        return gym.spaces.Dict(
            {
            "image": gym.spaces.Box(0, 255, img_shape + (3,), np.uint8),
            "vector": gym.spaces.Box(-np.inf, np.inf, shape = (vector_size,), dtype = np.float32)
            }
        )
    
    def transform_obs(self, observation):
        obs = {}
        obs["vector"] = observation
        return obs
    
    @property
    def action_space(self):
        return self._env.action_space
    

    ##Added method to calculate cost based off of position of end-effector position and the position of the wall
    #Cost set to  to 0.8
    def calculate_cost(self, observation):
        ee_pos = observation[:3]

        wall_pos = self._wall_position
        wall_half_size = self._wall_half_size

        closest_point = np.zeros(3)
        for i in range(3):
            closest_point[i] = max(wall_pos[i] - wall_half_size[i],
                                   min(ee_pos[i], wall_pos[i] + wall_half_size[i]))
        distance = np.linalg.norm(ee_pos - closest_point)

        if distance < self._collision_threshold:
            return 0.8
        return 0.0
    
    def step(self, action):
        assert np.isfinite(action).all(), action
        reward = 0
        cost = 0

        for _ in range(self._action_repeat):
            observation, cur_reward, terminated, truncated, info = self._env.step(action)
            reward +=cur_reward
            cost += self.calculate_cost(observation)
            if terminated or truncated:
                break
        
        img = self.render()

        obs = {}
        obs_dict = self.transform_obs(observation)

        obs["image"] = img
        obs["vector"] = obs_dict["vector"]
        obs["is_terminal"] = terminated or truncated
        obs["is_first"] = False

        info["cost"] = cost

        done = truncated or terminated
        return obs, reward, done, info
    
    def reset(self):
        observation, info = self._env.reset()
        obs_dict = self.transform_obs(observation)

        img = self.render()
        obs = {"is_terminal": False, "is_first": True}
        obs["image"] = img
        obs["vector"] = obs_dict["vector"]
        info["cost"] = 0
        return obs
    
    def render(self, *args, **kwargs):
        img = self._env.render()
        return img
    
    def close(self):
        return self._env.close()