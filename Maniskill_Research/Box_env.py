import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
import torch
from pathlib import Path
import numpy as np
import multiprocessing as mp

def aggregate_dct(dcts):
        full_dct = {}
        for dct in dcts:
            for key, value in dct.items():
                if key not in full_dct:
                    full_dct[key] = []
                full_dct[key].append(value)
        for key, value in full_dct.items():
            if isinstance(value[0], torch.Tensor):
                full_dct[key] = torch.stack(value)
            else:
                full_dct[key] = np.stack(value)
        return full_dct

class ManiskillWrapper(gym.Env):
    def __init__(
            self,
            env_name = "UnitreeG1TransportBox-v1",
            obs_mode = "rgb+depth",
            size = (224,224),
            
    ):
        self._cuda_initialized = False
        self._env = gym.make(env_name, obs_mode = obs_mode, 
                             render_backend = "cpu", 

                             sensor_configs = dict(width = size[0], height = size[1]))
        self._env = FlattenRGBDObservationWrapper(self._env)
        self.action_dim = self._env.action_space.shape[0]
        self.grasped = False
        
    def _initialize_cuda(self):
        if not self._cuda_initialized:
            # Perform CUDA-related operations here
            self._cuda_initialized = True

    @property
    def action_space(self):
        return self._env.action_space
    
    def calculate_cost(self, z_threshold = 0.6):
        cur_grasped = self._env.unwrapped.evaluate()["box_grasped"][0].item()
        dropped = self.grasped and not cur_grasped 
        self.grasped = cur_grasped

        box_z = self._env.unwrapped.box.pose.p[0,2].item()
        if dropped and box_z < z_threshold:
            return 0.8
        return 0.0
       
    def sample_random_init_goal_states(self,seed):
        rs = np.random.RandomState(seed)
        obs, _ = self._env.reset()
        full_state = self._env.unwrapped.get_state().cpu().numpy()

        init_state = full_state.copy()
        goal_state = full_state.copy()

        if len(init_state.shape) > 1:
            init_flat = init_state[0]
            goal_flat = goal_state[0] 
        else:
            init_flat = init_state
            goal_flat = goal_state

        box_xy_init = rs.uniform([-0.05, -0.05], [0.2, 0.05], 2) + np.array([-0.1, -0.37])
        box_z_init = 0.7508
        init_flat[0:3] = [box_xy_init[0],box_xy_init[1], box_z_init]

        box_xy_goal = rs.uniform([-0.78, 0.3],[0.78, 1.0],2)
        box_z_goal = rs.uniform(0.750, 0.751)
        goal_flat[0:3] = [box_xy_goal[0], box_xy_goal[1], box_z_goal]

        z_rotation_init = rs.uniform(0, np.pi/6)
        z_rotation_goal = rs.uniform(0, np.pi/6)
        
        init_flat[3:7] = [np.cos(z_rotation_init/2), 0, 0, np.sin(z_rotation_init/2)]
        goal_flat[3:7] = [np.cos(z_rotation_goal/2), 0, 0, np.sin(z_rotation_goal/2)]

        init_flat[20] = rs.uniform(-1.7, -1.4)  
        goal_flat[20] = rs.uniform(1.3, 1.5)

        return init_state, goal_state
    
    def eval_state(self, goal_state, cur_state):
        if isinstance(cur_state, torch.Tensor):
            cur_state = cur_state.cpu().numpy()
        if isinstance(goal_state, torch.Tensor):
            goal_state = goal_state.cpu().numpy()

        box_cur = cur_state[0:3]
        robot_cur = cur_state[13:76]
        cur = np.concatenate([box_cur, robot_cur])

        box_goal = goal_state[0:3]
        robot_goal = goal_state[13:76]
        goal = np.concatenate([box_goal, robot_goal])
        poss_diff = np.linalg.norm(cur - goal)
        success = poss_diff < 0.1

        state_dist = np.linalg.norm(goal_state - cur_state)
        return{
            'success': success,
            'state_dist' : state_dist
        }
    
    def prepare(self, seed, init_state):
        self._env.reset(seed = seed)
        if len(init_state.shape) ==1:
            state_with_batch = init_state.reshape(1, -1)
        else:
            state_with_batch = init_state
        self._env.unwrapped.set_state(state_with_batch)
        dummy_action = np.zeros(self.action_dim)
        obs, _, _, _, _ = self._env.step(dummy_action)
        state = self._env.get_state()
        obs = {
            'visual': obs['rgb'][0][:, :, 3:6],
            'proprio': obs['state'][0][13:64]
        }
        return obs, state
    
    def step_multiple(self, actions):
        obses = []
        rewards = []
        dones = []
        infos = []

        for action in actions:
            obs, reward, truncated, terminated, info = self._env.step(action)
            visual = obs['rgb'][0][:, :, 3:6]
            state = self._env.get_state()
            proprio = obs['state'][0][13:64]
            obs = {'visual' : visual, 'proprio': proprio}
            obses.append(obs)
            rewards.append(reward)
            done = terminated or truncated
            dones.append(done)
            info['state'] = state
            infos.append(info)
            if terminated or truncated:
                break

        obses = aggregate_dct(obses)
        rewards = np.array(rewards)
        dones = np.array(dones)
        infos = aggregate_dct(infos)

        return obses, rewards, dones, infos 
    
    def rollout(self, seed, init_state, actions):
        obs, state = self.prepare(seed, init_state)  

        obses, rewards, dones, infos = self.step_multiple(actions)
        
        initial_visual = obs['visual'].unsqueeze(0) 
        initial_proprio = obs['proprio'].unsqueeze(0)
    
        result_obses = {
            'visual': torch.cat([initial_visual, obses['visual']], dim=0),
            'proprio': torch.cat([initial_proprio, obses['proprio']], dim=0)
        }
    
        initial_state = state.unsqueeze(0)
        states = torch.cat([initial_state, infos['state']], dim=0)
    

        return result_obses, states
    
    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def step(self, action):
        return self._env.step(action)

    def render(self, mode="human"):
        return self._env.render(mode=mode)

    def close(self):
        return self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

        
         

      

            

