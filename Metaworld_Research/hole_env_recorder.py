import os 
from pathlib import Path
import argparse
import numpy as np
from PIL import Image
import metaworld
from metaworld.policies import SawyerPickOutOfHoleV2Policy
import torch

def calculate_cost(observation, hole_params):
    """Calculate collision cost for pick-out-of-hole environment"""
    ee_pos = observation[:3]
    obj_pos = observation[4:7]
    xy_distance = np.linalg.norm(ee_pos[:2] - hole_params['center'][:2])
    radius = hole_params['threshold_radius']
    # Physical parameters from .xml file
    hole_floor_z = obj_pos[2]
    table_surface_z = 0.054
    #Check for floor collision
    if xy_distance <= radius and ee_pos[2] < hole_floor_z - 0.005:
        return 0.8
    # check for collision with the imaginary cylinder 
    if hole_floor_z < ee_pos[2] < table_surface_z:
        allowed_radius = radius + np.exp((ee_pos[2] - 0.15) / 0.015)
        dist_to_wall = abs (xy_distance - allowed_radius)
        if dist_to_wall < hole_params['collision_threshold']:
            return 0.8
    return 0.0

def collect_episodes(env_name, num_episodes, max_steps, output_dir, image_size, seed = None):
    output_path = Path(output_dir)    
    output_path.mkdir(parents=True, exist_ok=True)
    obses_dir = output_path / "obses"
    obses_dir.mkdir(exist_ok=True)
    #Initializes environment and sets render mode and camera angle 
    mt1 = metaworld.MT1(env_name = env_name, seed = seed)
    env = mt1.train_classes[env_name](render_mode="rgb_array")
    task = mt1.train_tasks[0]
    env.set_task(task)
    env.camera_name = "corner1"
    #Set to built in policy
    policy = SawyerPickOutOfHoleV2Policy()

    hole_params = {
        'center': np.array([0.0, 0.6, 0.2]),
        'threshold_radius': 0.03,
        'collision_threshold': 0.001
    }
    
    all_actions = []
    all_states = []
    all_costs = []
    seq_lengths = []

    #Data collection
    for episode_idx in range(num_episodes):
        print(f"Episode {episode_idx +1}/{num_episodes}")

        episode_actions = []
        episode_states = []
        episode_costs = []
        episode_obs = []


        observation, _ = env.reset()

        for step in range(max_steps):
            episode_states.append(torch.tensor(observation, dtype = torch.float32))

            img = env.render()
            #Resizes image to 224x224 because Metaworld does not have built method for changing image resolutions
            if img.shape[0] != image_size or img.shape[1] != image_size:
                img_pil = Image.fromarray(img)
                img_pil = img_pil.resize((image_size, image_size), Image.LANCZOS)
                img = np.array(img_pil)

            
            episode_obs.append(torch.tensor(img.copy(), dtype = torch.float32))
            #Get actoin
            action = policy.get_action(observation)
            #Adds noise to action so that unsafe states are created and data is more varied 
            noise_std = 0.05
            noise = np.random.normal(0, noise_std, size = action.shape)
            action = np.clip(action + noise, -1.0, 1.0)


            episode_actions.append(torch.tensor(action, dtype = torch.float32))

            cost = calculate_cost(observation, hole_params)
            episode_costs.append(torch.tensor(cost, dtype = torch.float32))

            observation, reward, terminated, truncated, info = env.step(action)


            if terminated or truncated:
                break

        all_actions.append(torch.stack(episode_actions))
        all_states.append(torch.stack(episode_states))
        all_costs.append(torch.stack(episode_costs))
        seq_lengths.append(len(episode_actions))

        torch.save(torch.stack(episode_obs), obses_dir / f"episode_{episode_idx}.pth")

    env.close()
    #Pad sequences to the same length
    max_len = max(seq_lengths)
    
    padded_actions = torch.zeros(num_episodes, max_len, all_actions[0].shape[-1])
    padded_states = torch.zeros(num_episodes, max_len, all_states[0].shape[-1])
    padded_costs = torch.zeros(num_episodes, max_len)

    for i, (actions, states, costs) in enumerate(zip(all_actions, all_states, all_costs)):
        length = len(actions)
        padded_actions[i, :length] = actions
        padded_states[i, :length] = states
        padded_costs[i, :length] = costs
    #Save data
    torch.save(padded_actions, output_path / "actions.pth")
    torch.save(padded_states, output_path / "states.pth")
    torch.save(torch.tensor(seq_lengths), output_path / "seq_lengths.pth")
    torch.save(padded_costs, output_path / "costs.pth")

def main():
    #Added argparser so that changing variables is much easier 
    parser = argparse.ArgumentParser(description='Record Metaworld')
    parser.add_argument('--episodes', type=int, default=4, help='Number of episodes to run')
    parser.add_argument('--max-steps', type=int, default=500, help='max # of steps')
    parser.add_argument('--output-dir', type=str, default='/Users/maxwellastafyev/Desktop/Research_project/Manipulation_tasks/Dino_MW/MW_data', help='output directory')
    parser.add_argument('--image-size', type=int, default=224, help='size of images')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--env-name', type=str, default='pick-out-of-hole-v2')

    args = parser.parse_args()

    collect_episodes(
        env_name=args.env_name,
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        image_size=args.image_size,
        seed=args.seed
    )

if __name__ == "__main__":
    main()