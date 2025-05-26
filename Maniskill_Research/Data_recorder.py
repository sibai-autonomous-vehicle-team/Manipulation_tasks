import os
import gymnasium as gym
import mani_skill.envs
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper
from PIL import Image
import csv
import numpy as np
import torch

def get_data():
    all_data =[]
    #Initiate the environment
    env = gym.make("UnitreeG1PlaceAppleInBowl-v1", obs_mode = "rgb+depth")
    #adjust the obs value returned so that it contains the rendering and state in one variable
    env = FlattenRGBDObservationWrapper(env)
    obs, _  = env.reset()
    obs_data = {
        'state': obs['state'][0],
        'rgb': obs['rgb'][0],
        'depth' : obs['depth'][0],
        'step': 0,
        'action': None,
        'cost': 0
    }
    all_data.append(obs_data)
    #Run simulation
    for step in range(100):
        #Sample a random action. Change for actual policy
        action = env.action_space.sample()
        obs, _, terminated, truncated, _ = env.step(action)
        cost = calculate_cost(env)
        obs_data = {
            'action': action,
            'state': obs['state'][0],
            'rgb': obs['rgb'][0],
            'depth': obs['depth'][0],
            'step': step,
            'cost': cost
        }
        all_data.append(obs_data)
        if terminated or truncated:
            break
    env.close()
    return all_data

def save_data(all_data):
    # Change for your specific directory
    output_dir = os.path.join('/Users/maxwellastafyev/Desktop/Research_project/Maniskill_Research', 'maniskill_recordings')
    # Set up output directories
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir,'images')
    os.makedirs(images_dir, exist_ok=True)
    info_dir = os.path.join(output_dir, 'info')
    os.makedirs(info_dir, exist_ok=True)

    for step in range(len(all_data)):
        # Collect data and set up paths
        curr_data = all_data[step]
        img_path = os.path.join(images_dir, f"step__{step}.png")
        csv_path = os.path.join(info_dir, f"step__{step}.csv")
        #Convert rgb_array to image
        rgb_array = curr_data['rgb'].cpu().numpy()
        rgb_array = rgb_array[:, :, :3]
        if rgb_array.max() <= 1.0:
            rgb_array = (rgb_array * 255).astype('uint8')
        img = Image.fromarray(rgb_array)
        img.save(img_path)

        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['State'])
            writer.writerow([str(curr_data['state'])])

            writer.writerow(['Action'])
            writer.writerow([str(curr_data['action'])])

            writer.writerow(['Cost'])
            writer.writerow([str(curr_data['cost'])])


    
    return output_dir

def calculate_cost(env, collision_threshold = 1e-6):
    #Get objects from environment
    bowl = env.unwrapped.bowl
    scene = env.unwrapped.scene
    robot = env.unwrapped.agent.robot
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

def main():
    data = get_data()
    save_data(data)

if __name__ == "__main__":
    main()
