import os 
import csv
import argparse
import numpy as np
from PIL import Image
import metaworld
from metaworld.policies import SawyerPickPlaceWallV2Policy

def calculate_cost(observation, wall_params):
    #Determines cost based on position of end-effector compared to the arm
    ee_pos = observation[:3]
    wall_pos = wall_params['position']
    wall_half_size = wall_params['half_size']

    closest_point = np.zeros(3)
    for i in range(3):
        closest_point[i] = max(wall_pos[i] - wall_half_size[i],
                               min(ee_pos[i], wall_pos[i] + wall_half_size[i]))
        
    distance = np.linalg.norm(ee_pos - closest_point)

    if distance < wall_params['collision_threshold']:
        return 0.8
    
    return 0.0

def setup_folder(base_dir):
    ##Setup necessary directories
    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    info_dir = os.path.join(base_dir, 'info')
    os.makedirs(info_dir, exist_ok=True)

    return images_dir, info_dir

def record_step(env, observation, action, cost, step_count, images_dir, info_dir, image_size):
    step_filename = f"step_{step_count:04d}"
    # take a screenshot
    img = env.render()
    # Create image within image folder
    img_path = os.path.join(images_dir, f"{step_filename}.png")
    img = Image.fromarray(img)
    img.save(img_path)
    # Create info file within info folder
    csv_path = os.path.join(info_dir, f"{step_filename}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        
        writer.writerow(['State'])
        writer.writerow([str(list(observation))])

        writer.writerow(['Action'])
        writer.writerow([str(list(action))])

        writer.writerow(['Cost'])
        writer.writerow([str(cost)])

        return img_path, csv_path


def run_policy(env_name = 'pick-place-wall-v2', num_episodes = 1, max_steps = 500, output_dir = None ,image_size = 128, seed = None):

    if output_dir is None:
        output_dir = os.path.join('/Users/maxwellastafyev/Desktop/Metaworld_Research', 'metaworld_recordings')

    mt1 = metaworld.MT1(env_name = env_name, seed = seed)
    env = mt1.train_classes[env_name](render_mode="rgb_array")
    task = mt1.train_tasks[0]
    env.set_task(task)
    env.camera_name = "corner3"

    policy = SawyerPickPlaceWallV2Policy()


    wall_params = {
        'position': np.array([0.1, 0.75, 0.06]),
        'half_size': np.array([0.12, 0.01, 0.06]),
        'collision_threshold': 0.001
    }

    images_dir, info_dir = setup_folder(output_dir)

    total_steps = 0

    for episode in range(num_episodes):
        observation, _ = env.reset()
        episode_cost = 0
        
        for step in range(max_steps):

            action = policy.get_action(observation)
            # If using metaworld built in policy then will inject noise into action so we can produce unsafe states
            if policy == SawyerPickPlaceWallV2Policy():
                noise_std = 0.05
                noise = np.random.normal(0, noise_std, size = action.shape)
                action = np.clip(action + noise, -1.0, 1.0)
            else :
                action = action
            cost = calculate_cost(observation, wall_params)

            img_path, csv_path = record_step(env, observation, action, cost, total_steps, images_dir, info_dir, image_size)

            next_observation, reward, terminated, truncated, info = env.step(action)

            observation = next_observation
            episode_cost += cost
            total_steps += 1

            if terminated or truncated:
                break

    print("\nRecording complete")

    return output_dir

def main():
    desktop_path = os.path.join('/Users/maxwellastafyev/Desktop/Research_project', 'metaworld_recordings')

    parser = argparse.ArgumentParser(description = 'Record Metaworld')
    parser.add_argument('--episodes', type = int, default =1 , help = 'Number of episodes to run')
    parser.add_argument('--max-steps', type = int, default = 500, help = 'max # of steps')
    parser.add_argument('--output-dir', type = str, default = desktop_path, help = 'output directory')
    parser.add_argument('--image-size', type = int, default = 128, help = 'size of images')
    parser.add_argument('--seed', type = int, default = None, help = 'Random seed')

    args = parser.parse_args()

    run_policy(
        num_episodes=args.episodes,
        max_steps=args.max_steps,
        output_dir=args.output_dir,
        image_size= args.image_size,
        seed = args.seed
    )

if __name__ == "__main__":
    main()