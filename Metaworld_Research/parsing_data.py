import os 
import csv

def find_collision(info_dir):
    ground_collision = []
    wall_collision = []

    files = [f for f in os.listdir(info_dir) if f.endswith('.csv')]
    files.sort()

    for file in files:
        step_num = int(file.split('_')[1].split('.')[0])
        file_path = os.path.join(info_dir, file)

        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            lines = list(reader)

            if len(lines)>= 6 and 'Cost' in lines [4][0]:
                cost_value = float(lines[5][0])
                if cost_value== 0.5:
                    ground_collision.append(step_num)
                if cost_value == 0.8:
                    wall_collision.append(step_num)

    return ground_collision, wall_collision


info_dir = "/Users/maxwellastafyev/Desktop/Research_project/metaworld_recordings/info"
ground_collisions, wall_collisions = find_collision(info_dir)
total_collisions = len(ground_collisions) + len(wall_collisions)
print(f"Found {total_collisions} steps with collisions")
print(f"Ground: {ground_collisions}, Wall: {wall_collisions}")


