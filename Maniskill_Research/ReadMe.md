Maniskill Data Recorder

Data collection script for ManiSkill environment UnitreeG1AppleInBowl-v1, environment that captures RGB/depth observations, robot states, actions, and collision costs between robot's right hand and bowl

Cost function - returns a value of 0.8 if there is a collision between the robots right hand and the bowl 

Data_recorder.py saves RGB images, state vectors, actions, and collision costs
Dependencies
```bash
# Core requirements
pip install gymnasium
pip install torch
pip install Pillow
pip install numpy

# ManiSkill (robotics simulator)
pip install --upgrade mani_skill
```


test.py was used for determining the object names within the environment and if the objects exist

Env_file.py is the environment file for the set environment that I tested with. Added a cost function and removed render function because rgb_array is returned within obs when running _env.step(action)


Data_recorder.py is used for data collection. Everything is saved in a way that the Dino_MW processes it and images are rendered in 224x224 resolution.

```bash
#To get all argparser variables
python Data_recorder.py --help
```


