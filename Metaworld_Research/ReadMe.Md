#Metaworld Environment Wrapper

##Installation

```bash
conda create -n MW_Research python=3.9
conda activate MW_Research

conda install numpy
conda install pytorch torchvision -c pytorch
pip instal gymnasium
pip install git+https://github.com/Farama-Foundation/Metaworld.git

#If you encounter sympy conflicts with PyTorch
pip install sympy==1.13.1
```
hole_env.py is the environment file for the robot picking a puck out of a whole with a cost function that detects collisions

hole_env_recorder.py is for recording data within this environment 

wall_env.py is environment file for moving a puck while avoiding the walls with a cost function that detects collisions

wall_env_recorder.py is for recording data within this environment

parsing_data.py is for going through all of the data and returning which steps had a non-zero cost associated with them
