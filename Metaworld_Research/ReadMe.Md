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

wall_env.py is environment file for moving a puck while avoiding the walls with a cost function that detects collisions

parsing_data.py is for going through all of the data and returning which steps had a non-zero cost associated with them

The two recorders are for recording data in their respective environments if camera view is not to liking then change to "corner2" or "corner3". Images are saved in 224x224 format and data is saved in .pth files and are formatted for Dino_MW.

Run
```bash
python wall_env_recorder.py 
##For argparser variables
python wall_env_recorder.py --help
##or
python hole_env_recorder.py
##For argparser variables
python hole_env_recorderpy --help
```



