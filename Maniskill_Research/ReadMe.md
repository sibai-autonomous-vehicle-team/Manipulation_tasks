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
https://maniskill.readthedocs.io/en/latest/user_guide/getting_started/macos_install.html use instructions on this link for installation on mac

Output structure
maniskill_recordings/
├── images/
│   ├── step__0.png
│   ├── step__1.png
│   └── ...
└── info/
    ├── step__0.csv
    ├── step__1.csv
    └── ...


test.py was used for determining the object names within the environment and if the objects exist
