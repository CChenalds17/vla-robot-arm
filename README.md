# Supernumerary Robotic Arm with Project Aria vision system built on SmolVLA

## Installation
Tested on Python version 3.11.6
1. Clone the repository:

    `$ git clone https://github.com/CChenalds17/sra.git`
2. Initialize and add the Lerobot library as a submodule:

    `$ git submodule init`

    `$ git submodule update`
3. Install python packages:

    `$ pip install -r requirements.txt`

4. Install ffmpeg
5. Set up Project Aria Glasses: https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup

## Deployment
### Teleop/Recording
```
$ cd src
$ python -m sra.teleop.main
```
Be sure to set log in to huggingface first

### Autonomous loop
```
$ cd src
$ python -m sra.autonomous.main
```