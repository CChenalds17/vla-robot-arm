# VLA-Based Custom Robotic Arm
## Vision-Language-Action Model on Custom Robotic Arm
This library contains the code for a custom robotic arm setup based on SmolVLA. Our custom 1-DOF robotic arm uses the [LeRobot library](https://github.com/huggingface/lerobot) for data collection and [SmolVLA](https://huggingface.co/blog/smolvla) as its base model. We use the [Meta Project Aria Gen 1 Glasses](https://www.projectaria.com/) as our vision system. The arm is controlled by an Arduino Uno, which communicates with the Python script using a custom motor bus and communication protocol.

## Installation
Tested on Python version 3.11.6

Clone the repository:
```bash
git clone https://github.com/CChenalds17/sra.git
```

Initialize the LeRobot repository as a submodule and update it:
```bash
git submodule update --init --recursive
```

Create a virtual environment with Python 3.10 and activate it, e.g. with [`miniconda`](https://docs.anaconda.com/free/miniconda/index.html):

```bash
conda create -y -n lerobot python=3.11
conda activate lerobot
```

When using `miniconda`, install `ffmpeg` in your environment:

```bash
conda install ffmpeg -c conda-forge
```

> **NOTE:** This usually installs `ffmpeg 7.X` for your platform compiled with the `libsvtav1` encoder. If `libsvtav1` is not supported (check supported encoders with `ffmpeg -encoders`), you can:
>
> - _[On any platform]_ Explicitly install `ffmpeg 7.X` using:
>
> ```bash
> conda install ffmpeg=7.1.1 -c conda-forge
> ```
>
> - _[On Linux only]_ Install [ffmpeg build dependencies](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#GettheDependencies) and [compile ffmpeg from source with libsvtav1](https://trac.ffmpeg.org/wiki/CompilationGuide/Ubuntu#libsvtav1), and make sure you use the corresponding ffmpeg binary to your install with `which ffmpeg`.

Install python packages:
```bash
pip install -r requirements.txt
```

Set up Project Aria Glasses according to [Project Aria documentation](https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup)

## Deployment
Log in to huggingface-cli
```bash
huggingface-cli login
```
### Teleop/Recording
```bash
cd src
python -m sra.teleop.main --dataset_path=<dataset_path>
```

### Autonomous loop
```bash
cd src
python -m sra.autonomous.main --dataset_path=<dataset_path>
```