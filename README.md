# Reinforcement_Learning_Project
AIM240 Capstone Project Part 2 Code Submission

# Installation Instructions
Install Python v3.10.4: https://www.python.org/downloads/release/python-3104/

Clone this repository.
If using Git, you may run
```
git clone https://github.com/ScorelessPine/Reinforcement_Learning_Project && cd Reinforcement_Learning_Project
```
Otherwise download the Zip of this repository from https://github.com/ScorelessPine/Reinforcement_Learning_Project/archive/refs/heads/main.zip and unzip into a directory.

Optional: Create a python virtual environment
```
python -m venv RL
. RL/Scripts/activate
```

Install prerequisites

Note: Box2d is not officially supported on Python 3.10, so a package must be manually installed for it to work.
Navigate to https://www.lfd.uci.edu/~gohlke/pythonlibs/
Search for "Box2D-2.3.2-cp310-cp310-win_amd64.whl"
Download it and place it inside the repository.

```
pip install setuptools==66
pip install stable-baselines3[extra]
pip install "Box2D-2.3.2-cp310-cp310-win_amd64.whl"
pip install gym[box2d]                  (NOTE that this *will* error on python 3.10.4, however it installs some requirements for gym box2d that are needed)
pip install pyglet==1.5.27              (NOTE this specific version of pyglet is needed to render the CarRacing environment)
```

Run the evaluation and testing of the models

```
python EvaluateAndTestModels.py
```
