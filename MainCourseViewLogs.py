#====================================== 7. Viewing Logs in Tensorboard
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


log_path = os.path.join('Training','Logs')
training_log_path = os.path.join(log_path, 'PPO_5')
os.chdir(training_log_path)
os.system("pwd")
os.system("tensorboard --logdir=.")
