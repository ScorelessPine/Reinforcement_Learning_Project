import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

PPO_Path = os.path.join('Training','Saved Models','PPO_Model_Cartpole')

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
model = PPO.load(PPO_Path, env = env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()