#====================================== 6. Test Model
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v0'
env = gym.make(environment_name)
PPO_Path = os.path.join('Training','Saved Models','PPO_Model_Cartpole')
model = PPO.load(PPO_Path)

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()