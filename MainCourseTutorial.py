#====================================== 1. Import dependencies
import os
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

#====================================== 2. Load Environment
environment_name = 'CartPole-v0'
env = gym.make(environment_name)

episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0
    
    while not done:
        env.render()
        action = env.action_space.sample()
        n_state, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()

#====================================== 3. Train an RL Model
log_path = os.path.join('Training','Logs')

env = gym.make(environment_name)
env = DummyVecEnv([lambda:env])
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000)

#====================================== 4. Save and Reload Model
PPO_Path = os.path.join('Training','Saved Models','PPO_Model_Cartpole')
model.save(PPO_Path)
del model
model = PPO.load(PPO_Path, env=env)
model.learn(total_timesteps=1000)