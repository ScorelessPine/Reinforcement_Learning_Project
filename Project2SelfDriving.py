import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import os

environment_name = 'CarRacing-v0'
env = gym.make(environment_name)
log_path = os.path.join('Training','Logs')
ppo_path = os.path.join('Training','Saved Models','PPO_Driving_Model')

def test_environment():
    
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

def train_model():
    env=gym.make(environment_name)
    env=DummyVecEnv([lambda: env])
    model = PPO('CnnPolicy', env, verbose=1,tensorboard_log=log_path)
    
    model.learn(total_timesteps=100000)
    model.save(ppo_path)

def evaluate_and_test():
    model = PPO.load(ppo_path, env)
    evaluate_policy(model, env, n_eval_episodes=10, render=True)
    env.close()

if __name__ == '__main__':
    test_environment()
    train_model()
    evaluate_and_test()
    
    