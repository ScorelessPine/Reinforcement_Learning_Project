#====================================== 1. Import Dependencies
import gym
from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import os

environment_name = 'Breakout-v0'
env = gym.make(environment_name)
log_path = os.path.join('Training','Logs')
a2c_path = os.path.join('Training','Saved Models','A2C_Breakout_Model')

#====================================== 2. Test Environment
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

#====================================== 3. Vectorise Environment and Train Model
def vectorise_environment_and_train_model():
    
    env = make_atari_env(environment_name, n_envs=4, seed=0)
    env = VecFrameStack(env, n_stack=4)
    model = A2C('CnnPolicy', env, verbose=1, tensorboard_log=log_path)
    
#====================================== 4. Save and reload model
    model.learn(total_timesteps = 100000)
    model.save(a2c_path)

#====================================== 5. Evaluate and Test
def evaluate_and_test():
    
    env = make_atari_env(environment_name,n_envs=1,seed=0)
    env = VecFrameStack(env, n_stack=4)
    model=A2C.load(a2c_path, env)
    evaluate_policy(model, env, n_eval_episodes=10, render=True)
    env.close()

if __name__ == '__main__':
    
    test_environment()
    vectorise_environment_and_train_model()
    evaluate_and_test()