#====================================== 10. Using an Alternate Algorithm
import os
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

save_path = os.path.join('Training', 'Saved Models')
environment_name = 'CartPole-v0'
env = gym.make(environment_name)
log_path = os.path.join('Training','Logs')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)


#net_arch = [dict(pi=[128,128,128,128], vf = [128,128,128,128])]
model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000, callback=eval_callback)
