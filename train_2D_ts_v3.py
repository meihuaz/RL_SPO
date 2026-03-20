from stable_baselines3 import PPO
from gym_rec.envs_zmh.env_2D_ts_v4 import GridWorldEnv_2D_ts
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.ppo_mask.policies import MaskableActorCriticCnnPolicy
from torch import nn
import torch as th
import gym

from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import os
from model.custom_CNN import CustomCNN



# num agents
# num_agents = 10
# n_basis_modes_t=3
# n_basis_modes_s=3

# num_agents = 20
# n_basis_modes_t=5
# n_basis_modes_s=4

# num_agents = 30
# n_basis_modes_t=8
# n_basis_modes_s=6

# num_agents = 40
# n_basis_modes_t=8
# n_basis_modes_s=7

# num_agents = 50
# n_basis_modes_t=16
# n_basis_modes_s=13

num_agents = 60
n_basis_modes_t=15
n_basis_modes_s=17

# Grid size
grid_size = (30, 40)

env = GridWorldEnv_2D_ts(num_agents=num_agents, grid_size=grid_size,  n_basis_modes_t = n_basis_modes_t, n_basis_modes_s = n_basis_modes_s, ep_length=200)
 

base_dir = 'log/GridWorldEnv_2D_ts/'+str(grid_size[0])+"_"+str(grid_size[1])+'_'+str(num_agents)+'/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


log_file = base_dir+'reward.txt'
log_entries = [
    f"n_basis_modes_t: {n_basis_modes_t}",
    f"n_basis_modes_s: {n_basis_modes_s}",
    f"num_agents: {num_agents}"
]
with open(log_file, "a") as f:
    for entry in log_entries:
        f.write(entry + "\n")
    # 添加一个空行分隔不同更新步骤（可选）
    f.write("\n")



policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=env.action_space.n),
    net_arch= [dict(pi=[1024, 512, 256], vf=[1024, 512, 256])]
)

model = MaskablePPO("CnnPolicy",
        env,
        learning_rate = 0.0001,
        n_steps = 4000,
        batch_size = 1000,
        n_epochs = 20,
        gamma = 0.99,
        gae_lambda = 0.95,
        clip_range = 0.2,
        ent_coef = 0.01,
        verbose=1,
        policy_kwargs = policy_kwargs,
        log_file = base_dir+'reward.txt',
        checkpoint_path = base_dir+'checkpoint',
        obs_path = base_dir+'obs'    )

model.learn(total_timesteps=8000000)
model.save(base_dir+'checkpoint_final')


# # Test the trained agent
# # using the vecenv
# model.load("GridWorldEnv3040_agent3")
# model = MaskablePPO.load("GridWorldEnv3040_agent10")

# obs = env.reset()
# while True:
#     # Retrieve current action mask
#     action_masks = get_action_masks(env)
#     action, _states = model.predict(obs, action_masks=action_masks)
#     obs, reward, terminated, truncated, info = env.step(action)

#     print(obs, reward)
