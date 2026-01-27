from stable_baselines3 import PPO
from gym_rec.envs_zmh.env_2D_temperature_v2 import GridWorldEnv_2D_temperature
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
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter  # 导入格式化工具

plt.rcParams['font.size'] = 16  # 设置全局字体大小

# num agents
num_agents = 5
n_basis_modes = 3

# num_agents = 10
# n_basis_modes = 5

# Grid size 
grid_size = (30, 40)

env = GridWorldEnv_2D_temperature(num_agents=num_agents, n_basis_modes=n_basis_modes, grid_size=grid_size)


base_dir = 'save_img/GridWorldEnv_2D_t/'+str(grid_size[0])+"_"+str(grid_size[1])+'_'+str(num_agents)+'/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir)


# Test the trained agent
model = MaskablePPO.load('/public/home/yangnan/zhaomeihua/code/stable-baselines3-contrib-master_v3/log/GridWorldEnv_2D_temperature/30_40_5/checkpoint_928000_1819.3464', env=env)
# model = MaskablePPO.load('/public/home/yangnan/zhaomeihua/code/stable-baselines3-contrib-master_v3/log/GridWorldEnv_2D_temperature/30_40_10/checkpoint_996000_1223.46', env=env)


test_running_reward = 0
total_test_episodes = 1
max_ep_len = 90

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    
for ep in range(1, total_test_episodes+1):
    ep_reward = 0
    # obs, _ = env.reset()
    obs = np.load('/public/home/yangnan/zhaomeihua/code/stable-baselines3-contrib-master_v3/log/GridWorldEnv_2D_temperature/30_40_5/obs_928000_1819.3464.npy')
    # obs = np.load("/public/home/yangnan/zhaomeihua/code/stable-baselines3-contrib-master_v3/log/GridWorldEnv_2D_temperature/30_40_10/obs_996000_1223.46.npy")
    env.state = obs[0]
    reward_max = 0
    min_error = 1

    for t in range(1, max_ep_len+1):
        action_masks = get_action_masks(env)
        action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward

        error = env.inference(obs)

        if error<min_error:
            reward_max = reward
            min_error = error
            best_state = obs.copy()

        print(error)

    # error = env.reward_func(obs)
    # print('min_error: ', min_error)
    # print('reward_max: ', reward_max)

    test_running_reward += ep_reward
    print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
    ep_reward = 0

    error, reconstructed_pre, sensors = env.plot(best_state)
    print('min_error: ', min_error)
    reconstructed_X = env.data_orig

env.close()

print("============================================================================================")

avg_test_reward = test_running_reward / total_test_episodes
avg_test_reward = round(avg_test_reward, 2)
print("average test reward : " + str(avg_test_reward))

print("============================================================================================")

reconstructed_pre = reconstructed_pre * 36
reconstructed_X = reconstructed_X * 36

np.save(base_dir + "sensors.npy", sensors) 

############# 可视化重建结果###########
visu = True
if visu:
    padding = 30

    for i in range(2):
        save_dir = base_dir + str(i) + "_fig.png"

        # 将输入和目标图像转换为 numpy 数组
        gen_image = reconstructed_pre[i]
        input_image = reconstructed_X[i]


        # 1. 加载 .nc 文件
        data = input_image

        # 1. 加载 .nc 文件
        fine_field = gen_image

        # coordinates_path = "/home/zmh/toolkit/data/0_sensor_positions.npy"
        # coordinates = np.load(coordinates_path)

        # coordinates[:, 0] = coordinates[:, 0]/192*180-90
        # coordinates[:, 1] = coordinates[:, 1]/384*360


        fine_field[data==0] = np.nan
        data[data==0] = np.nan


        # channel = 0
        # filtered_coordinates = coordinates[coordinates[:, 2] == channel]

        # 提取数据
        data1 = data  # 第一个图像数据
        data2 = fine_field  # 第二个图像数据

        # 检查数据形状
        print("Data shape:", data1.shape, data2.shape)

        # 创建地图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), subplot_kw={'projection': ccrs.PlateCarree()})

        # 设置地图范围
        map_extent = [100, 180, 0, 60]

        # 添加海岸线、国家边界等
        for ax in [ax1, ax2]:
            # ax.add_feature(cfeature.COASTLINE)
            # 设置地图范围
            ax.set_extent([100, 180, 0, 60], crs=ccrs.PlateCarree())  # 西北太平洋范围

            # 设置经纬度刻度
            ax.set_xticks(np.arange(100, 181, 20), crs=ccrs.PlateCarree())  # 经度刻度
            ax.set_yticks(np.arange(0, 61, 10), crs=ccrs.PlateCarree())     # 纬度刻度
            ax.xaxis.set_major_formatter(LongitudeFormatter())
            ax.yaxis.set_major_formatter(LatitudeFormatter())

        # 设置全局的 vmin 和 vmax
        vmin = np.nanmin(data1)
        vmax = np.nanmax(data1)

        for i_p in range(len(sensors)):
            x, y = 60-sensors[i_p][0]*60/env.grid_size[0], sensors[i_p][1]*80/env.grid_size[1]+100
            # cv2.circle(gen_image, (x, y), radius=5, color=(0, 255, 0), thickness=2)
            ax1.scatter(y, x, color='black', s=100)

        # 可视化数据1
        im1 = ax1.imshow(data1, extent=[100, 180, 0, 60], transform=ccrs.PlateCarree(), cmap='bwr', alpha=1, vmin=vmin, vmax=vmax)
        cbar1 = plt.colorbar(im1, ax=ax1, orientation='vertical', pad=0.03, fraction=0.035)
        cbar1.set_label('Temperature (°C)')
        ax1.set_title("Ground Truth")

        # 在地图上标记坐标点
        # ax1.scatter(filtered_coordinates[:, 1], filtered_coordinates[:, 0], color='blue', s=10, 
        #             transform=ccrs.PlateCarree())
        
        # x_sens = x_sens
        # y_sens = y_sens            


        # 可视化数据2
        im2 = ax2.imshow(data2, extent=[100, 180, 0, 60], transform=ccrs.PlateCarree(), cmap='bwr', alpha=1, vmin=vmin, vmax=vmax)
        cbar2 = plt.colorbar(im2, ax=ax2, orientation='vertical', pad=0.03, fraction=0.035)
        cbar2.set_label('Temperature (°C)')
        ax2.set_title("Reconstruction")

        # 显示地图
        plt.tight_layout()
        plt.savefig(save_dir, dpi=600)
