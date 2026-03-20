import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data.data_NOAA import load_data, sea_n_sensors
import torch.nn.functional as F
import torch
import pysensors_zmh as ps
from pysensors_zmh.reconstruction import SSPOR

import matplotlib.pyplot as plt
import h5py


class GridWorldEnv_2D_temperature(gym.Env):
    """3x4 Grid Environment with a movable agent"""
    metadata = {"render_modes": ["console"]}

    def __init__(self, ep_length=100, n_basis_modes=3, num_agents=10, grid_size= (30, 40)):
        super(GridWorldEnv_2D_temperature, self).__init__()

        # num agents
        self.n_basis_modes = n_basis_modes
        self.num_agents = num_agents

        # Grid size
        self.grid_size = grid_size  # 

        self.action_space = spaces.Discrete(4 * self.grid_size[0] * self.grid_size[1])

        # Observation space: (x, y) coordinates of the agent
        self.observation_space = spaces.Box(
            low=0, 
            high=self.grid_size[1], 
            shape=(self.grid_size), 
            dtype=np.int32
        )

        self.self_init()

        self.ep_length = ep_length
        self.current_step = 0

    def self_init(self):
        
        #########################加载训练集##########################

        #########################读取数据##########################
        f = h5py.File('/public/home/yangnan/zhaomeihua/code/stable-baselines3-contrib-master_v3/data/sst_weekly.mat','r') 
        sst = np.nan_to_num( np.array(f['sst']) )
        
        num_frames = 1914

        sea = np.zeros((num_frames,180,360,1))
        for t in range(num_frames):
            sea[t,:,:,0] = sst[t,:].reshape(180,360,order='F')
        # sea /= 36.0

        data = np.flip(sea, axis=1).copy()
        data = torch.from_numpy(data)

        # 定义裁剪范围
        lat_start, lat_end = 30, 90  # 纬度索引范围 (0°N 到 60°N)
        lon_start, lon_end = 100, 180  # 经度索引范围 (100°E 到 180°E)

        # 裁剪西北太平洋区域
        data = data[:, lat_start:lat_end, lon_start:lon_end, :]
        
        # # reshape
        # data = data.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
        # data = F.interpolate(data,
        #                      size=(self.grid_size[0], self.grid_size[1]),
        #                      mode='nearest',
        #                      align_corners=None)
        # # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
        # data = data.permute(0, 2, 3, 1)

        data = data[:1040]
        #########################读取数据##########################


        mask = ~(data[0, :, :, 0] == 0)
        self.mask = mask
        self.data_orig = data.squeeze(-1)

        X = data.squeeze(-1).contiguous()
        X = X.view(X.shape[0], -1)

        mask_flat = mask.reshape(-1).numpy()  # 展平为一维掩码数组
        # 获取有效位置的索引
        valid_indices = np.where(mask_flat)[0]
        self.valid_indices = valid_indices

        X_valid = X[:, valid_indices]
        self.data = X_valid
        #########################加载训练集##########################
        
        ############# QR分解选择传感器位置###########
        model = SSPOR(basis=ps.basis.SVD(n_basis_modes=self.n_basis_modes),
                      n_sensors=self.num_agents)
        model.fit(X_valid.numpy())
        self.model = model

        sensors_valid = self.model.get_selected_sensors()
        sensors = valid_indices[sensors_valid]

        positions = []

        for i in range(sensors.shape[0]):
            # 提取当前智能体的位置索引
            
            idx = sensors[i]

            # 解码为 (x, y) 坐标
            x = idx // self.grid_size[1]
            y = idx % self.grid_size[1]
            positions.append((x, y))

        self.state = np.zeros(self.data_orig.shape[1:])
        for position in positions:
            self.state[position[0], position[1]] = 1


        #########################加载测试集##########################
        f = h5py.File('/public/home/yangnan/zhaomeihua/code/stable-baselines3-contrib-master_v3/data/sst_weekly.mat','r') 
        sst = np.nan_to_num( np.array(f['sst']) )
        
        num_frames = 1914

        sea = np.zeros((num_frames,180,360,1))
        for t in range(num_frames):
            sea[t,:,:,0] = sst[t,:].reshape(180,360,order='F')
        sea /= 36.0

        data = np.flip(sea, axis=1).copy()
        data = torch.from_numpy(data)

        # 定义裁剪范围
        lat_start, lat_end = 30, 90  # 纬度索引范围 (0°N 到 60°N)
        lon_start, lon_end = 100, 180  # 经度索引范围 (100°E 到 180°E)

        # 裁剪西北太平洋区域
        data = data[:, lat_start:lat_end, lon_start:lon_end, :]
        
        # # reshape
        # data = data.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
        # data = F.interpolate(data,
        #                      size=(self.grid_size[0], self.grid_size[1]),
        #                      mode='nearest',
        #                      align_corners=None)
        # # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
        # data = data.permute(0, 2, 3, 1)

        data_test = data[1040:]


        data_test_orig = data_test.squeeze(-1)
        self.data_test_orig = data_test_orig

        X_test = data_test.squeeze(-1).contiguous()
        X_test = X_test.view(X_test.shape[0], -1)
        self.X_test = X_test

        X_test_valid = X_test[:, valid_indices]
        self.X_test_valid = X_test_valid
        #########################加载测试集##########################

        
        pre = self.model.predict_zmh(X_valid[:, sensors_valid].numpy(), sensors_valid)
        error = np.sqrt(np.mean((pre - X_valid.numpy()) ** 2))
        self.train_reward = error
        self.w = round(0.025/error, 2)
        print('QR分解选择传感器位置,训练集重建误差: ', error)
        print('ws: ', self.w)


        pre = self.model.predict_zmh(X_test_valid[:, sensors_valid].numpy(), sensors_valid)
        error = np.sqrt(np.mean((pre - X_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(X_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((self.data_test_orig.shape[0], self.data_test_orig.shape[1], self.data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((self.data_test_orig.numpy()*36 - Xt_reconstructed_test*36) ** 2))
        print('QR分解选择传感器位置,温度重建误差: ', error1)

        self.state_init = self.state


    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        # x_sens, y_sens = sea_n_sensors(self.data_orig, self.num_agents)
        # sensors = np.zeros(self.data_orig.shape[1:])
        # if len(sensors.shape) == 2:
        #     sensors[x_sens, y_sens] = 1

        # state = sensors

        # self.state = state

        self.state = self.state_init.copy()

        self.current_step = 0

        return np.array(self.state, dtype=np.int32), {}
    

    def create_action_masks(self,  state_temp, mask):
        """
        mask: 二维布尔张量 (H, W), True表示可移动到该位置
        返回: (H, W, 4) 的布尔张量，对应[上, 下, 左, 右]是否有效
        source_mask: 该位置是否有传感器
        target_mask: 目标位置是否有传感器
        action_masks: 是否是边界
        """
        h, w = mask.shape
        action_masks = torch.zeros((h, w, 4), dtype=torch.bool)

        # 生成target_mask：目标位置是否有传感器
        target_mask = torch.from_numpy(state_temp == 1).to(device=mask.device)

        # 生成各方向掩码
        action_masks[1:, :, 0] = mask[:-1, :] & ~target_mask[:-1, :]     # 上：i>0且mask[i-1,j]=True
        action_masks[:-1, :, 1] = mask[1:, :] & ~target_mask[1:, :]   # 下：i<H-1且mask[i+1,j]=True
        action_masks[:, 1:, 2] = mask[:, :-1] & ~target_mask[:, :-1]    # 左：j>0且mask[i,j-1]=True
        action_masks[:, :-1, 3] = mask[:, 1:] & ~target_mask[:, 1:]    # 右：j<W-1且mask[i,j+1]=True

        source_mask = torch.from_numpy(state_temp == 1).unsqueeze(-1).repeat(1, 1, 4)
        action_masks = action_masks & source_mask
        action_masks = action_masks.float().permute(2, 0, 1)  # 源位置必须存在传感器

        return action_masks


    def action_masks(self) -> list[bool]:

        mask = self.create_action_masks(self.state, self.mask)
        return mask

    def step(self, action):
        """
        执行移动动作
        :param action: 元组(source_idx, target_idx) 
                       source_idx和target_idx是展平后的位置索引
        :return: (state, reward, done, info)
        """


        elements_per_action = self.grid_size[0]*self.grid_size[1]
        a = action // elements_per_action
        remainder = action % elements_per_action
        i = remainder // self.grid_size[1]
        j = remainder % self.grid_size[1]


        move = 0
        if self.state[i, j]==1:
            if a == 0 and i>0 and self.state[i-1, j]==0:  # Up
                self.state[i, j] = 0
                self.state[i-1, j] = 1
                move = 1
            elif a == 1 and i<(self.grid_size[0]-1) and self.state[i+1, j]==0:  # Down
                self.state[i, j] = 0
                self.state[i+1, j] = 1
                move = 1
            elif a == 2 and j>0 and self.state[i, j-1]==0:  # Left
                self.state[i, j] = 0
                self.state[i, j-1] = 1
                move = 1
            elif a == 3 and j<(self.grid_size[1]-1) and self.state[i, j+1]==0:  # Right
                self.state[i, j] = 0
                self.state[i, j+1] = 1
                move = 1

        if move == 0:
            reward = -1

        if move == 1:
            error = self.reward_func(self.state)

            if error>0.03:
                reward = -(error-0.03)   
            elif error>0.025:
                reward = -20*(error-0.03)
            elif error>0.02:
                reward = -40*error+1.1
            else:
                reward = -70*error+1.7

            if reward<-0.2:
                reward = -0.2

        done = False

        self.current_step += 1
        truncated = self.current_step >= self.ep_length

    
        # 构建info字典
        info = {}
        info['error'] = error

        return self.state, reward, done, truncated, info


    def reward_func(self, state):
        sensors = np.transpose(np.nonzero(state))

        for i in range(self.num_agents):
            assert self.mask[int(sensors[i][0]), int(sensors[i][1])], 'Invalid state'
        
        sensors_global = [i[0] * self.grid_size[1] + i[1] for i in sensors]

        # 将全局索引映射为有效位置索引
        sensor_valid = []
        for g in sensors_global:
            # 在有效位置索引中查找全局索引的位置
            idx = np.where(self.valid_indices == g)[0]
            if len(idx) > 0:
                sensor_valid.append(idx[0])
            else:
                # 如果找不到，抛出异常（理论上不会发生，因为mask已检查）
                raise ValueError(f"Global index {g} not found in valid indices")
            
        pre = self.model.predict_zmh(self.data[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.data.numpy())**2))

        return self.w*error
    

    def plot(self, state):
        sensors = np.transpose(np.nonzero(state))

        for i in range(self.num_agents):
            assert self.mask[int(sensors[i][0]), int(sensors[i][1])], 'Invalid state'
        
        sensors_global = [i[0] * self.grid_size[1] + i[1] for i in sensors]

        # 将全局索引映射为有效位置索引
        sensor_valid = []
        for g in sensors_global:
            # 在有效位置索引中查找全局索引的位置
            idx = np.where(self.valid_indices == g)[0]
            if len(idx) > 0:
                sensor_valid.append(idx[0])
            else:
                # 如果找不到，抛出异常（理论上不会发生，因为mask已检查）
                raise ValueError(f"Global index {g} not found in valid indices")
        
        pre = self.model.predict_zmh(self.X_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(self.X_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, self.valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((self.data_test_orig.shape[0], self.data_test_orig.shape[1], self.data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((self.data_test_orig.numpy()*36 - Xt_reconstructed_test*36) ** 2))
        # print('QR分解选择传感器位置,温度重建误差: ', error1)


        return error1, Xt_reconstructed_test, sensors

    def inference(self, state):
        sensors = np.transpose(np.nonzero(state))

        for i in range(self.num_agents):
            assert self.mask[int(sensors[i][0]), int(sensors[i][1])], 'Invalid state'
        
        sensors_global = [i[0] * self.grid_size[1] + i[1] for i in sensors]

        # 将全局索引映射为有效位置索引
        sensor_valid = []
        for g in sensors_global:
            # 在有效位置索引中查找全局索引的位置
            idx = np.where(self.valid_indices == g)[0]
            if len(idx) > 0:
                sensor_valid.append(idx[0])
            else:
                # 如果找不到，抛出异常（理论上不会发生，因为mask已检查）
                raise ValueError(f"Global index {g} not found in valid indices")
        
        pre = self.model.predict_zmh(self.X_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(self.X_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, self.valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((self.data_test_orig.shape[0], self.data_test_orig.shape[1], self.data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((self.data_test_orig.numpy()*36 - Xt_reconstructed_test*36) ** 2))
        # print('QR分解选择传感器位置,温度重建误差: ', error1)

        return error1

    def render(self, save_dir):
        state = self.state

        plt.figure(figsize=(20, 20))
        plt.imshow(
            self.data_orig[0],
            cmap='bwr',
        )  # 使用'bwr'颜色映射

        for i_p in range(len(state)):
            x, y = state[i_p][0], state[i_p][1]
            # cv2.circle(gen_image, (x, y), radius=5, color=(0, 255, 0), thickness=2)
            plt.scatter(y, x, color='black', s=900)

        #plt.imshow(combined_image, cmap='bwr', vmin=vmin, vmax=vmax)  # 使用'bwr'颜色映射
        # plt.colorbar()  # 显示颜色条
        plt.axis('off')  # 关闭坐标轴显示

        # 保存图像
        plt.savefig(save_dir, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close()  # 关闭当前图形，释放内存



if __name__ == '__main__':
    env = GridWorldEnv()
    state = env.reset()  # 重置环境
    env.step(1)