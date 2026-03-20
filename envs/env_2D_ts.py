import gymnasium as gym
from gymnasium import spaces
import numpy as np
from data.data_NOAA import load_data, sea_n_sensors
import torch.nn.functional as F
import torch
import pysensors_zmh as ps
from pysensors_zmh.reconstruction import SSPOR

import matplotlib.pyplot as plt



class GridWorldEnv_2D_ts(gym.Env):
    """3x4 Grid Environment with a movable agent"""
    metadata = {"render_modes": ["console"]}

    def __init__(self, ep_length=100, num_agents=10, grid_size= (30, 40),  n_basis_modes_t = 3, n_basis_modes_s = 3):
        super(GridWorldEnv_2D_ts, self).__init__()

        # num agents
        self.num_agents = num_agents
        self.n_basis_modes_t = n_basis_modes_t
        self.n_basis_modes_s = n_basis_modes_s


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
        t_data = np.load("/public/home/yangnan/zhaomeihua/data/oisst/oisst_highres_nw_pacific/train.npy")
        t_data = t_data/36.0
        t_data[np.isnan(t_data)] = 0
        t_data = torch.from_numpy(t_data)
        t_data = t_data.unsqueeze(-1)

        # reshape
        t_data = t_data.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
        t_data = F.interpolate(t_data,
                             size=(self.grid_size[0], self.grid_size[1]),
                             mode='nearest',
                             align_corners=None)
        # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
        t_data = t_data.permute(0, 2, 3, 1)

        s_data = np.load("/public/home/yangnan/zhaomeihua/data/Temperature_sanlity/remote/sss/train_nw_pacific.npy")
        s_data = (s_data-26.0)/(39.0-26.0)
        s_data[np.isnan(s_data)] = 0
        s_data = torch.from_numpy(s_data)
        s_data = s_data.unsqueeze(-1)
        s_data = torch.flip(s_data, dims=[1])
        
        # reshape
        s_data = s_data.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
        s_data = F.interpolate(s_data,
                             size=(self.grid_size[0], self.grid_size[1]),
                             mode='nearest',
                             align_corners=None)
        # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
        s_data = s_data.permute(0, 2, 3, 1)

        mask = ~(t_data[0, :, :, 0] == 0) | ~(s_data[0, :, :, 0] == 0)
        self.mask = mask
        self.t_data_orig = t_data.squeeze(-1)
        self.s_data_orig = s_data.squeeze(-1)

        mask_flat = mask.reshape(-1).numpy()  # 展平为一维掩码数组

        # 获取有效位置的索引
        valid_indices = np.where(mask_flat)[0]
        # print(f"有效位置总数: {len(valid_indices)}")
        # 只使用有效位置的数据
        self.valid_indices = valid_indices

        X_t = t_data.squeeze(-1)
        X_t = X_t.view(X_t.shape[0], -1)
        # self.t_data = X_t

        X_s = s_data.squeeze(-1)
        X_s = X_s.view(X_s.shape[0], -1)
        # self.s_data = X_s

        Xt_valid = X_t[:, valid_indices]
        Xs_valid = X_s[:, valid_indices]

        self.Xt_valid = Xt_valid
        self.Xs_valid = Xs_valid
        #########################加载训练集##########################

        #########################加载测试集##########################
        t_data_test = np.load("/public/home/yangnan/zhaomeihua/data/oisst/oisst_highres_nw_pacific/test.npy")
        t_data_test = t_data_test/36.0
        t_data_test[np.isnan(t_data_test)] = 0
        t_data_test = torch.from_numpy(t_data_test)
        t_data_test = t_data_test.unsqueeze(-1)

        # reshape
        t_data_test = t_data_test.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
        t_data_test = F.interpolate(t_data_test,
                            size=(self.grid_size[0], self.grid_size[1]),
                            mode='nearest',
                            align_corners=None)
        # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
        t_data_test = t_data_test.permute(0, 2, 3, 1)

        t_data_test_orig = t_data_test.squeeze(-1)
        self.t_data_test_orig = t_data_test_orig

        X_t_test = t_data_test.squeeze(-1)
        X_t_test = X_t_test.view(X_t_test.shape[0], -1)
        self.X_t_test = X_t_test

        X_t_test_valid = X_t_test[:, valid_indices]
        self.X_t_test_valid = X_t_test_valid


        s_data_test = np.load("/public/home/yangnan/zhaomeihua/data/Temperature_sanlity/remote/sss/test_nw_pacific.npy")
        s_data_test = (s_data_test-26.0)/(39.0-26.0)
        s_data_test[np.isnan(s_data_test)] = 0
        s_data_test = torch.from_numpy(s_data_test)
        s_data_test = s_data_test.unsqueeze(-1)
        s_data_test = torch.flip(s_data_test, dims=[1])

        # reshape
        s_data_test = s_data_test.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
        s_data_test = F.interpolate(s_data_test,
                            size=(self.grid_size[0], self.grid_size[1]),
                            mode='nearest',
                            align_corners=None)
        # 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
        s_data_test = s_data_test.permute(0, 2, 3, 1)

        s_data_test_orig = s_data_test.squeeze(-1)
        self.s_data_test_orig = s_data_test_orig

        X_s_test = s_data_test.squeeze(-1)
        X_s_test = X_s_test.view(X_s_test.shape[0], -1)
        self.X_s_test = X_s_test

        X_s_test_valid = X_s_test[:, valid_indices]
        self.X_s_test_valid = X_s_test_valid
        #########################加载测试集##########################


        model_t = SSPOR(basis=ps.basis.SVD(n_basis_modes=self.n_basis_modes_t),
                      n_sensors=self.num_agents)
        model_t.fit(Xt_valid.numpy())
        self.model_t = model_t


        model_s = SSPOR(basis=ps.basis.SVD(n_basis_modes=self.n_basis_modes_s),
                      n_sensors=self.num_agents)
        model_s.fit(Xs_valid.numpy())
        self.model_s = model_s


        model_t_2 = SSPOR(basis=ps.basis.SVD(n_basis_modes=self.n_basis_modes_t),
                      n_sensors=self.num_agents//2)
        model_t_2.fit(Xt_valid.numpy())
        self.model_t_2 = model_t_2


        model_s_2 = SSPOR(basis=ps.basis.SVD(n_basis_modes=self.n_basis_modes_s),
                      n_sensors=self.num_agents-self.num_agents//2)
        model_s_2.fit(Xs_valid.numpy())
        self.model_s_2 = model_s_2


        ############# QR分解选择传感器位置###########
        sensors_valid_t = self.model_t_2.get_selected_sensors()
        sensors_valid_s = self.model_s_2.get_selected_sensors()

        # sensors_valid = sensors_valid_t+sensors_valid_s
        sensors_valid = np.concatenate((sensors_valid_t, sensors_valid_s))
        if len(set(sensors_valid)) != self.num_agents:
            print('error')

        # 映射回原始全局索引
        sensors_global = valid_indices[sensors_valid]

        positions = []

        for i in range(sensors_global.shape[0]):
            # 提取当前智能体的位置索引
            
            idx = sensors_global[i]

            # 解码为 (x, y) 坐标
            x = idx // self.grid_size[1]
            y = idx % self.grid_size[1]
            positions.append((x, y))

        self.state = np.zeros(self.t_data_orig.shape[1:])
        for position in positions:
            self.state[position[0], position[1]] = 1

        # ###zmh test###
        # sensors = torch.nonzero(torch.from_numpy(self.state), as_tuple=False)
        # state = sensors.numpy()
        # state =  np.array(state, dtype=np.int32)
        # sensors = [i[0] * self.grid_size[1] + i[1] for i in state]


        # pre = self.model_t.predict_zmh(Xt_valid[:, sensors_valid].numpy(), sensors_valid)
        # error = np.sqrt(np.mean((pre - self.Xt_valid.numpy()) ** 2))
        # ###zmh test###

        # ###zmh test###
        # sensors = torch.nonzero(torch.from_numpy(self.state), as_tuple=False)
        # state = sensors.numpy()
        # state =  np.array(state, dtype=np.int32)
        # sensors = [i[0] * self.grid_size[1] + i[1] for i in state]


        # pre = self.model_s.predict_zmh(Xs_valid[:, sensors_valid].numpy(), sensors_valid)
        # error = np.sqrt(np.mean((pre - self.Xs_valid.numpy()) ** 2))
        # ###zmh test###

        # error = self.reward_func(self.state)



        sensors = np.transpose(np.nonzero(self.state))
        
        coords = []
        if len(sensors)<self.num_agents:
            self.mask1 = self.mask.clone()
            for n in range(self.num_agents-len(sensors)):
                while True:
                    new_x = np.random.randint(0, self.mask1.shape[0],1)[0]
                    new_y = np.random.randint(0, self.mask1.shape[1],1)[0]
                    if self.mask1[new_x,new_y] == True and self.state[new_x,new_y]==0:
                        coords.append([new_x,new_y])
                        self.mask1[new_x,new_y] = 0
                        self.state[new_x,new_y] = 1
                        break
            coords = np.array(coords)  
            sensors = np.concatenate([sensors, coords], axis=0)

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

        pre = self.model_t.predict_zmh(self.Xt_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.Xt_valid.numpy()) ** 2))
        self.train_t_reward = error
        self.wt = round(0.0125/error, 2)
 
        print('QR分解选择传感器位置,训练集温度重建误差: '+str(error))

        pre = self.model_s.predict_zmh(self.Xs_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.Xs_valid.numpy()) ** 2))
        self.train_s_reward = error
        self.ws = round(0.0125/error, 2)
        print('QR分解选择传感器位置,训练集盐度重建误差: '+str(error))


        pre = self.model_t.predict_zmh(self.X_t_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre*36 - self.X_t_test_valid.numpy()*36) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(self.X_t_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, self.valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((self.t_data_test_orig.shape[0], self.t_data_test_orig.shape[1], self.t_data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((self.t_data_test_orig.numpy()*36 - Xt_reconstructed_test*36) ** 2))
        print('QR分解选择传感器位置,温度重建误差: ', error1)

        pre = self.model_s.predict_zmh(self.X_s_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_s_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xs_reconstructed_test = np.zeros(self.X_s_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xs_reconstructed_test[:, self.valid_indices] = pre
        Xs_reconstructed_test = Xs_reconstructed_test.reshape((self.s_data_test_orig.shape[0], self.s_data_test_orig.shape[1], self.s_data_test_orig.shape[2]))
        # error2 = np.sqrt(np.mean((self.s_data_test_orig.numpy() - Xs_reconstructed_test) ** 2))

        s_data_test_orig_denormalized = self.s_data_test_orig.numpy() * (39.0 - 26.0) + 26.0
        Xs_reconstructed_test_denormalized = Xs_reconstructed_test * (39.0 - 26.0) + 26.0
        error2 = np.sqrt(np.mean((s_data_test_orig_denormalized - Xs_reconstructed_test_denormalized) ** 2))
        print('QR分解选择传感器位置,盐度重建误差: ', error2)

        self.state_init = self.state


    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        """
        # x_sens, y_sens = sea_n_sensors(self.mask, self.num_agents)
        # sensors = np.zeros(self.s_data_orig.shape[1:])
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

 
        # print(action, error, reward)


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

        pre_t = self.model_t.predict_zmh(self.Xt_valid[:, sensor_valid].numpy(), sensor_valid)
        error_t = np.sqrt(np.mean((pre_t - self.Xt_valid.numpy())**2))

        pre_s = self.model_s.predict_zmh(self.Xs_valid[:, sensor_valid].numpy(), sensor_valid)
        error_s = np.sqrt(np.mean((pre_s - self.Xs_valid.numpy())**2))

        # error = 0.5 * error_t + error_s
        error = self.wt * error_t + self.ws * error_s

        return error
    

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
        
        pre = self.model_t.predict_zmh(self.X_t_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_t_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(self.X_s_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, self.valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((self.t_data_test_orig.shape[0], self.t_data_test_orig.shape[1], self.t_data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((self.t_data_test_orig.numpy()*36 - Xt_reconstructed_test*36) ** 2))
        # print('QR分解选择传感器位置,温度重建误差: ', error1)

        pre = self.model_s.predict_zmh(self.X_s_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_s_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xs_reconstructed_test = np.zeros(self.X_s_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xs_reconstructed_test[:, self.valid_indices] = pre
        Xs_reconstructed_test = Xs_reconstructed_test.reshape((self.s_data_test_orig.shape[0], self.s_data_test_orig.shape[1], self.s_data_test_orig.shape[2]))
        # error2 = np.sqrt(np.mean((self.s_data_test_orig.numpy() - Xs_reconstructed_test) ** 2))

        s_data_test_orig_denormalized = self.s_data_test_orig.numpy() * (39.0 - 26.0) + 26.0
        Xs_reconstructed_test_denormalized = Xs_reconstructed_test * (39.0 - 26.0) + 26.0
        error2 = np.sqrt(np.mean((s_data_test_orig_denormalized - Xs_reconstructed_test_denormalized) ** 2))


        return error1, error2, Xt_reconstructed_test, Xs_reconstructed_test,  sensors

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
        
        pre = self.model_t.predict_zmh(self.X_t_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_t_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(self.X_s_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, self.valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((self.t_data_test_orig.shape[0], self.t_data_test_orig.shape[1], self.t_data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((self.t_data_test_orig.numpy()*36 - Xt_reconstructed_test*36) ** 2))
        # print('QR分解选择传感器位置,温度重建误差: ', error1)

        pre = self.model_s.predict_zmh(self.X_s_test_valid[:, sensor_valid].numpy(), sensor_valid)
        error = np.sqrt(np.mean((pre - self.X_s_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xs_reconstructed_test = np.zeros(self.X_s_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xs_reconstructed_test[:, self.valid_indices] = pre
        Xs_reconstructed_test = Xs_reconstructed_test.reshape((self.s_data_test_orig.shape[0], self.s_data_test_orig.shape[1], self.s_data_test_orig.shape[2]))
        # error2 = np.sqrt(np.mean((self.s_data_test_orig.numpy() - Xs_reconstructed_test) ** 2))

        s_data_test_orig_denormalized = self.s_data_test_orig.numpy() * (39.0 - 26.0) + 26.0
        Xs_reconstructed_test_denormalized = Xs_reconstructed_test * (39.0 - 26.0) + 26.0
        error2 = np.sqrt(np.mean((s_data_test_orig_denormalized - Xs_reconstructed_test_denormalized) ** 2))

        return error1, error2

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