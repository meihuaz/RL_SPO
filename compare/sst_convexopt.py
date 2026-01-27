from convexopt import cxp
import numpy as np
import torch
import pysensors as ps
from pysensors.reconstruction import SSPOR
import torch.nn.functional as F
import pysensors as ps
import os

num_agents = 10
n_basis_modes = 5
seed        = 0
grid_size = (30, 40)


#########################加载训练集##########################
data = np.load("data/train.npy")
data[np.isnan(data)] = 0
data = torch.from_numpy(data)
data = data.unsqueeze(-1)

# reshape
data = data.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
data = F.interpolate(data,
                        size=(grid_size[0], grid_size[1]),
                        mode='nearest',
                        align_corners=None)
# 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
data = data.permute(0, 2, 3, 1)

mask = ~(data[0, :, :, 0] == 0)
data_orig = data.squeeze(-1)

X = data.squeeze(-1)
X = X.view(X.shape[0], -1)

mask_flat = mask.reshape(-1).numpy()  # 展平为一维掩码数组
# 获取有效位置的索引
valid_indices = np.where(mask_flat)[0]

X_valid = X[:, valid_indices]
data = X_valid
#########################加载训练集##########################


#########################加载测试集##########################
data_test = np.load("data/test.npy")
data_test[np.isnan(data_test)] = 0
data_test = torch.from_numpy(data_test)
data_test = data_test.unsqueeze(-1)


# reshape
data_test = data_test.permute(0, 3, 1, 2)  # 将 D 移到第三维，变为 [B, C, D, H, W]
data_test = F.interpolate(data_test,
                    size=(grid_size[0], grid_size[1]),
                    mode='nearest',
                    align_corners=None)
# 将调整大小后的 Tensor permute 回 [B, C, H1, W1, D1]
data_test = data_test.permute(0, 2, 3, 1)

data_test_orig = data_test.squeeze(-1)

X_test = data_test.squeeze(-1)
X_test = X_test.view(X_test.shape[0], -1)

X_test_valid = X_test[:, valid_indices]
#########################加载测试集##########################


base_dir = 'log/convexopt/2d_t/'+str(grid_size[0])+"_"+str(grid_size[1])+'/'
if not os.path.exists(base_dir):
    os.makedirs(base_dir, exist_ok=True)

log_path = os.path.join(base_dir, 'log.txt')
log_file = open(log_path, 'a', buffering=1)
def write_to_log(text):
    if log_file is not None:
        print(text, file=log_file)
    print(text)


for num_agents in [5, 10, 20, 40]:
    min_error = 100

    for n_basis_modes in range(3, num_agents):

        write_to_log('\n')
        write_to_log('num_agents: ' + str(num_agents) )
        write_to_log('n_basis_modes: '+ str( n_basis_modes))

        ############# 定义模型###########
        model = SSPOR(basis=ps.basis.SVD(n_basis_modes=n_basis_modes),
                        n_sensors=num_agents)
        model.fit(X_valid.numpy())

        sensors_valid = cxp(X_valid.numpy(), num_agents, model.basis_matrix_)

        ############# 计算重建误差 ###########
        pre = model.predict_zmh(X_test_valid[:, sensors_valid].numpy(), sensors_valid)
        error = np.sqrt(np.mean((pre - X_test_valid.numpy()) ** 2))

        # 2. 初始化全零矩阵
        Xt_reconstructed_test = np.zeros(X_test.shape, dtype=np.float32)

        # 3. 将有效数据填充到对应位置
        Xt_reconstructed_test[:, valid_indices] = pre
        Xt_reconstructed_test = Xt_reconstructed_test.reshape((data_test_orig.shape[0], data_test_orig.shape[1], data_test_orig.shape[2]))
        error1 = np.sqrt(np.mean((data_test_orig.numpy() - Xt_reconstructed_test) ** 2))
        write_to_log('凸优化选择传感器位置,测试集重建误差: ' + str(error1))
        
        if error1<min_error:
            min_error=error1
            min_error_t = error1

            best_basis = n_basis_modes

    write_to_log('\n')
    write_to_log('num_agents: ' + str(num_agents) )
    write_to_log('min_error_t: ' + str(min_error_t) )
    write_to_log('best_basis: ' + str(best_basis) )


