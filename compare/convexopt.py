import numpy as np
import cvxpy as cp


def cxp(X_field, n_sensors, vt):
    """
    选择最佳传感器位置，使重建误差最小（D-optimal）

    Args:
        X_field (np.array): (T, N) 观测矩阵，T为时间步，N为位置数
        n_sensors (int): 要选择的传感器数量
        energy_threshold (float): 用于SVD截断的能量阈值（默认0.95）

    Returns:
        list: 传感器编号列表（长度为 n_sensors）
    """

    T, N = X_field.shape
    theta_vectors, r = vt, vt.shape[1]

    beta = cp.Variable(N)

    # 构造信息矩阵表达式（N个 r维向量的加权外积）
    # W = cp.multiply(beta, theta_vectors.T)
    # info_matrix = W @ theta_vectors
    info_matrix = sum(beta[i] *
                      (theta_vectors[i, :, None] @ theta_vectors[i, None, :])
                      for i in range(N))

    constraints = [beta >= 0, beta <= 1, cp.sum(beta) == n_sensors]
    # reg = 1e-6 * cp.pnorm(info_matrix, "fro").value * np.eye(r)
    objective = cp.Maximize(cp.log_det(info_matrix +
                                       1e-6 * np.eye(r)))  # 加噪防止奇异

    problem = cp.Problem(objective, constraints)

    try:
        problem.solve(solver=cp.SCS, verbose=True, max_iters=100000)
    except Exception as e:
        print(f"求解器异常：{e}")
        return None

    if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        beta_val = beta.value
        selected_indices = np.argsort(-beta_val)[:n_sensors]
        return selected_indices.tolist()
    else:
        print(f"优化失败，状态: {problem.status}")
        return None
