
# modifide by Xiaobo Wu 2025 10 28,29
import numpy as np
import matplotlib.pyplot as plt


def get_energy(P_grid, v_const=10.0, eps=1e-12, wind=None):
    """
    计算无人机沿给定路径的最小能量消耗（准静态模型）
    
    参数:
        P_grid : (N, 3) array, 网格坐标 [x_grid, y_grid, z_grid]
        v_const: 恒定地速 (m/s)
        eps     : 数值保护小量
        wind    : (Nx, Ny, Nz, 3) array, 风速场 (可选)
    
    返回:
        E : 总能量 (J)
    """
    
    # === 物理常数 ===
    rho = 1.18          # 空气密度 (kg/m³)
    Cd  = 1.0           # 机身阻力系数
    A_frame = 0.25      # 机身迎风面积 (m²)
    m = 4.0             # 无人机质量 (kg)
    A_rotor = 0.45      # 总旋翼盘面积 (m²)
    P_p_hover = 65.0    # 悬停轮廓功率 (W)
    g = 9.81            # 重力加速度 (m/s²)
    
    g_vec = np.array([0.0, -g, 0.0])

    # === 路径点映射到真实坐标 ===
    scale = np.array([4.0, 1.5, 4.0])  # 网格 → 真实距离 (m)
    P = P_grid * scale
    n = P.shape[0]

    # === 路径长度与时间 ===
    dP = P[1:] - P[:-1]
    ds = np.linalg.norm(dP, axis=1)
    ds = np.where(ds < eps, eps, ds)           # 防止除零
    s = np.zeros(n)
    s[1:] = np.cumsum(ds)

    dt = ds / v_const
    t = np.zeros(n)
    t[1:] = np.cumsum(dt)

    # === 地速 v_g：时间加权平均 + 强制恒速 ===
    v_g = np.zeros_like(P)
    
    # 端点：沿路径方向
    v_g[0]  = dP[0]  / ds[0]
    v_g[-1] = dP[-1] / ds[-1]

    # 中间点：加权平均（避免速度突变）
    if n > 2:
        dt1 = t[1:-1] - t[:-2]
        dt2 = t[2:]   - t[1:-1]
        dp1 = P[1:-1] - P[:-2]
        dp2 = P[2:]   - P[1:-1]
        
        v_prev = dp1 / dt1[:, None]
        v_next = dp2 / dt2[:, None]
        v_g[1:-1] = (v_prev * dt1[:, None] + v_next * dt2[:, None]) / (dt1 + dt2)[:, None]

    # 强制地速大小为 v_const
    v_g_norm = np.linalg.norm(v_g, axis=1, keepdims=True)
    v_g = np.where(v_g_norm > eps, v_g / v_g_norm * v_const, 0)

    # === 地加速度 a_g：中心差分 ===
    a_g = np.zeros_like(P)
    if n > 2:
        a_g[1:-1] =  2 * ( (dp2 / dt2[:, None]) - (dp1 / dt1[:, None]) ) / (dt1 + dt2)[:, None]
    a_g[0] = a_g[1] if n > 1 else 0
    a_g[-1] = a_g[-2] if n > 1 else 0

    # === 空速 v_a：考虑风场 ===
    if wind is None:
        v_a = v_g.copy()
    else:
        x_grid, z_grid, y_grid = np.round(P_grid).astype(int).T
        v_w = wind[x_grid, z_grid, y_grid]  # 假设 wind 索引顺序 (x, z, y)
        v_a = v_g - v_w

    v_a_norm = np.linalg.norm(v_a, axis=1, keepdims=True)
    v_a_unit = np.where(v_a_norm > eps, v_a / v_a_norm, 0)

    # === 寄生阻力 D（修正！）===
    D = -0.5 * rho * Cd * A_frame * v_a_norm * v_a  # 正确：||v_a|| * v_a

    # === 推力 T_vec ===
    T_vec = m * a_g - m * g_vec - D
    T = np.linalg.norm(T_vec, axis=1, keepdims=True)
    t_hat = np.where(T > eps, T_vec / T, 0)

    # === 爬升速度 v_c：空速在推力方向投影 ===
    v_c = np.sum(v_a * t_hat, axis=1, keepdims=True)

    # === 诱导速度 v_i（动量理论）===
    sqrt_arg = (0.5 * v_c)**2 + T / (2.0 * rho * A_rotor)
    v_i = -0.5 * v_c + np.sqrt(np.where(sqrt_arg < 0, 0, sqrt_arg))

    # === 功率分量 ===
    P_ideal = (T * (v_i + v_c))[:, 0]                                  # 诱导 + 爬升
    P_d = (0.5 * rho * Cd * A_frame * (v_a_norm**3))[:, 0]             # 寄生功率
    P_p = P_p_hover * ((T / (m * g)) ** 1.5)[:, 0]                      # 轮廓功率（经验）

    P_total = P_ideal + P_d + P_p

    # === 能量积分（梯形法则）===
    P_avg = 0.5 * (P_total[1:] + P_total[:-1])
    E = np.sum(P_avg * dt)

    return E