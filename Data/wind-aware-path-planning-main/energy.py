import numpy as np
import matplotlib.pyplot as plt


def get_energy(P_grid, v_const=10.0, eps=1e-12, wind = None):

    rho = 1.18        # Air density (kg/m^3)
    Cd  = 1.0         # Drag coefficient
    Aframe = 0.25     # Frontal area of the drone frame (m^2)
    m=4.0             # Mass of the drone (kg)
    A=0.45            # Total rotor disk area (m^2)
    P_p_hover=65.0    # Profile power at hover (Watts)
    g=9.81            # Gravitational acceleration (m/s^2)
    
    g_vec = np.array([0.0, -g, 0.0])

    P = P_grid * np.array([4, 1.5, 4]) # Scale grid coordinates to real-world coordinates (meters).
    n = P.shape[0] 

    dP = P[1:] - P[:-1]
    ds = np.linalg.norm(dP, axis=1) # Lengths for each path segment
    ds = np.where(ds < eps, eps, ds)
    s = np.zeros(n)
    s[1:] = np.cumsum(ds)

    dt = ds / v_const
    t = np.zeros(n)
    t[1:] = np.cumsum(dt) # Time to traverse each segment.

    # Calculate ground velocity (v_g) vector ----------------------------------------------------
    v_g = np.zeros_like(P)

    v_g[0]  = dP[0]  / ds[0]
    v_g[-1] = dP[-1] / ds[-1]

    dt1 = t[1:-1] - t[:-2]    
    dt2 = t[2:]   - t[1:-1]   
    dp1 = P[1:-1] - P[:-2]   
    dp2 = P[2:]   - P[1:-1]   

    v_g[1:-1] = ((dp2 / dt2[:, None]) * dt1[:, None] +
             (dp1 / dt1[:, None]) * dt2[:, None]) / (dt1 + dt2)[:, None]

    v_g_norm = np.linalg.norm(v_g, axis=1, keepdims=True)
    v_g = np.where(v_g_norm > 0, v_g / v_g_norm * v_const, 0)

    # Calculate ground acceleration (a_g) vector ----------------------------------------------------
    a_g = np.zeros_like(P)
    a_g[1:-1] = 2 * ((dp2 / dt2[:, None]) - (dp1 / dt1[:, None])) / (dt1 + dt2)[:, None]
    a_g[0] = a_g[1]
    a_g[-1] = a_g[-2]

    # Determine airspeed (v_a) by accounting for wind -------------------------------------------
    if wind is None:
        v_a = v_g
    else:
        x_grid, z_grid, y_grid = np.round(P_grid).astype(int).T
        v_w = wind[x_grid, z_grid, y_grid]
        v_a = v_g - v_w

    # Calculate energy  -------------------------------------------------------------------------
    v_a_norm = np.linalg.norm(v_a, axis=1, keepdims=True) 
    D = -0.5 * rho * Cd * Aframe * (v_a_norm * v_a) # Parasite drag force vector

    T_vec = m * a_g - m * g_vec - D # Calculate required thrust vector
    T = np.linalg.norm(T_vec, axis=1, keepdims=True)

    t_hat = T_vec / np.where(T < eps, eps, T) # Get the thrust unit vector

    v_c = np.sum(v_a * t_hat, axis=1, keepdims=True) # Calculate climb velocity (v_c): airspeed projected onto the thrust axis

    sqrt_arg = (0.5 * v_c)**2 + T / (2.0 * rho * A)
    v_i = -0.5 * v_c + np.sqrt(sqrt_arg) # Calculate induced velocity (v_i) using the momentum theory equation

    P_ideal = (T * (v_i + v_c))[:, 0] # Ideal power (P_ideal = induced power + climb power)
    P_d = (0.5 * rho * (v_a_norm**3) * Cd * Aframe)[:, 0] # Parasite power (power to overcome parasite drag)
    P_p = (P_p_hover * ((T / (m * g)) ** 1.5))[:, 0] # Profile power (power to overcome blade profile drag), modeled empirically.

    P_total = P_ideal + P_d + P_p # Add up the three main power components

    P_avg = 0.5 * (P_total[1:] + P_total[:-1])
    E = np.sum(P_avg * dt) # Integrate power over time to find total energy

    return E