
# Edited by Xiaobo Wu 2025 10 28,29
# Module for calculating energy consumption of UAV (unmanned aerial vehicle)
import numpy as np
import matplotlib.pyplot as plt


def get_energy(P_grid, v_const=10.0, eps=1e-12, wind=None):
    """
    Calculate minimum energy consumption of UAV along given flight path (quasi-static model)
    
    This model uses basic aerodynamic theory to calculate the required power
    for UAV flight operations, including:
    - Lift force (induced power)
    - Aerodynamic drag (parasitic drag)
    - Rotor blade power (profile power)
    
    Parameters:
        P_grid : array (N, 3), grid coordinates [x_grid, y_grid, z_grid]
                 Represents N points on the flight path in grid coordinate system
        v_const: constant ground velocity (m/s)
                 Desired movement speed of UAV relative to ground
        eps    : small numerical protection value
                 Prevent division by zero in calculations
        wind   : array (Nx, Ny, Nz, 3), wind field (optional)
                 Wind velocity vector at each point in 3D space
    
    Returns:
        E : total energy consumption (Joules)
    """
    
    # ============================================================================
    # PART 1: PHYSICAL CONSTANTS
    # ============================================================================
    # Physical parameters of UAV and environment
    
    rho = 1.18          # Air density at sea level, standard conditions (kg/m³)
    Cd  = 1.0           # Aerodynamic drag coefficient of aircraft body (dimensionless)
                        # Typical value for small UAVs
    A_frame = 0.25      # Frontal area of aircraft body (m²)
                        # Cross-sectional area perpendicular to flight direction
    m = 4.0             # Total mass of UAV including battery and payload (kg)
    A_rotor = 0.45      # Total swept area of rotor blades (m²)
                        # For quadcopter: π * (blade diameter)² * number of blades
    P_p_hover = 65.0    # Power consumption during hover (W)
                        # Includes friction losses and motor efficiency
    g = 9.81            # Gravitational acceleration (m/s²)
    
    # Gravity vector (pointing downward along negative y-axis)
    g_vec = np.array([0.0, -g, 0.0])

    # ============================================================================
    # PART 2: COORDINATE CONVERSION AND INITIALIZATION
    # ============================================================================
    # Convert from discrete grid coordinates to actual coordinates in continuous space
    
    scale = np.array([4.0, 1.5, 4.0])  # Scale factor: grid → actual distance (m)
                                        # [scale_x, scale_y, scale_z]
                                        # Number of meters per grid unit
    P = P_grid * scale  # Convert flight path points to actual coordinates (m)
    n = P.shape[0]      # Total number of waypoints on flight path

    # ============================================================================
    # PART 3: CALCULATE FLIGHT PATH LENGTH AND TIME
    # ============================================================================
    # Calculate Euclidean distance between consecutive waypoints
    
    dP = P[1:] - P[:-1]  # Displacement vector between consecutive points (m)
                         # Shape: (n-1, 3)
    
    ds = np.linalg.norm(dP, axis=1)  # Length of each flight segment (m)
                                       # ds[i] = ||P[i+1] - P[i]||
    
    ds = np.where(ds < eps, eps, ds)  # Protection: replace very short segments with eps
                                       # Prevent division by zero in subsequent calculations
    
    # Calculate cumulative distance along flight path (m)
    s = np.zeros(n)      # Distance from start point to each waypoint
    s[1:] = np.cumsum(ds)  # s[i] = total distance from point 0 to point i

    # Calculate flight time between waypoints (assume constant velocity)
    dt = ds / v_const    # Time for each segment = distance / velocity (s)
    
    # Calculate cumulative time
    t = np.zeros(n)      # Time from start to each waypoint
    t[1:] = np.cumsum(dt)  # t[i] = total time from point 0 to point i

    # ============================================================================
    # PART 4: CALCULATE GROUND VELOCITY
    # ============================================================================
    # Ground velocity is UAV velocity relative to ground (including wind effects)
    # Use weighted averaging method to ensure continuous velocity
    
    v_g = np.zeros_like(P)  # Initialize ground velocity array (m/s)
                            # Shape: (n, 3) - 3D velocity vector at each waypoint
    
    # Handle start and end points: velocity parallel to flight direction
    v_g[0]  = dP[0]  / ds[0]   # Start point: direction of first segment
    v_g[-1] = dP[-1] / ds[-1]  # End point: direction of last segment

    # Handle intermediate points: weighted average by time
    # Purpose: smooth velocity, avoid sudden changes that cause large acceleration
    if n > 2:
        # Time intervals between waypoints
        dt1 = t[1:-1] - t[:-2]  # Time from previous to current point
        dt2 = t[2:]   - t[1:-1]  # Time from current to next point
        
        # Displacement vectors
        dp1 = P[1:-1] - P[:-2]  # From point i-1 to i
        dp2 = P[2:]   - P[1:-1]  # From point i to i+1
        
        # Average velocity of previous and next segments
        v_prev = dp1 / dt1[:, None]  # Velocity on previous segment
        v_next = dp2 / dt2[:, None]  # Velocity on next segment
        
        # Weighted average: longer segment (in time) has higher weight
        v_g[1:-1] = (v_prev * dt1[:, None] + v_next * dt2[:, None]) / (dt1 + dt2)[:, None]

    # Normalize velocity magnitude to v_const
    # Ensure UAV flies at constant velocity as required
    v_g_norm = np.linalg.norm(v_g, axis=1, keepdims=True)  # Velocity magnitude ||v_g||
    v_g = np.where(v_g_norm > eps, v_g / v_g_norm * v_const, 0)  # v_g = (v_g/||v_g||) * v_const

    # ============================================================================
    # PART 5: CALCULATE GROUND ACCELERATION
    # ============================================================================
    # Acceleration needed to change ground velocity between waypoints
    # Use central difference to estimate derivative
    
    a_g = np.zeros_like(P)  # Initialize acceleration array (m/s²)
    
    # Calculate acceleration for intermediate points using central difference
    # a = dv/dt = (v_next - v_prev) / (dt_prev + dt_next)
    if n > 2:
        # Central difference formula is more accurate than forward or backward difference
        a_g[1:-1] = 2 * ( (dp2 / dt2[:, None]) - (dp1 / dt1[:, None]) ) / (dt1 + dt2)[:, None]
    
    # Handle boundaries: assume acceleration at start/end is same as adjacent point
    # (insufficient information to calculate central difference)
    a_g[0] = a_g[1] if n > 1 else 0    # Start point
    a_g[-1] = a_g[-2] if n > 1 else 0  # End point

    # ============================================================================
    # PART 6: CALCULATE AIRSPEED
    # ============================================================================
    # Airspeed is relative velocity between UAV and surrounding air
    # Relationship: v_airspeed = v_ground - v_wind
    
    if wind is None:
        # No wind case: airspeed = ground velocity
        v_a = v_g.copy()
    else:
        # Wind case: need to subtract wind velocity
        # Convert actual coordinates to grid coordinates to get wind values
        x_grid, z_grid, y_grid = np.round(P_grid).astype(int).T
        v_w = wind[x_grid, z_grid, y_grid]  # Get wind vector at each waypoint
                                             # Note: index order is (x, z, y)
        
        # Airspeed = ground velocity - wind velocity
        v_a = v_g - v_w  # Relative velocity vector of UAV relative to air

    # Calculate magnitude and direction of airspeed
    v_a_norm = np.linalg.norm(v_a, axis=1, keepdims=True)  # ||v_a|| (m/s)
    v_a_unit = np.where(v_a_norm > eps, v_a / v_a_norm, 0)  # Unit vector in flight direction

    # ============================================================================
    # PART 7: CALCULATE PARASITIC DRAG FORCE
    # ============================================================================
    # Drag force due to air friction when UAV moves
    # Formula: D = -0.5 * ρ * Cd * A * ||v||² * (direction of v)
    #        = -0.5 * ρ * Cd * A * ||v|| * v  (vector form)
    
    D = -0.5 * rho * Cd * A_frame * v_a_norm * v_a  # Drag force (N)
                                                     # Negative sign because it opposes motion
                                                     # Proportional to squared velocity

    # ============================================================================
    # PART 8: CALCULATE THRUST FORCE
    # ============================================================================
    # Apply Newton's second law: F_thrust = m*a - m*g - D
    # Thrust required to maintain motion along trajectory
    
    T_vec = m * a_g - m * g_vec - D  # Total thrust vector (N)
                                      # = inertial force - gravity - drag
    
    T = np.linalg.norm(T_vec, axis=1, keepdims=True)  # Magnitude of thrust ||T|| (N)
    t_hat = np.where(T > eps, T_vec / T, 0)  # Unit vector in thrust direction

    # ============================================================================
    # PART 9: CALCULATE CLIMB VELOCITY
    # ============================================================================
    # Climb velocity is the component of airspeed in thrust direction
    # v_c = v_a · t_hat (dot product)
    
    v_c = np.sum(v_a * t_hat, axis=1, keepdims=True)  # Climb velocity (m/s)
                                                        # Positive = climbing
                                                        # Negative = descending

    # ============================================================================
    # PART 10: CALCULATE INDUCED VELOCITY
    # ============================================================================
    # Use momentum theory for rotor
    # Air velocity through rotor disk due to lift force
    # Formula from Bernoulli's equation and momentum conservation
    
    sqrt_arg = (0.5 * v_c)**2 + T / (2.0 * rho * A_rotor)  # Ensure non-negative
    v_i = -0.5 * v_c + np.sqrt(np.where(sqrt_arg < 0, 0, sqrt_arg))  # Induced velocity (m/s)
                                                                        # According to momentum theory

    # ============================================================================
    # PART 11: CALCULATE POWER COMPONENTS
    # ============================================================================
    # Total power = Ideal power + Drag power + Profile power
    
    # 1. Ideal power (Ideal power / Induced + Climb power)
    # Energy to generate lift and change altitude
    P_ideal = (T * (v_i + v_c))[:, 0]  # P = F · v = T · (v_i + v_c)  [W]
                                        # v_i: induced velocity (due to lift)
                                        # v_c: climb velocity (altitude change)
    
    # 2. Parasitic drag power
    # Energy to overcome air drag of aircraft body
    P_d = (0.5 * rho * Cd * A_frame * (v_a_norm**3))[:, 0]  # P = D · v = 0.5·ρ·Cd·A·v³  [W]
                                                             # Proportional to v³
    
    # 3. Profile power
    # Energy to rotate rotor blades, overcome friction and efficiency losses
    # Empirical estimate based on load ratio
    P_p = P_p_hover * ((T / (m * g)) ** 1.5)[:, 0]  # P ∝ (T/W)^1.5  [W]
                                                     # W = m·g (weight)

    # Total power consumption at each waypoint
    P_total = P_ideal + P_d + P_p  # [W]

    # ============================================================================
    # PART 12: CALCULATE TOTAL ENERGY CONSUMPTION
    # ============================================================================
    # Integrate power over time: E = ∫ P(t) dt
    # Use trapezoidal rule for high accuracy
    
    # Average power between each pair of consecutive waypoints
    P_avg = 0.5 * (P_total[1:] + P_total[:-1])  # Arithmetic mean (W)
    
    # Energy = Average power × Time
    E = np.sum(P_avg * dt)  # Total energy consumption along entire flight path (J)
                            # 1 Joule = 1 Watt × 1 second

    return E  # Return total energy (Joules)