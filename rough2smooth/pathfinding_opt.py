
# Chỉnh sửa bởi Xiaobo Wu 2025 10 28,29
# Mô-đun tính toán năng lượng tiêu thụ của UAV (máy bay không người lái)
import numpy as np
import itertools

def calculate_global_threshold(wind_data, percentage=0.5):
    """
    Tính toán ngưỡng gió dựa trên phần trăm độ lớn trung bình của toàn bộ không gian.
    
    wind_data: Mảng 4D (dx, dy, dz, 3) chứa các thành phần u, v, w.
    percentage: Tỷ lệ phần trăm mong muốn (mặc định 0.2 tương đương 20%).
    """
    # 1. Tính độ lớn (Magnitude) của vector gió tại mỗi điểm (voxel)
    # Công thức: sqrt(u^2 + v^2 + w^2)
    magnitudes = np.linalg.norm(wind_data, axis=-1)
    
    # 2. Tính giá trị độ lớn trung bình của toàn bộ không gian
    global_avg_magnitude = np.mean(magnitudes)
    
    # 3. Tính ngưỡng theo tỷ lệ phần trăm
    threshold = global_avg_magnitude * percentage
    
    print(f"--- Báo cáo thông số gió toàn không gian ---")
    print(f"Độ lớn gió trung bình (Global Mean): {global_avg_magnitude:.4f} m/s")
    print(f"Ngưỡng đề xuất ({percentage*100}%): {threshold:.4f} m/s")
    
    return threshold


#---------------------------------------------------------------------------------------------------------------
#------------------------------------------------- Adaptive Box ------------------------------------------------
#---------------------------------------------------------------------------------------------------------------

class AdaptiveBox:
    def __init__(self, x_rng, y_rng, z_rng, is_obs, avg_w, w_std):
        self.x_rng, self.y_rng, self.z_rng = x_rng, y_rng, z_rng
        self.is_obstacle = is_obs
        self.avg_wind = avg_w
        self.wind_std = w_std
        
    def __repr__(self):
        type_str = "OBSTACLE" if self.is_obstacle else "FREE"
        size = (
            self.x_rng[1] - self.x_rng[0],
            self.y_rng[1] - self.y_rng[0],
            self.z_rng[1] - self.z_rng[0]
        )
        return (
                f"{type_str}\n"
                f"  x_range: {self.x_rng}\n"
                f"  y_range: {self.y_rng}\n"
                f"  z_range: {self.z_rng}\n"
                f"  size: {size}\n"
                f"  avg_wind: {self.avg_wind}\n"
                f"  wind_std: {self.wind_std}"
            )

def partition_space(
    obs_data,
    wind_data,
    x_rng,
    y_rng,
    z_rng,
    wind_threshold=0.5,
    min_cubic=(5, 5, 5),
    voxel_size=[1, 1, 1],
    min_cubic_unit="voxel"  # "voxel" hoặc "meter"
):
    x1, x2 = x_rng
    y1, y2 = y_rng
    z1, z2 = z_rng


    # Trích xuất vùng dữ liệu
    obs_reg = obs_data[x1:x2, y1:y2, z1:z2]
    wind_reg = wind_data[x1:x2, y1:y2, z1:z2, :]

    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1
    size_voxel = np.array([dx, dy, dz], dtype=float)
    size_meter = size_voxel * voxel_size

    # 1) Trạng thái vật cản (hỗ trợ cả bool và 0/1)
    has_obs = np.any(obs_reg)
    all_obs = np.all(obs_reg)

    # 2) Độ biến động gió
    vector_dispersion = 0.0
    avg_w = np.zeros(3, dtype=float)

    # Chỉ xử lý các voxel KHÔNG phải vật cản
    if wind_reg.size > 0:
        free_mask = ~obs_reg.astype(bool)  # True tại voxel tự do
        if np.any(free_mask):
            free_wind = wind_reg[free_mask]   # shape: (N_free, 3)
            avg_w = np.mean(free_wind, axis=0)

            diff = free_wind - avg_w
            sq_norms = np.sum(diff**2, axis=-1)
            vector_dispersion = float(np.sqrt(np.mean(sq_norms)))

    # 3) Điều kiện dừng theo min_cubic
    min_cubic = np.asarray(min_cubic, dtype=float)
    if min_cubic_unit == "meter":
        is_min_size = np.all(size_meter <= min_cubic)
    else:
        is_min_size = np.all(size_voxel <= min_cubic)

    is_stable = (not has_obs) and (vector_dispersion < wind_threshold)

    if all_obs or is_stable or is_min_size:
        return [AdaptiveBox(x_rng, y_rng, z_rng, has_obs, avg_w, vector_dispersion)]

    # 4) Chia theo trục dài nhất theo KÍCH THƯỚC THỰC (m)
    axis = int(np.argmax(size_meter))
    ranges = [x_rng, y_rng, z_rng]
    mid = ranges[axis][0] + (ranges[axis][1] - ranges[axis][0]) // 2

    # tránh chia rỗng gây đệ quy vô hạn
    if mid == ranges[axis][0] or mid == ranges[axis][1]:
        return [AdaptiveBox(x_rng, y_rng, z_rng, has_obs, avg_w, vector_dispersion)]

    r1, r2 = list(ranges), list(ranges)
    r1[axis] = (ranges[axis][0], mid)
    r2[axis] = (mid, ranges[axis][1])

    return (
        partition_space(
            obs_data, wind_data, *r1,
            wind_threshold=wind_threshold,
            min_cubic=min_cubic,
            voxel_size=voxel_size,
            min_cubic_unit=min_cubic_unit
        )
        + partition_space(
            obs_data, wind_data, *r2,
            wind_threshold=wind_threshold,
            min_cubic=min_cubic,
            voxel_size=voxel_size,
            min_cubic_unit=min_cubic_unit
        )
    )


#---------------------------------------------------------------------------------------------------------------
#------------------------------------------------- Edge Generation ---------------------------------------------
#---------------------------------------------------------------------------------------------------------------

def get_bounds_and_center(box):
    x1, y1, z1 = box["pos"]
    dx, dy, dz = box["size"]
    x2, y2, z2 = x1 + dx, y1 + dy, z1 + dz
    c = ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2)
    return (x1, x2, y1, y2, z1, z2), c

def overlap_positive(a1, a2, b1, b2, eps=1e-9):
    return min(a2, b2) - max(a1, b1) > eps

# face-adjacency
def face_touch(b1, b2, eps=1e-9):
    x1a, x2a, y1a, y2a, z1a, z2a = b1
    x1b, x2b, y1b, y2b, z1b, z2b = b2

    touch_x = (abs(x2a - x1b) <= eps or abs(x2b - x1a) <= eps) and \
              overlap_positive(y1a, y2a, y1b, y2b, eps) and \
              overlap_positive(z1a, z2a, z1b, z2b, eps)

    touch_y = (abs(y2a - y1b) <= eps or abs(y2b - y1a) <= eps) and \
              overlap_positive(x1a, x2a, x1b, x2b, eps) and \
              overlap_positive(z1a, z2a, z1b, z2b, eps)

    touch_z = (abs(z2a - z1b) <= eps or abs(z2b - z1a) <= eps) and \
              overlap_positive(x1a, x2a, x1b, x2b, eps) and \
              overlap_positive(y1a, y2a, y1b, y2b, eps)

    return touch_x or touch_y or touch_z

def face_adjacency(source_boxes_free):
    bounds_free = []
    centers_free = []
    for b in source_boxes_free:
        bd, c = get_bounds_and_center(b)
        bounds_free.append(bd)
        centers_free.append(c)

    centers_free = np.array(centers_free, dtype=float) if len(centers_free) else np.empty((0, 3), dtype=float)

    # Face-adjacency trên tập FREE
    face_pairs_free = []
    for i in range(len(source_boxes_free)):
        for j in range(i + 1, len(source_boxes_free)):
            if face_touch(bounds_free[i], bounds_free[j]):
                face_pairs_free.append((i, j))
    return face_pairs_free, centers_free, bounds_free

#edge-adjacency
def edge_touch_only(b1, b2, eps=1e-9):
    x1a, x2a, y1a, y2a, z1a, z2a = b1
    x1b, x2b, y1b, y2b, z1b, z2b = b2

    # overlap dương theo từng trục
    ox = overlap_positive(x1a, x2a, x1b, x2b, eps)
    oy = overlap_positive(y1a, y2a, y1b, y2b, eps)
    oz = overlap_positive(z1a, z2a, z1b, z2b, eps)

    # chạm mặt phẳng biên theo từng trục
    tx = (abs(x2a - x1b) <= eps) or (abs(x2b - x1a) <= eps)
    ty = (abs(y2a - y1b) <= eps) or (abs(y2b - y1a) <= eps)
    tz = (abs(z2a - z1b) <= eps) or (abs(z2b - z1a) <= eps)

    # Tiếp xúc theo cạnh:
    # - chạm ở 2 trục, overlap dương ở trục còn lại
    # - loại bỏ trường hợp chạm mặt
    edge_x = ty and tz and ox   # cạnh song song trục X
    edge_y = tx and tz and oy   # cạnh song song trục Y
    edge_z = tx and ty and oz   # cạnh song song trục Z

    return (edge_x or edge_y or edge_z) and (not face_touch(b1, b2, eps))

def edge_adjacency(source_boxes_free):
    bounds_free = []
    centers_free = []
    for b in source_boxes_free:
        bd, c = get_bounds_and_center(b)
        bounds_free.append(bd)
        centers_free.append(c)

    centers_free = np.array(centers_free, dtype=float) if len(centers_free) else np.empty((0, 3), dtype=float)
    edge_pairs_edge_free = []
    for i in range(len(source_boxes_free)):
        for j in range(i + 1, len(source_boxes_free)):
            if edge_touch_only(bounds_free[i], bounds_free[j]):
                edge_pairs_edge_free.append((i, j))

    return edge_pairs_edge_free, centers_free, bounds_free

#vertex-adjacency
def vertex_touch_only(b1, b2, eps=1e-9):
    x1a, x2a, y1a, y2a, z1a, z2a = b1
    x1b, x2b, y1b, y2b, z1b, z2b = b2

    tx = (abs(x2a - x1b) <= eps) or (abs(x2b - x1a) <= eps)
    ty = (abs(y2a - y1b) <= eps) or (abs(y2b - y1a) <= eps)
    tz = (abs(z2a - z1b) <= eps) or (abs(z2b - z1a) <= eps)

    # Chạm ở cả 3 trục => tiếp xúc tại đỉnh
    return tx and ty and tz and (not face_touch(b1, b2, eps)) and (not edge_touch_only(b1, b2, eps))


def shared_vertex_point(b1, b2, eps=1e-9):
    x1a, x2a, y1a, y2a, z1a, z2a = b1
    x1b, x2b, y1b, y2b, z1b, z2b = b2

    vx = x2a if abs(x2a - x1b) <= eps else (x1a if abs(x1a - x2b) <= eps else None)
    vy = y2a if abs(y2a - y1b) <= eps else (y1a if abs(y1a - y2b) <= eps else None)
    vz = z2a if abs(z2a - z1b) <= eps else (z1a if abs(z1a - z2b) <= eps else None)

    return (vx, vy, vz)

def vertex_adjacency(source_boxes_free):
    bounds_free = []
    centers_free = []
    for b in source_boxes_free:
        bd, c = get_bounds_and_center(b)
        bounds_free.append(bd)
        centers_free.append(c)
    # Tính các cặp chung đỉnh (vertex-touch only)
    vertex_pairs_free = []
    for i in range(len(source_boxes_free)):
        for j in range(i + 1, len(source_boxes_free)):
            if vertex_touch_only(bounds_free[i], bounds_free[j]):
                vertex_pairs_free.append((i, j))
    return vertex_pairs_free, centers_free, bounds_free


# eleminate interscts obstacle
# Lọc các edge bị cắt qua OBSTACLE cubic để tạo edges_data_nocut

def segment_intersects_aabb(p0, p1, bmin, bmax, eps=1e-9):
	"""
	Kiểm tra đoạn thẳng p0->p1 có giao với AABB [bmin, bmax] hay không (slab method).
	p0, p1, bmin, bmax: np.array shape (3,)
	"""
	d = p1 - p0
	tmin, tmax = 0.0, 1.0

	for k in range(3):
		if abs(d[k]) < eps:
			# Đoạn song song trục k -> phải nằm trong slab theo trục k
			if p0[k] < bmin[k] - eps or p0[k] > bmax[k] + eps:
				return False
		else:
			inv_d = 1.0 / d[k]
			t1 = (bmin[k] - p0[k]) * inv_d
			t2 = (bmax[k] - p0[k]) * inv_d
			if t1 > t2:
				t1, t2 = t2, t1
			tmin = max(tmin, t1)
			tmax = min(tmax, t2)
			if tmin > tmax:
				return False

	return True


    # Lấy edges source (phòng trường hợp bạn đặt tên edge_data thay vì edges_data)

    # Obstacle boxes -> AABB theo meter-space (cùng hệ với center_i_m/center_j_m)
def filter_edges_by_obstacles(edges_data, results, voxel_size_m):
    scale = np.asarray(voxel_size_m, dtype=float)
    obs_boxes = [b for b in results if b.is_obstacle]

    obs_aabbs_m = []
    for b in obs_boxes:
        bmin = np.array([b.x_rng[0], b.y_rng[0], b.z_rng[0]], dtype=float) * scale
        bmax = np.array([b.x_rng[1], b.y_rng[1], b.z_rng[1]], dtype=float) * scale
        obs_aabbs_m.append((bmin, bmax))

    # Lọc edge không cắt obstacle
    edges_data_nocut = []
    edges_data_cut = []
    cut_count = 0

    for e in edges_data:
        p0 = np.asarray(e["center_i_m"], dtype=float)
        p1 = np.asarray(e["center_j_m"], dtype=float)

        is_cut = False
        for bmin, bmax in obs_aabbs_m:
            if segment_intersects_aabb(p0, p1, bmin, bmax):
                is_cut = True
                break

        if is_cut:
            cut_count += 1
            edges_data_cut.append(e)
        else:
            edges_data_nocut.append(e)
    return edges_data_nocut, edges_data_cut, cut_count
#---------------------------------------------------------------------------------------------------------------
#------------------------------------------------- Energy calculation ------------------------------------------
#---------------------------------------------------------------------------------------------------------------
def calculate_drone_power(v_i, w_i, a_i, params):
    """
    Calculate total power consumption Pi according to the formula sequence.
    v_i, w_i, a_i are numpy array vectors with shape (3,)
    """
    # Get constants from params
    rho = params['rho']       # Air density
    Cd = params['Cd']         # Drag coefficient
    Af = params['Af']         # Frontal area
    m = params['m']           # Drone weight
    g_acc = params['g_acc']   # Gravitational acceleration vector [0, 0, -9.81]
    A = params['A']           # Rotor disk area
    Pp_hover = params['Pp_hover'] # Profile power when hovering
    mg_scalar = m * np.linalg.norm(g_acc)

    # (2) Velocity relative to wind
    v_a = v_i - w_i
    v_a_norm = np.linalg.norm(v_a)

    # (3) Aerodynamic drag
    Di = -0.5 * rho * Cd * Af * v_a_norm * v_a

    # (4) Thrust force vector
    # print("a_i:", np.array(a_i))
    Ti_vec = m * np.array(a_i) - m * g_acc - Di

    # (5) Magnitude of thrust force
    Ti_mag = np.linalg.norm(Ti_vec)

    # (6) Unit direction vector of thrust
    ti_hat = Ti_vec / Ti_mag if Ti_mag != 0 else np.array([0, 0, 1])

    # (7) Velocity component relative to thrust axis (scalar)
    vc_i = np.dot(v_a, ti_hat)

    # (8) Induced velocity (scalar)
    # Formula: v_ind = -1/2*vc + sqrt((1/2*vc)^2 + Ti/(2*rho*A))
    term1 = -0.5 * vc_i
    term2 = np.sqrt((0.5 * vc_i)**2 + Ti_mag / (2 * rho * A))
    v_ind = term1 + term2

    # (9) Useful power
    Pu_i = np.dot(Ti_vec, v_a)

    # (10) Induced power
    Pind_i = Ti_mag * v_ind

    # (11) Profile power
    Pp_i = Pp_hover * (Ti_mag / mg_scalar)**1.5

    # (12) Total power
    Pi = Pu_i + Pind_i + Pp_i

    return {
        "Pi": Pi,
        "Pu_i": Pu_i,
        "Pind_i": Pind_i,
        "Pp_i": Pp_i,
        "Ti_vec": Ti_vec,
        "Ti_mag": Ti_mag,
    }

def generate_directed_edge_dict(box_data, edge_data,speed_map,  parameters):
    """
    Create a 3D directed edge dictionary with:
    - box_data: {
    "pos": (x1, y1, z1),
    "size": (dx, dy, dz),
    "avg_wind": np.array([wx, wy, wz]) hoặc list 3 phần tử,
    "turbulence_level": float
    }
    - edge_data: {
    "type": "face | edge | vertex",
    "box_i": 12,
    "box_j": 57,
    "center_i_idx": [xi, yi, zi],
    "center_j_idx": [xj, yj, zj],
    "center_i_m": [Xi, Yi, Zi],
    "center_j_m": [Xj, Yj, Zj],
    "length_m": l
    }
    - parameters: dict with keys 'wind_field', 'cell_size'  (in meters), 'drone_params' (dict for power calculation)

    """
    moore_dict = {}
    for edge in edge_data:
        V_i = box_data[edge["box_i"]]["size"][0] * box_data[edge["box_i"]]["size"][1] * box_data[edge["box_i"]]["size"][2]
        V_j = box_data[edge["box_j"]]["size"][0] * box_data[edge["box_j"]]["size"][1] * box_data[edge["box_j"]]["size"][2]

        w_i = (V_i * box_data[edge["box_i"]]["avg_wind"] + V_j * box_data[edge["box_j"]]["avg_wind"]) / (V_i + V_j)
        for speed_level in speed_map.keys():
            speed=speed_map[speed_level]
            v_i = (speed / edge["length_m"]) * (np.array(edge["center_j_m"]) - np.array(edge["center_i_m"]))

            power_result = calculate_drone_power(v_i, w_i, np.array([0.0, 0.0, 0.0]), parameters)

            edge_key = f"{edge['box_i']}_{edge['box_j']}_{speed_level}"
            moore_dict[edge_key] = {
                "startbox_idx": edge["box_i"],
                "endbox_idx": edge["box_j"],
                "startpoint": edge["center_i_idx"],
                "startpoint_m": edge["center_i_m"],
                "endpoint": edge["center_j_idx"],
                "endpoint_m": edge["center_j_m"],
                "euclidean_distance": edge["length_m"],
                "v_level": speed_level ,
                "v": np.linalg.norm(v_i),
                "v_vector": v_i,
                "wind_vector": w_i,
                "Pi": power_result["Pi"],
                "Tmaneuver": power_result["Ti_mag"],
                "Energy": power_result["Pi"] * (edge["length_m"] / speed),
                "length_m": edge["length_m"],
            }

            v_i_reverse = -v_i
            power_result_reverse = calculate_drone_power(v_i_reverse, w_i, np.array([0.0, 0.0, 0.0]), parameters)

            edge_key_reverse = f"{edge['box_j']}_{edge['box_i']}_{speed_level}"
            moore_dict[edge_key_reverse] = {
                "startbox_idx": edge["box_j"],
                "endbox_idx": edge["box_i"],
                "startpoint": edge["center_j_idx"],
                "startpoint_m": edge["center_j_m"],
                "endpoint": edge["center_i_idx"],
                "endpoint_m": edge["center_i_m"],
                "euclidean_distance": edge["length_m"],
                "v_level": speed_level,
                "v": np.linalg.norm(v_i_reverse),
                "v_vector": v_i_reverse,
                "wind_vector": w_i,
                "Pi": power_result_reverse["Pi"],
                "Tmaneuver": power_result_reverse["Ti_mag"],
                "Energy": power_result_reverse["Pi"] * (edge["length_m"] / speed),
                "length_m": edge["length_m"],
            }

    return moore_dict

def calculate_energy_transition(edge_in_key, edge_out_key, wind, params,dt):
    """
    Calculate energy at the intersection point between 2 edges based on Vlevel
    """
    # 1. Decode edge information
    # edge_in_key=space_graph[edge_in_key]
    # edge_out_key=space_graph[edge_out_key]
    p_prev, p_curr, v_in_vec,v_mag_in = edge_in_key["startpoint"], edge_in_key["endpoint"], edge_in_key["v_vector"],edge_in_key["v"]
    _, p_next, v_out_vec,v_mag_out = edge_out_key["startpoint"], edge_out_key["endpoint"], edge_out_key["v_vector"],edge_out_key["v"]

  
    # 3. Calculate v_i and a_i at point p_curr
    # Assume dt is the transition time (average distance / average velocity)
    # dist_avg = (np.linalg.norm(p_curr - p_prev) + np.linalg.norm(p_next - p_curr)) / 2
    # dt = dist_avg / ((v_mag_in + v_mag_out) / 2)

    v_i = (v_in_vec + v_out_vec) / 2
    a_i = (v_out_vec - v_in_vec) / dt

    # --- Physical sequence ---
    rho, Cd, Af, m, A = params['rho'], params['Cd'], params['Af'], params['m'], params['A']
    g_vec = params['g_acc']
    Pp_hover = params['Pp_hover']
    mg_scalar = m * np.linalg.norm(g_vec)

    # (2) Relative velocity
    v_a = v_i - wind
    va_norm = np.linalg.norm(v_a)

    # (3) Drag force Di
    Di = -0.5 * rho * Cd * Af * va_norm * v_a

    # (4) Thrust force Ti (Vector)
    Ti_vec = m * a_i - m * g_vec - Di
    Ti_mag = np.linalg.norm(Ti_vec)

    # (6) Thrust direction
    ti_hat = Ti_vec / Ti_mag if Ti_mag > 1e-6 else np.array([0, 0, 1])

    # (7) Velocity along thrust axis
    vc_i = np.dot(v_a, ti_hat)

    # (8) Induced velocity
    v_ind = -0.5 * vc_i + np.sqrt(max(0, (0.5 * vc_i)**2 + Ti_mag / (2 * rho * A)))

    # (9-12) Power components
    Pu_i = np.dot(Ti_vec, v_a)
    Pind_i = Ti_mag * v_ind
    Pp_i = Pp_hover * (Ti_mag / mg_scalar)**1.5
    Pi = Pu_i + Pind_i + Pp_i

    return {
        # "edge_in": edge_in_key,
        # "edge_out": edge_out_key,
        "total_power_Pi": Pi,
        "Tmaneuver": Ti_mag,
        # "acceleration": a_i
    }


#---------------------------------------------------------------------------------------------------------------
#------------------------------------------------- QUBO matrix generation --------------------------------------
#---------------------------------------------------------------------------------------------------------------
def find_box_index(point, box_data):
    point = np.array(point, dtype=float)

    mins = np.array([box["pos_m"] for box in box_data], dtype=float)
    sizes = np.array([box["size_m"] for box in box_data], dtype=float)
    maxs = mins + sizes

    # tolerance tránh lỗi float
    eps = 1e-9

    inside = np.all((point >= mins - eps) & (point < maxs + eps), axis=1)

    indices = np.where(inside)[0]

    if len(indices) == 0:
        return None
    elif len(indices) == 1:
        return indices[0]
    else:
        # nếu overlap → chọn box nhỏ nhất
        volumes = sizes[indices].prod(axis=1)
        return indices[np.argmin(volumes)]
    
def QUBO_energy_matrix(box_data, space_graph, parameters, start_point_m, end_point_m):

    start_point = [250, 40, 0]      # meters
    start_box_idx = find_box_index(start_point_m, box_data)
    print(f"Start point {start_point} in box index: {start_box_idx}")
    end_point=[1000,1100,0]
    end_box_idx = find_box_index(end_point_m, box_data)
    print(f"End point {end_point} in box index: {end_box_idx}")

    # Filter all edges starting from point (0, 0, 0)
    edges_from_origin = {key: value for key, value in space_graph.items() if value['startbox_idx'] == start_box_idx}

    print(f"Number of edges starting: {len(edges_from_origin)}")
    for key, data in edges_from_origin.items():

        astart = data['v_vector']/parameters['dt_takeoff']
        Etakeoff = calculate_drone_power(data['v_vector']/2, box_data[data["startbox_idx"]]["avg_wind"], astart, parameters)["Pi"]*parameters['dt_takeoff']
        space_graph[key]["Etakeoff"] = Etakeoff  # Add takeoff energy to the edge
        # print(f"{key}:   Energy={data['Energy']:.2f}, Thrust force={data['Tmaneuver']:.3f} N, Etakeoff={Etakeoff:.2f} J")
    
        # Filter all edges ending at point (4, 4, 5)
    edges_to_destination = {key: value for key, value in space_graph.items() if value['endbox_idx'] == end_box_idx}

    print(f"Number of edges ending: {len(edges_to_destination)}")
    print("\nEdges to destination:")
    for key, data in edges_to_destination.items():
        astart = data['v_vector']/parameters['dt_landing']
        Elanding = calculate_drone_power(data['v_vector']/2, box_data[data["endbox_idx"]]["avg_wind"], astart, parameters)["Pi"]*parameters['dt_landing']
        space_graph[key]["Elanding"] = Elanding  # Add landing energy to the edge
        # print(f"{key}: {data['endpoint']}  Energy={data['Energy']:.2f}, Thrust force={data['Tmaneuver']:.3f} N, Elanding={Elanding:.2f} J")
    
    
    edges_exceeded_Tmax={
    key: data
    for key, data in space_graph.items()
    if data["Tmaneuver"] > parameters["Tmax"]
    }
    edges_to_start = {
    key: data
    for key, data in space_graph.items()
    if data["endbox_idx"] == start_box_idx
    }

    edges_from_end = {
        key: data
        for key, data in space_graph.items()
        if data["startbox_idx"] == end_box_idx
    }

    # Remove edges in space_graph whose key is in edges_to_start or edges_from_end
    keys_to_remove = set(edges_to_start.keys()) | set(edges_from_end.keys() | set(edges_exceeded_Tmax.keys()))
    removed = 0
    for k in keys_to_remove:
        if k in space_graph:
            del space_graph[k]
            removed += 1
    print(f"Deleted {removed} keys from space_graph")
    print(f"Remaining edges: {len(space_graph)}")


    # Create QUBO matrix from space_graph
    edge_keys = list(space_graph.keys())
    N = len(edge_keys)
    edge_index = {key: idx for idx, key in enumerate(edge_keys)}
    print(f"Number of edges (N): {N}")
    print(f"QUBO matrix size: {N} x {N}")
    # Initialize QUBO matrix
    Q = np.zeros((N, N), dtype=float)

    list_edges_start = {
        key: space_graph[key]
        for key in edge_index
        if space_graph[key]["startbox_idx"] == start_box_idx
    }
    max_energy_start = 0
    for key in list_edges_start:
        idx = edge_index[key]  # get key index in graph
        bufVal= 0.5*space_graph[key]["Energy"] + space_graph[key].get("Etakeoff", 0.0)
        Q[idx, idx] += bufVal/1000  # Scale down energy to fit into QUBO range
        # print(f"Edge {key}: Energy={space_graph[key]['Energy']:.2f} J, Etakeoff={space_graph[key].get('Etakeoff', 0.0):.2f} J, Q[{idx},{idx}]={Q[idx, idx]:.2f}")
        if bufVal > max_energy_start:
            max_energy_start = bufVal

    print(f"Max energy for edges from start point: {max_energy_start:.2f} J")
    # Assign diagonal values Q[i,i] = 3 for edges with the endpoint
    list_edges_end = {
        key: space_graph[key]
        for key in edge_index
        if space_graph[key]["endbox_idx"] == end_box_idx
    }
    max_energy_end = 0
    for key in list_edges_end:
        idx = edge_index[key]  # get key index in graph
        bufVal = 0.5*space_graph[key]["Energy"] + space_graph[key].get("Elanding", 0.0)
        Q[idx, idx] += bufVal/1000  # Scale down energy to fit into QUBO range 
        # print(f"Edge {key}: Energy={space_graph[key]['Energy']:.2f} J, Elanding={space_graph[key].get('Elanding', 0.0):.2f} J, Q[{idx},{idx}]={Q[idx, idx]:.2f}")
        if bufVal > max_energy_end:
            max_energy_end = bufVal
    print(f"Max energy for edges to end point: {max_energy_end:.2f} J")

    all_mid_boxes = []
    for  box_idx, box in enumerate(box_data):
        if box_idx == start_box_idx or box_idx == end_box_idx: 
            continue
        all_mid_boxes.append(box_idx)
    print(f"Total mid boxes: {len(all_mid_boxes)}")
    max_pair_energy = 0
    max_pair_thrust = 0
    for box_idx in all_mid_boxes:
        edges_out_list= {
            k: v
            for k, v in space_graph.items()
            if v["startbox_idx"] == box_idx
        }
        edges_in_list= {
            k: v
            for k, v in space_graph.items()
            if v["endbox_idx"] == box_idx
        }
        # print(f"Box {box_idx}: {len(edges_in_list)} incoming edges, {len(edges_out_list)} outgoing edges")
        for edges_in in edges_in_list:
            idx_in = edge_index[edges_in]
            for edges_out in edges_out_list:
                idx_out = edge_index[edges_out]

                V_i = box_data[space_graph[edges_in]["startbox_idx"]]["size"][0] * box_data[space_graph[edges_in]["startbox_idx"]]["size"][1] * box_data[space_graph[edges_in]["startbox_idx"]]["size"][2]
                V_j = box_data[space_graph[edges_out]["endbox_idx"]]["size"][0] * box_data[space_graph[edges_out]["endbox_idx"]]["size"][1] * box_data[space_graph[edges_out]["endbox_idx"]]["size"][2]

                w_i = (V_i * box_data[space_graph[edges_in]["startbox_idx"]]["avg_wind"] + V_j * box_data[space_graph[edges_out]["endbox_idx"]]["avg_wind"]) / (V_i + V_j)

                change_dir = calculate_energy_transition(space_graph[edges_in], space_graph[edges_out], w_i, parameters, parameters['dt'])
                bufVal = (0.5*space_graph[edges_in]["Energy"] + 0.5*space_graph[edges_out]["Energy"] + change_dir["total_power_Pi"]*(parameters['dt']))/1000  # Scale down to kJ for QUBO
                Q[min(idx_in, idx_out), max(idx_in, idx_out)] += bufVal  # Convert J to kJ for better scaling
                if bufVal > max_pair_energy:
                    max_pair_energy = bufVal
                if change_dir["Tmaneuver"] > max_pair_thrust:
                    max_pair_thrust = change_dir["Tmaneuver"]
                # print(f"Pair: {edges_in}(id: {idx_in}) -> {edges_out}(id: {idx_out}), Transition Energy={change_dir['total_power_Pi']:.2f} J, Total Pair Energy={bufVal:.2f} J, Thrust={change_dir['Tmaneuver']:.2f} N")

    #constraint
    for key in list_edges_start:
        idx = edge_index[key]  # get key index in graph
        Q[idx, idx] -= max_energy_start
    for i in range(len(list_edges_start)):
        for j in range(i+1, len(list_edges_start)):
            idx_i = edge_index[list(list_edges_start.keys())[i]]
            idx_j = edge_index[list(list_edges_start.keys())[j]]

            Q[min(idx_i, idx_j), max(idx_i, idx_j)] += max_energy_start

    for key in list_edges_end:
        idx = edge_index[key]  # get key index in graph
        Q[idx, idx] -= max_energy_end
    for i in range(len(list_edges_end)):
        for j in range(i+1, len(list_edges_end)):
            idx_i = edge_index[list(list_edges_end.keys())[i]]
            idx_j = edge_index[list(list_edges_end.keys())[j]]

            Q[min(idx_i, idx_j), max(idx_i, idx_j)] += max_energy_end

    for box_idx in all_mid_boxes:
        edges_out_list= {
            k: v
            for k, v in space_graph.items()
            if v["startbox_idx"] == box_idx
        }
        edges_in_list= {
            k: v
            for k, v in space_graph.items()
            if v["endbox_idx"] == box_idx
        }
        for edges_in in edges_in_list:
            idx_in = edge_index[edges_in]
            Q[idx_in, idx_in] += 10*max_pair_energy

        for edges_out in edges_out_list:
            idx_out = edge_index[edges_out]
            Q[idx_out, idx_out] += 10*max_pair_energy

        for i in range(len(edges_in_list)):
            for j in range(i+1, len(edges_in_list)):
                idx_i = edge_index[list(edges_in_list.keys())[i]]
                idx_j = edge_index[list(edges_in_list.keys())[j]]

                Q[min(idx_i, idx_j), max(idx_i, idx_j)] += 20*max_pair_energy

        for i in range(len(edges_out_list)):
            for j in range(i+1, len(edges_out_list)):
                idx_i = edge_index[list(edges_out_list.keys())[i]]
                idx_j = edge_index[list(edges_out_list.keys())[j]]

                Q[min(idx_i, idx_j), max(idx_i, idx_j)] += 20*max_pair_energy

        for edges_in in edges_in_list:
            idx_in = edge_index[edges_in]
            for edges_out in edges_out_list:
                idx_out = edge_index[edges_out]
                Q[min(idx_in, idx_out), max(idx_in, idx_out)] +=-20*max_pair_energy

        num_violations = 0
    for box_idx in all_mid_boxes:
        edges_out_list= {
            k: v
            for k, v in space_graph.items()
            if v["startbox_idx"] == box_idx
        }
        edges_in_list= {
            k: v
            for k, v in space_graph.items()
            if v["endbox_idx"] == box_idx
        }
        for edges_in in edges_in_list:
            idx_in = edge_index[edges_in]
            for edges_out in edges_out_list:
                V_i = box_data[space_graph[edges_in]["startbox_idx"]]["size"][0] * box_data[space_graph[edges_in]["startbox_idx"]]["size"][1] * box_data[space_graph[edges_in]["startbox_idx"]]["size"][2]
                V_j = box_data[space_graph[edges_out]["endbox_idx"]]["size"][0] * box_data[space_graph[edges_out]["endbox_idx"]]["size"][1] * box_data[space_graph[edges_out]["endbox_idx"]]["size"][2]

                w_i = (V_i * box_data[space_graph[edges_in]["startbox_idx"]]["avg_wind"] + V_j * box_data[space_graph[edges_out]["endbox_idx"]]["avg_wind"]) / (V_i + V_j)

                idx_out = edge_index[edges_out]
                change_dir = calculate_energy_transition(space_graph[edges_in], space_graph[edges_out], w_i, parameters, parameters['dt'])
                if change_dir["Tmaneuver"] > parameters['Tmax']:
                    # print(f"Violation: Transition from {edges_in} to {edges_out} has Tmaneuver={change_dir['Tmaneuver']:.2f} N > Tmax={parameters['Tmax']} N")
                    Q[min(idx_in, idx_out), max(idx_in, idx_out)] += 10*max_pair_energy
                    num_violations += 1
    print(f"Number of pairs with Tmaneuver > Tmax: {num_violations}")

    return Q, edge_index



#---------------------------------------------------------------------------------------------------------------
#------------------------------------------------- Visualization -----------------------------------------------
#---------------------------------------------------------------------------------------------------------------

def idx_to_m(xyz,voxel_size_m=np.array([4, 4, 1.5])):
    # xyz: (..., 3) in grid-index units -> meters
    return xyz * voxel_size_m


def wireframe_adaptive(boxes):
    xs, ys, zs = [], [], []
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7)
    ]
    for b in boxes:
        x1, x2 = b.x_rng
        y1, y2 = b.y_rng
        z1, z2 = b.z_rng

        v_idx = np.array([
            [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],
            [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]
        ], dtype=float)

        v = idx_to_m(v_idx)  # convert box corners to meters

        for a, c in edges:
            xs += [v[a, 0], v[c, 0], None]
            ys += [v[a, 1], v[c, 1], None]
            zs += [v[a, 2], v[c, 2], None]
    return xs, ys, zs