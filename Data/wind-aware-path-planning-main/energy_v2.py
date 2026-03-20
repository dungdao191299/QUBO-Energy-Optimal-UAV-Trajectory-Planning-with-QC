
# Chỉnh sửa bởi Xiaobo Wu 2025 10 28,29
# Mô-đun tính toán năng lượng tiêu thụ của UAV (máy bay không người lái)
import numpy as np
import matplotlib.pyplot as plt


def get_energy(P_grid, v_const=10.0, eps=1e-12, wind=None):
    """
    Tính toán năng lượng tiêu thụ tối thiểu của UAV dọc theo đường bay cho trước (mô hình cận tĩnh)
    
    Mô hình này sử dụng lý thuyết khí động học cơ bản để tính toán công suất cần thiết
    cho các hoạt động bay của UAV, bao gồm:
    - Lực nâng (induced power)
    - Lực cản khí động (parasitic drag)
    - Công suất quay cánh quạt (profile power)
    
    Tham số:
        P_grid : mảng (N, 3), tọa độ lưới [x_grid, y_grid, z_grid]
                 Đại diện cho N điểm trên đường bay trong hệ tọa độ lưới
        v_const: vận tốc đất không đổi (m/s)
                 Tốc độ di chuyển mong muốn của UAV so với mặt đất
        eps    : giá trị nhỏ bảo vệ số học
                 Tránh chia cho 0 trong các phép tính
        wind   : mảng (Nx, Ny, Nz, 3), trường gió (tùy chọn)
                 Véc-tơ vận tốc gió tại mỗi điểm trong không gian 3D
    
    Trả về:
        E : tổng năng lượng tiêu thụ (Joules)
    """
    
    # ============================================================================
    # PHẦN 1: CÁC HẰNG SỐ VẬT LÝ
    # ============================================================================
    # Các thông số vật lý của UAV và môi trường
    
    rho = 1.18          # Mật độ không khí ở mực nước biển, điều kiện tiêu chuẩn (kg/m³)
    Cd  = 1.0           # Hệ số cản khí động của thân máy bay (không thứ nguyên)
                        # Giá trị điển hình cho UAV cỡ nhỏ
    A_frame = 0.25      # Diện tích chắn gió của thân máy bay (m²)
                        # Diện tích hình chiếu vuông góc với hướng bay
    m = 4.0             # Khối lượng tổng của UAV bao gồm pin và payload (kg)
    A_rotor = 0.45      # Tổng diện tích quét của các cánh quạt rotor (m²)
                        # Cho quadcopter: π * (đường kính cánh quạt)² * số cánh
    P_p_hover = 65.0    # Công suất tiêu thụ khi lơ lửng (W)
                        # Bao gồm tổn thất ma sát và hiệu suất động cơ
    g = 9.81            # Gia tốc trọng trường (m/s²)
    
    # Véc-tơ trọng lực (hướng xuống theo trục y âm)
    g_vec = np.array([0.0, -g, 0.0])

    # ============================================================================
    # PHẦN 2: CHUYỂN ĐỔI TỌA ĐỘ VÀ KHỞI TẠO
    # ============================================================================
    # Chuyển đổi từ tọa độ lưới rời rạc sang tọa độ thực trong không gian liên tục
    
    scale = np.array([4.0, 1.5, 4.0])  # Hệ số tỉ lệ: lưới → khoảng cách thực (m)
                                        # [scale_x, scale_y, scale_z]
                                        # Mỗi đơn vị lưới tương ứng với bao nhiêu mét
    P = P_grid * scale  # Chuyển đổi các điểm đường bay sang tọa độ thực (m)
    n = P.shape[0]      # Tổng số điểm waypoint trên đường bay

    # ============================================================================
    # PHẦN 3: TÍNH TOÁN CHIỀU DÀI ĐƯỜNG BAY VÀ THỜI GIAN
    # ============================================================================
    # Tính khoảng cách Euclid giữa các điểm waypoint liên tiếp
    
    dP = P[1:] - P[:-1]  # Véc-tơ chuyển vị giữa các điểm liên tiếp (m)
                         # Kích thước: (n-1, 3)
    
    ds = np.linalg.norm(dP, axis=1)  # Độ dài mỗi đoạn bay (m)
                                       # ds[i] = ||P[i+1] - P[i]||
    
    ds = np.where(ds < eps, eps, ds)  # Bảo vệ: thay các đoạn quá ngắn bằng eps
                                       # Tránh chia cho 0 trong các phép tính sau
    
    # Tính độ dài tích lũy theo đường bay (m)
    s = np.zeros(n)      # Khoảng cách từ điểm xuất phát đến mỗi waypoint
    s[1:] = np.cumsum(ds)  # s[i] = tổng độ dài từ điểm 0 đến điểm i

    # Tính thời gian bay giữa các waypoint (giả sử tốc độ không đổi)
    dt = ds / v_const    # Thời gian mỗi đoạn = khoảng cách / tốc độ (s)
    
    # Tính thời gian tích lũy
    t = np.zeros(n)      # Thời gian từ lúc xuất phát đến mỗi waypoint
    t[1:] = np.cumsum(dt)  # t[i] = tổng thời gian từ điểm 0 đến điểm i

    # ============================================================================
    # PHẦN 4: TÍNH VẬN TỐC ĐẤT (GROUND VELOCITY)
    # ============================================================================
    # Vận tốc đất là vận tốc của UAV so với mặt đất (bao gồm cả ảnh hưởng của gió)
    # Sử dụng phương pháp trung bình có trọng số để đảm bảo vận tốc liên tục
    
    v_g = np.zeros_like(P)  # Khởi tạo mảng vận tốc đất (m/s)
                            # Kích thước: (n, 3) - véc-tơ vận tốc 3D tại mỗi waypoint
    
    # Xử lý điểm đầu và điểm cuối: vận tốc song song với hướng bay
    v_g[0]  = dP[0]  / ds[0]   # Điểm xuất phát: hướng theo đoạn đầu tiên
    v_g[-1] = dP[-1] / ds[-1]  # Điểm đích: hướng theo đoạn cuối cùng

    # Xử lý các điểm giữa: trung bình có trọng số theo thời gian
    # Mục đích: làm mịn vận tốc, tránh thay đổi đột ngột gây gia tốc lớn
    if n > 2:
        # Khoảng thời gian giữa các waypoint
        dt1 = t[1:-1] - t[:-2]  # Thời gian từ điểm trước đến điểm hiện tại
        dt2 = t[2:]   - t[1:-1]  # Thời gian từ điểm hiện tại đến điểm sau
        
        # Véc-tơ chuyển vị
        dp1 = P[1:-1] - P[:-2]  # Từ điểm i-1 đến i
        dp2 = P[2:]   - P[1:-1]  # Từ điểm i đến i+1
        
        # Vận tốc trung bình của đoạn trước và đoạn sau
        v_prev = dp1 / dt1[:, None]  # Vận tốc trên đoạn trước
        v_next = dp2 / dt2[:, None]  # Vận tốc trên đoạn sau
        
        # Trung bình có trọng số: đoạn nào dài hơn (về thời gian) có trọng số cao hơn
        v_g[1:-1] = (v_prev * dt1[:, None] + v_next * dt2[:, None]) / (dt1 + dt2)[:, None]

    # Chuẩn hóa độ lớn vận tốc về v_const
    # Đảm bảo UAV bay với tốc độ không đổi như yêu cầu
    v_g_norm = np.linalg.norm(v_g, axis=1, keepdims=True)  # Độ lớn vận tốc ||v_g||
    v_g = np.where(v_g_norm > eps, v_g / v_g_norm * v_const, 0)  # v_g = (v_g/||v_g||) * v_const

    # ============================================================================
    # PHẦN 5: TÍNH GIA TỐC ĐẤT (GROUND ACCELERATION)
    # ============================================================================
    # Gia tốc cần thiết để thay đổi vận tốc đất giữa các waypoint
    # Sử dụng sai phân trung tâm để ước lượng đạo hàm
    
    a_g = np.zeros_like(P)  # Khởi tạo mảng gia tốc (m/s²)
    
    # Tính gia tốc cho các điểm giữa bằng sai phân trung tâm
    # a = dv/dt = (v_next - v_prev) / (dt_prev + dt_next)
    if n > 2:
        # Công thức sai phân trung tâm chính xác hơn sai phân tiến hoặc lùi
        a_g[1:-1] = 2 * ( (dp2 / dt2[:, None]) - (dp1 / dt1[:, None]) ) / (dt1 + dt2)[:, None]
    
    # Xử lý biên: giả sử gia tốc ở đầu/cuối giống điểm kế bên
    # (do không có đủ thông tin để tính sai phân trung tâm)
    a_g[0] = a_g[1] if n > 1 else 0    # Điểm xuất phát
    a_g[-1] = a_g[-2] if n > 1 else 0  # Điểm đích

    # ============================================================================
    # PHẦN 6: TÍNH VẬN TỐC KHÔNG KHÍ (AIRSPEED)
    # ============================================================================
    # Vận tốc không khí là vận tốc tương đối giữa UAV và không khí xung quanh
    # Quan hệ: v_airspeed = v_ground - v_wind
    
    if wind is None:
        # Trường hợp không có gió: vận tốc không khí = vận tốc đất
        v_a = v_g.copy()
    else:
        # Trường hợp có gió: cần trừ đi vận tốc gió
        # Chuyển tọa độ thực về tọa độ lưới để lấy giá trị gió
        x_grid, z_grid, y_grid = np.round(P_grid).astype(int).T
        v_w = wind[x_grid, z_grid, y_grid]  # Lấy véc-tơ gió tại mỗi waypoint
                                             # Lưu ý: thứ tự index là (x, z, y)
        
        # Vận tốc không khí = vận tốc đất - vận tốc gió
        v_a = v_g - v_w  # Véc-tơ vận tốc tương đối UAV so với không khí

    # Tính độ lớn và hướng của vận tốc không khí
    v_a_norm = np.linalg.norm(v_a, axis=1, keepdims=True)  # ||v_a|| (m/s)
    v_a_unit = np.where(v_a_norm > eps, v_a / v_a_norm, 0)  # Véc-tơ đơn vị hướng bay

    # ============================================================================
    # PHẦN 7: TÍNH LỰC CẢN KHÍ ĐỘNG (PARASITIC DRAG)
    # ============================================================================
    # Lực cản do ma sát không khí khi UAV di chuyển
    # Công thức: D = -0.5 * ρ * Cd * A * ||v||² * (hướng v)
    #          = -0.5 * ρ * Cd * A * ||v|| * v  (dạng véc-tơ)
    
    D = -0.5 * rho * Cd * A_frame * v_a_norm * v_a  # Lực cản (N)
                                                     # Dấu âm vì ngược hướng chuyển động
                                                     # Tỉ lệ với bình phương tốc độ

    # ============================================================================
    # PHẦN 8: TÍNH LỰC ĐẨY (THRUST)
    # ============================================================================
    # Áp dụng định luật Newton II: F_thrust = m*a - m*g - D
    # Lực đẩy cần thiết để duy trì chuyển động theo quỹ đạo
    
    T_vec = m * a_g - m * g_vec - D  # Véc-tơ lực đẩy tổng (N)
                                      # = lực quán tính - trọng lực - lực cản
    
    T = np.linalg.norm(T_vec, axis=1, keepdims=True)  # Độ lớn lực đẩy ||T|| (N)
    t_hat = np.where(T > eps, T_vec / T, 0)  # Véc-tơ đơn vị hướng lực đẩy

    # ============================================================================
    # PHẦN 9: TÍNH VẬN TỐC LEO/HẠ (CLIMB VELOCITY)
    # ============================================================================
    # Vận tốc leo là thành phần vận tốc không khí theo hướng lực đẩy
    # v_c = v_a · t_hat (tích vô hướng)
    
    v_c = np.sum(v_a * t_hat, axis=1, keepdims=True)  # Vận tốc leo (m/s)
                                                        # Dương = leo lên
                                                        # Âm = hạ xuống

    # ============================================================================
    # PHẦN 10: TÍNH VẬN TỐC CẢM ỨNG (INDUCED VELOCITY)
    # ============================================================================
    # Sử dụng lý thuyết động lượng (momentum theory) cho rotor
    # Vận tốc không khí đi qua đĩa quay rotor do lực nâng gây ra
    # Công thức từ phương trình Bernoulli và bảo toàn động lượng
    
    sqrt_arg = (0.5 * v_c)**2 + T / (2.0 * rho * A_rotor)  # Đảm bảo không âm
    v_i = -0.5 * v_c + np.sqrt(np.where(sqrt_arg < 0, 0, sqrt_arg))  # Vận tốc cảm ứng (m/s)
                                                                        # Theo lý thuyết động lượng

    # ============================================================================
    # PHẦN 11: TÍNH CÁC THÀNH PHẦN CÔNG SUẤT
    # ============================================================================
    # Tổng công suất = Công suất lý tưởng + Công suất cản + Công suất hồ sơ
    
    # 1. Công suất lý tưởng (Ideal power / Induced + Climb power)
    # Năng lượng để tạo lực nâng và thay đổi độ cao
    P_ideal = (T * (v_i + v_c))[:, 0]  # P = F · v = T · (v_i + v_c)  [W]
                                        # v_i: vận tốc cảm ứng (do lực nâng)
                                        # v_c: vận tốc leo (thay đổi độ cao)
    
    # 2. Công suất cản ký sinh (Parasitic power)
    # Năng lượng để thắng lực cản không khí của thân máy bay
    P_d = (0.5 * rho * Cd * A_frame * (v_a_norm**3))[:, 0]  # P = D · v = 0.5·ρ·Cd·A·v³  [W]
                                                             # Tỉ lệ với v³
    
    # 3. Công suất hồ sơ (Profile power)
    # Năng lượng để quay cánh quạt, vượt qua ma sát và tổn thất hiệu suất
    # Ước lượng kinh nghiệm dựa trên tỉ số tải trọng
    P_p = P_p_hover * ((T / (m * g)) ** 1.5)[:, 0]  # P ∝ (T/W)^1.5  [W]
                                                     # W = m·g (trọng lượng)

    # Tổng công suất tiêu thụ tại mỗi waypoint
    P_total = P_ideal + P_d + P_p  # [W]

    # ============================================================================
    # PHẦN 12: TÍNH TỔNG NĂNG LƯỢNG TIÊU THỤ
    # ============================================================================
    # Tích phân công suất theo thời gian: E = ∫ P(t) dt
    # Sử dụng quy tắc hình thang (trapezoidal rule) cho độ chính xác cao
    
    # Công suất trung bình giữa mỗi cặp waypoint liên tiếp
    P_avg = 0.5 * (P_total[1:] + P_total[:-1])  # Trung bình số học (W)
    
    # Năng lượng = Công suất trung bình × Thời gian
    E = np.sum(P_avg * dt)  # Tổng năng lượng tiêu thụ trên toàn đường bay (J)
                            # 1 Joule = 1 Watt × 1 giây

    return E  # Trả về tổng năng lượng (Joules)