import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from IPython.display import HTML

# --- パラメータ設定 ---
L = 1000.0          # 道路の長さ [m]
T_max = 400.0       # シミュレーション時間 [s]
nx = 200            # 空間分割数
dx = L / nx
dt = 0.1            # 時間刻み

rho_max = 0.10      # 最大密度 [veh/m]
v_free_base = 20.0  # ベース自由流速度 [m/s]

bn_start = 600.0    # ボトルネック開始位置
bn_end = 800.0      # ボトルネック終了位置
bn_severity = 0.6   # 速度制限 (60%ダウン)

rho_initial = 0.025 # 初期流入密度

# --- 空間設定 ---
x = np.linspace(0, L, nx)
v_max_field = np.ones(nx) * v_free_base
bn_indices = np.where((x >= bn_start) & (x <= bn_end))[0]
v_max_field[bn_indices] *= (1.0 - bn_severity)

# トレーサー粒子（車両）の初期化
num_particles = 40
particle_positions = np.linspace(0, L, num_particles)

# --- LWRモデル関数 ---
def flux(rho, v_m):
    rho = np.maximum(rho, 0.0)
    return v_m * rho * (1 - rho / rho_max)

def compute_godunov_flux(rho_L, rho_R, v_m_interface):
    rho_crit = rho_max / 2.0
    q_max_local = flux(rho_crit, v_m_interface)
    D = flux(rho_L, v_m_interface) if rho_L < rho_crit else q_max_local
    S = q_max_local if rho_R < rho_crit else flux(rho_R, v_m_interface)
    return min(D, S)

# --- 衝撃波速度の計算関数 ---
def calculate_shock_info(current_rho, current_v_max_field):
    # ボトルネックより手前(上流)で、密度が急激に上がっている場所を探す
    # 探索範囲: 0m ～ ボトルネック開始位置
    search_idx_end = int(bn_start / dx)
    search_rho = current_rho[:search_idx_end]
    
    # 密度の勾配（変化率）を計算
    gradient = np.diff(search_rho)
    
    # 勾配が最大になる場所＝渋滞の境界面（自由流→渋滞への突入点）
    if len(gradient) == 0: return None, None
    shock_idx = np.argmax(gradient)
    
    # ノイズ対策: 密度の差が小さすぎる場合は「渋滞なし」と判定
    rho_up = current_rho[shock_idx]     # 上流側（自由流）密度
    rho_down = current_rho[shock_idx+1] # 下流側（渋滞）密度
    
    if (rho_down - rho_up) < 0.02:
        return None, None
        
    # ランキン・ユゴニオの条件で速度Uを計算
    # U = (Q2 - Q1) / (rho2 - rho1)
    q_up = flux(rho_up, current_v_max_field[shock_idx])
    q_down = flux(rho_down, current_v_max_field[shock_idx+1])
    
    u_shock = (q_down - q_up) / (rho_down - rho_up)
    
    shock_pos_x = x[shock_idx]
    return shock_pos_x, u_shock

# --- シミュレーションループ ---
rho = np.ones(nx) * rho_initial
rho += np.random.uniform(-0.005, 0.005, nx)
rho = np.clip(rho, 0.0, rho_max)

history_rho = []
history_particles = []
history_shock = [] # 衝撃波情報の履歴

t_steps = int(T_max / dt)

for t in range(t_steps):
    history_rho.append(rho.copy())
    history_particles.append(particle_positions.copy())
    
    # 衝撃波情報の計算と保存
    s_pos, s_vel = calculate_shock_info(rho, v_max_field)
    history_shock.append((s_pos, s_vel))
    
    # LWR更新
    current_v_field = v_max_field * (1 - rho / rho_max)
    current_v_field = np.maximum(current_v_field, 0.0)

    rho_new = np.zeros_like(rho)
    F = np.zeros(nx + 1)
    for i in range(1, nx):
        F[i] = compute_godunov_flux(rho[i-1], rho[i], v_max_field[i])
    F[0] = compute_godunov_flux(rho_initial, rho[0], v_max_field[0])
    F[nx] = compute_godunov_flux(rho[nx-1], 0.0, v_max_field[nx-1])
    
    for i in range(nx):
        rho_new[i] = rho[i] - (dt / dx) * (F[i+1] - F[i])
    rho = rho_new
    
    # 粒子更新
    particle_velocities = np.interp(particle_positions, x, current_v_field)
    particle_positions += particle_velocities * dt
    particle_positions = particle_positions % L

# --- アニメーション表示 ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})
plt.subplots_adjust(hspace=0.4)

# === 上段: 密度グラフ ===
bn_rect1 = Rectangle((bn_start, 0), bn_end - bn_start, rho_max*1.1, color='gray', alpha=0.3)
ax1.add_patch(bn_rect1)
line_rho, = ax1.plot([], [], 'b-', lw=2, label='Density')
# ax1.axhline(rho_max/2, color='r', linestyle='--', alpha=0.3, label='Critical Density')

# 衝撃波を示す垂直線とテキスト
shock_line = ax1.axvline(x=-10, color='red', linestyle='-', lw=2, label='Shock Front')
shock_text = ax1.text(0.05, 0.8, '', transform=ax1.transAxes, color='red', fontweight='bold')

ax1.set_xlim(0, L)
ax1.set_ylim(0, rho_max * 1.1)
ax1.set_ylabel('Density [veh/m]')
ax1.set_title('LWR Model: Density & Shock Wave Speed')
ax1.legend(loc='upper right', fontsize='small')
ax1.grid(True)

# === 下段: 車両移動 ===
ax2.add_patch(Rectangle((0, -0.1), L, 0.2, color='lightgray', alpha=0.5))
ax2.add_patch(Rectangle((bn_start, -0.1), bn_end - bn_start, 0.2, color='gray', alpha=0.3))
points, = ax2.plot([], [], 'ko', markersize=6, alpha=0.8)

# 衝撃波位置を下段にも表示
shock_line_2 = ax2.axvline(x=-10, color='red', linestyle=':', lw=2, alpha=0.7)

ax2.set_xlim(0, L)
ax2.set_ylim(-0.2, 0.2)
ax2.set_xlabel('Position x [m]')
ax2.set_yticks([])
ax2.spines['left'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.set_title('Vehicle Trajectories')

time_text = ax1.text(0.02, 0.92, '', transform=ax1.transAxes)
plt.close()

def init():
    line_rho.set_data([], [])
    points.set_data([], [])
    time_text.set_text('')
    shock_line.set_xdata([-10]) # 画面外へ
    shock_line_2.set_xdata([-10])
    shock_text.set_text('')
    return line_rho, points, time_text, shock_line, shock_line_2, shock_text

def update(frame):
    # 密度更新
    line_rho.set_data(x, history_rho[frame])
    
    # 粒子更新
    current_particles = history_particles[frame]
    points.set_data(current_particles, np.zeros_like(current_particles))
    
    # 衝撃波情報の表示更新
    s_pos, s_vel = history_shock[frame]
    if s_pos is not None:
        shock_line.set_xdata([s_pos])
        shock_line_2.set_xdata([s_pos])
        # 速度を表示 (負の値なら「Backwards」と注釈)
        direction = "(Backwards)" if s_vel < 0 else "(Forwards)"
        shock_text.set_text(f'Shock Speed: {s_vel:.2f} m/s\n{direction}')
    else:
        shock_line.set_xdata([-10])
        shock_line_2.set_xdata([-10])
        shock_text.set_text('No Shock Wave')

    time_text.set_text(f'Time: {frame * dt:.1f} s')
    return line_rho, points, time_text, shock_line, shock_line_2, shock_text

step_skip = 5
ani = animation.FuncAnimation(fig, update, frames=range(0, t_steps, step_skip),
                              init_func=init, interval=30, blit=True)
HTML(ani.to_jshtml())
plt.show()