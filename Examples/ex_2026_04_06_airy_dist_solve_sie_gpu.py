import numpy as np
import matplotlib.pyplot as plt

# GPU (CuPy) の利用確認
try:
    import cupy as cp
    xp = cp
    HAS_GPU = True
    print("🚀 GPU (CuPy) モードで実行します。")
except ImportError:
    xp = np
    HAS_GPU = False
    print("⚠️ CuPyが見つからないため、CPU (NumPy) で実行します。")

# Risa/Asirのデータをインポート (setprec(30) で出力された高精度データ)
import airy_dist_sie_data as data

def solve_sie_linear_robust(alpha=1.0, beta=1e10, gamma=1e-15):
    """
    線形最小二乗法 (lstsq) と列スケーリングを用いた最速・最高精度のソルバー
    """
    W_j = np.array(data.W_j, dtype=np.float64)
    L_ek_Tj = np.array(data.L_ek_Tj, dtype=np.float64)
    E_k_Pi = np.array(data.E_k_Pi, dtype=np.float64)
    Q_i = np.array(data.Q_i, dtype=np.float64)

    print('Number of basis = ',E_k_Pi.shape[0])
    
    M = L_ek_Tj.shape[0]
    N_plus_1 = L_ek_Tj.shape[1]
    
    # ODE行列の全体スケーリング (オーバーフロー防止)
    max_L = np.max(np.abs(L_ek_Tj))
    if max_L > 0:
        L_ek_Tj_scaled = L_ek_Tj / max_L
    else:
        L_ek_Tj_scaled = L_ek_Tj

    # 1. 行列 A と ベクトル b の構築
    A_ode = np.sqrt(alpha * W_j)[:, None] * L_ek_Tj_scaled.T
    A_const = np.sqrt(beta) * E_k_Pi.T
    A_reg = np.sqrt(gamma) * np.eye(M)
    
    A = np.vstack((A_ode, A_const, A_reg))
    b = np.concatenate((np.zeros(N_plus_1), np.sqrt(beta) * Q_i, np.zeros(M)))
    
    # 🌟【重要】列スケーリング (Preconditioning)
    col_scales = np.max(np.abs(A), axis=0)
    col_scales[col_scales == 0] = 1.0
    A_scaled = A / col_scales
    
    # 2. GPU/CPU による線形最小二乗法の実行
    print("GPU/CPU 線形最小二乗法 (lstsq) で最適化を実行中...")
    A_xp = xp.array(A_scaled)
    b_xp = xp.array(b)
    
    f_k_scaled, _, _, _ = xp.linalg.lstsq(A_xp, b_xp, rcond=None)
    
    if HAS_GPU:
        f_k_scaled = cp.asnumpy(f_k_scaled)
        
    # スケーリングを元に戻して真の係数を取得
    f_k_opt = f_k_scaled / col_scales
    
    # 損失関数の計算
    cost = np.sum((np.dot(A_scaled, f_k_scaled) - b)**2)
    
    return f_k_opt, cost

def plot_final_result(f_k):
    T_j = np.array(data.T_j)
    E_k_Tj = np.array(data.E_k_Tj)
    f_val = np.dot(f_k, E_k_Tj)
    num_basis=E_k_Tj.shape[0]
    
    print("\n=== 最適化された係数 f_k (最初の6項) ===")
    for i in range(6):
        print(f"f_{i} = {f_k[i]:.6f}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(T_j, f_val, label='Approximated NC Z(t)]$ ($F_{'+str(num_basis-1)+'}$)', color='green', linewidth=2)
    plt.scatter(data.P_i, data.Q_i, color='red', marker='x', s=80, zorder=5, label='Monte-Carlo Points')
    
    plt.xlabel('$t$')
    plt.ylabel('NC Z(t)')
    plt.title('SIE Method B: Target B (Rank 3 ODE) - Linear Solver')

# 描画範囲は plt に任せる.    
#    plt.xlim(np.min(T_j), np.max(T_j))
#    plt.ylim(0, 0.07)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    # 最適なハイパーパラメータで実行
#    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1e-13, gamma=1e-20)   # cheb の場合. 
    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1, gamma=0)   # cheb の場合. 

    print(f"\n最終コスト (ペナルティ込): {final_cost:.4e}")
    plot_final_result(f_k_opt)
