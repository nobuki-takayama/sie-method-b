#This program contains Japanese comments; please use machine translation to translate them into your preferred language if necessary.
#Parts of the input data or simulation data that need to be changed are marked with the comment "change here".


#  input-ode.rr (ann3.txt) tk_sie_b.test_sie_cheb() を lsq で
# https://www.math.kobe-u.ac.jp/OpenXM/Math/defusing/ec/tryb6/2021_07_09_tryb6_tmpb.py の simulation data との比較も.
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# GPU (CuPy) の利用確認
try:
    import cupy as cp
    xp = cp
    HAS_GPU = True
    print("Executing on GPU (CuPy).")
except ImportError:
    xp = np
    HAS_GPU = False
    print("No CuPy, executing on CPU (NumPy).")

# Risa/Asirのデータをインポート (setprec(30) で出力された高精度データ)
#import sie_method_b_data_targetB as data
import sie_method_b_data_cheb as data

def solve_sie_linear_robust(alpha=1.0, beta=1e10, gamma=1e-15):
    """
    線形最小二乗法 (lstsq) と列スケーリングを用いた最速・最高精度のソルバー
    """
    W_j = np.array(data.W_j, dtype=np.float64)
    L_ek_Tj = np.array(data.L_ek_Tj, dtype=np.float64)
    E_k_Pi = np.array(data.E_k_Pi, dtype=np.float64)
    Q_i = np.array(data.Q_i, dtype=np.float64)
    
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
    
    print("\n=== coeffs f_k (first 6 terms) ===")
    for i in range(6):
        print(f"f_{i} = {f_k[i]:.6f}")
        
    plt.figure(figsize=(10, 6))
    plt.plot(T_j, f_val, label='Approximated $E[\chi(M_t)]$ ($F_{29}$)', color='green', linewidth=2)
    plt.scatter(data.P_i, data.Q_i, color='red', marker='x', s=80, zorder=5, label='Monte-Carlo Points')
    
    plt.xlabel('$t$')
    plt.ylabel('$E[\chi(M_t)]$')
    plt.title('SIE Method B: Target B (Rank 11 ODE) - Linear Solver')

    #value by simulation: h-mle/rk-of-misc-2018/graph-yiex5c.r
    plt.scatter([3.8,3.82,3.84,3.86,3.88,3.9,3.92,3.94,3.96,3.98,4],[0.067223,0.044484,0.028353,0.017448,0.010471,0.006083,0.003348,0.001824,0.00087,0.000446,0.000214])
    plt.scatter([3.8,3.801,3.802,3.803,3.804,3.805,3.806,3.807,3.808,3.809],[0.06716,0.065485,0.064732,0.063315,0.061814,0.060477,0.059611,0.058257,0.05752,0.055971],marker='x')
    
    # 論文と同等の描画範囲
    plt.xlim(3.8, 4.0)
    plt.ylim(0, 0.07)
    
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def plot_relerror(f_k):
    T_j = np.array(data.T_j)
    E_k_Tj = np.array(data.E_k_Tj)
    f_val = np.dot(f_k, E_k_Tj)

    # xをソートした時の「インデックス（順序）」を取得
    sort_indices = np.argsort(T_j)
    # その順序で両方の配列を並べ替える
    x_sorted = T_j[sort_indices]
    y_sorted = f_val[sort_indices]

    sol_func = CubicSpline(x_sorted,y_sorted)
    # https://www.math.kobe-u.ac.jp/OpenXM/Math/defusing/ec/tryb6/2021_07_09_tryb6_tmpb.py
    # change here (the following 2 lines).
    simx=np.array([3.8,3.82,3.84,3.86,3.88,3.9,3.92,3.94,3.96,3.98,4])
    simy=np.array([0.067223,0.044484,0.028353,0.017448,0.010471,0.006083,0.003348,0.001824,0.00087,0.000446,0.000214])
    print("Relative errors");
    plt.plot(simx,(sol_func(simx)-simy)/simy)
    plt.show()

if __name__ == "__main__":
    # 最適なハイパーパラメータで実行
#    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1e8, gamma=1e-15) # it does not work
#    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1e6, gamma=1e-12) # it does not work

#import sie_method_b_data_targetB as data
#    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1e-1, gamma=1e-12)  # it works well. 台形公式, モノミアル基底の場合.

# import sie_method_b_data_cheb as data 
#    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1e-13, gamma=1e-20)   # cheb の場合. 
    f_k_opt, final_cost = solve_sie_linear_robust(alpha=1.0, beta=1e-30, gamma=0)   # cheb の場合. より精度が高い.

    print(f"\nfinal cost, 最終コスト (ペナルティ込): {final_cost:.4e}")
    plot_final_result(f_k_opt)
    plot_relerror(f_k_opt)  # change here. If no comparison data, comment it out.

