#This program contains Japanese comments; please use machine translation to translate them into your preferred language if necessary.
#Parts of the input data or simulation data that need to be changed are marked with the comment "change here".

import numpy as np
import matplotlib.pyplot as plt

x, dx = var('x dx')

def apply_op(Op, BaseF, X, Dx):
    ans = 0
    for coeff, degree in Op.coefficients(Dx):
        deg = int(degree)  # 前回修正した型キャスト
        if deg == 0:
            ans += coeff * BaseF
        else:
            ans += coeff * diff(BaseF, X, deg)
    return ans

def cheb_data(T_s, T_e, Num_basis, N, X):
    S = (2 * X - (T_s + T_e)) / (T_e - T_s)
    basis = [1]
    if Num_basis > 1:
        basis.append(S)
    for j in range(2, Num_basis):
        basis.append((2 * S * basis[j-1] - basis[j-2]).expand())
    
    T_list = []
    W_list = []
    for j in range(N):
        arg = pi * (2 * j + 1) / (2 * N)
        s_val = cos(arg)
        w_raw = (pi / N) * sin(arg)
        
        t_val = (T_e - T_s)/2 * s_val + (T_s + T_e)/2
        w_val = w_raw * (T_e - T_s)/2
        T_list.append(t_val)
        W_list.append(w_val)
        
    return basis, T_list, W_list

def solve_sie_method_b(Op, basis, P_list, Q_list, T_list, W_list, X, Dx, alpha=1.0, beta=1e-1, gamma=1e-12):
    M = len(basis)
    N_pts = len(T_list)
    
    print("1. 微分作用素を基底関数に適用中 (Symbolic Differentiation)...")
    op_applied = [apply_op(Op, b, X, Dx) for b in basis]

    print("2. 評価行列の構築中 (Fast High-precision Evaluation)...")
    # 🌟【最重要修正】100ビットから 500ビット(約150桁) へ精度を極限まで引き上げ、桁落ちをねじ伏せる
    R500 = RealField(500)
    R500_x = R500['x']
    
    # 記号式を 150桁精度の多項式に変換
    poly_op_applied = [R500_x(expr) for expr in op_applied]
    poly_basis = [R500_x(expr) for expr in basis]
    
    # 評価点も 500ビット精度に事前変換
    T_list_R500 = [R500(t) for t in T_list]
    P_list_R500 = [R500(p) for p in P_list]
    
    L_ek_Tj = np.zeros((M, N_pts), dtype=np.float64)
    E_k_Pi = np.zeros((M, len(P_list)), dtype=np.float64)
    
    # Horner法による超高速評価
    for k in range(M):
        for j in range(N_pts):
            L_ek_Tj[k, j] = float(poly_op_applied[k](T_list_R500[j]))
            
        for i in range(len(P_list)):
            E_k_Pi[k, i] = float(poly_basis[k](P_list_R500[i]))
# 修正はここまで    
            
    W_j = np.array([float(R500(w)) for w in W_list], dtype=np.float64)
    Q_i = np.array([float(q) for q in Q_list], dtype=np.float64)
    
    print("3. 最適化問題の構築と求解 (Linear Least Squares)...")
    max_L = np.max(np.abs(L_ek_Tj))
    L_ek_Tj_scaled = L_ek_Tj / max_L if max_L > 0 else L_ek_Tj
    
    A_ode = np.sqrt(alpha * W_j)[:, None] * L_ek_Tj_scaled.T
    A_const = np.sqrt(beta) * E_k_Pi.T
    A_reg = np.sqrt(gamma) * np.eye(M)
    
    A = np.vstack((A_ode, A_const, A_reg))
    b = np.concatenate((np.zeros(N_pts), np.sqrt(beta) * Q_i, np.zeros(M)))
    
    col_scales = np.max(np.abs(A), axis=0)
    col_scales[col_scales == 0] = 1.0
    A_scaled = A / col_scales
    
    f_k_scaled, residuals, rank, s_vals = np.linalg.lstsq(A_scaled, b, rcond=None)
    f_k_opt = f_k_scaled / col_scales
    
    cost = np.sum((np.dot(A_scaled, f_k_scaled) - b)**2)
    return f_k_opt, cost, poly_basis, T_list_R500

def plot_result(f_k, poly_basis, T_list_R500, P_list, Q_list):
    N_pts = len(T_list_R500)
    M = len(poly_basis)
    
    T_j_np = np.array([float(t) for t in T_list_R500])
    E_k_Tj = np.zeros((M, N_pts))
    for k in range(M):
        for j in range(N_pts):
            # ここも 500ビット精度で評価して float に落とす
            E_k_Tj[k, j] = float(poly_basis[k](T_list_R500[j]))
            
    f_val = np.dot(f_k, E_k_Tj)
    
    plt.figure(figsize=(10, 6))
    plt.plot(T_j_np, f_val, label='Approximated $E[\\chi(M_t)]$', color='green', linewidth=2)
    plt.scatter([float(p) for p in P_list], Q_list, color='red', marker='x', s=80, zorder=5, label='Simulation Points')
    
    plt.xlabel('$t$')
    plt.ylabel('$E[\\chi(M_t)]$')
    plt.title('SIE Method B: Target B (SageMath Standalone)')
    plt.xlim(3.8, 4.0)  # change here
    plt.ylim(0, 0.07)   # change here
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

def run_target_b():
    try:
        with open('input-ode.txt', 'r') as f:  # change here
            ode_str = f.read().strip()
            ode_str = ode_str.replace('ODE=', '').replace(';;', '').strip()
    except FileNotFoundError:
        print("エラー: input-ode.rr が見つかりません。")
        return

    print("ODEをパース中...")
    Op_raw = SR(ode_str)
    Op = Op_raw * dx  # change here
    
    # 🌟【重要修正】すべてを「厳密な有理数」として定義し、浮動小数の混入をブロック！
    # change here (4 lines).
    T_s = QQ(38)/10
    T_e = QQ(4)
    Num_basis = 30
    N = 200
    
    print("基底関数と求積データを生成中...")
    basis, T_list, W_list = cheb_data(T_s, T_e, Num_basis, N, x)

    # change here (4 lines)    
    # 🌟 参照点も厳密な有理数で生成
    P_list = [QQ(38)/10 + QQ(i)/1000 for i in range(10)]
    Q_list = [1679/25000, 13097/200000, 16183/250000, 12663/200000, 30907/500000, 
              60477/1000000, 59611/1000000, 58257/1000000, 719/12500, 55971/1000000]
              
    f_k_opt, cost, poly_basis, T_list_R500 = solve_sie_method_b(
        Op, basis, P_list, Q_list, T_list, W_list, x, dx, 
        alpha=1.0, beta=1e-13, gamma=1e-20 # 昨日成功したノイズ回避設定
    )
    
    print(f"\n最終コスト: {cost:.4e}")
    plot_result(f_k_opt, poly_basis, T_list_R500, P_list, Q_list)

if __name__ == "__main__":
    run_target_b()
