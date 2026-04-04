#This program contains Japanese comments; please use machine translation to translate them into your preferred language if necessary.
#Parts of the input data or simulation data that need to be changed are marked with the comment "change here".

using Oscar
using LinearAlgebra
using Plots

# 🌟 グローバルスコープで多項式環 (Polynomial Rings) を厳密に定義
# まず有理数体 QQ 上の変数 x の多項式環を作成
QQx, x = polynomial_ring(QQ, "x")
# 次に、その多項式環 QQx を係数とする変数 dx の多項式環を作成 (反復多項式環)
QQx_dx, dx = polynomial_ring(QQx, "dx")

# 🌟 追加: OSCARの有理数係数多項式に、浮動小数を安全に代入・評価する関数
function eval_poly_bf(poly, val::BigFloat)
    ans = big(0.0)
    for i in degree(poly):-1:0
        c = coeff(poly, i)
        c_bf = BigFloat(numerator(c)) / BigFloat(denominator(c))
        ans = ans * val + c_bf
    end
    return ans
end

# 1. 微分作用素の適用関数
function apply_op(Op, BaseF)
    ans = zero(parent(BaseF))
    # Op は dx の多項式。degree(Op) で dx の最大次数を取得
    for i in 0:degree(Op)
        c = coeff(Op, i) # i次 (dx^i) の係数 (xの多項式) を取得
        if !iszero(c)
            deriv = BaseF
            # i階微分を適用 (OSCARの derivative 関数)
            for k in 1:i
                deriv = derivative(deriv)
            end
            ans += c * deriv
        end
    end
    return ans
end

# 2. チェビシェフデータの生成
function cheb_data(T_s, T_e, Num_basis, N, var_x)
    # 基底は完全な有理数 (QQ) の多項式として生成
    S = (2 * var_x - (T_s + T_e)) / (T_e - T_s)
    basis = typeof(S)[]
    push!(basis, one(parent(var_x)))
    if Num_basis > 1
        push!(basis, S)
    end
    for j in 3:Num_basis
        push!(basis, 2 * S * basis[j-1] - basis[j-2])
    end

    T_list = BigFloat[]
    W_list = BigFloat[]
    for j in 1:N
        # Juliaは1-originなので、分点の式は (2j - 1)
        arg = big(pi) * (2*j - 1) / (2*N)
        s_val = cos(arg)
        w_raw = (big(pi) / N) * sin(arg)

	# --- 修正前 ---
        # t_val = (T_e - T_s)/2 * s_val + (T_s + T_e)/2
        # w_val = w_raw * (T_e - T_s)/2

        # 🌟 --- 修正後 ---
        # OSCARの有理数から分子・分母を抽出し、BigFloatの分数として明示的に実体化する
        diff_bf = BigFloat(numerator(T_e - T_s)) / BigFloat(denominator(T_e - T_s))
         mid_bf  = BigFloat(numerator(T_s + T_e)) / BigFloat(denominator(T_s + T_e))

         t_val = (diff_bf / 2) * s_val + (mid_bf / 2)
         w_val = w_raw * (diff_bf / 2)

        push!(T_list, t_val)
        push!(W_list, w_val)
    end
    return basis, T_list, W_list
end

# 3. メインルーチン
function run_target_b()
    # 🌟 計算精度を 500ビット(約150桁) に設定し、評価時の桁落ちを完全に無効化
    setprecision(BigFloat, 500)

    # Risa/Asir のファイルを読み込みパース
    ode_str = ""
    try
        ode_str = read("input-ode.txt", String)  # change here
        ode_str = replace(ode_str, "ODE=" => "", ";;" => "", "\n" => "", "\r" => "")
    catch
        println("エラー: input-ode.rr が読み込めません。")
        return
    end

    println("ODEをパース中...")
    # 文字列の式を評価。グローバルの x, dx が適用され、OSCARの多項式オブジェクトとして実体化する
    Op_raw = eval(Meta.parse(ode_str))
    
    # Rank 11 に修正
    Op = Op_raw * dx  # change here

    # change here (4 lines)
    T_s = QQ(38,10)
    T_e = QQ(4)
    Num_basis = 30
    N_pts = 200

    # change here (4 lines)
    println("基底関数と求積データを生成中...")
    basis, T_list, W_list = cheb_data(T_s, T_e, Num_basis, N_pts, x)

    P_list = [3.8 + i/1000 for i in 0:9]
    Q_list = [1679/25000, 13097/200000, 16183/250000, 12663/200000, 30907/500000, 
              60477/1000000, 59611/1000000, 58257/1000000, 719/12500, 55971/1000000]

    M = Num_basis

    println("1. 微分作用素を基底関数に適用中 (OSCAR Symbolic Differentiation)...")
    op_applied = [apply_op(Op, b) for b in basis]

    println("2. 評価行列の構築中 (BigFloat Evaluation)...")

    L_ek_Tj = zeros(Float64, M, N_pts)
    for k in 1:M
        for j in 1:N_pts
            # 500ビット精度の BigFloat を直接代入して評価し、最後に Float64 (64bit) に落とす
            L_ek_Tj[k, j] = Float64(eval_poly_bf(op_applied[k], T_list[j]))
        end
    end

    E_k_Pi = zeros(Float64, M, length(P_list))
    for k in 1:M
        for i in 1:length(P_list)
            E_k_Pi[k, i] = Float64(eval_poly_bf(basis[k], BigFloat(P_list[i])))
        end
    end

    W_j = Float64.(W_list)
    Q_i = Float64.(Q_list)

    println("3. 最適化問題の構築と求解 (Julia Linear Algebra)...")
    alpha = 1.0
    beta_w = 1e-13
    gamma_w = 1e-20

    max_L = maximum(abs, L_ek_Tj)
    if max_L > 0
        L_ek_Tj ./= max_L
    end

    A_ode = zeros(Float64, N_pts, M)
    for j in 1:N_pts
        for k in 1:M
            A_ode[j, k] = sqrt(alpha * W_j[j]) * L_ek_Tj[k, j]
        end
    end

    A_const = zeros(Float64, length(P_list), M)
    for i in 1:length(P_list)
        for k in 1:M
            A_const[i, k] = sqrt(beta_w) * E_k_Pi[k, i]
        end
    end

    # 正則化行列
    A_reg = Matrix{Float64}(I, M, M) .* sqrt(gamma_w)

    # 行列の縦結合 (vcat) とベクトル結合
    A = vcat(A_ode, A_const, A_reg)
    b = vcat(zeros(Float64, N_pts), sqrt(beta_w) .* Q_i, zeros(Float64, M))

    # 列スケーリング
    col_scales = maximum(abs, A, dims=1)
    replace!(y -> iszero(y) ? 1.0 : y, col_scales)
    A_scaled = A ./ col_scales

    # 🌟 Julia 組み込みの最小二乗法ソルバー ( LAPACK の \ 演算子による SVD/QR 自動解決)
    f_k_scaled = A_scaled \ b
    f_k_opt = f_k_scaled ./ vec(col_scales)

    cost = sum((A_scaled * f_k_scaled .- b).^2)
    println("\n最終コスト: ", cost)

    println("4. 結果のプロット...")
    T_j_np = Float64.(T_list)
    E_k_Tj = zeros(Float64, M, N_pts)
    for k in 1:M
        for j in 1:N_pts
            E_k_Tj[k, j] = Float64(eval_poly_bf(basis[k], T_list[j]))
        end
    end

    # 行列の積で関数値を一括計算
    f_val = E_k_Tj' * f_k_opt  

    # change here (plot range)
    p = plot(T_j_np, f_val, label="Approximated E[chi(M_t)]", color=:green, linewidth=2,
             xlabel="t", ylabel="E[chi(M_t)]", title="SIE Method B: Target B (Julia / OSCAR)",
             xlims=(3.8, 4.0), ylims=(0, 0.07), grid=true, legend=:topright)
    scatter!(p, P_list, Q_list, label="Simulation Points", color=:red, markershape=:cross, markersize=8)

    display(p)
end

# 実行
run_target_b()