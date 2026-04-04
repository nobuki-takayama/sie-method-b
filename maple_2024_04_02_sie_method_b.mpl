# sie_method_b.mpl
# Maple Standalone Implementation for SIE Method B

#This program contains Japanese comments; please use machine translation to translate them into your preferred language if necessary.
#Parts of the input data or simulation data that need to be changed are marked with the comment "change here".


restart;
with(LinearAlgebra):
with(plots):
with(PolynomialTools):
with(StringTools):

# 1. 作用素の適用関数 (Symbolic Differentiation)
apply_op := proc(Op, BaseF, X, Dx)
    local ans, coeff_list, i, deg;
    ans := 0;
    
    # PolynomialTools を用いて Dx の多項式として係数をリスト化
    # 例: a*dx^2 + b*dx + c -> [c, b, a]
    coeff_list := CoefficientList(Op, Dx);
    
    for i from 1 to nops(coeff_list) do
        deg := i - 1; # Maple のリストは 1-origin
        if coeff_list[i] <> 0 then
            if deg = 0 then
                ans := ans + coeff_list[i] * BaseF;
            else
                # X で deg 回微分 (X$deg)
                ans := ans + coeff_list[i] * diff(BaseF, X$deg);
            end if;
        end if;
    end do;
    
    return ans;
end proc:

# 2. チェビシェフデータ生成
cheb_data := proc(T_s, T_e, Num_basis, N, X)
    local S, basis_arr, T_arr, W_arr, j, arg, s_val, w_raw;
    
    # 基底の生成 (1-origin に注意)
    S := (2*X - (T_s + T_e)) / (T_e - T_s);
    basis_arr := Array(1..Num_basis);
    basis_arr[1] := 1;
    if Num_basis > 1 then
        basis_arr[2] := S;
    end if;
    for j from 3 to Num_basis do
        basis_arr[j] := expand(2 * S * basis_arr[j-1] - basis_arr[j-2]);
    end do;

    # 分点と重みの生成 (厳密な数式ツリーとして保持)
    T_arr := Array(1..N);
    W_arr := Array(1..N);
    for j from 1 to N do
        # インデックスのズレ (j-1) に注意して数式を構築
        arg := Pi * (2*(j-1) + 1) / (2*N);
        s_val := cos(arg);
        w_raw := (Pi / N) * sin(arg);
        
        T_arr[j] := (T_e - T_s)/2 * s_val + (T_s + T_e)/2;
        W_arr[j] := w_raw * (T_e - T_s)/2;
    end do;

    return convert(basis_arr, list), convert(T_arr, list), convert(W_arr, list);
end proc:

# 3. メインルーチン
run_target_b := proc()
    local ode_str, Op_raw, Op, X, Dx;
    local T_s, T_e, Num_basis, N_pts, basis, T_list, W_list;
    local P_list, Q_list, M, op_applied;
    local L_ek_Tj, E_k_Pi, W_j, Q_i;
    local alpha, beta_w, gamma_w, max_L, A_ode, A_const, A_reg, A, b;
    local col_scales, A_scaled, f_k_scaled, f_k_opt;
    local p_plot, line_plot, f_val_func, cost;
    local j, k;

    # 🌟【重要】計算精度を 150桁 (約500ビット) に設定し、桁落ちを無効化
    Digits := 150;

    X := 'x';
    Dx := 'dx';

    # Risa/Asir の出力ファイルを読み込みパースする
    try
        ode_str := FileTools[Text][ReadFile]("input-ode.txt"); #change here
        ode_str := SubstituteAll(ode_str, "ODE=", "");
        ode_str := SubstituteAll(ode_str, ";;", "");
        Op_raw := parse(ode_str);
    catch:
        printf("エラー: input-ode.txt が読み込めません。\n");
        return;
    end try;

    # Rank 11 に修正
    Op := Op_raw * Dx; # chage here.

    #change here (the following 4 lines)
    T_s := 38/10;
    T_e := 4;
    Num_basis := 30;
    N_pts := 200;

    #change here (The following 5 lines).
    printf("基底関数と求積データを生成中...\n");
    basis, T_list, W_list := cheb_data(T_s, T_e, Num_basis, N_pts, X);

    P_list := [seq(38/10 + i/1000, i=0..9)];
    Q_list := [1679/25000, 13097/200000, 16183/250000, 12663/200000, 30907/500000, 
               60477/1000000, 59611/1000000, 58257/1000000, 719/12500, 55971/1000000];
    
    M := Num_basis;

    printf("1. 微分作用素を基底関数に適用中 (Symbolic Differentiation)...\n");
    op_applied := [seq(apply_op(Op, basis[k], X, Dx), k=1..M)];

    printf("2. 評価行列の構築中 (High-precision Evaluation)...\n");
    L_ek_Tj := Matrix(M, N_pts, datatype=float);
    for k from 1 to M do
        for j from 1 to N_pts do
            # evalf を呼ぶことで Digits=150 の精度で高速評価される
            L_ek_Tj[k, j] := evalf(subs(X = T_list[j], op_applied[k]));
        end do;
    end do;

    E_k_Pi := Matrix(M, nops(P_list), datatype=float);
    for k from 1 to M do
        for j from 1 to nops(P_list) do
            E_k_Pi[k, j] := evalf(subs(X = P_list[j], basis[k]));
        end do;
    end do;

    W_j := Vector([seq(evalf(w), w in W_list)], datatype=float);
    Q_i := Vector([seq(evalf(q), q in Q_list)], datatype=float);

    printf("3. 最適化問題の構築と求解 (Linear Least Squares)...\n");
    alpha := evalf(1.0);
    beta_w := evalf(10^(-13));
    gamma_w := evalf(10^(-20));

    max_L := max(map(abs, convert(L_ek_Tj, list)));
    if max_L > 0 then
        L_ek_Tj := L_ek_Tj / max_L;
    end if;

    A_ode := Matrix(N_pts, M, datatype=float);
    for j from 1 to N_pts do
        for k from 1 to M do
            A_ode[j, k] := sqrt(alpha * W_j[j]) * L_ek_Tj[k, j];
        end do;
    end do;

    A_const := Matrix(nops(P_list), M, datatype=float);
    for j from 1 to nops(P_list) do
        for k from 1 to M do
            A_const[j, k] := sqrt(beta_w) * E_k_Pi[k, j];
        end do;
    end do;

    A_reg := Matrix(M, M, datatype=float);
    for k from 1 to M do
        A_reg[k, k] := sqrt(gamma_w);
    end do;

    # 行列の縦結合 (vstack に相当) とベクトル結合
    A := <<A_ode> , <A_const> , <A_reg>>;
    b := <Vector(N_pts, 0) , sqrt(beta_w)*Q_i , Vector(M, 0)>;

    # 列スケーリング
    col_scales := Vector(M);
    A_scaled := Matrix(RowDimension(A), M, datatype=float);
    for k from 1 to M do
        col_scales[k] := max(map(abs, convert(Column(A, k), list)));
        if col_scales[k] = 0 then col_scales[k] := 1.0; end if;
        A_scaled[.., k] := Column(A, k) / col_scales[k];
    end do;

    # 線形最小二乗法 (SVD 等を自動選択して安全に解く)
    #f_k_scaled := LeastSquares(A_scaled, b);

    #  修正後：SVDの反復限界を回避するため、正規方程式 (Normal Equations) を直接解く
    # Digits=150 の圧倒的精度があるからこそ許される「禁忌の力技（Direct Solve）」です
    Normal_A := Transpose(A_scaled) . A_scaled;
    Normal_b := Transpose(A_scaled) . b;
    f_k_scaled := LinearSolve(Normal_A, Normal_b);
    
    f_k_opt := Vector(M);
    for k from 1 to M do
        f_k_opt[k] := f_k_scaled[k] / col_scales[k];
    end do;

    # コスト計算
    cost := Norm(A_scaled . f_k_scaled - b, 2)^2;
    printf("\n最終コスト: %e\n", cost);

    printf("4. 結果のプロット...\n");
    # f_k * basis_k を足し合わせた近似関数を構築
    f_val_func := unapply(add(f_k_opt[k] * basis[k], k=1..M), X);

    ## change here (plot range)
    line_plot := plot(f_val_func(X), X=3.8..4.0, color=green, thickness=2, legend="Approximated E[chi(M_t)]");
    p_plot := pointplot([seq([P_list[i], Q_list[i]], i=1..nops(P_list))], color=red, symbol=cross, symbolsize=20, legend="Simulation Points");

    display(line_plot, p_plot, view=[3.8..4.0, 0..0.07], title="SIE Method B: Target B (Maple Standalone)", gridlines=true);

end proc:

# 実行
run_target_b();
