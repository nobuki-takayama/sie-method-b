(* SIE Method B: Target B (Mathematica Standalone) *)
(* This program contains Japanese comments; please use machine translation to translate them into your preferred language if necessary.
#Parts of the input data or simulation data that need to be changed are marked with the comment "change here".
*)

ClearAll["Global`*"];
SieF=0;
(* 1. 微分作用素の適用 (Symbolic Differentiation) *)
ApplyOp[Op_, BaseF_, X_, Dx_] := Module[
  {coeffs, ans = 0, deg},
  (* Dx の多項式として係数をリスト抽出 (定数項が 1 番目) *)
  coeffs = CoefficientList[Op, Dx];
  
  For[i = 1, i <= Length[coeffs], i++,
    deg = i - 1; (* 1-origin なので 1 を引く *)
    If[coeffs[[i]] =!= 0,
      ans += coeffs[[i]] * D[BaseF, {X, deg}];
    ];
  ];
  ans
];

(* 2. チェビシェフデータ生成 *)
ChebData[Ts_, Te_, NumBasis_, Npts_, X_] := Module[
  {S, basis, Tlist, Wlist, arg, sVal, wRaw},
  
  (* 基底の生成 *)
  S = (2*X - (Ts + Te)) / (Te - Ts);
  basis = ConstantArray[0, NumBasis];
  basis[[1]] = 1;
  If[NumBasis > 1, basis[[2]] = S];
  For[j = 3, j <= NumBasis, j++,
    basis[[j]] = Expand[2 * S * basis[[j-1]] - basis[[j-2]]];
  ];

  (* 分点と重みの生成 (厳密な数式として保持) *)
  Tlist = ConstantArray[0, Npts];
  Wlist = ConstantArray[0, Npts];
  For[j = 1, j <= Npts, j++,
    (* Mathematica は 1-origin なので (2j-1) となる *)
    arg = Pi * (2*j - 1) / (2*Npts); 
    sVal = Cos[arg];
    wRaw = (Pi / Npts) * Sin[arg];
    
    Tlist[[j]] = (Te - Ts)/2 * sVal + (Ts + Te)/2;
    Wlist[[j]] = wRaw * (Te - Ts)/2;
  ];

  {basis, Tlist, Wlist}
];

(* 3. メインルーチン *)
RunTargetB[] := Module[
  {odeStr, OpRaw, Op, Ts, Te, NumBasis, Npts, basis, Tlist, Wlist,
   Plist, Qlist, M, opApplied, LekTj, EkPi, Wj, Qi,
   alpha, betaW, gammaW, maxL, Aode, Aconst, Areg, A, b,g,
   colScales, Ascaled, fkScaled, fkOpt, cost, fValFunc,
   prec = 150}, (* 🌟 計算精度を 150桁 に設定 *)

  (* Risa/Asir のファイルを読み込み、Mathematica の式にパース *)
  odeStr = Import["input-ode.txt", "Text"];  (* change here. ファイルの場所を確認 *) 
  odeStr = StringReplace[odeStr, {"ODE=" -> "", ";;" -> ""}];
  OpRaw = ToExpression[odeStr];
  
  (* Rank 11 に修正 *)
  Op = OpRaw * dx; (* change here *)

  (* change here (two lines) *)
  Ts = 38/10; Te = 4;  (* 有理数で, 小数はだめ *)
  NumBasis = 30; Npts = 200;

  (* change here (4 lines) *)
  Print["基底関数と求積データを生成中..."];
  {basis, Tlist, Wlist} = ChebData[Ts, Te, NumBasis, Npts, x];

  Plist = Table[38/10 + i/1000, {i, 0, 9}];
  Qlist = {1679/25000, 13097/200000, 16183/250000, 12663/200000, 30907/500000, 
           60477/1000000, 59611/1000000, 58257/1000000, 719/12500, 55971/1000000};
           
  M = NumBasis;

  Print["1. 微分作用素を基底関数に適用中 (Symbolic Differentiation)..."];
  opApplied = Table[ApplyOp[Op, basis[[k]], x, dx], {k, 1, M}];

  Print["2. 評価行列の構築中 (High-precision Evaluation)..."];
  (* 🌟 N[expr, 150] を用いて、置換ルール (/. x -> ...) で150桁の厳密評価を行う *)
  LekTj = Table[
    N[opApplied[[k]] /. x -> SetPrecision[Tlist[[j]], prec], prec],
    {k, 1, M}, {j, 1, Npts}
  ];

  EkPi = Table[
    N[basis[[k]] /. x -> SetPrecision[Plist[[i]], prec], prec],
    {k, 1, M}, {i, 1, Length[Plist]}
  ];

  Wj = N[Wlist, prec];
  Qi = N[Qlist, prec];

  Print["3. 最適化問題の構築と求解 (Linear Least Squares)..."];
  alpha = 1; betaW = 10^-13; gammaW = 10^-20;

  maxL = Max[Abs[LekTj]];
  If[maxL > 0, LekTj = LekTj / maxL];

  Aode = Table[Sqrt[alpha * Wj[[j]]] * LekTj[[k, j]], {j, 1, Npts}, {k, 1, M}];
  Aconst = Table[Sqrt[betaW] * EkPi[[k, i]], {i, 1, Length[Plist]}, {k, 1, M}];
  Areg = DiagonalMatrix[ConstantArray[Sqrt[gammaW], M]];

  (* 行列の結合 (Join) *)
  A = Join[Aode, Aconst, Areg];
  b = Join[ConstantArray[0, Npts], Sqrt[betaW] * Qi, ConstantArray[0, M]];

  (* 列スケーリング *)
  colScales = Max[Abs[#]] & /@ Transpose[A];
  colScales = Replace[colScales, x_ /; x == 0 -> 1, {1}];
  
  Ascaled = Transpose[Transpose[A] / colScales];

  (* 最小二乗法の実行 *)
  fkScaled = LeastSquares[Ascaled, b];
  fkOpt = fkScaled / colScales;

  cost = Norm[Ascaled . fkScaled - b]^2;
  Print["\n最終コスト: ", cost];

  fkOpt=Rationalize[fkOpt,0];  (* 有理数に変換しないとPlotで precision error *)
  Print["4. 結果のプロット..."];
  fValFunc = Sum[fkOpt[[k]] * basis[[k]], {k, 1, M}];
  SieF=fValFunc; (* 結果関数を大域変数で保存 *)

  (* change here (plot ranges) *)
  g=Show[
    Plot[fValFunc, {x, 38/10, 4}, 
         WorkingPrecision -> prec,  (* 🌟 ここを追加！描画時の桁落ちを防ぐ *)
         PlotStyle -> {Green, Thickness[0.005]},
         PlotRange -> {{3.8, 4.0}, {0, 0.07}},
         PlotLegends -> {"Approximated E[chi(Mt)]"},
         Frame -> True, GridLines -> Automatic, 
         FrameLabel -> {"t", "E[chi(Mt)]"},
         PlotLabel -> "SIE Method B: Target B (Mathematica Standalone)"],
    ListPlot[Transpose[{Plist, Qlist}], 
             PlotStyle -> {Red, PointSize[0.02]},
             PlotMarkers -> {"\[Cross]", 15}, 
             PlotLegends -> {"Simulation Points"}]
  ];
  Return[g]
];

(* 実行, ; をつけない.*)
SieG=RunTargetB[]
