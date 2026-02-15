"""
Capítulo 5.2 – Resultados del VaR (toolbox para la tesis)
----------------------------------------------------------
Este módulo complementa `capitulo5_datos_descriptiva.py` y provee funciones para:

1) **Estimaciones de VaR** bajo distintos métodos:
   - Histórico (rolling quantile)
   - Paramétrico Normal (rolling μ, σ)
   - Paramétrico t-Student (rolling fit)
   - EWMA-RiskMetrics (σ_t con lambda, Normal)
2) **Backtesting y excepciones**:
   - Conteo de excepciones
   - Test de Kupiec (POF)
   - Test de Christoffersen (Independencia) y Cobertura Condicional (CC)
   - Señal de “semaforización” tipo **Basel traffic-light** (generalizada vía Binomial)
3) **Comparación emergentes vs desarrollados** mediante resúmenes por grupo.

Supuestos y convenciones:
- Trabajamos con **rendimientos diarios logarítmicos**.
- **Signo** del VaR: lo reportamos como **número positivo** (pérdida umbral). Una **excepción** ocurre cuando `r_t < -VaR_t`.
- Ventana rolling por defecto: 500 días. Ajustable.

Uso típico (en Jupyter):

    from capitulo5_datos_descriptiva import run_capitulo5_data
    prices, returns, tabla = run_capitulo5_data()

    from capitulo5_var_backtesting import run_var_pipeline
    out = run_var_pipeline(returns, alphas=[0.95, 0.99], window=500,
                           methods=("hist","norm","t","ewma"),
                           ewma_lambda=0.94, t_step=5)

    # Tablas finales por método/alpha:
    out["backtests"]["norm"][0.99]   # DataFrame con resultados por activo
    out["groups"]["hist"][0.99]      # Resumen por grupo (EM vs DM)

Salida en disco:
- Carpeta ./output/var/ con CSVs de VaR por activo/método/alpha y tablas de backtesting.

Autor: Finanzas investigacion VaR y CVaR
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t, chi2, binom

# -------------------------------------------------------------
# Utilidades
# -------------------------------------------------------------

def ensure_outdir(path: str = "output/var") -> str:
    os.makedirs(path, exist_ok=True)
    return path

# Grupos sugeridos para tu tesis
GROUP_MAP = {
    "S&P 500": "Desarrollado",
    "Euro Stoxx 50": "Desarrollado",
    "FTSE 100": "Desarrollado",
    "MSCI Emerging Markets": "Emergente",
    "Merval (Argentina)": "Emergente",
    "Bovespa (Brasil)": "Emergente",
}

# -------------------------------------------------------------
# 1) Estimación de VaR
# -------------------------------------------------------------

def var_hist(returns: pd.DataFrame, alpha: float = 0.99, window: int = 500) -> pd.DataFrame:
    """VaR histórico rolling. Devuelve **positivo** (umbral de pérdida)."""
    q = returns.rolling(window, min_periods=window).quantile(1 - alpha)
    var = -q
    var.columns = returns.columns
    return var


def var_norm(returns: pd.DataFrame, alpha: float = 0.99, window: int = 500) -> pd.DataFrame:
    """VaR Normal rolling. c = μ + σ * z_{1-α} (cuantil de cola izquierda); VaR = -c > 0."""
    mu = returns.rolling(window, min_periods=window).mean()
    sd = returns.rolling(window, min_periods=window).std(ddof=1)
    z = norm.ppf(1 - alpha)
    c = mu + sd * z
    var = -c
    var.columns = returns.columns
    return var


def _rolling_t_params(s: pd.Series, window: int = 500, step: int = 5) -> pd.DataFrame:
    """Ajusta t-Student rolling por ventanas con salto `step` (para performance).
    Devuelve DataFrame con columnas [df, loc, scale]."""
    idx = s.index
    df_list, loc_list, scale_list = [np.nan] * len(idx), [np.nan] * len(idx), [np.nan] * len(idx)
    i = window - 1
    while i < len(s):
        x = s.iloc[i - window + 1 : i + 1].dropna()
        if len(x) >= window * 0.9:  # tolera hasta 10% NaNs
            try:
                df_, loc_, scale_ = student_t.fit(x.values)
                # Limitar df por estabilidad numérica
                df_ = np.clip(df_, 3.01, 200.0)
            except Exception:
                df_, loc_, scale_ = np.nan, np.nan, np.nan
        else:
            df_, loc_, scale_ = np.nan, np.nan, np.nan
        df_list[i] = df_; loc_list[i] = loc_; scale_list[i] = scale_
        i += step
    # Forward-fill parámetros entre estimaciones y back-fill inicial
    params = pd.DataFrame({"df": df_list, "loc": loc_list, "scale": scale_list}, index=idx)
    params = params.ffill().bfill()
    return params


def var_t(returns: pd.DataFrame, alpha: float = 0.99, window: int = 500, step: int = 5) -> pd.DataFrame:
    """VaR rolling con t-Student (parámetros rolling por ventana con salto `step`)."""
    vars_ = {}
    for col in returns.columns:
        params = _rolling_t_params(returns[col], window=window, step=step)
        q = student_t.ppf(1 - alpha, df=params["df"]) * params["scale"] + params["loc"]
        vars_[col] = -q
    var = pd.DataFrame(vars_, index=returns.index)
    var.columns = returns.columns
    return var


def var_ewma(returns: pd.DataFrame, alpha: float = 0.99, lam: float = 0.94) -> pd.DataFrame:
    """VaR con volatilidad EWMA (RiskMetrics). Asume media 0; c = σ_t * z_{1-α}; VaR = -c > 0."""
    z = norm.ppf(1 - alpha)
    sigmas = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        r = returns[col].fillna(0.0)
        s2 = np.zeros(len(r))
        # inicializar con var muestral de primeros 60 días
        init = r[:60].var(ddof=1) if r[:60].shape[0] >= 2 else r.var(ddof=1)
        if not np.isfinite(init) or init <= 0:
            init = 1e-6
        s2[0] = init
        for i in range(1, len(r)):
            s2[i] = lam * s2[i-1] + (1 - lam) * (r.iloc[i-1] ** 2)
        sigmas[col] = np.sqrt(s2)
    c = sigmas * z
    var = -c
    var.columns = returns.columns
    return var


# Wrapper para computar varias combinaciones

def compute_vars_all(
    returns: pd.DataFrame,
    methods: Iterable[str] = ("hist", "norm", "t", "ewma"),
    alphas: Iterable[float] = (0.95, 0.99),
    window: int = 500,
    ewma_lambda: float = 0.94,
    t_step: int = 5,
) -> Dict[str, Dict[float, pd.DataFrame]]:
    out: Dict[str, Dict[float, pd.DataFrame]] = {}
    for m in methods:
        out[m] = {}
        for a in alphas:
            if m == "hist":
                out[m][a] = var_hist(returns, alpha=a, window=window)
            elif m == "norm":
                out[m][a] = var_norm(returns, alpha=a, window=window)
            elif m == "t":
                out[m][a] = var_t(returns, alpha=a, window=window, step=t_step)
            elif m == "ewma":
                out[m][a] = var_ewma(returns, alpha=a, lam=ewma_lambda)
            else:
                raise ValueError(f"Método no soportado: {m}")
    return out

# -------------------------------------------------------------
# 2) Backtesting
# -------------------------------------------------------------

def backtest_var(returns: pd.Series, var_pos: pd.Series, alpha: float) -> Dict[str, float]:
    """
    Backtesting robusto para una serie. VaR es positivo (umbral de pérdida).
    Evita divisiones por cero en POF/Christoffersen usando clamps numéricos.
    """
    EPS = 1e-12
    clamp01 = lambda p: float(np.clip(p, EPS, 1 - EPS))
    safe_log = lambda x: np.log(np.maximum(x, EPS))

    df = pd.concat({"r": returns, "VaR": var_pos}, axis=1).dropna()
    if df.empty:
        return {"N": 0, "excepciones": np.nan, "rate": np.nan, "esperadas": 0.0,
                "kupiec_p": np.nan, "christ_p": np.nan, "cc_p": np.nan, "zona": "NA"}

    exc = (df["r"] < -df["VaR"]).astype(int)
    n = int(exc.shape[0])
    x = int(exc.sum())
    rate = (x / n) if n else np.nan
    p_tail = 1 - alpha

    # --- Kupiec POF ---
    if n == 0:
        kupiec_p = np.nan
        LR_pof_val = 0.0
    else:
        # pi_hat efectivo (evita 0 y 1 exactos)
        if x == 0:
            pi_eff = clamp01(0.0)
        elif x == n:
            pi_eff = clamp01(1.0)
        else:
            pi_eff = clamp01(rate)
        L0 = (p_tail ** x) * ((1 - p_tail) ** (n - x))
        L1 = (pi_eff ** x) * ((1 - pi_eff) ** (n - x))
        LR_pof_val = -2 * (safe_log(L0) - safe_log(L1))
        kupiec_p = 1 - chi2.cdf(LR_pof_val, df=1)

    # --- Christoffersen Independencia ---
    trans = pd.DataFrame({"e": exc}).assign(e1=lambda d: d["e"].shift(1)).dropna().astype(int)
    if trans.empty:
        christ_p = np.nan
        LR_ind = 0.0
    else:
        n00 = int(((trans.e1 == 0) & (trans.e == 0)).sum())
        n01 = int(((trans.e1 == 0) & (trans.e == 1)).sum())
        n10 = int(((trans.e1 == 1) & (trans.e == 0)).sum())
        n11 = int(((trans.e1 == 1) & (trans.e == 1)).sum())
        tot = n00 + n01 + n10 + n11
        n0_ = n00 + n01
        n1_ = n10 + n11
        if tot == 0:
            christ_p, LR_ind = np.nan, 0.0
        else:
            pi0 = clamp01(n01 / n0_) if n0_ > 0 else clamp01(0.0)
            pi1 = clamp01(n11 / n1_) if n1_ > 0 else clamp01(0.0)
            pi_  = clamp01((n01 + n11) / tot)
            L0 = ((1 - pi_) ** (n00 + n10)) * (pi_ ** (n01 + n11))
            L1 = ((1 - pi0) ** n00) * (pi0 ** n01) * ((1 - pi1) ** n10) * (pi1 ** n11)
            LR_ind = -2 * (safe_log(L0) - safe_log(L1))
            christ_p = 1 - chi2.cdf(LR_ind, df=1)

    # --- Cobertura Condicional (POF + IND) ---
    if np.isnan(kupiec_p) or np.isnan(christ_p):
        cc_p = np.nan
    else:
        LR_cc_val = LR_pof_val + LR_ind
        cc_p = 1 - chi2.cdf(LR_cc_val, df=2)

    # --- Semáforo Basel (binomial) ---
    from scipy.stats import binom
    green_max  = int(binom.ppf(0.95 , n, p_tail)) if n > 0 else 0
    yellow_max = int(binom.ppf(0.999, n, p_tail)) if n > 0 else 0
    zona = "Green" if x <= green_max else ("Yellow" if x <= yellow_max else "Red")

    return {
        "N": n,
        "excepciones": x,
        "rate": rate,
        "esperadas": n * p_tail,
        "kupiec_p": kupiec_p,
        "christ_p": christ_p,
        "cc_p": cc_p,
        "zona": zona,
    }


def backtest_table(returns: pd.DataFrame, var_df: pd.DataFrame, alpha: float) -> pd.DataFrame:
    rows = []
    for col in returns.columns:
        res = backtest_var(returns[col], var_df[col], alpha)
        res["Índice"] = col
        res["alpha"] = alpha
        rows.append(res)
    out = pd.DataFrame(rows).set_index("Índice").sort_values("rate", ascending=False)
    return out

# -------------------------------------------------------------
# 3) Comparación EM vs DM
# -------------------------------------------------------------

def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Grupo"] = [GROUP_MAP.get(c, "Otro") for c in df.index]
    return df


def group_summary(backtest_df: pd.DataFrame) -> pd.DataFrame:
    g = add_groups(backtest_df)
    agg = g.groupby("Grupo").agg(
        activos=("N", "count"),
        n_total=("N", "sum"),
        exc_total=("excepciones", "sum"),
        rate_media=("rate", "mean"),
        kupiec_p_med=("kupiec_p", "mean"),
        christ_p_med=("christ_p", "mean"),
        cc_p_med=("cc_p", "mean"),
        green_share=("zona", lambda s: (s == "Green").mean()),
        yellow_share=("zona", lambda s: (s == "Yellow").mean()),
        red_share=("zona", lambda s: (s == "Red").mean()),
    )
    return agg

# -------------------------------------------------------------
# 4) Pipeline de VaR + Backtesting + Resumen por grupo
# -------------------------------------------------------------

def run_var_pipeline(
    returns: pd.DataFrame,
    methods: Iterable[str] = ("hist", "norm", "t", "ewma"),
    alphas: Iterable[float] = (0.95, 0.99),
    window: int = 500,
    ewma_lambda: float = 0.94,
    t_step: int = 5,
    outdir: str = "output/var",
):
    outdir = ensure_outdir(outdir)

    # 1) VaR
    vars_all = compute_vars_all(returns, methods=methods, alphas=alphas,
                                window=window, ewma_lambda=ewma_lambda, t_step=t_step)

    # 2) Guardar VaR por método/alpha
    for m in vars_all:
        for a in vars_all[m]:
            f = os.path.join(outdir, f"VaR_{m}_{a:.2f}.csv")
            vars_all[m][a].to_csv(f, index_label="Date")

    # 3) Backtesting por método/alpha
    backtests: Dict[str, Dict[float, pd.DataFrame]] = {}
    for m in vars_all:
        backtests[m] = {}
        for a in vars_all[m]:
            bt = backtest_table(returns, vars_all[m][a], a)
            backtests[m][a] = bt
            bt.to_csv(os.path.join(outdir, f"backtest_{m}_{a:.2f}.csv"))

    # 4) Resumen grupos EM vs DM
    groups: Dict[str, Dict[float, pd.DataFrame]] = {}
    for m in backtests:
        groups[m] = {}
        for a in backtests[m]:
            groups[m][a] = group_summary(backtests[m][a])
            groups[m][a].to_csv(os.path.join(outdir, f"groups_{m}_{a:.2f}.csv"))

    return {
        "vars": vars_all,        # dict[método][alpha] -> DataFrame de VaR_t
        "backtests": backtests,  # dict[método][alpha] -> DataFrame resultados por índice
        "groups": groups,        # dict[método][alpha] -> resumen por grupo
    }
