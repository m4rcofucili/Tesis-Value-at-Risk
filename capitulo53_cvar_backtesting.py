#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capítulo 5.3 – Resultados del CVaR (Expected Shortfall)
-------------------------------------------------------
Calcula CVaR (ES) para múltiples índices, niveles de confianza y métodos;
realiza un backtesting orientado a colas (tail coverage y tail loss ratio),
y guarda salidas en CSV y gráficos.

Métodos:
 - hist   : CVaR histórico (no paramétrico)
 - norm   : CVaR paramétrico Normal
 - t      : CVaR paramétrico t-Student
 - ewma   : CVaR paramétrico Normal con σ_t por EWMA (λ)

Salidas:
 - ./output/cvar/cvar_estimates.csv
 - ./output/cvar/cvar_backtesting.csv
 - ./output/cvar/plots/
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# MAPEO DE ÍNDICES: alias -> nombre "lindo" (como en capitulo5_datos)
# -------------------------------------------------------------
INDEX_NAME_MAP = {
    "SP500": "S&P 500",
    "EUROSTOXX50": "Euro Stoxx 50",
    "FTSE100": "FTSE 100",
    "MSCIEM": "MSCI Emerging Markets",
    "MERVAL": "Merval (Argentina)",
    "BOVESPA": "Bovespa (Brasil)",
}

# Tickers por defecto para yfinance si NO se usa returns_csv
YF_TICKER_MAP = {
    "SP500": "^GSPC",
    "EUROSTOXX50": "^STOXX50E",
    "FTSE100": "^FTSE",
    "MSCIEM": "EEM",
    "MERVAL": "^MERV",
    "BOVESPA": "^BVSP",
}


def _parse_list(x: str):
    return [s.strip() for s in x.split(",") if s.strip()]


def map_indices(indices):
    """Mapea alias como 'SP500' a los nombres usados en capitulo5_datos."""
    return [INDEX_NAME_MAP.get(i, i) for i in indices]


def _ensure_dirs():
    os.makedirs("./output/cvar/plots", exist_ok=True)


def load_returns(indices, start, end, returns_csv=None):
    """
    Carga retornos desde CSV o descarga con yfinance si no hay CSV.

    - Si returns_csv no es None:
        * intenta primero con los nombres mapeados (S&P 500, etc.)
        * si no encuentra, intenta con los alias (SP500, etc.)
    - Si returns_csv es None:
        * descarga con yfinance usando YF_TICKER_MAP
        * nombra las columnas con los nombres mapeados
    """
    idx_names = map_indices(indices)

    if returns_csv:
        df = pd.read_csv(returns_csv, parse_dates=[0])
        df = df.set_index(df.columns[0]).sort_index()

        # 1) Intento con nombres "lindos" (como en capitulo5_datos)
        keep = [c for c in idx_names if c in df.columns]

        # 2) Si no hay coincidencias, intento con alias crudos
        if not keep:
            keep = [c for c in indices if c in df.columns]

        if not keep:
            raise ValueError(
                "El CSV no contiene los índices solicitados.\n"
                f"Buscados (mapeados): {idx_names}\n"
                f"Buscados (alias):    {indices}\n"
                f"Columnas CSV:        {list(df.columns)}"
            )

        return df.loc[start:end, keep].dropna(how="all")

    # Si NO hay CSV, usamos yfinance con los alias -> tickers
    import yfinance as yf

    data = {}
    for alias in indices:
        ticker = YF_TICKER_MAP.get(alias, alias)
        s = yf.download(ticker, start=start, end=end)["Adj Close"]
        name = INDEX_NAME_MAP.get(alias, alias)
        data[name] = s

    px = pd.DataFrame(data).dropna(how="all")
    rets = np.log(px / px.shift(1))
    return rets.dropna()


# -------------------------------------------------------------
# FUNCIONES DE VAR/CVAR
# -------------------------------------------------------------
def var_cvar_hist(losses, alpha):
    var = np.quantile(losses, alpha)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail) else var
    return var, cvar


def var_cvar_norm(mu, sigma, alpha):
    z = norm.ppf(alpha)
    var = mu + sigma * z
    cvar = mu + sigma * (norm.pdf(z) / (1 - alpha))
    return var, cvar


def var_cvar_t(mu, sigma, nu, alpha):
    q = student_t.ppf(alpha, df=nu)
    f_q = student_t.pdf(q, df=nu)
    var = mu + sigma * q
    cvar = mu + sigma * (f_q * (nu + q**2)) / ((nu - 1) * (1 - alpha))
    return var, cvar


def ewma_sigma(losses, lam=0.94):
    s2 = np.zeros_like(losses)
    s2[0] = np.var(losses[:50]) if len(losses) >= 50 else np.var(losses)
    for t in range(1, len(losses)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * (losses[t - 1] ** 2)
    return np.sqrt(s2)


# -------------------------------------------------------------
# ESTIMACIÓN ROLLING
# -------------------------------------------------------------
def estimate_cvar(df, alphas, methods, window, t_step, ewma_lambda):
    rows = []
    df = df.sort_index()
    for idx in df.columns:
        L = -df[idx].dropna().values
        if len(L) < window:
            # Si querés debug: print(f"[Aviso] {idx}: len={len(L)} < window={window}, se omite.")
            continue
        ewma_all = ewma_sigma(L, ewma_lambda) if "ewma" in methods else None
        for i in range(window, len(L), t_step):
            losses = L[i - window:i]
            mu, sigma = losses.mean(), losses.std(ddof=1)
            nu = 8
            if "t" in methods:
                try:
                    g2 = pd.Series(losses).kurtosis(fisher=True)
                    if g2 and g2 > 0:
                        nu = max(5.1, 6 / g2 + 4)
                except Exception:
                    nu = 8
            date = df.index[i - 1]
            for a in alphas:
                if "hist" in methods:
                    v, c = var_cvar_hist(losses, a)
                    rows.append((date, idx, "hist", a, v, c))
                if "norm" in methods:
                    v, c = var_cvar_norm(mu, sigma, a)
                    rows.append((date, idx, "norm", a, v, c))
                if "t" in methods:
                    v, c = var_cvar_t(mu, sigma, nu, a)
                    rows.append((date, idx, "t", a, v, c))
                if "ewma" in methods and ewma_all is not None:
                    sig = ewma_all[i - 1]
                    v, c = var_cvar_norm(0, sig, a)
                    rows.append((date, idx, "ewma", a, v, c))
    out = pd.DataFrame(rows, columns=["date", "index", "method", "alpha", "var", "cvar"])
    return out.sort_values(["index", "date", "method", "alpha"]).reset_index(drop=True)


# -------------------------------------------------------------
# BACKTESTING CVAR
# -------------------------------------------------------------
def backtest_cvar(est, df):
    recs = []
    for (idx, met, a), g in est.groupby(["index", "method", "alpha"]):
        s = df[[idx]].rename(columns={idx: "ret"})
        s["loss"] = -s["ret"]
        j = s.join(g.set_index("date")[["var", "cvar"]], how="inner")
        if j.empty:
            continue
        j["breach"] = (j["loss"] >= j["var"]).astype(int)
        tail = j[j["breach"] == 1]
        n_total = len(j)
        n_tail = len(tail)
        tcov = n_tail / n_total if n_total else np.nan
        tail_mean_real = tail["loss"].mean() if n_tail else np.nan
        tlr = tail_mean_real / tail["cvar"].mean() if n_tail else np.nan
        recs.append({
            "index": idx,
            "method": met,
            "alpha": a,
            "n_total": n_total,
            "n_tail": n_tail,
            "tail_coverage": tcov,
            "expected_tail": 1 - a,
            "tail_mean_real": tail_mean_real,
            "tail_loss_ratio": tlr,
        })
    return pd.DataFrame(recs)


# -------------------------------------------------------------
# PLOTS
# -------------------------------------------------------------
def plot_overlays(est, df, outdir="./output/cvar/plots"):
    os.makedirs(outdir, exist_ok=True)
    for idx, gidx in est.groupby("index"):
        s = df[[idx]].rename(columns={idx: "ret"})
        plt.figure(figsize=(12, 4))
        plt.plot(s.index, s["ret"], color="gray", lw=0.8, label="Retorno diario")
        for (m, a), gm in gidx.groupby(["method", "alpha"]):
            gm = gm.set_index("date")
            plt.plot(gm.index, -gm["var"], lw=1.0, label=f"-VaR {m} α={a}")
            plt.plot(gm.index, -gm["cvar"], lw=1.0, ls=":", label=f"-CVaR {m} α={a}")
        plt.axhline(0, color="black", lw=0.8)
        plt.legend(fontsize=8, ncol=4)
        plt.title(f"Retornos vs VaR/CVaR – {idx}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"overlay_{idx}.png"), dpi=140)
        plt.close()
        
        
def plot_var_cvar_area(
    est: pd.DataFrame,
    df: pd.DataFrame,
    outdir: str = "./output/cvar/plots",
    methods: list[str] | None = None,
    alphas: list[float] | None = None,
):
    """
    Gráficos de retornos + VaR–CVaR con área sombreada entre ambas curvas.
    - est: DataFrame con columnas [date, index, method, alpha, var, cvar]
    - df:  DataFrame de retornos (índice fecha, columnas índices)
    """
    os.makedirs(outdir, exist_ok=True)

    # Si no se filtran métodos/niveles, usamos todos los disponibles
    if methods is None:
        methods = sorted(est["method"].unique().tolist())
    if alphas is None:
        alphas = sorted(est["alpha"].unique().tolist())

    for idx, gidx in est.groupby("index"):
        # Serie de retornos para este índice
        s = df[[idx]].rename(columns={idx: "ret"})

        for m in methods:
            for a in alphas:
                gm = gidx[(gidx["method"] == m) & (gidx["alpha"] == a)]
                if gm.empty:
                    continue

                gm = gm.sort_values("date").set_index("date")

                plt.figure(figsize=(12, 4))

                # Retornos diarios
                plt.plot(s.index, s["ret"], color="gray", lw=0.8, label="Retorno diario")

                # Curvas -VaR y -CVaR (en retornos)
                var_line = -gm["var"]
                cvar_line = -gm["cvar"]

                plt.plot(gm.index, var_line, lw=1.0, label=f"-VaR {m} α={a}")
                plt.plot(gm.index, cvar_line, lw=1.0, ls=":", label=f"-CVaR {m} α={a}")

                # Área entre -VaR y -CVaR (amplitud de pérdidas extremas)
                plt.fill_between(
                    gm.index,
                    var_line,
                    cvar_line,
                    alpha=0.20,
                    label="Área VaR–CVaR",
                )

                plt.axhline(0, color="black", lw=0.8)
                plt.title(f"Retornos vs VaR/CVaR (área) – {idx} – {m}, α={a}")
                plt.legend(fontsize=8, ncol=2)
                plt.tight_layout()

                fname = f"overlay_area_{idx}_{m}_a{int(round(a * 100))}.png"
                plt.savefig(os.path.join(outdir, fname), dpi=140)
                plt.close()

def plot_heatmap_var_cvar_gap(
    est: pd.DataFrame,
    outdir: str = "./output/cvar/plots",
    method: str = "t",
    alpha: float = 0.99,
    freq: str = "M",
    relative: bool = True,
):
    """
    Heatmap del diferencial VaR–CVaR a lo largo del tiempo.
    - Filtra por método y nivel de confianza.
    - Agrega por períodos (freq = 'M' mensual, 'Q' trimestral, etc.).
    - Mide el gap entre CVaR y VaR:
        * relativo:  cvar/var - 1
        * absoluto:  cvar - var
    """
    os.makedirs(outdir, exist_ok=True)

    if est.empty:
        print("[heatmap] 'est' está vacío, no se genera gráfico.")
        return

    est = est.copy()
    est["date"] = pd.to_datetime(est["date"])

    # Intento con method/alpha pedidos
    g = est[(est["method"] == method) & (est["alpha"] == alpha)]

    if g.empty:
        # DEBUG: mostrar combinaciones disponibles
        combos = est[["method", "alpha"]].drop_duplicates().sort_values(["method", "alpha"])
        print("[heatmap] No hay filas para method/alpha pedidos:"
              f" method={method}, alpha={alpha}")
        print("[heatmap] Combinaciones disponibles:")
        print(combos)

        # fallback: usar la primera combinación disponible
        if not combos.empty:
            method_fallback = combos.iloc[0]["method"]
            alpha_fallback = combos.iloc[0]["alpha"]
            print(f"[heatmap] Usando fallback method={method_fallback}, alpha={alpha_fallback}")
            g = est[(est["method"] == method_fallback) & (est["alpha"] == alpha_fallback)]
        else:
            return

    # Gap VaR–CVaR
    if relative:
        g["gap"] = g["cvar"] / g["var"] - 1.0
        gap_label = "Gap relativo CVaR/VaR - 1"
    else:
        g["gap"] = g["cvar"] - g["var"]
        gap_label = "Gap absoluto CVaR - VaR"

    g["period"] = g["date"].dt.to_period(freq)

    mat = (
        g.groupby(["index", "period"])["gap"]
         .mean()
         .unstack("period")
         .sort_index()
    )

    if mat.empty:
        print("[heatmap] Matriz vacía después de agrupar, no se genera gráfico.")
        return

    mat = mat.reindex(sorted(mat.columns), axis=1)

    plt.figure(figsize=(14, 4 + 0.3 * len(mat.index)))

    im = plt.imshow(
        mat.values,
        aspect="auto",
        origin="lower",
    )

    plt.yticks(
        ticks=np.arange(len(mat.index)),
        labels=mat.index,
    )

    periods = mat.columns.to_timestamp()
    xticks = np.arange(len(periods))

    if len(periods) > 12:
        step = max(1, len(periods) // 12)
    else:
        step = 1

    plt.xticks(
        ticks=xticks[::step],
        labels=[p.strftime("%Y-%m") for p in periods[::step]],
        rotation=45,
        ha="right",
    )

    plt.colorbar(im, label=gap_label)
    plt.title(f"Heatmap gap VaR–CVaR – método={method}, α={alpha}, freq={freq}")
    plt.tight_layout()

    fname = f"heatmap_gap_{method}_a{int(round(alpha * 100))}_{freq}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=140)
    plt.close()

    print(f"[heatmap] Guardado {os.path.join(outdir, fname)}")

def compute_rolling_var_cvar_corr(
    est: pd.DataFrame,
    method: str = "t",
    alpha: float = 0.99,
    window: int = 250,
) -> pd.DataFrame:
    """
    Calcula la correlación rolling entre VaR y CVaR para cada índice.
    Devuelve un DataFrame con índice fecha y columnas = índices.
    Cada valor es corr_t( VaR_t, CVaR_t ) en una ventana de longitud `window`.
    """
    if est.empty:
        print("[corr] 'est' está vacío, no se calcula correlación.")
        return pd.DataFrame()

    est = est.copy()
    est["date"] = pd.to_datetime(est["date"])

    # Filtrar por método y nivel de confianza
    g = est[(est["method"] == method) & (est["alpha"] == alpha)].copy()
    if g.empty:
        combos = est[["method", "alpha"]].drop_duplicates().sort_values(["method", "alpha"])
        print("[corr] No hay filas para method/alpha pedidos:"
              f" method={method}, alpha={alpha}")
        print("[corr] Combinaciones disponibles:")
        print(combos)
        return pd.DataFrame()

    # Estructura de salida: fechas comunes
    dates = sorted(g["date"].unique())
    corr_panel = pd.DataFrame(index=dates)

    for idx in sorted(g["index"].unique()):
        sub = (
            g[g["index"] == idx]
            .sort_values("date")
            .set_index("date")[["var", "cvar"]]
        )
        if sub.shape[0] < window:
            continue

        # Correlación rolling entre var y cvar
        roll_corr = sub["var"].rolling(window).corr(sub["cvar"])
        corr_panel[idx] = roll_corr

    corr_panel = corr_panel.dropna(how="all")
    return corr_panel
def plot_var_cvar_corr_heatmap(
    est: pd.DataFrame,
    outdir: str = "./output/cvar/plots",
    method: str = "t",
    alpha: float = 0.99,
    window: int = 250,
    freq: str = "M",
):
    """
    Heatmap de la correlación condicional VaR–CVaR.
    - Calcula corr rolling entre VaR y CVaR (ventana `window`).
    - Agrega por períodos (freq='M' mensual, 'Q' trimestral, etc.).
    - Muestra matriz: filas = índices, columnas = períodos, color = corr media.
    """
    os.makedirs(outdir, exist_ok=True)

    corr_panel = compute_rolling_var_cvar_corr(
        est=est,
        method=method,
        alpha=alpha,
        window=window,
    )
    if corr_panel.empty:
        print("[corr] Panel de correlaciones vacío, no se genera heatmap.")
        return

    # Pasar a periodo (mes, trimestre, etc.) y promediar
    df_corr = corr_panel.copy()
    df_corr.index = pd.to_datetime(df_corr.index)
    df_corr["period"] = df_corr.index.to_period(freq)

    mat = (
        df_corr.groupby("period")
        .mean()                 # promedio de corr en la ventana dentro del período
        .T                      # filas = índices, columnas = periodos
    )

    if mat.empty:
        print("[corr] Matriz vacía después de agrupar, no se genera heatmap.")
        return

    # Ordenar períodos en el eje X
    mat = mat.reindex(sorted(mat.columns), axis=1)

    plt.figure(figsize=(14, 4 + 0.3 * len(mat.index)))

    im = plt.imshow(
        mat.values,
        aspect="auto",
        origin="lower",
        vmin=-1.0,
        vmax=1.0,
    )

    # Eje Y: índices
    plt.yticks(
        ticks=np.arange(len(mat.index)),
        labels=mat.index,
    )

    # Eje X: períodos
    periods = mat.columns.to_timestamp()
    xticks = np.arange(len(periods))
    if len(periods) > 12:
        step = max(1, len(periods) // 12)
    else:
        step = 1

    plt.xticks(
        ticks=xticks[::step],
        labels=[p.strftime("%Y-%m") for p in periods[::step]],
        rotation=45,
        ha="right",
    )

    plt.colorbar(im, label="Corr(VaR, CVaR)")
    plt.title(
        f"Correlación condicional VaR–CVaR (rolling={window}) – método={method}, α={alpha}, freq={freq}"
    )
    plt.tight_layout()

    fname = f"heatmap_corr_var_cvar_{method}_a{int(round(alpha * 100))}_w{window}_{freq}.png"
    plt.savefig(os.path.join(outdir, fname), dpi=140)
    plt.close()

    print(f"[corr] Guardado {os.path.join(outdir, fname)}")


# -------------------------------------------------------------
# ORQUESTADOR PRINCIPAL
# -------------------------------------------------------------
def run_capitulo53(
    start: str,
    end: str,
    indices,
    alphas,
    methods,
    window=500,
    ewma_lambda=0.94,
    t_step=5,
    returns_csv=None,
    save_outputs=True,
):
    _ensure_dirs()
    df = load_returns(indices, start, end, returns_csv)
    est = estimate_cvar(df, alphas, methods, window, t_step, ewma_lambda)
    bt = backtest_cvar(est, df)

    if save_outputs:
        outdir = "./output/cvar"
        os.makedirs(outdir, exist_ok=True)
        est.to_csv(os.path.join(outdir, "cvar_estimates.csv"), index=False)
        bt.to_csv(os.path.join(outdir, "cvar_backtesting.csv"), index=False)

        plot_overlays(est, df, outdir=os.path.join(outdir, "plots"))
        plot_var_cvar_area(est, df, outdir=os.path.join(outdir, "plots"))

        # Heatmap gap VaR–CVaR
        plot_heatmap_var_cvar_gap(
            est,
            outdir=os.path.join(outdir, "plots"),
            method="t",
            alpha=0.99,
            freq="M",
            relative=True,
        )

        # NUEVO: Heatmap correlación condicional VaR–CVaR
        plot_var_cvar_corr_heatmap(
            est,
            outdir=os.path.join(outdir, "plots"),
            method="t",
            alpha=0.99,
            window=250,   # ~1 año de ruedas
            freq="M",
        )

    return {"estimates": est, "backtesting": bt}


