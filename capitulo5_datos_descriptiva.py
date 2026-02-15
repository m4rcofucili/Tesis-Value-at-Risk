"""
Capítulo 5 – Descripción de los datos y estadística descriptiva (script reproducible)
-----------------------------------------------------------------------------------
Este script descarga precios diarios desde **APIs públicas** (prioridad: Yahoo Finance vía `yfinance`; 
respaldo: Stooq vía `pandas_datareader`) para los índices/ETFs definidos, calcula **rendimientos logarítmicos**,
construye **estadística descriptiva** (media, desvío, asimetría, curtosis, Jarque–Bera, Shapiro–Wilk),
y genera **gráficos** (series, histogramas con densidad y normal teórica, QQ-plots).

> Nota importante sobre fuentes: algunas series de índices puros no están disponibles libremente; 
> en esos casos uso **ETFs como proxies** (p.ej., EEM para MSCI EM). Podés ajustar los mapeos abajo.

Requisitos (instalación):
    pip install yfinance pandas numpy scipy matplotlib seaborn pandas_datareader statsmodels

Cómo ejecutar (por defecto 2008-01-01 a 2023-12-31):
    python capitulo5_datos_descriptiva.py

Con argumentos:
    python capitulo5_datos_descriptiva.py --start 2008-01-01 --end 2023-12-31 \
        --indices "SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA"

Salida:
    - Carpeta ./output/ con
        * prices.csv, returns.csv
        * tabla_descriptiva.csv (para la Tabla 5.1)
        * Plots por activo: *_precios.png, *_retornos.png, *_hist.png, *_qq.png

Autor: Finanzas investigacion VaR y CVaR
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import jarque_bera, shapiro, skew, kurtosis, norm
import statsmodels.api as sm

# Data readers
import yfinance as yf
from pandas_datareader import data as pdr

sns.set(style="whitegrid", context="talk")
plt.rcParams.update({"figure.figsize": (11, 6), "axes.titlesize": 16, "axes.labelsize": 13})

@dataclass
class AssetConfig:
    nombre: str
    candidatos: List[str]  # lista de tickers candidatos a probar en orden
    fuente: str = "auto"   # "yahoo", "stooq" o "auto"

# ---------------------------------------------------------------------------
# 1) Define universo: índices y proxies
# ---------------------------------------------------------------------------
UNIVERSO: Dict[str, AssetConfig] = {
    # Desarrollados (índices directos en Yahoo; si fallan, proxies)
    "SP500": AssetConfig("S&P 500", ["^GSPC", "SPY"]),
    "EUROSTOXX50": AssetConfig("Euro Stoxx 50", ["^STOXX50E", "FEZ"]),
    "FTSE100": AssetConfig("FTSE 100", ["^FTSE", "UKX.L", "VUKE.L", "ISF.L", "EWU", "^UKX", "FTSE", "UKX"]),
    # Emergentes (usamos ETFs si los índices no están disponibles libremente)
    "MSCIEM": AssetConfig("MSCI Emerging Markets", ["^MSCIEF", "EEM"]),
    "MERVAL": AssetConfig("Merval (Argentina)", ["^MERV", "IMV.BA", "ARGT"]),
    "BOVESPA": AssetConfig("Bovespa (Brasil)", ["^BVSP", "IBOV.SA", "EWZ"]),
}

# ---------------------------------------------------------------------------
# 2) Descarga de datos
# ---------------------------------------------------------------------------

def try_download_yahoo(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            s = df["Close"].rename(ticker).dropna()
            return s
    except Exception:
        pass
    return None


def try_download_stooq(ticker: str, start: str, end: str) -> Optional[pd.Series]:
    try:
        df = pdr.DataReader(ticker, "stooq", start=start, end=end)
        # Stooq devuelve reverse-chronological; ordenamos
        if isinstance(df, pd.DataFrame) and not df.empty and "Close" in df.columns:
            df = df.sort_index()
            s = df["Close"].rename(ticker).dropna()
            return s
    except Exception:
        pass
    return None


def descargar_activo(cfg: AssetConfig, start: str, end: str) -> pd.Series:
    """Itera sobre candidatos y fuentes para devolver una serie de precios de cierre."""
    errores = []
    for tk in cfg.candidatos:
        # Prioridad Yahoo
        if cfg.fuente in ("auto", "yahoo"):
            s = try_download_yahoo(tk, start, end)
            if s is not None:
                s.name = cfg.nombre
                return s
            else:
                errores.append(f"Yahoo falló para {tk}")
        # Respaldo Stooq (algunos tickers difieren; probamos tal cual)
        if cfg.fuente in ("auto", "stooq"):
            s = try_download_stooq(tk, start, end)
            if s is not None:
                s.name = cfg.nombre
                return s
            else:
                errores.append(f"Stooq falló para {tk}")
    raise RuntimeError(f"No se pudo descargar {cfg.nombre}. Intentos: {', '.join(errores)}")


def descargar_universo(indices: List[str], start: str, end: str) -> pd.DataFrame:
    precios = []
    for key in indices:
        if key not in UNIVERSO:
            raise KeyError(f"Índice desconocido: {key}. Opciones: {list(UNIVERSO.keys())}")
        cfg = UNIVERSO[key]
        print(f"Descargando {cfg.nombre} ...")
        try:
            s = descargar_activo(cfg, start, end)
            precios.append(s)
        except Exception as e:
            print(f"[Aviso] Se omitió {cfg.nombre} por error: {e}")
            continue
    if not precios:
        raise RuntimeError("No se pudo descargar ningún activo; revise conexión, tickers o conectividad.")
    df = pd.concat(precios, axis=1)
    # Alinear calendarios: intersección de fechas con forward-fill mínimo para pequeños desfasajes
    df = df.dropna(how="all").sort_index()
    return df

# ---------------------------------------------------------------------------
# 3) Transformaciones y estadísticas
# ---------------------------------------------------------------------------

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna(how="all")


def tabla_descriptiva(returns: pd.DataFrame) -> pd.DataFrame:
    filas = []
    for col in returns.columns:
        x = returns[col].dropna()
        if x.empty:
            continue
        mu = x.mean() * 100  # % diario
        sigma = x.std(ddof=1) * 100  # % diario
        sk = skew(x, bias=False)
        kt = kurtosis(x, fisher=True, bias=False) + 3  # curtosis total (no-exceso)
        jb_stat, jb_p = jarque_bera(x)
        sh_stat, sh_p = shapiro(x.sample(n=min(5000, len(x)), random_state=123))  # límite de Shapiro
        filas.append({
            "Índice": col,
            "Media (%)": mu,
            "Desv. Est. (%)": sigma,
            "Asimetría": sk,
            "Curtosis": kt,
            "JB p-value": jb_p,
            "SW p-value": sh_p,
            "n": len(x)
        })
    out = pd.DataFrame(filas).set_index("Índice")
    # Orden sugerido: por volatilidad descendente
    out = out.sort_values(by="Desv. Est. (%)", ascending=False)
    return out

# ---------------------------------------------------------------------------
# 4) Gráficos
# ---------------------------------------------------------------------------

def ensure_outdir(path: str = "output"):
    os.makedirs(path, exist_ok=True)
    return path


def plot_series(prices: pd.DataFrame, outdir: str):
    for col in prices.columns:
        ax = prices[col].plot(color="#2a9d8f", lw=1.5)
        ax.set_title(f"Serie de precios – {col}")
        ax.set_xlabel("Fecha"); ax.set_ylabel("Precio (escala original)")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{col}_precios.png")); plt.clf()


def plot_returns(returns: pd.DataFrame, outdir: str):
    for col in returns.columns:
        ax = returns[col].plot(color="#264653", lw=1)
        ax.set_title(f"Rendimientos diarios (log) – {col}")
        ax.set_xlabel("Fecha"); ax.set_ylabel("r_t")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{col}_retornos.png")); plt.clf()


def plot_hist_density(returns: pd.DataFrame, outdir: str):
    for col in returns.columns:
        x = returns[col].dropna()
        if x.empty:
            continue
        # Hist + KDE
        sns.histplot(x, bins=80, stat="density", color="#457b9d", alpha=0.35)
        sns.kdeplot(x, color="#1d3557", lw=2, label="Densidad empírica")
        # Normal teórica con misma media y desvío
        mu, sd = x.mean(), x.std(ddof=1)
        xs = np.linspace(x.quantile(0.001), x.quantile(0.999), 400)
        plt.plot(xs, norm.pdf(xs, mu, sd), "r--", lw=2, label="Normal(μ,σ)")
        plt.title(f"Histograma y densidad – {col}")
        plt.xlabel("r_t"); plt.ylabel("Densidad")
        plt.legend()
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{col}_hist.png")); plt.clf()


def plot_qq(returns: pd.DataFrame, outdir: str):
    for col in returns.columns:
        x = returns[col].dropna()
        if x.empty:
            continue
        sm.qqplot(x, line="s")
        plt.title(f"QQ-plot vs Normal – {col}")
        plt.tight_layout(); plt.savefig(os.path.join(outdir, f"{col}_qq.png")); plt.clf()

# ---------------------------------------------------------------------------
# 4.1) Helper para uso en Notebooks / Jupyter (sin CLI)
# ---------------------------------------------------------------------------

def run_capitulo5(start: str = "2008-01-01", end: str = "2023-12-31",
                  indices: str = "SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA") -> pd.DataFrame:
    """Ejecuta todo el pipeline y devuelve la tabla descriptiva lista para reporte.
    Pensado para Jupyter: ignora argumentos de línea y usa parámetros explícitos.
    """
    indices_list = [s.strip().upper() for s in indices.split(",") if s.strip()]
    outdir = ensure_outdir("output")

    prices = descargar_universo(indices_list, start, end)
    prices.to_csv(os.path.join(outdir, "prices.csv"), index_label="Date")

    rets = log_returns(prices)
    rets.to_csv(os.path.join(outdir, "returns.csv"), index_label="Date")

    tabla = tabla_descriptiva(rets)
    tabla_rep = tabla.copy()
    tabla_rep[["Media (%)", "Desv. Est. (%)"]] = tabla_rep[["Media (%)", "Desv. Est. (%)"]].round(2)
    tabla_rep[["Asimetría", "Curtosis"]] = tabla_rep[["Asimetría", "Curtosis"]].round(2)
    tabla_rep[["JB p-value", "SW p-value"]] = tabla_rep[["JB p-value", "SW p-value"]].applymap(lambda v: f"{v:.4f}")
    tabla_rep.to_csv(os.path.join(outdir, "tabla_descriptiva.csv"))

    plot_series(prices, outdir)
    plot_returns(rets, outdir)
    plot_hist_density(rets, outdir)
    plot_qq(rets, outdir)

    print("=== Tabla 5.1 – Estadística descriptiva (ordenada por volatilidad) ===")
    print(tabla_rep)
    print("Archivos guardados en:", os.path.abspath(outdir))

    return tabla_rep

# ---------------------------------------------------------------------------
# 4.2) Helper que devuelve DataFrames (para seguir trabajando en memoria)
# ---------------------------------------------------------------------------

def run_capitulo5_data(
    start: str = "2008-01-01",
    end: str = "2023-12-31",
    indices: str = "SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA",
    strict: bool = False,
):
    """Ejecuta el pipeline y devuelve (prices, returns, tabla_rep) como DataFrames.
    Guarda además los CSV/PNGs en ./output/ para trazabilidad.
    """
    import os
    import pandas as pd

    indices_list = [s.strip().upper() for s in indices.split(",") if s.strip()]
    outdir = ensure_outdir("output")

    # Llamada compatible con versiones de descargar_universo con/sin `strict`
    try:
        prices = descargar_universo(indices_list, start, end, strict=strict)
    except TypeError as e:
        if "unexpected keyword argument 'strict'" in str(e):
            prices = descargar_universo(indices_list, start, end)  # reintenta sin `strict`
        else:
            raise

    # Calcular retornos y tabla descriptiva
    rets = log_returns(prices)
    tabla = tabla_descriptiva(rets)
    tabla_rep = tabla.copy()

    # --- Limpieza y redondeo robusto ---
    pct_cols = ["Media (%)", "Desv. Est. (%)"]
    stat_cols = ["Asimetría", "Curtosis"]
    pval_cols = ["JB p-value", "SW p-value"]

    # Convierte a numérico silenciosamente (por si vino texto)
    for cols in (pct_cols, stat_cols, pval_cols):
        keep = [c for c in cols if c in tabla_rep.columns]
        if keep:
            tabla_rep[keep] = tabla_rep[keep].apply(
                lambda s: pd.to_numeric(s, errors="coerce")
            )

    # Redondeos finales
    cols = [c for c in pct_cols if c in tabla_rep.columns]
    if cols:
        tabla_rep[cols] = tabla_rep[cols].round(2)

    cols = [c for c in stat_cols if c in tabla_rep.columns]
    if cols:
        tabla_rep[cols] = tabla_rep[cols].round(2)

    cols = [c for c in pval_cols if c in tabla_rep.columns]
    if cols:
        tabla_rep[cols] = tabla_rep[cols].round(4)

    # --- Persistencia a disco ---
    prices.to_csv(os.path.join(outdir, "prices.csv"), index_label="Date")
    rets.to_csv(os.path.join(outdir, "returns.csv"), index_label="Date")
    tabla_rep.to_csv(
        os.path.join(outdir, "tabla_descriptiva.csv"),
        float_format="%.4f",  # formato para floats
    )

    # --- Gráficos ---
    plot_series(prices, outdir)
    plot_returns(rets, outdir)
    plot_hist_density(rets, outdir)
    plot_qq(rets, outdir)

    return prices, rets, tabla_rep


# ---------------------------------------------------------------------------
# 5) Main
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Capítulo 5 – Descarga y estadística descriptiva")
    parser.add_argument("--start", type=str, default="2008-01-01", help="Fecha inicio YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2023-12-31", help="Fecha fin YYYY-MM-DD")
    parser.add_argument(
        "--indices",
        type=str,
        default="SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA",
        help=f"Lista separada por comas. Opciones: {','.join(UNIVERSO.keys())}",
    )
    # Jupyter/IPython agrega argumentos como --f=...; los ignoramos de forma segura
    if argv is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(argv)
    if unknown:
        print("[Aviso] Argumentos no reconocidos ignorados:", unknown)
    return args

def main():
    args = parse_args()
    indices = [s.strip().upper() for s in args.indices.split(",") if s.strip()]
    outdir = ensure_outdir("output")

    # 1) Descarga
    prices = descargar_universo(indices, args.start, args.end)
    prices.to_csv(os.path.join(outdir, "prices.csv"), index_label="Date")

    # 2) Rendimientos logarítmicos
    rets = log_returns(prices)
    rets.to_csv(os.path.join(outdir, "returns.csv"), index_label="Date")

    # 3) Tabla descriptiva
    tabla = tabla_descriptiva(rets)
    # Redondeo amigable para reporte
    tabla_rep = tabla.copy()
    tabla_rep[["Media (%)", "Desv. Est. (%)"]] = tabla_rep[["Media (%)", "Desv. Est. (%)"]].round(2)
    tabla_rep[["Asimetría", "Curtosis"]] = tabla_rep[["Asimetría", "Curtosis"]].round(2)
    tabla_rep[["JB p-value", "SW p-value"]] = tabla_rep[["JB p-value", "SW p-value"]].applymap(lambda v: f"{v:.4f}")
    tabla_rep.to_csv(os.path.join(outdir, "tabla_descriptiva.csv"))

    # 4) Gráficos
    plot_series(prices, outdir)
    plot_returns(rets, outdir)
    plot_hist_density(rets, outdir)
    plot_qq(rets, outdir)
    
    # 5) Preview por consola
    print("\n=== Tabla 5.1 – Estadística descriptiva (ordenada por volatilidad) ===")
    print(tabla_rep)
    print("\nArchivos guardados en:", os.path.abspath(outdir))


if __name__ == "__main__":
    main()
