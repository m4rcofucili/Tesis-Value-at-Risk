from __future__ import annotations
import argparse
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from capitulo5_var_backtesting import run_var_pipeline

# Intentamos importar la ruta de datos "caja de herramientas"
try:
    from capitulo5_datos_descriptiva import run_capitulo5_data
    HAVE_DATA_HELPER = True
except Exception:
    HAVE_DATA_HELPER = False

# -----------------------------
# Plots auxiliares
# -----------------------------

def ensure_outdir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(s: str) -> str:
    return (
        s.replace("/", "-").replace("\\", "-")
        .replace(" ", "_").replace("(", "").replace(")", "")
        .replace("%", "pct").replace("&", "y")
    )


def plot_var_overlay(returns: pd.DataFrame, var_df: pd.DataFrame, idx_name: str, title: str, outpath: str):
    # Evitar "columns overlap" renombrando explícitamente
    r_df = returns[[idx_name]].rename(columns={idx_name: "r"})
    v_df = var_df[[idx_name]].rename(columns={idx_name: "VaR"})  # VaR positivo (umbral)
    common = pd.concat([r_df, v_df], axis=1, join="inner").dropna()
    if common.empty:
        return

    r = common["r"]
    barrier = -common["VaR"]  # umbral de pérdida

    plt.figure(figsize=(12, 6))
    plt.plot(common.index, r, color="#264653", lw=0.8, label="r_t")
    plt.plot(common.index, barrier, color="#e76f51", lw=1.2, label="-VaR_t")
    exc = r < barrier
    if exc.any():
        plt.scatter(common.index[exc], r[exc], color="#e63946", s=10, zorder=3, label="Excepciones")
    plt.title(title)
    plt.xlabel("Fecha"); plt.ylabel("Rendimiento diario (log)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

# -----------------------------
# Core runner
# -----------------------------

def run_capitulo52(start: str = "2008-01-01", end: str = "2023-12-31",
                   indices: str = "SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA",
                   alphas: str = "0.95,0.99",
                   methods: str = "hist,norm,t,ewma",
                   window: int = 500,
                   ewma_lambda: float = 0.94,
                   t_step: int = 5,
                   outdir: str = "output/var"):
    # 1) Obtener datos curados (prices, returns)
    if HAVE_DATA_HELPER:
        from capitulo5_datos_descriptiva import run_capitulo5_data
        prices, returns, tabla = run_capitulo5_data(start=start, end=end, indices=indices)
    else:
        # Fallback: leer returns del CSV generado por tu script Cap.5 (ejecutalo antes)
        returns = pd.read_csv("./output/returns.csv", parse_dates=["Date"], index_col="Date")
        if returns.empty:
            raise RuntimeError("No se encontraron returns en ./output/returns.csv. Ejecuta el script de Cap.5 primero.")

    # 2) VaR + Backtesting + Resúmenes
    alphas_list = [float(a.strip()) for a in alphas.split(",") if a.strip()]
    methods_list = [m.strip() for m in methods.split(",") if m.strip()]

    out = run_var_pipeline(
        returns,
        methods=methods_list,
        alphas=alphas_list,
        window=window,
        ewma_lambda=ewma_lambda,
        t_step=t_step,
        outdir=outdir,
    )

    # 3) Plots de overlay r_t vs -VaR_t
    plots_dir = ensure_outdir(os.path.join(outdir, "plots"))
    for m in methods_list:
        for a in alphas_list:
            var_df = out["vars"][m][a]
            for idx_name in returns.columns:
                if idx_name in var_df.columns:
                    title = f"{idx_name} – {m.upper()} – VaR@{a:.2f}"
                    fname = f"overlay_{safe_name(idx_name)}_{m}_{a:.2f}.png"
                    outpath = os.path.join(plots_dir, fname)
                    plot_var_overlay(returns, var_df, idx_name, title, outpath)

    # 4) Mensaje final breve
    print("\nListo: resultados guardados en:", os.path.abspath(outdir))
    print(" - CSVs de VaR: VaR_<método>_<alpha>.csv")
    print(" - Backtesting: backtest_<método>_<alpha>.csv (excepciones, p-values, zona)")
    print(" - Grupos EM vs DM: groups_<método>_<alpha>.csv")
    print(" - Plots overlay: ./output/var/plots/*.png")

    return out

# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser(description="Runner Cap. 5.2 – VaR + Backtesting")
    p.add_argument("--start", type=str, default="2008-01-01")
    p.add_argument("--end", type=str, default="2023-12-31")
    p.add_argument("--indices", type=str, default="SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA")
    p.add_argument("--alphas", type=str, default="0.95,0.99")
    p.add_argument("--methods", type=str, default="hist,norm,t,ewma")
    p.add_argument("--window", type=int, default=500)
    p.add_argument("--ewma_lambda", type=float, default=0.94)
    p.add_argument("--t_step", type=int, default=5)
    if argv is None:
        args, unknown = p.parse_known_args()
    else:
        args, unknown = p.parse_known_args(argv)
    if unknown:
        print("[Aviso] Argumentos no reconocidos ignorados:", unknown)
    return args


def main():
    args = parse_args()
    run_capitulo52(
        start=args.start,
        end=args.end,
        indices=args.indices,
        alphas=args.alphas,
        methods=args.methods,
        window=args.window,
        ewma_lambda=args.ewma_lambda,
        t_step=args.t_step,
    )


if __name__ == "__main__":
    main()
