#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capítulo 5.3 – Integración (CVaR): Datos -> CVaR -> Backtesting -> EM vs DM
----------------------------------------------------------------------------

Orquesta el flujo del Capítulo 5.3 reutilizando el pipeline anterior:
 - Toma retornos diarios ya curados (opcional vía --returns_csv)
 - Llama a `run_capitulo53(...)` para estimar CVaR y realizar backtesting
 - Resume resultados por grupos (EM vs DM) y guarda los archivos de salida

Salidas:
 - ./output/cvar/cvar_estimates.csv
 - ./output/cvar/cvar_backtesting.csv
 - ./output/cvar/cvar_groups_summary.csv
 - ./output/cvar/plots/

Uso (ejemplo):
    python capitulo53_integracion.py --start 2008-01-01 --end 2023-12-31 \
        --indices "SP500,EUROSTOXX50,FTSE100,MSCIEM,MERVAL,BOVESPA" \
        --alphas "0.95,0.99" --methods "hist,norm,t,ewma" \
        --window 500 --ewma_lambda 0.94 --t_step 5 \
        --em "MSCIEM,MERVAL,BOVESPA" --dm "SP500,EUROSTOXX50,FTSE100" \
        --returns_csv "./output/var/returns_clean.csv"

Autor: Finanzas investigacion VaR y CVaR
"""
from __future__ import annotations
import os
import argparse
from typing import List, Dict
import pandas as pd

# Import estándar: debe existir capitulo53_cvar_backtesting.py en la misma carpeta
from capitulo53_cvar_backtesting import run_capitulo53, _parse_list, map_indices


# =====================================================================
# CONFIGURACIÓN POR DEFECTO (en alias)
# =====================================================================
DEFAULT_EM = ["MSCIEM", "MERVAL", "BOVESPA"]
DEFAULT_DM = ["SP500", "EUROSTOXX50", "FTSE100"]


# =====================================================================
# FUNCIONES
# =====================================================================
def summarize_groups(bt: pd.DataFrame, em_aliases: List[str], dm_aliases: List[str]) -> pd.DataFrame:
    """
    Construye un resumen por grupo (EM vs DM) promediando métricas clave del backtesting.
    Métricas: tail_coverage, expected_tail, gap_coverage = tail_coverage - expected_tail,
              tail_mean_real, tail_loss_ratio.

    em_aliases y dm_aliases se pasan en alias (SP500, MSCIEM, etc.), pero se mapean a los
    nombres usados en los datos (S&P 500, MSCI Emerging Markets, etc.).
    """
    bt = bt.copy()

    em_names = set(map_indices(em_aliases)) | set(em_aliases)
    dm_names = set(map_indices(dm_aliases)) | set(dm_aliases)

    bt["group"] = bt["index"].apply(
        lambda x: "EM" if x in em_names else ("DM" if x in dm_names else "OTROS")
    )

    agg = (
        bt.groupby(["group", "method", "alpha"], as_index=False)
        .agg(
            n_total=("n_total", "sum"),
            n_tail=("n_tail", "sum"),
            tail_coverage=("tail_coverage", "mean"),
            expected_tail=("expected_tail", "mean"),
            tail_mean_real=("tail_mean_real", "mean"),
            tail_loss_ratio=("tail_loss_ratio", "mean"),
        )
    )
    agg["gap_coverage"] = agg["tail_coverage"] - agg["expected_tail"]
    return agg.sort_values(["group", "method", "alpha"]).reset_index(drop=True)


def run_capitulo53_integracion(
    start: str,
    end: str,
    indices: List[str],
    alphas: List[float],
    methods: List[str],
    window: int = 500,
    ewma_lambda: float = 0.94,
    t_step: int = 5,
    em: List[str] | None = None,
    dm: List[str] | None = None,
    returns_csv: str | None = None,
    save_outputs: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Ejecuta todo el flujo del capítulo 5.3:
      - Calcula CVaR y backtesting
      - Resume resultados por grupo EM vs DM
    """
    em = em or DEFAULT_EM
    dm = dm or DEFAULT_DM

    # Ejecutar cálculo principal de CVaR
    results = run_capitulo53(
        start=start,
        end=end,
        indices=indices,
        alphas=alphas,
        methods=methods,
        window=window,
        ewma_lambda=ewma_lambda,
        t_step=t_step,
        returns_csv=returns_csv,
        save_outputs=save_outputs,
    )

    # Resumen por grupos (em y dm en alias, se mapean internamente)
    bt = results["backtesting"]
    groups = summarize_groups(bt, em_aliases=em, dm_aliases=dm)

    if save_outputs:
        outdir = "./output/cvar"
        os.makedirs(outdir, exist_ok=True)
        groups.to_csv(os.path.join(outdir, "cvar_groups_summary.csv"), index=False)

    return {**results, "groups_summary": groups}


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser(description="Capítulo 5.3 – Integración CVaR (EM vs DM)")
    ap.add_argument("--start", type=str, required=True)
    ap.add_argument("--end", type=str, required=True)
    ap.add_argument("--indices", type=str, required=True)
    ap.add_argument("--alphas", type=str, default="0.95,0.99")
    ap.add_argument("--methods", type=str, default="hist,norm,t,ewma")
    ap.add_argument("--window", type=int, default=500)
    ap.add_argument("--ewma_lambda", type=float, default=0.94)
    ap.add_argument("--t_step", type=int, default=5)
    ap.add_argument("--em", type=str, default=",".join(DEFAULT_EM))
    ap.add_argument("--dm", type=str, default=",".join(DEFAULT_DM))
    ap.add_argument(
        "--returns_csv",
        type=str,
        default=None,
        help="Ruta a CSV con retornos diarios ya curados (opcional).",
    )

    args = ap.parse_args()
    indices = _parse_list(args.indices)
    alphas = [float(a) for a in _parse_list(args.alphas)]
    methods = _parse_list(args.methods)
    em = _parse_list(args.em)
    dm = _parse_list(args.dm)

    run_capitulo53_integracion(
        start=args.start,
        end=args.end,
        indices=indices,
        alphas=alphas,
        methods=methods,
        window=args.window,
        ewma_lambda=args.ewma_lambda,
        t_step=args.t_step,
        em=em,
        dm=dm,
        returns_csv=args.returns_csv,
        save_outputs=True,
    )


if __name__ == "__main__":
    main()
