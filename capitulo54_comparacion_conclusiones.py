#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Capítulo 5.4 – Comparación y Conclusiones (GitHub-ready)
--------------------------------------------------------
- Descubre resultados en output/var** y output/cvar** (fallback a artifacts).
- Normaliza columnas comunes y deriva hit_rate.
- Corrige duplicaciones y mejora regla alpha/nivel.
- Genera:
  - artifacts/comparacion_var_cvar.csv
  - artifacts/resumen_var_cvar.csv
  - artifacts/fig_comparacion_hit_rate.png (si hay datos)
"""

from pathlib import Path
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ART = Path("artifacts")
ART.mkdir(parents=True, exist_ok=True)

VAR_GLOBS = [
    "output/var/**/*.csv",
    "output/var_fast/**/*.csv",
    "output/var_fast2/**/*.csv",
]
VAR_FALLBACK = Path("artifacts/var_backtest_results.csv")

CVAR_GLOBS = [
    "output/cvar/**/*.csv",
]
CVAR_FALLBACK = Path("artifacts/cvar_backtest_results.csv")

FILENAME_RE = re.compile(r"backtest_(?P<metodo>[^_]+)_(?P<nivel>[0-9]+(?:\.[0-9]+)?)\.csv$", re.IGNORECASE)

TAILCOV_COL_CANDIDATES = ["tail_coverage", "tail_cov", "coverage_emp", "emp_coverage"]
INDEX_COL_CANDIDATES = ["indice", "Index", "index", "activo", "ticker", "symbol", "serie", "asset"]
EXC_COL_CANDIDATES = ["n_excepciones", "excepciones", "n_exceed", "num_exceedances"]
NOBS_COL_CANDIDATES = ["n_obs", "N", "n", "num_obs", "total"]
RATIO_COL_CANDIDATES = ["ratio_excepciones", "exceedance_rate", "exceed_rate", "ratio"]

HITRATE_COL = "hit_rate"


def discover_files(globs, fallback: Path):
    files = []
    for g in globs:
        files.extend(Path(".").glob(g))
    if not files and fallback.exists():
        files = [fallback]
    return sorted(set([f for f in files if f.is_file()]))


def coalesce_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns:
            return n
    return None


def ensure_indice(df: pd.DataFrame, src_path: Path) -> pd.DataFrame:
    idx_col = coalesce_col(df, INDEX_COL_CANDIDATES)
    if idx_col:
        if idx_col != "indice":
            df.rename(columns={idx_col: "indice"}, inplace=True)
        return df
    df["indice"] = src_path.stem
    return df


def ensure_numerics(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def normalize_nivel_from_alpha(df: pd.DataFrame) -> pd.DataFrame:
    """
    Si existe 'alpha' (cola) y no existe 'nivel', o si 'nivel' parece cola,
    lo normaliza a nivel de confianza en (0,1).
    Reglas:
      - Si valor <= 0.5 -> se interpreta como alpha cola => nivel = 1 - alpha
      - Si valor > 0.5 -> ya es nivel de confianza
    """
    if "nivel" not in df.columns and "alpha" in df.columns:
        df["nivel"] = df["alpha"]

    if "nivel" in df.columns:
        df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")
        m = df["nivel"].notna()
        df.loc[m & (df["nivel"] <= 0.5), "nivel"] = 1.0 - df.loc[m & (df["nivel"] <= 0.5), "nivel"]
    return df


def derive_hit_rate(df: pd.DataFrame) -> pd.DataFrame:
    if HITRATE_COL in df.columns:
        return ensure_numerics(df, [HITRATE_COL])

    exc_col = coalesce_col(df, EXC_COL_CANDIDATES)
    nobs_col = coalesce_col(df, NOBS_COL_CANDIDATES)
    ratio_col = coalesce_col(df, RATIO_COL_CANDIDATES)
    tailcov_col = coalesce_col(df, TAILCOV_COL_CANDIDATES)

    if exc_col and nobs_col:
        df = ensure_numerics(df, [exc_col, nobs_col])
        df[HITRATE_COL] = df[exc_col] / df[nobs_col]
        df.loc[~np.isfinite(df[HITRATE_COL]), HITRATE_COL] = np.nan
        df.loc[df[nobs_col] == 0, HITRATE_COL] = np.nan
        return df

    if ratio_col:
        df = ensure_numerics(df, [ratio_col])
        df[HITRATE_COL] = df[ratio_col] / 100.0 if df[ratio_col].dropna().gt(1).any() else df[ratio_col]
        return df

    if tailcov_col:
        df = ensure_numerics(df, [tailcov_col])
        df[HITRATE_COL] = df[tailcov_col] / 100.0 if df[tailcov_col].dropna().gt(1).any() else df[tailcov_col]
        return df

    return df


def infer_from_filename(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    m = FILENAME_RE.search(path.name)
    if not m:
        return df
    metodo_from_name = m.group("metodo")
    nivel_from_name = m.group("nivel")

    if "metodo" not in df.columns or df["metodo"].isna().all():
        df["metodo"] = metodo_from_name
    if "nivel" not in df.columns or df["nivel"].isna().all():
        try:
            df["nivel"] = float(nivel_from_name)
        except Exception:
            df["nivel"] = nivel_from_name
    return df


def construir_resumen_var_cvar(merged: pd.DataFrame, out_path: Path):
    if not {"medida", "hit_rate"}.issubset(merged.columns):
        print("No se puede generar resumen: faltan columnas básicas en merged.")
        return

    df = merged.copy()
    df["hit_rate"] = pd.to_numeric(df["hit_rate"], errors="coerce")
    if "nivel" in df.columns:
        df["nivel"] = pd.to_numeric(df["nivel"], errors="coerce")
    else:
        df["nivel"] = pd.NA

    df = df.dropna(subset=["hit_rate", "medida"])

    df["nivel_prob"] = df["nivel"].where((df["nivel"] > 0) & (df["nivel"] < 1))
    df["hit_teorico"] = 1 - df["nivel_prob"]

    df["diff_hit"] = df["hit_rate"] - df["hit_teorico"]
    df["abs_diff_hit"] = df["diff_hit"].abs()
    df["overcoverage"] = df["diff_hit"] > 0

    group_keys = ["medida"]
    if "metodo" in df.columns:
        group_keys.append("metodo")
    if "nivel" in df.columns:
        group_keys.append("nivel")

    agg_dict = {
        "hit_rate_medio": ("hit_rate", "mean"),
        "hit_rate_std": ("hit_rate", "std"),
        "hit_teorico_medio": ("hit_teorico", "mean"),
        "diff_promedio": ("diff_hit", "mean"),
        "abs_diff_promedio": ("abs_diff_hit", "mean"),
        "proporcion_overcoverage": ("overcoverage", "mean"),
    }

    if "indice" in df.columns:
        agg_dict["n_indices"] = ("indice", "nunique")
    else:
        agg_dict["n_filas"] = ("hit_rate", "size")

    resumen = df.groupby(group_keys, dropna=False).agg(**agg_dict).reset_index()
    resumen.to_csv(out_path, index=False)
    print(f"Resumen VaR/CVaR guardado en {out_path}")


def load_and_annotate(paths, medida_label: str) -> pd.DataFrame:
    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] No pude leer {p}: {e}")
            continue

        if "nivel_conf" in df.columns and "nivel" not in df.columns:
            df.rename(columns={"nivel_conf": "nivel"}, inplace=True)
        if "method" in df.columns and "metodo" not in df.columns:
            df.rename(columns={"method": "metodo"}, inplace=True)

        df = ensure_indice(df, p)
        df = infer_from_filename(df, p)
        df = normalize_nivel_from_alpha(df)
        df = derive_hit_rate(df)

        df["medida"] = medida_label
        df["__archivo"] = str(p)
        frames.append(df)

    if frames:
        return pd.concat(frames, axis=0, ignore_index=True)
    return pd.DataFrame()


def main():
    var_paths = discover_files(VAR_GLOBS, VAR_FALLBACK)
    cvar_paths = discover_files(CVAR_GLOBS, CVAR_FALLBACK)

    if not var_paths and not cvar_paths:
        print("No se encontraron resultados de VaR/CVaR. Genera los CSVs primero.")
        return

    var_df = load_and_annotate(var_paths, "VaR") if var_paths else pd.DataFrame()
    cvar_df = load_and_annotate(cvar_paths, "CVaR") if cvar_paths else pd.DataFrame()

    if var_df.empty and cvar_df.empty:
        print("No se pudieron cargar resultados de VaR/CVaR (archivos vacíos o ilegibles).")
        return

    merged = pd.concat([d for d in [var_df, cvar_df] if not d.empty], axis=0, ignore_index=True)

    out_csv = ART / "comparacion_var_cvar.csv"
    merged.to_csv(out_csv, index=False)
    print(f"Tabla de comparación guardada en {out_csv}")

    resumen_path = ART / "resumen_var_cvar.csv"
    construir_resumen_var_cvar(merged, resumen_path)

    needed = {"metodo", "nivel", "medida", "hit_rate"}
    if needed.issubset(set(merged.columns)) and merged["hit_rate"].notna().any():
        dfp = merged.dropna(subset=["hit_rate"]).copy()

        plt.figure(figsize=(10, 6))
        for medida in sorted(dfp["medida"].dropna().unique()):
            sub = dfp[dfp["medida"] == medida]
            grp = sub.groupby(["metodo", "nivel"], as_index=False)["hit_rate"].mean()
            xs = [f"{m}-{n}" for m, n in zip(grp["metodo"].astype(str), grp["nivel"].astype(str))]
            plt.plot(xs, grp["hit_rate"], marker="o", label=medida)

        niveles = sorted(dfp["nivel"].dropna().unique())
        if len(niveles) == 1 and isinstance(niveles[0], (int, float)) and 0 < niveles[0] < 1:
            plt.axhline(1 - niveles[0], linestyle="--", linewidth=1)

        plt.title("Comparación de tasa de excepciones (hit_rate) por método y nivel")
        plt.xlabel("método-nivel")
        plt.ylabel("hit rate")
        plt.legend()
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        out_fig = ART / "fig_comparacion_hit_rate.png"
        plt.savefig(out_fig, dpi=160)
        print(f"Figura guardada en {out_fig}")
    else:
        print("No se generó figura: faltan columnas o hit_rate está vacío.")


if __name__ == "__main__":
    main()
