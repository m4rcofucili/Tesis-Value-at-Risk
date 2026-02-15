"""
Capítulo 5.4 – VaR por Simulación Monte Carlo (MC–GARCH) – Versión extendida (GitHub-ready)
------------------------------------------------------------------------------------------

Implementa:
1) Lectura de ./output/returns.csv
2) Selección de un activo
3) Backtesting rolling de VaR por Monte Carlo GARCH (AR(1)–GARCH(1,1))
4) Pruebas Kupiec / Christoffersen / Cobertura Condicional
5) Gráficos ilustrativos

Mejoras para repositorio:
- Determinismo opcional con --seed
- Sin seaborn (menos dependencias)
- Logging en lugar de prints
"""

from __future__ import annotations

import argparse
import os
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from arch.univariate import arch_model
from scipy.stats import chi2

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s"
)
logger = logging.getLogger("mc_garch")

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------

OUTDIR_DEFAULT = os.path.join("output", "mc_garch_ext")
RETURNS_PATH_DEFAULT = os.path.join("output", "returns.csv")

plt.rcParams.update({
    "figure.figsize": (11, 6),
    "axes.titlesize": 16,
    "axes.labelsize": 13
})


@dataclass
class McGarchConfig:
    asset: str
    start: Optional[str]
    end: Optional[str]
    nsims: int
    alpha_levels: Tuple[float, ...]
    train_size: int
    step: int
    dist: str = "t"
    seed: Optional[int] = None


def ensure_outdir(path: str = OUTDIR_DEFAULT) -> str:
    os.makedirs(path, exist_ok=True)
    return path


ASSET_NAME_MAP: Dict[str, str] = {
    "SP500": "S&P 500",
    "EUROSTOXX50": "Euro Stoxx 50",
    "FTSE100": "FTSE 100",
    "MSCIEM": "MSCI Emerging Markets",
    "MERVAL": "Merval (Argentina)",
    "BOVESPA": "Bovespa (Brasil)",
}


def load_returns(path: str = RETURNS_PATH_DEFAULT,
                 asset: str = "SP500",
                 start: Optional[str] = None,
                 end: Optional[str] = None) -> pd.Series:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró {path}. Generá primero returns.csv con capitulo5_datos_descriptiva.py."
        )

    df = pd.read_csv(path, index_col=0, parse_dates=True)
    asset_col = ASSET_NAME_MAP.get(asset, asset)

    if asset_col not in df.columns:
        raise KeyError(
            f"Activo {asset} (mapeado a '{asset_col}') no encontrado en returns.csv. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    s = df[asset_col].dropna()
    if start is not None:
        s = s[s.index >= pd.to_datetime(start)]
    if end is not None:
        s = s[s.index <= pd.to_datetime(end)]

    if s.empty:
        raise ValueError("La serie de retornos está vacía tras aplicar filtros de fecha.")

    return s


def fit_ar1_garch11(returns: pd.Series, dist: str = "t"):
    if dist not in ("t", "normal"):
        raise ValueError("dist debe ser 't' o 'normal'.")

    am = arch_model(
        returns,
        mean="AR",
        lags=1,
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist=dist
    )
    res = am.fit(disp="off")
    return res


def simulate_one_step_mc(res, last_return: float, nsims: int, rng: np.random.Generator) -> np.ndarray:
    params = res.params

    mu_const = params.get("mu", params.get("Const", 0.0))
    phi1 = params.get("ar.L1", 0.0)

    fcast = res.forecast(horizon=1)
    sigma2_t1 = fcast.variance.iloc[-1, 0]
    sigma_t1 = float(np.sqrt(sigma2_t1))

    mu_t1 = float(mu_const + phi1 * last_return)

    if res.model.distribution.name.lower().startswith("t"):
        nu = float(params.get("nu", 10.0))
        shocks = rng.standard_t(df=nu, size=nsims)
    else:
        shocks = rng.standard_normal(size=nsims)

    return mu_t1 + sigma_t1 * shocks


def mc_var_from_simulated(simulated_returns: np.ndarray, alpha_levels: Tuple[float, ...]) -> Dict[str, float]:
    vars_dict: Dict[str, float] = {}
    for cl in alpha_levels:
        alpha_tail = 1.0 - cl
        q = float(np.quantile(simulated_returns, alpha_tail))
        key = f"VaR_{int(round(cl * 100))}"
        vars_dict[key] = q
    return vars_dict


def kupiec_test(exceptions: np.ndarray, alpha_tail: float) -> Tuple[float, float]:
    N = len(exceptions)
    x = int(exceptions.sum())
    if N == 0:
        return np.nan, np.nan

    pi_hat = x / N
    if pi_hat in (0, 1):
        return np.nan, np.nan

    num = (1 - alpha_tail) ** (N - x) * (alpha_tail ** x)
    den = (1 - pi_hat) ** (N - x) * (pi_hat ** x)
    lr_uc = -2 * np.log(num / den)
    pvalue = 1 - chi2.cdf(lr_uc, df=1)
    return float(lr_uc), float(pvalue)


def christoffersen_test(exceptions: np.ndarray) -> Tuple[float, float]:
    if len(exceptions) < 2:
        return np.nan, np.nan

    x = exceptions.astype(int)
    N00 = N01 = N10 = N11 = 0

    for t in range(1, len(x)):
        prev, curr = x[t - 1], x[t]
        if prev == 0 and curr == 0:
            N00 += 1
        elif prev == 0 and curr == 1:
            N01 += 1
        elif prev == 1 and curr == 0:
            N10 += 1
        elif prev == 1 and curr == 1:
            N11 += 1

    N0 = N00 + N01
    N1 = N10 + N11
    if N0 == 0 or N1 == 0:
        return np.nan, np.nan

    pi0 = N01 / N0
    pi1 = N11 / N1
    pi = (N01 + N11) / (N0 + N1)

    L0 = ((1 - pi) ** (N00 + N10)) * (pi ** (N01 + N11))
    L1 = ((1 - pi0) ** N00) * (pi0 ** N01) * ((1 - pi1) ** N10) * (pi1 ** N11)

    lr_ind = -2 * np.log(L0 / L1)
    pvalue = 1 - chi2.cdf(lr_ind, df=1)
    return float(lr_ind), float(pvalue)


def conditional_coverage_test(exceptions: np.ndarray, alpha_tail: float) -> Tuple[float, float, float, float, float, float]:
    lr_uc, p_uc = kupiec_test(exceptions, alpha_tail)
    lr_ind, p_ind = christoffersen_test(exceptions)

    if np.isnan(lr_uc) or np.isnan(lr_ind):
        return lr_uc, p_uc, lr_ind, p_ind, np.nan, np.nan

    lr_cc = lr_uc + lr_ind
    p_cc = 1 - chi2.cdf(lr_cc, df=2)
    return lr_uc, p_uc, lr_ind, p_ind, float(lr_cc), float(p_cc)


def decision_from_p(p: float, alpha: float = 0.05) -> str:
    if p is None or np.isnan(p):
        return "NA"
    return "Rechaza H0" if p < alpha else "No se rechaza H0"


def mc_garch_backtest(returns: pd.Series, cfg: McGarchConfig, rng: np.random.Generator) -> pd.DataFrame:
    series = returns.dropna()
    if len(series) <= cfg.train_size:
        raise ValueError(f"Se requieren más de train_size={cfg.train_size} observaciones ({len(series)} disponibles).")

    out_index = series.index[cfg.train_size:]
    results: List[Dict[str, float]] = []

    res = None

    for i, date in enumerate(out_index):
        end_loc = series.index.get_loc(date)
        train_slice = series.iloc[:end_loc]

        if (res is None) or ((i % cfg.step) == 0):
            logger.info("Recalibrando GARCH en %s | i=%d | n_train=%d", date.date(), i, len(train_slice))
            res = fit_ar1_garch11(train_slice, dist=cfg.dist)

        last_ret = float(train_slice.iloc[-1])
        sim_ret = simulate_one_step_mc(res, last_return=last_ret, nsims=cfg.nsims, rng=rng)
        var_dict = mc_var_from_simulated(sim_ret, cfg.alpha_levels)

        row: Dict[str, float] = {"Date": date, "return": float(series.loc[date])}
        row.update(var_dict)
        results.append(row)

    df = pd.DataFrame(results).set_index("Date")

    for cl in cfg.alpha_levels:
        key = f"VaR_{int(round(cl * 100))}"
        exc_key = f"exception_{int(round(cl * 100))}"
        df[exc_key] = (df["return"] < df[key]).astype(int)

    return df


def plot_mc_distribution(simulated_returns: np.ndarray, var_dict: Dict[str, float], asset: str, outdir: str):
    plt.hist(simulated_returns, bins=80, density=True, alpha=0.35, label="Sim MC")
    for key, v in var_dict.items():
        plt.axvline(v, linestyle="--", linewidth=2, label=key)

    plt.title(f"Distribución simulada de retornos – {asset} (MC–GARCH)")
    plt.xlabel("Retorno simulado")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"mc_dist_{asset}.png"))
    plt.clf()


def plot_mc_paths(res, last_return: float, asset: str, outdir: str, rng: np.random.Generator, horizon: int = 20, n_paths: int = 200):
    params = res.params
    mu_const = params.get("mu", params.get("Const", 0.0))
    phi1 = params.get("ar.L1", 0.0)
    omega = params.get("omega", 0.0)
    alpha = params.get("alpha[1]", params.get("alpha.L1", 0.0))
    beta = params.get("beta[1]", params.get("beta.L1", 0.0))

    sigma_prev = float(res.conditional_volatility.iloc[-1])
    eps_prev = float(res.resid.iloc[-1])

    use_t = res.model.distribution.name.lower().startswith("t")
    nu = float(params.get("nu", 10.0)) if use_t else None

    paths = np.zeros((horizon, n_paths))

    for j in range(n_paths):
        r_prev = last_return
        sigma_t = sigma_prev
        eps_t = eps_prev
        for t in range(horizon):
            sigma_t = float(np.sqrt(omega + alpha * eps_t**2 + beta * sigma_t**2))
            z = float(rng.standard_t(df=nu, size=1)[0]) if use_t else float(rng.standard_normal(size=1)[0])
            eps_t = sigma_t * z
            r_t = float(mu_const + phi1 * r_prev + eps_t)
            paths[t, j] = r_t
            r_prev = r_t

    plt.plot(paths, alpha=0.3, linewidth=1)
    plt.axhline(0.0, linewidth=1)
    plt.title(f"Trayectorias sintéticas de retornos – {asset} (MC–GARCH)")
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Retorno simulado")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"mc_paths_{asset}.png"))
    plt.clf()


def plot_var_timeseries(df: pd.DataFrame, asset: str, cl_for_plot: float, outdir: str):
    key = f"VaR_{int(round(cl_for_plot * 100))}"
    exc_key = f"exception_{int(round(cl_for_plot * 100))}"

    plt.plot(df.index, df["return"], linewidth=1, label="Retorno diario")
    plt.plot(df.index, df[key], linewidth=2, label=f"{key} (MC–GARCH)")

    breaches = df[df[exc_key] == 1]
    if not breaches.empty:
        plt.scatter(breaches.index, breaches["return"], s=30, label="Violaciones")

    plt.title(f"Retornos vs VaR MC – {asset} (conf. {int(cl_for_plot*100)}%)")
    plt.xlabel("Fecha")
    plt.ylabel("Retorno")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"mc_var_ts_{asset}_{int(cl_for_plot*100)}.png"))
    plt.clf()


def run_mc_garch_pipeline(cfg: McGarchConfig, returns_path: str = RETURNS_PATH_DEFAULT, outdir: str = OUTDIR_DEFAULT):
    outdir = ensure_outdir(outdir)

    rng = np.random.default_rng(cfg.seed)

    logger.info("Cargando retornos: %s", os.path.abspath(returns_path))
    rets = load_returns(path=returns_path, asset=cfg.asset, start=cfg.start, end=cfg.end)
    logger.info("Serie cargada (%s). Observaciones: %d", cfg.asset, len(rets))

    logger.info("Iniciando backtesting MC–GARCH...")
    df_mc = mc_garch_backtest(rets, cfg, rng=rng)
    logger.info("Backtesting completado. Filas: %d", len(df_mc))

    csv_path = os.path.join(outdir, f"mc_var_{cfg.asset}.csv")
    df_mc.to_csv(csv_path, float_format="%.6f")
    logger.info("Serie VaR MC guardada: %s", csv_path)

    rows_bt: List[Dict[str, float]] = []
    for cl in cfg.alpha_levels:
        alpha_tail = 1.0 - cl
        exc_key = f"exception_{int(round(cl * 100))}"
        exc = df_mc[exc_key].values
        lr_uc, p_uc, lr_ind, p_ind, lr_cc, p_cc = conditional_coverage_test(exc, alpha_tail)

        rows_bt.append({
            "asset": cfg.asset,
            "conf_level": cl,
            "alpha_tail": alpha_tail,
            "LR_uc": lr_uc,
            "p_uc": p_uc,
            "LR_ind": lr_ind,
            "p_ind": p_ind,
            "LR_cc": lr_cc,
            "p_cc": p_cc,
            "n": len(exc),
            "exceptions": int(exc.sum()),
            "Kupiec_5pc": decision_from_p(p_uc, alpha=0.05),
            "Christoffersen_5pc": decision_from_p(p_ind, alpha=0.05),
            "CC_5pc": decision_from_p(p_cc, alpha=0.05),
        })

    bt_df = pd.DataFrame(rows_bt)
    bt_path = os.path.join(outdir, f"mc_backtesting_{cfg.asset}.csv")
    bt_df.to_csv(bt_path, index=False, float_format="%.6f")
    logger.info("Backtesting guardado: %s", bt_path)

    logger.info("Generando gráficos ilustrativos...")
    res_full = fit_ar1_garch11(rets.iloc[:-1], dist=cfg.dist)
    last_ret_full = float(rets.iloc[-2])
    sim_last = simulate_one_step_mc(res_full, last_return=last_ret_full, nsims=cfg.nsims, rng=rng)
    var_last = mc_var_from_simulated(sim_last, cfg.alpha_levels)

    plot_mc_distribution(sim_last, var_last, cfg.asset, outdir)
    plot_mc_paths(res_full, last_return=last_ret_full, asset=cfg.asset, outdir=outdir, rng=rng, horizon=20, n_paths=200)

    cl_plot = max(cfg.alpha_levels)
    plot_var_timeseries(df_mc, cfg.asset, cl_plot, outdir)

    logger.info("OK: outputs guardados en %s", os.path.abspath(outdir))
    return bt_df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capítulo 5.4 – VaR por Simulación Monte Carlo (MC–GARCH)")
    parser.add_argument("--asset", type=str, default="SP500")
    parser.add_argument("--assets", type=str, nargs="*", default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--nsims", type=int, default=3000)
    parser.add_argument("--alpha-levels", type=float, nargs="+", default=[0.95, 0.99])
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--dist", type=str, default="t", choices=["t", "normal"])
    parser.add_argument("--returns-path", type=str, default=RETURNS_PATH_DEFAULT)
    parser.add_argument("--outdir", type=str, default=OUTDIR_DEFAULT)
    parser.add_argument("--seed", type=int, default=42, help="Semilla para reproducibilidad (default: 42).")

    if argv is None:
        args, unknown = parser.parse_known_args()
    else:
        args, unknown = parser.parse_known_args(argv)
    if unknown:
        logger.warning("Argumentos no reconocidos ignorados: %s", unknown)
    return args


def main(argv: Optional[List[str]] = None):
    args = parse_args(argv)

    assets = args.assets if args.assets else [args.asset]

    all_bt: List[pd.DataFrame] = []
    for asset in assets:
        logger.info("Procesando activo: %s", asset)
        cfg = McGarchConfig(
            asset=asset,
            start=args.start,
            end=args.end,
            nsims=args.nsims,
            alpha_levels=tuple(args.alpha_levels),
            train_size=args.train_size,
            step=args.step,
            dist=args.dist,
            seed=args.seed,
        )
        bt_df = run_mc_garch_pipeline(cfg, returns_path=args.returns_path, outdir=args.outdir)
        if "asset" not in bt_df.columns:
            bt_df = bt_df.copy()
            bt_df["asset"] = asset
        all_bt.append(bt_df)

    if all_bt:
        bt_all = pd.concat(all_bt, ignore_index=True)
        summary_path = os.path.join(args.outdir, "mc_backtesting_ALL.csv")
        bt_all.to_csv(summary_path, index=False, float_format="%.6f")
        logger.info("Tabla conjunta guardada: %s", summary_path)


if __name__ == "__main__":
    main()
