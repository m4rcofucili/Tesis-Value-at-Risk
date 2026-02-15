#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Runner estricto para el proyecto (GitHub-ready)
-----------------------------------------------
- Sin placeholders.
- Sin fallbacks opacos.
- Falla rápido si algún paso rompe.
- Asume que los scripts están en ./src
- Estándar canónico de datos: output/returns.csv
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

STEPS = [
    "capitulo5_datos_descriptiva.py",
    "capitulo5_var_backtesting.py",
    "capitulo52_integracion.py",
    "capitulo53_cvar_backtesting.py",
    "capitulo53_integracion.py",
    "capitulo54_montecarlo_extendido.py",
    "capitulo54_comparacion_conclusiones.py",
]

def run_step(pyfile: str) -> None:
    path = SRC / pyfile
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")

    cmd = [sys.executable, str(path)]
    print(f"\n== Ejecutando: {pyfile} ==")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if rc != 0:
        raise RuntimeError(f"Fallo {pyfile} (return code {rc})")

def main() -> None:
    for s in STEPS:
        run_step(s)
    print("\nOK: pipeline finalizado sin errores.")

if __name__ == "__main__":
    main()
