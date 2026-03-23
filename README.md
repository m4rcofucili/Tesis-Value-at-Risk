# Tesis – Value at Risk & Conditional Value at Risk

## 📌 Descripción
Repositorio asociado a la tesis de Maestría en Finanzas que analiza la complementariedad entre VaR y CVaR en mercados desarrollados y emergentes (2008–2023).

## 🎯 Objetivo
Evaluar la capacidad de distintas metodologías para capturar riesgo extremo y proponer el indicador Spread VaR–CVaR como medida de profundidad de cola.

## 📊 Metodología
- VaR: Histórico, Normal, t-Student, EWMA, MC-GARCH
- CVaR: Histórico, Paramétrico y EWMA
- Backtesting: Kupiec, Christoffersen
- Indicador propuesto: Spread VaR–CVaR

## 🧠 Estructura del código
- `src/`: scripts principales
- `outputs/`: resultados (tablas y gráficos)
- `data/`: datos utilizados

## ▶️ Cómo correr el proyecto

```bash
pip install -r requirements.txt
python run_pipeline.py
