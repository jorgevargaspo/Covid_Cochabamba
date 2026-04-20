import numpy as np
import pandas as pd
from src.diffusion_reaction import DiffusionReactionGraph

def generate_synthetic_data(adj_matrix, beta, gamma, D, K, u0, t_span=120, dt=0.1, noise_std=0.1):
    """
    Simula el modelo EDP y añade ruido para obtener observaciones.
    Retorna:
        - true_cases_total: array de casos totales reales (suma provincias) en días discretos.
        - obs_cases_total: observaciones con ruido.
        - true_by_province: matriz (días, provincias) de casos reales.
    """
    model = DiffusionReactionGraph(adj_matrix, beta, gamma, D, K)
    true_u = model.simulate(u0, t_span, dt)   # pasos finos
    # Muestrear cada día (dt_dia = 1.0)
    time_points = np.arange(0, t_span + 1, 1.0)
    idx_dias = (time_points / dt).astype(int)
    true_daily = true_u[idx_dias, :]          # shape (días, provincias)
    true_total = true_daily.sum(axis=1)
    # Añadir ruido de medición (proporcional)
    obs_total = true_total * (1 + np.random.normal(0, noise_std, size=len(true_total)))
    obs_total = np.maximum(obs_total, 0)      # sin negativos
    return true_total, obs_total, true_daily, time_points.astype(int)