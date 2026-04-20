import numpy as np
import pandas as pd
from src.diffusion_reaction import DiffusionReactionGraph

def generate_synthetic_data(adj, beta, gamma, D, K, u0, t_span=120, dt=0.1, noise_std=0.1, random_seed=42):
    np.random.seed(random_seed)
    model = DiffusionReactionGraph(adj, beta, gamma, D, K)
    u = model.simulate(u0, t_span, dt)
    time_days = np.arange(0, t_span + 1, 1.0)
    idx_days = (time_days / dt).astype(int)
    true_daily = u[idx_days, :]
    true_total = true_daily.sum(axis=1)
    obs_total = true_total * (1 + np.random.normal(0, noise_std, size=len(true_total)))
    obs_total = np.maximum(obs_total, 0)
    return true_total, obs_total, true_daily, time_days.astype(int)

def load_real_data_custom(filepath, t_span=120):
    """Carga un CSV con columnas 'dia' y 'observaciones' (opcional 'casos_reales')."""
    df = pd.read_csv(filepath)
    obs = df['observaciones'].values[:t_span+1]
    true = df['casos_reales'].values[:t_span+1] if 'casos_reales' in df.columns else None
    return true, obs