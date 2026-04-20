import numpy as np
from src.diffusion_reaction import DiffusionReactionGraph

def simulate_base(adj, beta, gamma, D, K, u0, t_span=120, dt=0.1):
    model = DiffusionReactionGraph(adj, beta, gamma, D, K)
    u = model.simulate(u0, t_span, dt)
    return u, u.sum(axis=1)

def simulate_local_quarantine(adj, beta, gamma, D, K, u0, t_span=120, dt=0.1,
                               quarantine_start=30, quarantine_end=90,
                               quarantine_provinces=[0,1], beta_reduction=0.3):
    """
    Caso 1: Cuarentena estricta en Cercado y Quillacollo (provincias 0 y 1)
    desde el día 30 hasta el día 90, reduciendo beta en 70% solo en esas provincias.
    """
    # Crear una clase temporal que modifique beta según tiempo y provincia
    class ModelWithLocalQuarantine(DiffusionReactionGraph):
        def get_beta(self, t, province_idx=None):
            if province_idx is None:
                return self._beta_base
            if quarantine_start <= t <= quarantine_end and province_idx in quarantine_provinces:
                return self._beta_base[province_idx] * beta_reduction
            return self._beta_base[province_idx]
    model = ModelWithLocalQuarantine(adj, beta, gamma, D, K)
    u = model.simulate(u0, t_span, dt)
    return u, u.sum(axis=1)

def simulate_temporal_mobility(adj, beta, gamma, D, K, u0, t_span=120, dt=0.1,
                                mobility_start=45, mobility_end=50, D_factor=3):
    """
    Caso 2: Aumento de movilidad (D×3) durante Semana Santa (días 45 a 50).
    """
    def D_func(t):
        if mobility_start <= t <= mobility_end:
            return D * D_factor
        return D
    model = DiffusionReactionGraph(adj, beta, gamma, D_func, K)
    u = model.simulate(u0, t_span, dt)
    return u, u.sum(axis=1)

def predict_peak_from_initial_data(observations, true_total, Q, R, initial_days=30, t_span=120):
    """
    Caso 3: Usa solo los primeros 'initial_days' días para predecir el pico máximo.
    Retorna día del pico real, día del pico predicho y error absoluto.
    """
    from src.kalman_filter import KalmanFilter1D
    obs_trunc = observations[:initial_days+1]
    kf = KalmanFilter1D(Q, R, [obs_trunc[0], 0.0], np.eye(2)*100)
    # Filtrar hasta el día initial_days
    for z in obs_trunc:
        kf.predict()
        kf.update(z)
    # Predecir desde initial_days+1 hasta t_span
    pred_full = np.full(t_span+1, np.nan)
    pred_full[:initial_days+1] = obs_trunc  # opcional
    x_pred = kf.x.copy()
    for i in range(initial_days+1, t_span+1):
        x_pred = kf.F @ x_pred
        pred_full[i] = x_pred[0]
    # Encontrar picos
    peak_real = np.argmax(true_total)
    peak_pred = np.argmax(pred_full)
    error = abs(peak_pred - peak_real)
    return peak_real, peak_pred, error, pred_full