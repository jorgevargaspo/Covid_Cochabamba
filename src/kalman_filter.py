"""
Filtro de Kalman 1D para estimación de casos activos y tendencia.
Incluye búsqueda en grilla de Q y R para minimizar RMSE.
"""
import numpy as np
from sklearn.metrics import mean_squared_error

class KalmanFilter1D:
    def __init__(self, Q, R, initial_state, initial_P):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * Q
        self.R = R
        self.x = np.array(initial_state, dtype=float).flatten()
        self.P = np.array(initial_P, dtype=float)
        self.history = {'x': [], 'P': []}

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K.flatten() * y
        self.P = (np.eye(2) - np.outer(K.flatten(), self.H)) @ self.P
        innov = y / np.sqrt(S)
        self.history['x'].append(self.x.copy())
        self.history['P'].append(self.P.copy())
        return innov

    def filter_series(self, observations):
        estimates = []
        stds = []
        innovations = []
        for z in observations:
            self.predict()
            innov = self.update(z)
            estimates.append(self.x[0])
            stds.append(np.sqrt(self.P[0,0]))
            innovations.append(innov)
        return np.array(estimates), np.array(stds), np.array(innovations)

    def predict_n_steps(self, n):
        x_pred = self.x.copy()
        preds = []
        for _ in range(n):
            x_pred = self.F @ x_pred
            preds.append(x_pred[0])
        return np.array(preds)

def optimize_kalman_params(observations, true_values, Q_range, R_range, initial_state, initial_P):
    """Búsqueda en grilla de Q y R que minimiza RMSE entre estimación y verdad."""
    best_rmse = np.inf
    best_Q, best_R = None, None
    for Q in Q_range:
        for R in R_range:
            kf = KalmanFilter1D(Q, R, initial_state, initial_P)
            est, _, _ = kf.filter_series(observations)
            rmse = np.sqrt(mean_squared_error(true_values, est))
            if rmse < best_rmse:
                best_rmse = rmse
                best_Q, best_R = Q, R
    return best_Q, best_R, best_rmse