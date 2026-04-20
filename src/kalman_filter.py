import numpy as np

class KalmanFilter1D:
    """
    Estado: [casos, tendencia]
    Transición: F = [[1, 1], [0, 1]]
    Observación: H = [1, 0]
    """
    def __init__(self, Q, R, initial_state=None, initial_P=None):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * Q
        self.R = R
        if initial_state is None:
            self.x = np.array([0.0, 0.0])
        else:
            self.x = np.array(initial_state, dtype=float)
        if initial_P is None:
            self.P = np.eye(2) * 100.0
        else:
            self.P = np.array(initial_P)

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[0]  # predicción de casos

    def update(self, z):
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T / S
        self.x = self.x + K * y
        self.P = (np.eye(2) - K @ self.H) @ self.P
        innovacion = y / np.sqrt(S)   # residuo normalizado
        return innovacion

    def predict_n_steps(self, n):
        """Predice n pasos adelante (sin actualizar)."""
        x_pred = self.x.copy()
        preds = []
        for _ in range(n):
            x_pred = self.F @ x_pred
            preds.append(x_pred[0])
        return np.array(preds)