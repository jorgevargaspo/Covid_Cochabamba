import numpy as np

class DiffusionReactionGraph:
    """
    Modelo de difusión-reacción sobre un grafo (provincias).
    dU/dt = D * L * U + beta * U * (1 - U/K) - gamma * U
    donde L es el Laplaciano del grafo (L = A - Dg, con Dg matriz de grados).
    """
    def __init__(self, adjacency_matrix, beta, gamma, D, K):
        self.adj = np.array(adjacency_matrix, dtype=float)
        self.n = self.adj.shape[0]
        self.degrees = np.sum(self.adj, axis=1)
        self.laplacian = self.adj - np.diag(self.degrees)  # forma negativa (difusión: D * L * U)
        self.beta = beta
        self.gamma = gamma
        self.D = D
        self.K = K

    def reaction(self, u):
        """Término de reacción logístico + recuperación."""
        return self.beta * u * (1 - u / self.K) - self.gamma * u

    def derivative(self, u):
        """Derivada temporal: difusión + reacción."""
        diff = self.D * self.laplacian @ u
        react = self.reaction(u)
        return diff + react

    def euler_step(self, u, dt):
        """Un paso de Euler explícito."""
        return u + dt * self.derivative(u)

    def simulate(self, u0, t_span, dt):
        """Simula desde t=0 hasta t_span con paso dt. Retorna matriz (tiempo, provincias)."""
        n_steps = int(t_span / dt) + 1
        u = np.zeros((n_steps, self.n))
        u[0] = u0.copy()
        for i in range(1, n_steps):
            u[i] = self.euler_step(u[i-1], dt)
        return u