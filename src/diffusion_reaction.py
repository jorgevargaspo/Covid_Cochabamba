"""
Módulo: diffusion_reaction.py
Modelo de difusión-reacción sobre un grafo de provincias.
Implementa la ecuación: du/dt = D * L u + beta*u*(1-u/K) - gamma*u
con L = laplaciano del grafo.
Incluye soporte para beta variable por provincia y tiempo, y D variable en el tiempo.
"""
import numpy as np

class DiffusionReactionGraph:
    """
    Modelo EDP en grafo con capacidades avanzadas:
    - Beta puede ser escalar o array por provincia.
    - D puede ser escalar o función del tiempo.
    - Se puede aplicar cuarentena localizada y temporal.
    """
    def __init__(self, adjacency, beta, gamma, D, K):
        """
        adjacency: matriz de adyacencia (n_prov x n_prov)
        beta: escalar o array de longitud n (tasa de contagio por provincia)
        gamma: escalar (tasa de recuperación)
        D: escalar o función que devuelve escalar (coeficiente de difusión)
        K: capacidad máxima por provincia (escalar)
        """
        self.adj = np.array(adjacency, dtype=float)
        self.n = self.adj.shape[0]
        self.degrees = np.sum(self.adj, axis=1)
        self.laplacian = self.adj - np.diag(self.degrees)   # forma negativa (difusión: D * L * u)
        self.gamma = gamma
        self.K = K
        self._beta_base = np.asarray(beta) if np.isscalar(beta) else np.array(beta, dtype=float)
        if self._beta_base.shape == ():
            self._beta_base = np.full(self.n, self._beta_base)
        self._D_base = D

    def get_beta(self, t, province_idx=None):
        """Permite sobreescribir en subclases o durante simulación. Por defecto constante."""
        if province_idx is None:
            return self._beta_base
        return self._beta_base[province_idx]

    def get_D(self, t):
        """Permite D variable en el tiempo."""
        if callable(self._D_base):
            return self._D_base(t)
        return self._D_base

    def reaction(self, u, t):
        """Término de reacción con beta posiblemente por provincia."""
        beta_t = self.get_beta(t)
        return beta_t * u * (1 - u / self.K) - self.gamma * u

    def derivative(self, u, t):
        D_t = self.get_D(t)
        diff = D_t * self.laplacian @ u
        react = self.reaction(u, t)
        return diff + react

    def euler_step(self, u, t, dt):
        return u + dt * self.derivative(u, t)

    def simulate(self, u0, t_span, dt, callback=None):
        """
        Simula desde t=0 hasta t_span con paso dt.
        callback: función opcional que se llama en cada paso (t, u) para modificar parámetros.
        Retorna matriz (n_steps, n_provincias).
        """
        n_steps = int(t_span / dt) + 1
        u = np.zeros((n_steps, self.n))
        u[0] = u0.copy()
        time = np.linspace(0, t_span, n_steps)
        for i in range(1, n_steps):
            t_curr = time[i-1]
            # Aplicar callback antes del paso (para modificar beta, D, etc.)
            if callback is not None:
                callback(t_curr, u[i-1], self)
            u[i] = self.euler_step(u[i-1], t_curr, dt)
        return u