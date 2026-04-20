import numpy as np
from scipy.integrate import odeint

def sir_model(y, t, beta, gamma, N):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def simulate_sir(beta, gamma, N, I0, t_span=120, dt=0.1):
    t = np.arange(0, t_span+dt, dt)
    S0 = N - I0
    y0 = [S0, I0, 0]
    sol = odeint(sir_model, y0, t, args=(beta, gamma, N))
    return t, sol[:, 1]  # retorna infectados