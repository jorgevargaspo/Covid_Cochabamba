import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1. MODELO DE DIFUSIÓN-REACCIÓN EN GRAFO (PROVINCIAS)
# ============================================================
class DiffusionReactionGraph:
    def __init__(self, adjacency, beta, gamma, D, K):
        self.adj = np.array(adjacency, dtype=float)
        self.n = self.adj.shape[0]
        self.degrees = np.sum(self.adj, axis=1)
        self.laplacian = self.adj - np.diag(self.degrees)
        self.beta = beta
        self.gamma = gamma
        self.D = D
        self.K = K

    def reaction(self, u):
        return self.beta * u * (1 - u / self.K) - self.gamma * u

    def derivative(self, u):
        diff = self.D * self.laplacian @ u
        react = self.reaction(u)
        return diff + react

    def euler_step(self, u, dt):
        return u + dt * self.derivative(u)

    def simulate(self, u0, t_span, dt):
        n_steps = int(t_span / dt) + 1
        u = np.zeros((n_steps, self.n))
        u[0] = u0.copy()
        for i in range(1, n_steps):
            u[i] = self.euler_step(u[i-1], dt)
        return u

# ============================================================
# 2. FILTRO DE KALMAN 1D (CASOS + TENDENCIA)
# ============================================================
class KalmanFilter1D:
    def __init__(self, Q, R, initial_state, initial_P):
        self.F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.H = np.array([[1.0, 0.0]])
        self.Q = np.eye(2) * Q
        self.R = R
        self.x = np.array(initial_state, dtype=float).flatten()
        self.P = np.array(initial_P, dtype=float)

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
        return innov

    def predict_n_steps(self, n):
        x_pred = self.x.copy()
        preds = []
        for _ in range(n):
            x_pred = self.F @ x_pred
            preds.append(x_pred[0])
        return np.array(preds)

# ============================================================
# 3. GENERACIÓN DE DATOS SINTÉTICOS
# ============================================================
def generate_synthetic_data(adj, beta, gamma, D, K, u0, t_span=120, dt=0.1, noise_std=0.1):
    model = DiffusionReactionGraph(adj, beta, gamma, D, K)
    u = model.simulate(u0, t_span, dt)
    time_days = np.arange(0, t_span + 1, 1.0)
    idx_days = (time_days / dt).astype(int)
    true_daily = u[idx_days, :]
    true_total = true_daily.sum(axis=1)
    obs_total = true_total * (1 + np.random.normal(0, noise_std, size=len(true_total)))
    obs_total = np.maximum(obs_total, 0)
    return true_total, obs_total, true_daily, time_days.astype(int)

# ============================================================
# 4. FUNCIONES DE VISUALIZACIÓN
# ============================================================
def plot_temporal(true, obs, kalman_est, kalman_std, title="Evolución de casos activos"):
    plt.figure(figsize=(12,5))
    plt.plot(true, label="Real (sin ruido)", linewidth=2)
    plt.plot(obs, 'ro', markersize=3, label="Observaciones (ruidosas)")
    plt.plot(kalman_est, 'g-', label="Estimación Kalman", linewidth=2)
    plt.fill_between(range(len(kalman_est)), 
                     kalman_est - 2*kalman_std, 
                     kalman_est + 2*kalman_std, 
                     alpha=0.3, label="Banda ±2σ")
    plt.xlabel("Días")
    plt.ylabel("Casos activos")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt.gcf()

def plot_heatmap_by_province(province_data, province_names, days_labels, title="Mapa de calor"):
    """
    province_data: matriz (n_dias_seleccionados, n_provincias)
    days_labels: lista de etiquetas de días (ej. [0,30,60,90,120])
    """
    fig, axes = plt.subplots(1, len(days_labels), figsize=(15,4))
    for i, day_label in enumerate(days_labels):
        data = province_data[i, :]
        sns.heatmap([data], annot=True, fmt=".0f", xticklabels=province_names, 
                    yticklabels=[f"Día {day_label}"], cmap="Reds", cbar=True, ax=axes[i])
        axes[i].set_title(f"Día {day_label}")
    plt.tight_layout()
    return fig

def plot_comparison_scenarios(scenarios_data, labels, title="Comparación de escenarios"):
    plt.figure(figsize=(12,5))
    for data, label in zip(scenarios_data, labels):
        plt.plot(data, label=label)
    plt.xlabel("Días")
    plt.ylabel("Casos activos totales")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    return plt.gcf()

# ============================================================
# 5. CONFIGURACIÓN INICIAL
# ============================================================
provincias = ["Cercado", "Quillacollo", "Chapare", "Punata", "Mizque"]
n_prov = len(provincias)

adj = [
    [0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 0, 0, 0],
    [0, 1, 0, 0, 0]
]

beta = 0.3
gamma = 0.1
D = 0.05
K = 10000
u0 = np.zeros(n_prov)
u0[0] = 100

t_span = 120
dt = 0.1
noise_std = 0.1

# ============================================================
# 6. GENERAR DATOS
# ============================================================
true_total, obs_total, true_by_province, time_days = generate_synthetic_data(
    adj, beta, gamma, D, K, u0, t_span, dt, noise_std
)

# ============================================================
# 7. FILTRO DE KALMAN
# ============================================================
Q = 0.01
R = 0.1
kf = KalmanFilter1D(Q, R, initial_state=[obs_total[0], 0.0], initial_P=np.eye(2)*100)

kalman_estimates = []
kalman_stds = []
innovations = []

for z in obs_total:
    kf.predict()
    innov = kf.update(z)
    kalman_estimates.append(kf.x[0])
    kalman_stds.append(np.sqrt(kf.P[0,0]))
    innovations.append(innov)

kalman_estimates = np.array(kalman_estimates)
kalman_stds = np.array(kalman_stds)

pred_7 = kf.predict_n_steps(7)
print("Predicción de casos activos para los próximos 7 días:", pred_7)

# ============================================================
# 8. GRÁFICAS
# ============================================================
fig1 = plot_temporal(true_total, obs_total, kalman_estimates, kalman_stds,
                     title="Evolución de casos activos en Cochabamba")
plt.savefig("temporal_evolution.png", dpi=150)

# Seleccionar días específicos para el mapa de calor
days_to_plot_indices = [0, 30, 60, 90, 120]
valid_indices = [i for i in days_to_plot_indices if i < len(true_by_province)]
province_data_selected = true_by_province[valid_indices, :]   # shape (5,5)
fig2 = plot_heatmap_by_province(province_data_selected, provincias, valid_indices,
                                title="Propagación espacial de COVID-19")
plt.savefig("spatial_heatmap.png", dpi=150)

# ============================================================
# 9. ESCENARIOS (Base, Cuarentena, Alta movilidad)
# ============================================================
model_base = DiffusionReactionGraph(adj, beta, gamma, D, K)
u_base = model_base.simulate(u0, t_span, dt)
total_base = u_base.sum(axis=1)

beta_low = beta * 0.3
model_quar = DiffusionReactionGraph(adj, beta_low, gamma, D, K)
u_quar = model_quar.simulate(u0, t_span, dt)
total_quar = u_quar.sum(axis=1)

D_high = D * 3
model_mob = DiffusionReactionGraph(adj, beta, gamma, D_high, K)
u_mob = model_mob.simulate(u0, t_span, dt)
total_mob = u_mob.sum(axis=1)

scenarios = [total_base, total_quar, total_mob]
labels = ["Base (sin intervención)", "Cuarentena (β 70% menor)", "Alta movilidad (D×3)"]
fig3 = plot_comparison_scenarios(scenarios, labels, title="Comparación de escenarios")
plt.savefig("scenario_comparison.png", dpi=150)

# ============================================================
# 10. ANÁLISIS DE SENSIBILIDAD
# ============================================================
beta_vals = [0.2, 0.3, 0.4, 0.5]
D_vals = [0.01, 0.05, 0.1, 0.2]

peak_times_beta = []
for b in beta_vals:
    model = DiffusionReactionGraph(adj, b, gamma, D, K)
    u = model.simulate(u0, t_span, dt)
    total = u.sum(axis=1)
    peak_times_beta.append(np.argmax(total) * dt)

peak_times_D = []
for d in D_vals:
    model = DiffusionReactionGraph(adj, beta, gamma, d, K)
    u = model.simulate(u0, t_span, dt)
    total = u.sum(axis=1)
    peak_times_D.append(np.argmax(total) * dt)

data_heat = np.array([peak_times_beta, peak_times_D])
plt.figure(figsize=(8,5))
sns.heatmap(data_heat, annot=True, xticklabels=beta_vals, yticklabels=['β', 'D'], cmap='coolwarm')
plt.title("Tiempo del pico (días) en función de β y D")
plt.tight_layout()
plt.savefig("sensitivity_heatmap.png", dpi=150)

# ============================================================
# 11. TOMA DE DECISIONES
# ============================================================
print("\n--- TOMA DE DECISIONES ---")
threshold = 2.5
alert_days = np.where(np.abs(innovations) > threshold)[0]
if len(alert_days) > 0:
    print(f"⚠️ Alerta: Cambio de régimen detectado en días {alert_days} (innovación > {threshold})")
else:
    print("No se detectaron cambios de régimen significativos.")

peak_day = np.argmax(total_base) * dt
peak_province = np.argmax(u_base[int(peak_day/dt), :])
print(f"El pico de casos ocurre alrededor del día {peak_day:.0f} en la provincia de {provincias[peak_province]}.")
print("Recomendación: Concentrar recursos sanitarios en esa provincia durante la ventana de pico ±10 días.")

for i, prov in enumerate(provincias):
    max_cases = np.max(u_base[:, i])
    if max_cases > 0.8 * K:
        print(f"⚠️ {prov} supera el 80% de capacidad (K={K}). Requiere refuerzo inmediato.")
    else:
        print(f"✓ {prov} dentro de capacidad (max={max_cases:.0f}).")

# Guardar CSV
df = pd.DataFrame({
    'dia': time_days[:len(true_total)],
    'casos_reales': true_total,
    'observaciones': obs_total,
    'estimacion_kalman': kalman_estimates,
    'incertidumbre': kalman_stds
})
df.to_csv("synthetic_cases.csv", index=False)
print("\nResultados guardados: temporal_evolution.png, spatial_heatmap.png, scenario_comparison.png, sensitivity_heatmap.png, synthetic_cases.csv")