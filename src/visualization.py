import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    plt.tight_layout()
    return plt.gcf()

def plot_heatmap_by_province(province_data, province_names, days_to_plot, title="Mapa de calor de infectados por provincia"):
    """
    province_data: matriz (días, provincias)
    days_to_plot: lista de índices de días a mostrar
    """
    fig, axes = plt.subplots(1, len(days_to_plot), figsize=(15,4))
    for idx, day in enumerate(days_to_plot):
        data = province_data[day, :]
        sns.heatmap([data], annot=True, fmt=".0f", xticklabels=province_names, yticklabels=[f"Día {day}"],
                    cmap="Reds", cbar=True, ax=axes[idx])
        axes[idx].set_title(f"Día {day}")
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