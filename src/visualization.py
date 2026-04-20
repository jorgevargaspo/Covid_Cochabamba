import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.animation import FuncAnimation

def plot_temporal(true, obs, kalman_est, kalman_std, title="Evolución de casos activos", save_path=None):
    plt.figure(figsize=(12,5))
    if true is not None:
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
    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt.gcf()

def plot_heatmap_by_province(province_data, province_names, days_labels, title="Mapa de calor", save_path=None):
    fig, axes = plt.subplots(1, len(days_labels), figsize=(15,4))
    for i, day_label in enumerate(days_labels):
        data = province_data[i, :]
        sns.heatmap([data], annot=True, fmt=".0f", xticklabels=province_names, 
                    yticklabels=[f"Día {day_label}"], cmap="Reds", cbar=True, ax=axes[i])
        axes[i].set_title(f"Día {day_label}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig

def plot_comparison_scenarios(scenarios_data, labels, title="Comparación de escenarios", save_path=None):
    plt.figure(figsize=(12,5))
    for data, label in zip(scenarios_data, labels):
        plt.plot(data, label=label)
    plt.xlabel("Días")
    plt.ylabel("Casos activos totales")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=150)
    return plt.gcf()

def create_animation_spread(true_by_province, province_names, output_gif="spread_animation.gif", fps=5):
    """Crea una animación GIF de la propagación espacial por provincia."""
    fig, ax = plt.subplots(figsize=(8,6))
    def update(day):
        ax.clear()
        data = true_by_province[day, :].reshape(1, -1)
        sns.heatmap(data, annot=True, fmt=".0f", xticklabels=province_names,
                    yticklabels=[f"Día {day}"], cmap="Reds", cbar=True, ax=ax)
        ax.set_title(f"Propagación del COVID-19 - Día {day}")
    ani = FuncAnimation(fig, update, frames=range(0, len(true_by_province), 5), repeat=False)
    ani.save(output_gif, writer='pillow', fps=fps)
    plt.close()