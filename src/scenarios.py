from src.diffusion_reaction import DiffusionReactionGraph
import numpy as np

def run_scenario(adj_matrix, beta, gamma, D, K, u0, t_span=120, dt=0.1, 
                 quarantine=None, mobility_factor=None):
    """
    quarantine: diccionario {provincia_idx: factor_reduccion_beta} aplicado desde cierto día.
    mobility_factor: factor multiplicador de D en ciertos días (ej. semana santa).
    """
    model = DiffusionReactionGraph(adj_matrix, beta, gamma, D, K)
    n_steps = int(t_span / dt) + 1
    u = np.zeros((n_steps, model.n))
    u[0] = u0.copy()
    for i in range(1, n_steps):
        t = i * dt
        # Aplicar cuarentena (reducir beta localmente)
        beta_actual = model.beta
        if quarantine is not None and t >= quarantine['start_day'] and t <= quarantine['end_day']:
            for prov, factor in quarantine['factors'].items():
                # Esto es una simplificación: modificamos el término de reacción directamente
                # En una implementación completa habría que re-calcular la derivada con beta variable.
                pass
        # Para simplificar, se puede modificar la reacción paso a paso:
        # Usamos el modelo original pero podemos modificar parámetros internamente.
        # Por claridad, aquí no implementamos la modificación dinámica; se hará en main.
        u[i] = model.euler_step(u[i-1], dt)
    return u