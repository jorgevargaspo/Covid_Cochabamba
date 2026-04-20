import numpy as np
from src.diffusion_reaction import DiffusionReactionGraph

def sensitivity_analysis(adj_matrix, base_params, param_ranges, u0, t_span=120, metric='peak_time'):
    """
    Itera sobre valores de un parámetro y calcula la métrica.
    param_ranges: {'beta': [0.2,0.3,0.4], ...}
    """
    results = {}
    for param_name, values in param_ranges.items():
        metric_values = []
        for val in values:
            params = base_params.copy()
            params[param_name] = val
            model = DiffusionReactionGraph(adj_matrix, params['beta'], params['gamma'], params['D'], params['K'])
            u = model.simulate(u0, t_span, dt=0.1)
            total_cases = u.sum(axis=1)
            if metric == 'peak_time':
                peak = np.argmax(total_cases) * 0.1   # dt=0.1 -> días
                metric_values.append(peak)
            elif metric == 'peak_value':
                metric_values.append(np.max(total_cases))
        results[param_name] = metric_values
    return results