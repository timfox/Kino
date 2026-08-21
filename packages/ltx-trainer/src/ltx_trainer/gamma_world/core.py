import numpy as np

def simplex_agent_phases(n_agents: int) -> np.ndarray:
    """Stub: regular-simplex phases for permutation-symmetric agents."""
    return np.linspace(0.0, 2.0 * np.pi, n_agents, endpoint=False)
