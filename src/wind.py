# ══════════════════════════════════════════════════════
#  wind.py
#  Wind speed profile generators
# ══════════════════════════════════════════════════════

import numpy as np
from parameters import v_cutin, v_cutout, v_rated


def generate_wind(t, profile='smooth_step'):
    """
    Generate wind speed time series.

    Parameters
    ----------
    t       : np.ndarray — time array (s)
    profile : str
        'smooth_step' — ramped steps across Region 2 and 3 (default)
        'instant_step' — instantaneous steps (stress test)
        'ramp'         — linear ramp from cut-in to above rated
        'turbulent'    — mean wind + low-pass filtered noise

    Returns
    -------
    v : np.ndarray — wind speed (m/s)
    """
    if profile == 'smooth_step':
        waypoints = [
            (0,   5.0), (50,  5.0), (70,  8.0),
            (100, 8.0), (120, 11.4),(150, 11.4),
            (175, 14.0),(200, 14.0),(230, 18.0),
            (250, 18.0)
        ]
        times  = [w[0] for w in waypoints]
        speeds = [w[1] for w in waypoints]
        return np.interp(t, times, speeds)

    elif profile == 'instant_step':
        v = np.ones_like(t) * 5.0
        v[t >= 50]  = 8.0
        v[t >= 100] = 11.4
        v[t >= 150] = 14.0
        v[t >= 200] = 18.0
        return v

    elif profile == 'ramp':
        return np.interp(t, [0, 250], [3.0, 20.0])

    elif profile == 'turbulent':
        np.random.seed(42)
        v_mean = 10.0
        noise  = np.random.randn(len(t))
        dt     = t[1] - t[0]
        tau    = 5.0
        alpha  = dt / (tau + dt)
        v_turb = np.zeros_like(t)
        for i in range(1, len(t)):
            v_turb[i] = alpha * noise[i] + (1 - alpha) * v_turb[i - 1]
        v = v_mean + 1.5 * v_turb
        return np.clip(v, v_cutin, v_cutout)

    else:
        raise ValueError(f"Unknown profile: '{profile}'. "
                         f"Choose from: smooth_step, instant_step, ramp, turbulent")
