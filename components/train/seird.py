import numpy as np
from scipy.integrate import odeint

# Running
async def process(equation, df_params):
  N, days, S0, E0, I0, R0, D0, gamma, delta, alpha, omega, R_0, zeta, beta = df_params
  y0 = S0, E0, I0, R0, D0
  time = np.linspace(0, days, days) # Grid of time period (in days)
  # Integrate the SIR equations over the time grid, t.
  ret = odeint(equation, y0, time, args=(N, beta, gamma, delta, alpha, omega, zeta))
  return ret.T, time