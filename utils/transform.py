import numpy as np

def accumulativeToNon(arr):
  lists = []
  for i in range(len(arr)):
    if(i == 0):
      lists.append(arr[i])
      continue
    lists.append(arr[i] - arr[i-1])
  return np.array(lists)

def revariable(df):
  return df.confirmed.tolist(), df.death.tolist(), df.recovered.tolist(), len(df)

def seird_params(df_params, wave):
  N = df_params.loc[df_params.wave == wave]['N'].tolist()[0]
  gamma = df_params.loc[df_params.wave == wave]['gamma'].tolist()[0]
  delta = df_params.loc[df_params.wave == wave]['delta'].tolist()[0]
  alpha = df_params.loc[df_params.wave == wave]['alpha'].tolist()[0]
  omega = df_params.loc[df_params.wave == wave]['omega'].tolist()[0]
  R_0 = df_params.loc[df_params.wave == wave]['R_0'].tolist()[0]
  zeta = df_params.loc[df_params.wave == wave]['zeta'].tolist()[0]
  # zeta = 10 # wave1
  # # zeta = 13 # wave2
  # # zeta = 15 # wave3
  # # zeta = 20 # wave4
  beta = df_params.loc[df_params.wave == wave]['beta'].tolist()[0]
  days = df_params.loc[df_params.wave == wave]['days'].tolist()[0]

  S0 = df_params.loc[df_params.wave == wave]['S0'].tolist()[0]
  E0 = df_params.loc[df_params.wave == wave]['E0'].tolist()[0]
  I0 = df_params.loc[df_params.wave == wave]['I0'].tolist()[0]
  R0 = df_params.loc[df_params.wave == wave]['R0'].tolist()[0]
  D0 = df_params.loc[df_params.wave == wave]['D0'].tolist()[0]
  return [ N, days, S0, E0, I0, R0, D0, gamma, delta, alpha, omega, R_0, zeta, beta ]