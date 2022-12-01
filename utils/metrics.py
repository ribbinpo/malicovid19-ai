import numpy as np

def smape(forcast, actual):
  if (len(actual) == len(forcast)):
    n = len(actual)
    sums = 0
    for i in range(n):
      sums += (np.abs(forcast[i] - actual[i]) / ((np.abs(actual[i]) + np.abs(forcast[i]))/2))
      print(sums)
    return (1/n) * sums