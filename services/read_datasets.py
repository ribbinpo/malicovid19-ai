import pandas as pd

# wave1: 12/03/63-19/12/63 Total: 343 Days [0-342]
# wave2: 20/12/63-01/04/64 Total: 103 Days [343-445]
# wave3: 02/04/64-12/12/64 Total: 255 Days [446-700]
# wave4: 13/12/64-30/08/65 Total: 261 Days [701-961]

df_total = pd.read_csv('assets/datasets/thai_d4.csv')
waves = {
  '1': [0, 343],
  '2': [343, 446],
  '3': [446, 701],
  '4': [701, 962]
}

def all_df():
  return df_total

def df_wave(wave):
  range = waves[str(wave)]
  return df_total[range[0]:range[1]].reset_index(drop=True)