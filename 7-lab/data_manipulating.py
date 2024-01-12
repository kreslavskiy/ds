import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def show_data(*args, title):
  plt.clf()
  for arg in args:
    plt.plot(arg)
  plt.title(title)
  plt.show()

def data_sum_sales(data):
  result = {}
  for i in range(len(data[3])):
    current_key = data[1][i]
    sales = data[3][i] * data[5][i]

    if current_key in result:
      result[current_key] += sales
    else:
      result[current_key] = sales
  return result

def region_clustering(data, region_dict):
  result = {}

  for i in range(len(data[3])):
    magaz_code = data[0][i]
    region = region_dict.get(magaz_code, "")

    sales = [data[:, i][1], data[:, i][3] * data[:, i][5]]

    if region not in result:
      result[region] = {'sales': []}

    result[region]['sales'].append(sales)

  for region in result:
    clustered_dates = {}
    for entry in result[region]['sales']:
      if entry[0] not in clustered_dates:
        clustered_dates[entry[0]] = entry[1]
      else:
        clustered_dates[entry[0]] += entry[1]
    sorted_dates = dict(sorted(clustered_dates.items(), key=lambda x: pd.to_datetime(x[0])))
    result[region]['sales'] = sorted_dates

  return result

def predict_MNK(S0, koef):
  iter = len(S0)
  y_output_extrapol = np.zeros((koef, 1))
  y_input = np.zeros((iter, 1))
  F = np.ones((iter, 3))
  for i in range(iter):  
    y_input[i, 0] = float(S0[i])  
    F[i, 1] = float(i)
    F[i, 2] = float(i * i)
  FT=F.T
  FFT = FT.dot(F)
  FFTI=np.linalg.inv(FFT)
  FFTIFT=FFTI.dot(FT)
  C=FFTIFT.dot(y_input)
  for i in range(koef):
    y_output_extrapol[i, 0] = C[0, 0]+C[1, 0]*i+(C[2, 0]*i*i)
  return y_output_extrapol
