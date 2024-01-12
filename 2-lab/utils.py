import re
import io
import requests
import numpy as np
import pandas as pd

def parse(url):
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
  }
  response = requests.get(url, headers=headers).text
  html = re.sub(r'<.*?>', lambda g: g.group(0).upper(), response)
  dataframe = pd.read_html(io.StringIO(html))[0]
  dataframe.to_csv('parsed.csv', index=False)
  return dataframe

def clear_data(url):
  items_name = 'Курс (грн.)'
  dataframe = (parse(url).filter(items=[items_name])[:-1].astype(float) / 10000).to_numpy()[::-1]

  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)

  return dataframe

def MNK (S0):
    iter = len(S0)
    y_input = np.zeros((iter, 1))
    F = np.ones((iter, 3))
    for i in range(iter): 
        y_input[i, 0] = float(S0[i])  
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
    FT = F.T
    FFT = FT.dot(F)
    FFTI = np.linalg.inv(FFT)
    FFTIFT = FFTI.dot(FT)
    C = FFTIFT.dot(y_input)
    y_output = F.dot(C)
    print('Регресійна модель:')
    print('y(t) = ', C[0,0], ' + ', C[1,0], ' * t', ' + ', C[2,0], ' * t^2')
    return y_output

def predict_MNK (S0, koef):
    iter = len(S0)
    y_output_extrapol = np.zeros((iter+koef, 1))
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
    print('Регресійна модель:')
    print('y(t)= ', C[0, 0], ' + ', C[1, 0], ' * t', ' + ', C[2, 0], ' * t^2')
    for i in range(iter+koef):
        y_output_extrapol[i, 0] = C[0, 0]+C[1, 0]*i+(C[2, 0]*i*i)
    return y_output_extrapol

def ABF (S0):
    iter = len(S0)
    y_input = np.zeros((iter, 1))
    y_output_AB = np.zeros((iter, 1))
    T0=1
    for i in range(iter):
        y_input[i, 0] = float(S0[i])

    y_speed_retro=(y_input[1, 0]-y_input[0, 0])/T0
    y_extra=y_input[0, 0]+y_speed_retro
    alfa=2*(2*1-1)/(1*(1+1))
    beta=(6/1)*(1+1)
    y_output_AB[0, 0]=y_input[0, 0]+alfa*(y_input[0, 0])

    for i in range(1, iter):
        y_output_AB[i,0]=y_extra+alfa*(y_input[i, 0]- y_extra)
        y_speed_retro += (beta/T0)*(y_input[i, 0]- y_extra)
        y_extra = y_output_AB[i,0] + y_speed_retro
        alfa = (2 * (2 * i - 1)) / (i * (i + 1))
        beta = 6 /(i* (i + 1))
    y_output_AB[0] = S0[0]
    return y_output_AB

def smooth_data(data, n_wind):
    smoothed_data = np.zeros(len(data))
    window_data = np.zeros(n_wind)

    for i in range(len(data) - n_wind + 1):
        for j in range(n_wind): window_data[j] = data[i + j]
        smoothed_data[i + n_wind - 1] = np.median(window_data)

    for i in range(n_wind):
        smoothed_data[i] = data[i]

    return smoothed_data