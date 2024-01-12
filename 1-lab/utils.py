import re
import requests
import numpy as np
import pandas as pd

def parse(url):
  headers = {
      'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36'
  }
  response = requests.get(url, headers=headers).text
  html = re.sub(r'<.*?>', lambda g: g.group(0).upper(), response)
  dataframe = pd.read_html(html)[0]
  dataframe.to_csv('parsed.csv', index=False)
  return dataframe

def clear_data(url):
  items_name = 'Курс (грн.)'
  dataframe = (parse(url).filter(items=[items_name])[:-1].astype(float) / 10000).to_numpy()[::-1]

  pd.set_option('display.max_rows', None)
  pd.set_option('display.max_columns', None)

  return dataframe

def generate_square_model(n, a, b, c):
    res = np.zeros(n)
    koeffs = [c, b, a]
    for i in range(n):
        for j, koef in enumerate(koeffs):
            res[i] += koef*(i**j)
    return res

def add_noise(dataset, noise):
    n = len(dataset)
    if len(noise) != n: raise ValueError('Datasets must have the same length')

    data = np.zeros(n)
    for i in range(n):
        data[i] = dataset[i] + noise[i]
    return data

def add_anomalies(data, loc, scale, percentage, q):
    result = data.copy()
    n = len(data)
    indexes = np.zeros(int(n * (percentage / 100)))
    indexes = np.vectorize(lambda _: np.random.randint(1, n))(indexes)
    anomaly = np.random.normal(loc, (q * scale), len(indexes))

    for i in range(len(indexes)):
        k=indexes[i]
        result[k] = data[k] + anomaly[i]
    return result
