import numpy as np
import math as mt
from output import print_MNK_characteristics, plot_data, r2_score, print_predict_MNK_characteristics
from utils import clear_data, ABF, predict_MNK, smooth_data

n = 10_000
n_wind = 3
extrapolation_koeff = 0.5

data = clear_data('https://index.minfin.com.ua/ua/exchange/nbu/curr/ils/')
print_MNK_characteristics(data, 'Курс шекеля до гривні')
plot_data('Курс шекеля до гривні', 1, data)

smoothed = smooth_data(data, n_wind)
plot_data('Згладжено', 1, smoothed)
y_output = print_MNK_characteristics(smoothed, 'Згладжено')
plot_data('Модель', 1, smoothed, y_output)
r2_score(smoothed, y_output, 'Модель')

koef = mt.ceil(len(smoothed) * extrapolation_koeff)
predict = predict_MNK(smoothed, koef)
plot_data('Прогонозовано', 1, smoothed, predict)
print_predict_MNK_characteristics(koef, predict, 'Прогонозовано')

abf = ABF(smoothed)
r2_score(smoothed, abf, 'alfa-beta фільтр')
plot_data('alfa-beta фільтр', 1, smoothed, abf)
