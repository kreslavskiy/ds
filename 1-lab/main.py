import numpy as np
from output import print_MNK_characteristics, plot_data
from utils import add_noise, add_anomalies, clear_data, generate_square_model

n = 10_000

data = clear_data('https://index.minfin.com.ua/ua/exchange/nbu/curr/ils/')
meanSquare, median, var, a, b, c = print_MNK_characteristics(data, 'Курс шекеля до гривні')
plot_data('Курс шекеля до гривні', 1, data)

normal_noise = np.random.normal(0, meanSquare, n)
print_MNK_characteristics(normal_noise, 'Нормальний шум')
plot_data('Нормальний шум', 2, normal_noise)

square_model = generate_square_model(n, a/((n/len(data))**2), b/(n/len(data)), c)
plot_data('Квадратична модель', 1, square_model)

data_with_noise = add_noise(square_model, normal_noise)
print_MNK_characteristics(data_with_noise, 'Модель із нормальним шумом')
plot_data('Модель із нормальним шумом', 1, data_with_noise, square_model)

data_with_noise_and_anomalies = add_anomalies(data_with_noise, 0, 1.5, 5, 1)
print_MNK_characteristics(data_with_noise_and_anomalies, 'Модель із нормальним шумом та аномальними вимірами')
plot_data('Модель із нормальним шумом та аномальними вимірами', 1, data_with_noise_and_anomalies, square_model)
