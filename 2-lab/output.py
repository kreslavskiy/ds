import math as mt
import numpy as np
import matplotlib.pyplot as plt

from utils import MNK

def print_MNK_characteristics(data, title):
    num_iterations = len(data)
    y_output = MNK(data) 
    result_data = np.zeros((num_iterations))
    for i in range(num_iterations):
        result_data[i] = data[i] - y_output[i, 0]

    median = np.median(result_data)
    var = np.var(result_data)
    meanSquare = mt.sqrt(var)

    print('------------', title ,'-------------')
    print('Мат. очікуання =', median)
    print('Дисперсія =', var)
    print('Середньоквадратичне відхилення =', meanSquare)
    print('-----------------------------------------------------\n\n')

    return y_output

def print_predict_MNK_characteristics(koef, data, title):
    num_iterations = len(data)
    y_output = MNK(data) 
    result_data = np.zeros((num_iterations))
    for i in range(num_iterations):
        result_data[i] = data[i] - y_output[i, 0]

    median = np.median(result_data)
    var = np.var(result_data)
    meanSquare = mt.sqrt(var)
    
    scvS_extrapol = meanSquare * koef
    print('------------', title ,'-------------')
    print('Кількість елементів вибірки =', len(data))
    print('Мат. очікування =', median)
    print('Дисперсія =', var)
    print('Середньоквадратичне відхилення =', meanSquare)
    print('Довірчий інтервал прогнозованих значень =', scvS_extrapol)
    print('-----------------------------------------------------\n\n')

def plot_data(label, type, *args):
    if type == 1:
        for data in args: plt.plot(data)
    elif type == 2: plt.hist(args[0], bins=20)

    plt.ylabel(label)
    plt.show()

def r2_score(smoothed, y_poutput, title):
    output_length = len(y_poutput)
    numerator = 0
    denominator_1 = sum(smoothed)
    denominator_2 =  0
    for i in range(output_length):
        numerator += (smoothed[i] - y_poutput[i, 0]) ** 2
        denominator_2 += (smoothed[i] - (denominator_1 / output_length)) ** 2

    r2_score = 1 - (numerator / denominator_2)
    print('------------', title ,'-------------')
    print('Кількість елементів вибірки =', output_length)
    print('Коефіцієнт детермінації =', r2_score)
    print('-----------------------------------------------------\n\n')

    return r2_score
