import math as mt
import numpy as np
import matplotlib.pyplot as plt

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
    a, b, c = C[2,0], C[1,0], C[0,0]
    print('Регресійна модель:')
    print('y(t) = ', C[0,0], ' + ', C[1,0], ' * t', ' + ', C[2,0], ' * t^2')
    return y_output, a, b, c

def print_MNK_characteristics(data, title):
    num_iterations = len(data)
    y_output, a, b, c = MNK(data) 
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

    return meanSquare, median, var, a, b, c

def plot_data(label, type, *args):
    if type == 1:
        for data in args: plt.plot(data)
    elif type == 2: plt.hist(args[0], bins=20)

    plt.ylabel(label)
    plt.show()
