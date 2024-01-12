import pandas as pd
import numpy as np

def generate_matrix(dataframe: pd.DataFrame):
  rows = dataframe.shape[0] # rows last index (if 10 rows then 9) ===== 9
  columns = dataframe.shape[1] # columns number ======= 13

  matrix = np.zeros((rows, columns-3))
  titles = dataframe.columns
  for i in range(3, (columns), 1):
    column_matrix = dataframe[titles[i]]
    for j in range(len(column_matrix)):
      matrix[j, (i-3)] = column_matrix[j]
  return matrix

def get_row_by_index(matrix, row_index):
  matrix_shape = np.shape(matrix)
  row = np.zeros((matrix_shape[1]))
  for j in range(matrix_shape[1]):
    row[j] =  matrix[row_index, j]
  return row

def find_optimal():
  data = pd.read_excel('dataset.xlsx')
  weights = list(data['Weight'])
  criteria_types = list(data['Criteria'])
  matrix = generate_matrix(data)

  matrix_shape = np.shape(matrix) # 9 criterias at index 0, 10 cars at index 1
  Integro = np.zeros((matrix_shape[1]))

  criterias = []
  criterias_normalized = []
  weights_normalized = []
  for i in range(matrix_shape[0]):
    criteria = get_row_by_index(matrix, i)
    criterias.append(criteria)
    criterias_normalized.append(np.zeros((matrix_shape[1])))
    weights_normalized.append(weights[i] / sum(weights))

  for i in range(matrix_shape[1]):
    for j in range(matrix_shape[0]):
      if criteria_types[j] == 'min':
        criterias_normalized[j][i] = criterias[j][i] / sum(list(criterias[j]))
      elif criteria_types[j] == 'max':
        criterias_normalized[j][i] = (1/ criterias[j][i]) / sum(list(1/criterias[j]))

    for k in range(matrix_shape[0]):
      Integro[i] += (weights_normalized[k] * (1 - criterias_normalized[k][i]) ** (-1))
  
  optimal_index = list(Integro).index(min(Integro))
  integro_sorted = sorted(Integro)

  print(data)
  print('\n\nOptimal electric car:', data.columns[optimal_index + 3])
  print('\nRated by integrated criteria:')
  for i in range(len(integro_sorted)):
    element_index = list(Integro).index(integro_sorted[i])
    print(f'{i+1}. {data.columns[element_index + 3]}: {integro_sorted[i]}')

  return

if __name__ == '__main__':
  find_optimal()