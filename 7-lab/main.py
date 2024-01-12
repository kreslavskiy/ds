import numpy as np
import pandas as pd
from data_manipulating import show_data, data_sum_sales, region_clustering, predict_MNK

if __name__ == '__main__':
  df = pd.read_excel('Pr_1.xls')
  data = np.transpose(df.iloc[:, 0:6].values)
  district_dict = dict(zip(df.iloc[:, 9].dropna().values, df['Регіон'].dropna().values))

  sales = data_sum_sales(data)

  show_data(sales.values(), title='Продажі по днях')

  cluster = region_clustering(data, district_dict)

  for region in cluster:
    sales_data = cluster[region]['sales'].values()
    MNK_sales = predict_MNK(list(sales_data), int(len(sales_data) * 0.5))

    show_data(MNK_sales, title=f'Прогноз на 6 місяців регіону: {region}')
