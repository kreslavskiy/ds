import geopandas as gpd
import matplotlib.pyplot as plt

def map_plot_exemple(filename):
  df = gpd.read_file(filename)
  print('df.head():', df.head())

  _, ax = plt.subplots(1, 1, figsize=(8, 5))
  df.plot(ax=ax)
  plt.show()

def map_filtr_plot_exemple_Bulgaria(filname):
  df = gpd.read_file(filname)

  def myfilter(x):
    return x in tc

  tc = ['spring']
  df['delete'] = df['fclass'].apply(lambda x: myfilter(x))
  df = df[df['delete']]

  _, ax = plt.subplots(1, 1, figsize=(8, 5))
  df.plot(ax=ax, column='fclass', legend=True, cmap='viridis')
  plt.show()

if __name__ == '__main__':
  map_plot_exemple('Bulgaria/gis_osm_natural_free_1.shp')
  map_filtr_plot_exemple_Bulgaria('Bulgaria/gis_osm_natural_free_1.shp')
