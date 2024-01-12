import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

def clusterization(dataset):
  plt.scatter(dataset[:, 0], dataset[:, 1], s=50)
  plt.show()

  kmeans = KMeans(n_clusters=3)
  kmeans.fit(dataset)
  colors = kmeans.predict(dataset)
  centers = kmeans.cluster_centers_

  plt.scatter(dataset[:, 0], dataset[:, 1], c=colors, s=50, cmap='viridis')
  plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5, marker='x')
  plt.show()

  return centers

if __name__ == '__main__':
  # First dataset
  dataset, _ = make_blobs(n_samples=600, cluster_std=1.5)
  clusterization(dataset)

  # Second dataset
  dataset, _ = make_blobs(n_samples=400, cluster_std=4.2)
  clusterization(dataset)

  # Third dataset
  expected_centers = [[4, -1], [5, 3], [9, 0]]
  dataset, _ = make_blobs(n_samples=750, cluster_std=1.2, centers=expected_centers)
  actual_centers = clusterization(dataset)
  
  print(f'Expected: {expected_centers}\nActual: {actual_centers}')
