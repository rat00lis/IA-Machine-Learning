import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

# Datos de los puntos y configuración inicial
points = {'A': [2, 10], 'B': [2, 5], 'C': [8, 4], 'D': [5, 8], 'E': [7, 5], 'F': [6, 4], 'G': [1, 2], 'H': [4, 9]}
data = np.array(list(points.values()))
labels = np.array(list(points.keys()))
initial_centroids = {'A': points['A'], 'D': points['D'], 'G': points['G']}

# Funciones para calcular nuevos centroides y asignar puntos a clusters
def calculate_new_centroids(clusters, data):
    return {cluster: np.mean(data[clusters[cluster]], axis=0) for cluster in clusters}

def assign_clusters(centroids, data):
    distances = cdist(data, list(centroids.values()))
    cluster_assignments = np.argmin(distances, axis=1)
    return {label: np.where(cluster_assignments == i)[0].tolist() for i, label in enumerate(centroids)}

# Inicialización y ejecución de iteraciones
centroids = initial_centroids
iterations = 0
results = []

while iterations < 10:
    clusters = assign_clusters(centroids, data)
    new_centroids = calculate_new_centroids(clusters, data)
    
    # Guardar resultados para CSV
    for label, points_indices in clusters.items():
        for idx in points_indices:
            results.append({'Punto': labels[idx], 'Coordenadas': data[idx], 'Cluster': label, 'Iteración': iterations + 1})
    
    if np.allclose(list(centroids.values()), list(new_centroids.values())):
        break
    
    centroids = new_centroids
    iterations += 1

# Guardar resultados en CSV
df_results = pd.DataFrame(results)
df_results.to_csv('ejercicio1.csv', sep=';', index=False)

# Preparar y guardar el gráfico
plt.figure(figsize=(10, 10))  # Cambiar el tamaño de la figura a 10x10

# Definir intervalos más pequeños para el grid, extendiendo a 11x11
plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 12, 1))

colors = ['red', 'green', 'blue']  # Colores para los clusters

for i, (cluster_label, centroid_coords) in enumerate(centroids.items()):
    cluster_points_indices = clusters[cluster_label]
    cluster_points = data[cluster_points_indices]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {cluster_label}')
    plt.scatter(centroid_coords[0], centroid_coords[1], color=colors[i], marker='x', s=100)
    # Ajustar la posición del texto para que esté arriba del centroide con un margen
    plt.text(centroid_coords[0], centroid_coords[1] + 0.3, f'({centroid_coords[0]:.2f}, {centroid_coords[1]:.2f})', color='black', ha='center')

plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.legend()
plt.title('Clusters K-means')

plt.grid(True, which='both', linestyle='-', linewidth=0.5)  # Hacer el grid más fino

plt.xlim(0, 11)  # Definir los límites del eje X de 0 a 11
plt.ylim(0, 11)  # Definir los límites del eje Y de 0 a 11

plt.savefig('ejercicio1.png')

