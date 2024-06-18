import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np
from datos import *

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
    
    # Generar y guardar el gráfico para la iteración actual
    plt.figure(figsize=(10, 10))
    plt.xticks(np.arange(0, 12, 1))
    plt.yticks(np.arange(0, 12, 1))
    colors = ['red', 'green', 'blue']
    for i, (cluster_label, centroid_coords) in enumerate(centroids.items()):
        cluster_points_indices = clusters[cluster_label]
        cluster_points = data[cluster_points_indices]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color=colors[i], label=f'Cluster {cluster_label}')
        plt.scatter(centroid_coords[0], centroid_coords[1], color=colors[i], marker='x', s=100)
        plt.text(centroid_coords[0], centroid_coords[1] + 0.3, f'({centroid_coords[0]:.2f}, {centroid_coords[1]:.2f})', color='black', ha='center')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.legend()
    plt.title(f'Clusters K-means Iteración {iterations + 1}')
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xlim(0, 11)
    plt.ylim(0, 11)
    plt.savefig(f'ejercicio1_iteracion{iterations + 1}.png')
    plt.close()
    
    if np.allclose(list(centroids.values()), list(new_centroids.values())):
        break
    
    centroids = new_centroids
    iterations += 1

# Guardar resultados en CSV
df_results = pd.DataFrame(results)
df_results.to_csv('ejercicio1.csv', sep=';', index=False)