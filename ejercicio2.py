from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
from datos import data  # Asumiendo que 'data' es un array de numpy con los puntos del ejercicio 1

# Función para dibujar los resultados de DBSCAN
def plot_dbscan(eps, min_samples, iteration):
    # Aplicar DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
    labels = db.labels_
    
    # Dibujar grilla de 10x10 y puntos
    plt.figure(figsize=(10, 10))
    plt.xticks(np.arange(0, 11, 1))
    plt.yticks(np.arange(0, 11, 1))
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    
    # Colores para los clusters, ruido en negro
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            col = [0, 0, 0, 1]  # Color negro para ruido
        
        class_member_mask = (labels == k)
        xy = data[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=14)
    
    plt.title(f'DBSCAN con eps={eps} y min_samples={min_samples}')
    # Guardar el gráfico en lugar de mostrarlo
    plt.savefig(f'ejercicio2_iteracion{iteration}.png')
    plt.close()  # Cerrar la figura para liberar memoria

# Dibujar y guardar DBSCAN para eps=2 y min_samples=2
plot_dbscan(eps=2, min_samples=2, iteration=1)

# Dibujar y guardar DBSCAN para eps=sqrt(10) y min_samples=2
plot_dbscan(eps=np.sqrt(10), min_samples=2, iteration=2)