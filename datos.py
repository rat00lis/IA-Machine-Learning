import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import numpy as np

# Datos de los puntos y configuración inicial
points = {'A': [2, 10], 'B': [2, 5], 'C': [8, 4], 'D': [5, 8], 'E': [7, 5], 'F': [6, 4], 'G': [1, 2], 'H': [4, 9]}
data = np.array(list(points.values()))
labels = np.array(list(points.keys()))
initial_centroids = {'A': points['A'], 'D': points['D'], 'G': points['G']}
