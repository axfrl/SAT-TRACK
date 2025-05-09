import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Données des articulations
joints = [
    [0.931, -0.042,  4.252],
    [0.987,  0.061,  4.224],
    [0.864,  0.050,  4.284],
    [0.952, -0.154,  4.281],
    [0.991,  0.447,  4.143],
    [0.724,  0.423,  4.223],
    [0.954, -0.299,  4.248],
    [1.035,  0.839,  4.280],
    [0.727,  0.839,  4.268],
    [0.939, -0.345,  4.207],
    [1.012,  0.913,  4.166],
    [0.634,  0.889,  4.177],
    [0.950, -0.566,  4.166],
    [1.020, -0.463,  4.155],
    [0.883, -0.484,  4.235],
    [0.914, -0.608,  4.099],
    [1.090, -0.439,  4.097],
    [0.786, -0.496,  4.260],
    [1.132, -0.173,  4.134],
    [0.695, -0.255,  4.316],
    [0.947, -0.034,  4.028],
    [0.645, -0.091,  4.116],
    [0.886,  0.014,  3.989],
    [0.634, -0.043,  4.044],
    [0.835, -0.629,  4.016],
    [0.833, -0.681,  4.049],
    [0.880, -0.670,  4.003],
    [0.872, -0.695,  4.145],
    [0.977, -0.669,  4.042],
    [0.955,  0.921,  4.111],
    [1.039,  0.925,  4.127],
    [1.065,  0.865,  4.338],
    [0.623,  0.890,  4.098],
    [0.588,  0.898,  4.178],
    [0.753,  0.884,  4.321],
    [0.809, -0.055,  4.005],
    [0.789,  0.017,  3.951],
    [0.800,  0.051,  3.938],
    [0.826,  0.069,  3.945],
    [0.866,  0.082,  3.965],
    [0.716, -0.095,  3.993],
    [0.653, -0.022,  3.947],
    [0.628,  0.005,  3.953],
    [0.612,  0.019,  3.980],
    [0.603,  0.027,  4.027]
]

# Extraire x, y, z
x = np.array([j[0] for j in joints])
y = np.array([j[1] for j in joints])
z = np.array([j[2] for j in joints])
indices = range(len(joints))

# Normalisation des coordonnées
# 1. Recentrer autour de zéro
x_centered = x - np.mean(x)
y_centered = y - np.mean(y)
z_centered = z - np.mean(z)

# 2. Mettre à l'échelle pour que la plage max soit dans [-1, 1]
max_range = np.max([np.ptp(x_centered), np.ptp(y_centered), np.ptp(z_centered)])
if max_range > 0:  # Éviter la division par zéro
    x_normalized = x_centered / (max_range / 2)
    y_normalized = y_centered / (max_range / 2)
    z_normalized = z_centered / (max_range / 2)
else:
    x_normalized = x_centered
    y_normalized = y_centered
    z_normalized = z_centered

# Créer la figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Tracer les points
ax.scatter(x_normalized, y_normalized, z_normalized, c='r', marker='o')

# Ajouter les indices comme étiquettes
for i, (xi, yi, zi) in enumerate(zip(x_normalized, y_normalized, z_normalized)):
    ax.text(xi, yi, zi, f'{indices[i]}', size=10, zorder=1, color='k')

# Définir les labels des axes
ax.set_xlabel('X (normalisé)')
ax.set_ylabel('Y (normalisé)')
ax.set_zlabel('Z (normalisé)')

# Forcer une échelle normalisée : même plage pour tous les axes
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

# S'assurer que les axes ont la même échelle
ax.set_box_aspect([1, 1, 1])  # Échelle égale pour X, Y, Z

# Afficher le graphe
plt.show()