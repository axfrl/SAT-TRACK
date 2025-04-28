import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein, bottleneck
import time
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
import time

# Initialiser le mode interactif de matplotlib
plt.ion()

def compute_persistence_diagrams(vertices_list, homology_dimensions=[0, 1], max_points=1000):
    """
    Calcule les diagrammes de persistance pour une liste de nuages de points.
    
    Args:
        vertices_list (dict): Dictionnaire {id: joints_3d} où joints_3d est un tableau NumPy de forme (n_points, 3).
        homology_dimensions (list): Dimensions homologiques à calculer (ex: [0, 1] pour composantes et boucles).
        max_points (int): Nombre maximum de points par nuage pour limiter le coût de calcul.
    
    Returns:
        tuple: (diagrams, id_list) où diagrams est une liste de diagrammes de persistance et id_list est la liste des IDs.
    """
    # Vérifier que la liste n'est pas vide
    if not vertices_list:
        return [], []
    
    # Sous-échantillonnage si nécessaire pour limiter le coût
    sampled_vertices = []
    id_list = []
    for id, joints_3d in vertices_list.items():
        id_list.append(id)
        if joints_3d.shape[0] > max_points:
            indices = np.random.choice(joints_3d.shape[0], max_points, replace=False)
            sampled_vertices.append(joints_3d[indices])
        else:
            sampled_vertices.append(joints_3d)
    
    # Convertir en format attendu par giotto-tda (tableau 3D)
    vertices_array = np.array(sampled_vertices)
    
    # Calculer les diagrammes de persistance
    VR = VietorisRipsPersistence(homology_dimensions=homology_dimensions, n_jobs=-1)
    diagrams = VR.fit_transform(vertices_array)
    
    return diagrams, id_list

def compute_distance_matrix(diagrams, distance_type="wasserstein"):
    """
    Calcule la matrice de distances entre les diagrammes de persistance.
    
    Args:
        diagrams (list): Liste de diagrammes de persistance, chaque élément est une liste de dimensions (par ex : [H0, H1]).
        distance_type (str): Type de distance à utiliser ("wasserstein" ou "bottleneck").
    
    Returns:
        ndarray: Matrice de distances (n_diagrams x n_diagrams).
    """
    if len(diagrams) < 2:
        return np.zeros((0, 0))
    
    n_diagrams = len(diagrams)
    distance_matrix = np.zeros((n_diagrams, n_diagrams))
    
    for i in range(n_diagrams):
        for j in range(i + 1, n_diagrams):
            distances = []
            for dim_i, dim_j in zip(diagrams[i], diagrams[j]):
                if distance_type.lower() == "wasserstein":
                    if dim_i.size == 0 and dim_j.size == 0:
                        distances.append(0)
                    else:
                        distances.append(wasserstein(dim_i, dim_j, matching=False))
                
                elif distance_type.lower() == "bottleneck":
                    if dim_i.size == 0 and dim_j.size == 0:
                        distances.append(0)
                    else:
                        distances.append(bottleneck(dim_i, dim_j))
                
                else:
                    raise ValueError("distance_type doit être 'wasserstein' ou 'bottleneck'")
            
            distance = np.sum(distances)  # Somme des distances sur toutes les dimensions
            distance_matrix[i, j] = distance
            distance_matrix[j, i] = distance
    
    return distance_matrix

def compute_cross_distance_matrix(diagrams_D, diagrams_P, distance_type="wasserstein"):
    """
    Calcule la matrice des distances entre deux ensembles de diagrammes.

    Args:
        diagrams_D (list of np.ndarray): Liste de diagrammes D_i.
        diagrams_P (list of np.ndarray): Liste de diagrammes P_j.
        distance_type (str): Type de distance ("wasserstein" ou "bottleneck").

    Returns:
        np.ndarray: Matrice M où M[i, j] = distance(Di, Pj).
    """
    n_D = len(diagrams_D)
    n_P = len(diagrams_P)
    
    M = np.zeros((n_D, n_P))
    
    for i in range(n_D):
        for j in range(n_P):
            if distance_type == "wasserstein":
                dist = wasserstein(diagrams_D[i], diagrams_P[j])
            elif distance_type == "bottleneck":
                dist = bottleneck(diagrams_D[i], diagrams_P[j])
            else:
                raise ValueError(f"Distance type '{distance_type}' non supporté.")
            M[i, j] = dist
    return M