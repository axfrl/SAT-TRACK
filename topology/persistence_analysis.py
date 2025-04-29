import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import wasserstein, bottleneck
import time
import numpy as np
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
import torch

def compute_persistence_diagrams(joint_dict, max_dimension=2):
    """
    Calcule les diagrammes de persistance pour chaque ensemble de joints 3D dans un dictionnaire
    en utilisant Giotto-TDA. Les valeurs du dictionnaire peuvent être des tenseurs PyTorch ou des tableaux NumPy.

    Args:
        joint_dict (dict): Dictionnaire avec des clés entières (IDs) et des valeurs torch.Tensor ou NumPy
                          de forme (num_joints, 3) représentant des joints 3D.
        max_dimension (int): Dimension maximale pour la persistance (par défaut 2).

    Returns:
        dict: Dictionnaire avec les mêmes clés et les diagrammes de persistance comme valeurs.
              Chaque diagramme est un tableau NumPy de forme (n_points, 3) avec [birth, death, dim].
    """
    # Initialise le calculateur de persistance avec Vietoris-Rips
    VR = VietorisRipsPersistence(
        homology_dimensions=list(range(max_dimension + 1)),  # Dimensions 0, 1, ..., max_dimension
        metric="euclidean",
        n_jobs=1  # Peut être ajusté pour parallélisme CPU
    )

    persistence_diagrams = {}

    for id, joints in joint_dict.items():
        # Convertit en tableau NumPy si c'est un tenseur PyTorch
        if isinstance(joints, torch.Tensor):
            joints = joints.cpu().numpy()  # Transfert GPU->CPU et conversion en NumPy
        elif not isinstance(joints, np.ndarray):
            raise ValueError(f"Les données pour l'ID {id} ne sont ni un tenseur PyTorch ni un tableau NumPy, type trouvé : {type(joints)}")

        # Vérifie la forme (num_joints, 3)
        if joints.ndim != 2 or joints.shape[1] != 3:
            raise ValueError(f"Forme invalide pour l'ID {id} : {joints.shape}, attendu (num_joints, 3)")

        # gtda attend une liste de nuages de points, donc reshape en (1, num_joints, 3)
        joints = joints[None, :, :]

        # Calcule le diagramme de persistance
        diagram = VR.fit_transform(joints)[0]  # shape (n_points, 3) : [birth, death, dim]

        # Stocke le diagramme dans le dictionnaire de sortie
        persistence_diagrams[id] = diagram

    return persistence_diagrams

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

def compute_cross_distance_matrix(diagrams_D, diagrams_P, distance_type="wasserstein", epsilon=0.0):
    """
    Calcule la matrice des distances entre deux ensembles de diagrammes, en ne prenant que les points
    dont la naissance dépasse un seuil epsilon.

    Args:
        diagrams_D (list of np.ndarray): Liste de diagrammes D_i.
        diagrams_P (list of np.ndarray): Liste de diagrammes P_j.
        distance_type (str): Type de distance ("wasserstein" ou "bottleneck").
        epsilon (float): Seuil pour la naissance des points (birth > epsilon, par défaut : 0.0).

    Returns:
        np.ndarray: Matrice M où M[i, j] = distance(Di, Pj) après filtrage.
    """
    n_D = len(diagrams_D)
    n_P = len(diagrams_P)
    
    M = np.zeros((n_D, n_P))
    
    # Filtrer les diagrammes pour ne garder que les points avec birth > epsilon
    filtered_diagrams_D = [
        diag[diag[:, 0] > epsilon] if diag.size > 0 else np.empty((0, diag.shape[1]))
        for diag in diagrams_D
    ]
    filtered_diagrams_P = [
        diag[diag[:, 0] > epsilon] if diag.size > 0 else np.empty((0, diag.shape[1]))
        for diag in diagrams_P
    ]
    
    for i in range(n_D):
        for j in range(n_P):
            if filtered_diagrams_D[i].size == 0 or filtered_diagrams_P[j].size == 0:
                # Si l'un des diagrammes est vide après filtrage, assigner une distance infinie ou 0
                M[i, j] = np.inf if distance_type == "bottleneck" else 0.0
                continue
                
            if distance_type == "wasserstein":
                dist = wasserstein(filtered_diagrams_D[i], filtered_diagrams_P[j])
            elif distance_type == "bottleneck":
                dist = bottleneck(filtered_diagrams_D[i], filtered_diagrams_P[j])
            else:
                raise ValueError(f"Distance type '{distance_type}' non supporté.")
            M[i, j] = dist
    
    return M