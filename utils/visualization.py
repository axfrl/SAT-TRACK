import os
import io
import torch
import numpy as np
from termcolor import colored
import time
try:
    # os.environ["PYOPENGL_PLATFORM"] = "osmesa"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    import pyrender
except:
    print(colored('pyrender is not correctly imported.', 'red'))
import matplotlib
from matplotlib import colors
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import cv2
import trimesh
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as R
import torchvision
from .transforms import adjust_colors
from gtda.homology import VietorisRipsPersistence
from matplotlib.backends.backend_agg import FigureCanvasAgg

BASE_COLORS = np.loadtxt(os.path.abspath(os.path.join(__file__, "../colors.txt")), skiprows=0)/255.
BASE_COLORS = adjust_colors(BASE_COLORS,
                            saturation_threshold = 0.3, 
                            brightness_threshold = 0.8)


def get_colors_rgb(size):
    # np.random.seed(131)
    return BASE_COLORS[np.random.choice(BASE_COLORS.shape[0], size=size, replace=False)]

def tensor_to_BGR(img_tensor):
    img = img_tensor.numpy()*255
    img = img.astype(np.uint8).transpose((1,2,0))[:,:,::-1].copy()
    return img

def pad_img(img, pad_size = None, pad_color_offset = 127):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    if pad_size is None:
        pad_size = max(img.shape[0],img.shape[1])

    pad = np.zeros((pad_size,pad_size,img.shape[-1]), dtype=img.dtype) + pad_color_offset
    pad[:img.shape[0], :img.shape[1]] = img.copy()
    return pad


def vis_scale_img(img, scale_map, conf_thresh = 0.3, patch_size=14):
    cmap = plt.get_cmap('coolwarm')

    vis_map = np.zeros((scale_map.shape[0]*patch_size, scale_map.shape[1]*patch_size, 3), dtype=np.uint8)
    loc_i, loc_j = torch.where(scale_map[:,:,0] > conf_thresh)
    for (i, j) in zip(loc_i, loc_j):
        scale = round(math.sqrt(scale_map[i,j,1].item()),2)
        vis_map[i*patch_size: (i+1)*patch_size, j*patch_size: (j+1)*patch_size] = (np.array(cmap(1-scale)[:3][::-1])*255).astype(np.uint8)
    vis_map = pad_img(vis_map, pad_color_offset=0)
    img = pad_img(img)
    # print(img.shape, vis_map.shape)
    assert img.shape == vis_map.shape

    white_img = 0.6*img + 0.4*np.array((255,255,255))

    valid_mask = (vis_map > 0)
    visible_weight = 0.8
    img = vis_map * valid_mask * visible_weight +\
                img * valid_mask * (1-visible_weight)+\
                white_img * (1-valid_mask)

    # draw patches
    loc_i, loc_j = torch.where(scale_map[:,:,0]+1)
    for (i, j) in zip(loc_i.tolist(), loc_j.tolist()):
        cv2.rectangle(img, (j*patch_size, i*patch_size), ((j+1)*patch_size, (i+1)*patch_size),
                        color=(255,255,255), thickness = 2 )

    return img

def vis_meshes_img(img, verts, smpl_faces, cam_intrinsics, colors = None, padding = True):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())

    if padding:
        pad_size = max(img.shape[0],img.shape[1])
        img = pad_img(img, pad_size)

    if colors is not None:
        assert len(colors) == len(verts)

    if len(cam_intrinsics.flatten()) == 9:
        cam_intrinsics = cam_intrinsics.reshape(3,3)
        rgb, depth = render_mesh(img.shape[0],img.shape[1],verts,smpl_faces,cam_intrinsics,colors)
        valid_mask = (depth > 0)[:,:,None] 
        visible_weight = 1.
        rendered_img = rgb[:,:,::-1] * valid_mask * visible_weight +\
                        img * valid_mask * (1-visible_weight)+\
                        img * (1-valid_mask)
    else:
        rendered_img = img
        for i, cam_int in enumerate(cam_intrinsics):
            rgb, depth = render_mesh(img.shape[0],img.shape[1],[verts[i]],smpl_faces,cam_int,colors)
            valid_mask = (depth > 0)[:,:,None] 
            visible_weight = 0.8
            rendered_img = rgb[:,:,::-1] * valid_mask * visible_weight +\
                            rendered_img * valid_mask * (1-visible_weight)+\
                            rendered_img * (1-valid_mask)
    rendered_img = rendered_img.astype(np.uint8)

    return rendered_img


def vis_joints_img(img, j2ds):
    pass

def vis_sat(img, input_size, patch_size, sat_dict, bid, padding=True):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())

    assert max(img.shape[0], img.shape[1]) == input_size
    if padding:
        img = pad_img(img, input_size)

    # visualize patches
    pos_y, pos_x = sat_dict['pos_y'][bid], sat_dict['pos_x'][bid]
    pos_y = (pos_y * input_size).detach().int().cpu().numpy()
    pos_x = (pos_x * input_size).detach().int().cpu().numpy()

    lvls = sat_dict['lvl']
    if lvls is None:
        lvl = np.zeros(len(pos_x),dtype=int)
    else:
        lvl = lvls[bid].detach().int().cpu().numpy()

    for (cx, cy, l) in zip(pos_x, pos_y, lvl):
        if l == 0:
            half_patch = patch_size//2
            # color = (139, 97, 233)
            color = (173,178,241)
        elif l == 1:
            half_patch = patch_size
            # color = (246, 222, 118)
            color = (239,198,175)
        elif l >= 2:
            half_patch = patch_size*(2**(l-1))
            # color = (0,0,0)
            color = (255, 255, 255)
        else:
            raise NotImplementedError

        x1, x2 = cx - half_patch, cx + half_patch
        y1, y2 = cy - half_patch, cy + half_patch

        if l>0:
            k = 7*l
            img[y1:y2,x1:x2] = 0.5*cv2.blur(img[y1:y2,x1:x2].copy(),(k,k)) + 0.5*np.array(color)
        else:
            # pass
            img[y1:y2,x1:x2] = 0.5*img[y1:y2,x1:x2].copy() + 0.5*np.array(color)


    for (cx, cy, l) in zip(pos_x, pos_y, lvl):
        if l == 0:
            half_patch = patch_size//2
            color = (139, 97, 233)
        elif l == 1:
            half_patch = patch_size
            color = (246, 222, 118)
        elif l >= 2:
            half_patch = patch_size*(2**(l-1))
            color = (255, 255, 255)
        else:
            raise NotImplementedError
        
        x1, x2 = cx - half_patch, cx + half_patch
        y1, y2 = cy - half_patch, cy + half_patch

        cv2.rectangle(img, (x1, y1), (x2, y2),
            color=(255,255,255), thickness = 2 )
        

    return img

def vis_boxes(img, boxes, padding=True, color = (0,0,255)):
    if not isinstance(img, np.ndarray):
        img = tensor_to_BGR(img.detach().cpu())
    if padding:
        pad_size = max(img.shape[0],img.shape[1])
        img = pad_img(img, pad_size)
    
    for bbox in boxes:
        bbox = bbox.int().tolist()
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
            color=color, thickness = 2 )
    
    return img


def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, transparent=False, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def render_mesh(height, width, meshes, face, cam_intrinsics, colors = None):
    
    # renderer
    scene = pyrender.Scene(ambient_light=(0.3, 0.3, 0.3))
    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height, point_size=1.0)

    # light
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)
    light_pose = np.eye(4)
    light_pose[:3, 3] = np.array([0, -1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([0, 1, 1])
    scene.add(light, pose=light_pose)
    light_pose[:3, 3] = np.array([1, 1, 2])
    scene.add(light, pose=light_pose)

    # mesh
    if colors is None:
        colors = get_colors_rgb(len(meshes))

    for i, mesh in enumerate(meshes):
        mesh = trimesh.Trimesh(mesh, face)
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        material = pyrender.MetallicRoughnessMaterial(metallicFactor=0.0, alphaMode='OPAQUE', baseColorFactor=(*colors[i], 1.0))
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        scene.add(mesh, f'mesh_{i}')


    # camera
    f=np.array([cam_intrinsics[0,0],cam_intrinsics[1,1]])
    c=cam_intrinsics[0:2,2]
    camera = pyrender.camera.IntrinsicsCamera(fx=f[0], fy=f[1], cx=c[0], cy=c[1])
    scene.add(camera)

    # render
    rgb, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    rgb = rgb[:,:,:3].astype(np.float32)
    renderer.delete()
    return rgb, depth

RADIUS_VERT = 1
KERNEL_VERT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*RADIUS_VERT+1, 2*RADIUS_VERT+1))

RADIUS_JOINT = 10
KERNEL_JOINT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*RADIUS_JOINT+1, 2*RADIUS_JOINT+1))

ORIG_W, ORIG_H = 1280, 720
PADDED_H = ORIG_W
PAD_TOP = (PADDED_H - ORIG_H) // 2
SCALE_FACTOR = 1288 / PADDED_H  # input_size / padded_h
EXPECTED_CY = 1288 / 2

def vis_vertices_img(frame, verts_cam_list, cam_intrinsics, frame_size):
    frame_h, frame_w = frame.shape[:2]
    # 1) concat tous les points en numpy directement 
    all_verts = np.vstack([
        v.squeeze(0) if v.ndim == 3 else v
        for v in verts_cam_list
    ]).astype(np.float32)  # (N,3)

    # 2) projection homogène en numpy
    K = cam_intrinsics.cpu().numpy().astype(np.float32)
    verts_homo = all_verts @ K.T                # (N,3)
    pts2d = verts_homo[:, :2] / (verts_homo[:, 2:] + 1e-6)  # (N,2)

    # 3) ajustement de l’offset en y
    predicted_cy = K[1,2]
    cy_offset = (EXPECTED_CY - predicted_cy) * (frame_h / (1288 - 2 * PAD_TOP * SCALE_FACTOR))

    # 4) passage à l’échelle vers la taille de la frame
    xs = pts2d[:,0] * (frame_w / 1288)
    ys = (pts2d[:,1] - PAD_TOP * SCALE_FACTOR) * (frame_h / (1288 - 2 * PAD_TOP * SCALE_FACTOR))
    ys += cy_offset

    # 5) clipping et entiers
    ix = np.clip(xs, 0, frame_w-1).astype(np.int32)
    iy = np.clip(ys, 0, frame_h-1).astype(np.int32)

    # 6) masque binaire + dilatation pour faire des “cercles”
    mask = np.zeros((frame_h, frame_w), np.uint8)
    mask[iy, ix] = 255
    mask = cv2.dilate(mask, KERNEL_VERT, iterations=1)

    # 7) application du rendu vert
    frame[mask==255] = (0,255,0)
    return frame

def vis_vertices_img_with_tracked_pose(frame, tracked_poses, cam_intrins, frame_size, colors, input_size=1288, point_radius=8):
    frame = frame.copy()
    frame_w, frame_h = frame_size
    K = cam_intrins.float()
    device = K.device

    padded_h = frame_w
    pad_top = (padded_h - frame_h) // 2 if padded_h > frame_h else 0
    scale_factor = input_size / padded_h if padded_h > 0 else 1.0
    expected_cy = input_size / 2
    predicted_cy = K[1, 2]
    cy_offset = (expected_cy - predicted_cy) * (frame_h / (input_size - 2 * pad_top * scale_factor + 1e-6))

    # D'abord, on prépare un dictionnaire {color: mask}
    color_to_mask = {}

    for id, joints_3d in tracked_poses.items():
        if not isinstance(joints_3d, np.ndarray) or joints_3d.shape[-1] != 3:
            print(f"Joints ID {id} rejetés : Type={type(joints_3d)}, Shape={getattr(joints_3d, 'shape', 'N/A')}")
            if isinstance(joints_3d, torch.Tensor):
                joints_3d = joints_3d.cpu().numpy()
            else:
                continue

        joints = torch.from_numpy(joints_3d).float().to(device)
        if joints.dim() == 3:
            joints = joints.squeeze(0)

        joints_homo = joints @ K.T
        joints_2d_resized = joints_homo[:, :2] / (joints_homo[:, 2:3] + 1e-6)

        joints_2d_adjusted = joints_2d_resized.clone()
        joints_2d_adjusted[:, 0] = joints_2d_resized[:, 0] * (frame_w / input_size)
        joints_2d_adjusted[:, 1] = (joints_2d_resized[:, 1] - pad_top * scale_factor) * \
                                  (frame_h / (input_size - 2 * pad_top * scale_factor + 1e-6)) + cy_offset

        coords = joints_2d_adjusted.cpu().numpy()
        coords = np.clip(coords, [0, 0], [frame_w - 1, frame_h - 1]).astype(np.int32)

        color = colors.get(id, (255, 255, 255))

        # Initie un masque pour cette couleur si pas encore créé
        if color not in color_to_mask:
            color_to_mask[color] = np.zeros((frame_h, frame_w), np.uint8)

        # Place tous les points pour cet ID sur son masque
        ix = coords[:, 0]
        iy = coords[:, 1]
        color_to_mask[color][iy, ix] = 255

    # Ensuite pour chaque couleur, on dilate le masque puis on l'applique
    for color, mask in color_to_mask.items():
        dilated_mask = cv2.dilate(mask, KERNEL_JOINT, iterations=1)
        frame[dilated_mask == 255] = color

    return frame

def add_left_border_to_frame(frame, border_width=300, border_color=(0, 0, 0)):
    """
    Ajoute un bandereau vertical à gauche de la frame vidéo.

    Args:
        frame (np.ndarray): Frame vidéo (hauteur, largeur, 3).
        border_width (int): Largeur du bandereau en pixels (par défaut : 300).
        border_color (tuple): Couleur du bandereau en BGR (par défaut : noir (0, 0, 0)).

    Returns:
        np.ndarray: Frame avec le bandereau ajouté à gauche.
    """
    height, width, channels = frame.shape
    # Créer une image pour le bandereau (même hauteur, largeur spécifiée, même nombre de canaux)
    border = np.full((height, border_width, channels), border_color, dtype=np.uint8)
    # Concaténer le bandereau à gauche de la frame
    frame_with_border = np.hstack((border, frame))
    return frame_with_border

def display_persistence_diagrams(diagrams, color_dict, track_ids, fig=None, ax=None):
    """
    Affiche les diagrammes de persistance dans une fenêtre continue avec une couleur par humain.
    
    Args:
        diagrams (list): Liste de diagrammes de persistance.
        frame_number (int): Numéro de la frame pour le titre.
        color_dict (dict): Dictionnaire associant track_id à un tuple de couleur (r, g, b) ou (r, g, b, a).
        track_ids (list): Liste des track_id correspondant aux diagrammes.
        fig (matplotlib.figure.Figure): Figure existante (optionnel, pour réutilisation).
        ax (matplotlib.axes.Axes): Axes existants (optionnel, pour réutilisation).
    
    Returns:
        tuple: (fig, ax) pour réutilisation dans la boucle.
    """
    # Créer une nouvelle figure si aucune n'est fournie
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    # Effacer les axes pour la mise à jour
    ax.clear()
    
    # Déterminer la plage pour les axes
    max_death = max([dg[:, 1].max() for dg in diagrams if dg.size > 0]) if len(diagrams) > 0 else 1.0
    max_birth = max([dg[:, 0].max() for dg in diagrams if dg.size > 0]) if len(diagrams) > 0 else 1.0
    max_val = max(max_birth, max_death) * 1.1  # Ajouter une marge
    
    # Tracer la diagonale
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Tracer les diagrammes avec les couleurs correspondantes
    for i, (dg, track_id) in enumerate(zip(diagrams, track_ids)):
        if dg.size > 0 and track_id in color_dict:
            births = dg[:, 0]
            deaths = dg[:, 1]
            color = color_dict[track_id]
            # Normaliser la couleur si nécessaire (matplotlib attend des valeurs entre 0 et 1)
            if max(color) > 1:
                color = tuple(c / 255 for c in color[:3]) + ((color[3] / 255,) if len(color) > 3 else (1.0,))
            ax.scatter(births, deaths, color=color, label=f'Humain {track_id}', s=50, alpha=0.7)
    
    # Configurer les axes et la légende
    ax.set_xlabel('Naissance')
    ax.set_ylabel('Mort')
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_title(f'Diagrammes de Persistance')
    if len(diagrams) > 0:
        ax.legend()
    
    # Forcer la mise à jour de l'affichage
    fig.canvas.draw()
    fig.canvas.flush_events()
    
    return fig, ax

def visualize_distance_matrix(distance_matrix, ids_set1, ids_set2, title="Cross-Distance Matrix Heatmap"):
    """
    Affiche ou met à jour une heatmap pour une matrice de distances croisées entre deux ensembles
    de diagrammes de persistance. Gère les matrices vides.

    Args:
        distance_matrix (np.ndarray or torch.Tensor): Matrice de distances de forme (n_set1, n_set2).
        ids_set1 (list): Liste des IDs pour le premier ensemble (lignes).
        ids_set2 (list): Liste des IDs pour le deuxième ensemble (colonnes).
        title (str): Titre de la heatmap.
    """
    # Convertit en NumPy si c'est un tenseur PyTorch
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.cpu().numpy()

    # Vérifie que la matrice est un tableau NumPy
    if not isinstance(distance_matrix, np.ndarray):
        raise TypeError(f"La matrice doit être un np.ndarray ou torch.Tensor, type trouvé : {type(distance_matrix)}")

    # Crée une figure si elle n'existe pas
    if not hasattr(visualize_distance_matrix, 'fig'):
        plt.ion()  # Mode interactif
        visualize_distance_matrix.fig, visualize_distance_matrix.ax = plt.subplots()
        visualize_distance_matrix.heatmap = None
        visualize_distance_matrix.cbar = None

    # Nettoie l'axe
    visualize_distance_matrix.ax.clear()

    # Vérifie si la matrice est vide
    if distance_matrix.size == 0 or distance_matrix.shape[0] == 0 or distance_matrix.shape[1] == 0:
        visualize_distance_matrix.ax.text(
            0.5, 0.5, "Matrice vide : aucune donnée détectée",
            horizontalalignment='center', verticalalignment='center',
            transform=visualize_distance_matrix.ax.transAxes
        )
        visualize_distance_matrix.ax.set_xticks([])
        visualize_distance_matrix.ax.set_yticks([])
        visualize_distance_matrix.ax.set_xlabel('IDs Set 2')
        visualize_distance_matrix.ax.set_ylabel('IDs Set 1')
        visualize_distance_matrix.ax.set_title(f"{title} (Vide)")
        visualize_distance_matrix.fig.canvas.draw()
        visualize_distance_matrix.fig.canvas.flush_events()
        plt.pause(0.001)
        print(f"Warning: Matrice vide pour {title}, shape: {distance_matrix.shape}")
        return

    # Vérifie la compatibilité des dimensions
    if distance_matrix.shape[0] != len(ids_set1) or distance_matrix.shape[1] != len(ids_set2):
        raise ValueError(f"Dimensions incohérentes : matrice {distance_matrix.shape}, "
                        f"ids_set1 {len(ids_set1)}, ids_set2 {len(ids_set2)}")

    # Crée une normalisation pour la colormap
    norm = colors.Normalize(vmin=np.min(distance_matrix), vmax=np.max(distance_matrix))

    # Affiche la heatmap
    visualize_distance_matrix.heatmap = visualize_distance_matrix.ax.imshow(
        distance_matrix,
        cmap='viridis',
        norm=norm,
        interpolation='nearest'
    )

    # Ajoute ou met à jour la colorbar
    if visualize_distance_matrix.cbar is None:
        visualize_distance_matrix.cbar = visualize_distance_matrix.fig.colorbar(
            visualize_distance_matrix.heatmap, ax=visualize_distance_matrix.ax
        )
    else:
        visualize_distance_matrix.cbar.update_normal(visualize_distance_matrix.heatmap)

    # Configure les étiquettes des axes
    visualize_distance_matrix.ax.set_xticks(np.arange(len(ids_set2)))
    visualize_distance_matrix.ax.set_yticks(np.arange(len(ids_set1)))
    visualize_distance_matrix.ax.set_xticklabels(ids_set2)
    visualize_distance_matrix.ax.set_yticklabels(ids_set1)
    visualize_distance_matrix.ax.set_xlabel('IDs Set 2')
    visualize_distance_matrix.ax.set_ylabel('IDs Set 1')
    visualize_distance_matrix.ax.set_title(title)

    # Met à jour la figure
    visualize_distance_matrix.fig.canvas.draw()
    visualize_distance_matrix.fig.canvas.flush_events()
    plt.pause(0.001)

def update_cross_distance_matrix_plot(fig, ax, distance_matrix, diagrams_id_D, diagrams_id_P, title="Matrice de Distances"):
    """Mise à jour du graphique de la matrice de distances avec surlignage de la valeur maximale par ligne."""
    
    # Clear the axis and re-plot the matrix
    ax.clear()
    cax = ax.matshow(distance_matrix, cmap='viridis')  # Afficher la matrice avec un colormap
    fig.colorbar(cax)  # Ajouter une barre de couleur
    
    # Ajouter un titre
    ax.set_title(title)

    # Ajouter des labels pour les axes
    ax.set_xticks(np.arange(len(diagrams_id_P)))
    ax.set_yticks(np.arange(len(diagrams_id_D)))
    ax.set_xticklabels(diagrams_id_P)
    ax.set_yticklabels(diagrams_id_D)
    
    # Surligner la valeur la plus haute de chaque ligne
    for i in range(distance_matrix.shape[0]):  # Pour chaque ligne
        max_index = np.argmax(distance_matrix[i])  # Trouver l'indice de la valeur maximale
        max_value = distance_matrix[i, max_index]  # Obtenir la valeur maximale
        ax.text(max_index, i, f'{max_value:.2f}', ha='center', va='center', color='red', fontsize=12, fontweight='bold')

    # Rafraîchir l'affichage
    fig.canvas.draw()
    fig.canvas.flush_events()

from matplotlib.backends.backend_agg import FigureCanvasAgg

def generate_heatmap_image(distance_matrix, ids_set1, ids_set2, title="Matrice de Distances Croisées"):
    """
    Génère une image de heatmap à partir de la matrice de distances croisées.

    Args:
        distance_matrix (np.ndarray or torch.Tensor): Matrice de distances.
        ids_set1 (list): Liste des IDs pour les lignes (par ex., hist_tracked_pose).
        ids_set2 (list): Liste des IDs pour les colonnes (par ex., detec_tracked_pose).
        title (str): Titre de la heatmap.

    Returns:
        np.ndarray: Image de la heatmap (hauteur, largeur, 3) en RGB.
    """
    # Convertit en NumPy si c'est un tenseur PyTorch
    if isinstance(distance_matrix, torch.Tensor):
        distance_matrix = distance_matrix.cpu().numpy()

    # Créer une figure Matplotlib
    fig, ax = plt.subplots(figsize=(4, 3))  # Taille réduite pour superposition

    if distance_matrix.size == 0 or distance_matrix.shape[0] == 0 or distance_matrix.shape[1] == 0:
        # Si la matrice est vide, afficher un message
        ax.text(0.5, 0.5, "Matrice vide", horizontalalignment='center', verticalalignment='center')
        ax.set_title(title)
        ax.axis('off')
    else:
        # Afficher la heatmap avec les données
        norm = plt.Normalize(vmin=np.min(distance_matrix), vmax=np.max(distance_matrix))
        im = ax.imshow(distance_matrix, cmap='viridis', norm=norm, interpolation='nearest')
        ax.set_xticks(np.arange(len(ids_set2)))
        ax.set_yticks(np.arange(len(ids_set1)))
        ax.set_xticklabels(ids_set2)
        ax.set_yticklabels(ids_set1)
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    # Convertir la figure en image NumPy
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # Obtenir les données RGBA
    rgba = np.asarray(canvas.buffer_rgba())
    
    # Convertir RGBA en RGB (supprimer le canal alpha)
    image = rgba[:, :, :3]  # Prendre les 3 premiers canaux (RGB)
    
    plt.close(fig)  # Fermer la figure pour libérer la mémoire
    return image

def overlay_heatmap_on_frame(frame, heatmap_image, position=(10, 10), alpha=0.7, brightness_factor=1.5, size_factor=1.5):
    """
    Superpose l'image de la heatmap sur la frame vidéo, dans le bandereau à gauche, avec luminosité et taille ajustées.

    Args:
        frame (np.ndarray): Frame vidéo (hauteur, largeur, 3).
        heatmap_image (np.ndarray): Image de la heatmap (hauteur_hm, largeur_hm, 3).
        position (tuple): Position (x, y) où placer la heatmap (par défaut : (10, 10)).
        alpha (float): Transparence de la heatmap.
        brightness_factor (float): Facteur de luminosité pour la heatmap.
        size_factor (float): Facteur d'agrandissement (par défaut : 1.5 pour 1/3 de la frame).

    Returns:
        tuple: (frame avec heatmap superposée, hauteur redimensionnée de la heatmap).
    """
    frame_copy = frame.copy()
    x_hm, y_hm = position
    h_hm, w_hm, _ = heatmap_image.shape

    # Augmenter la luminosité de la heatmap
    heatmap_bright = increase_brightness(heatmap_image, brightness_factor)

    # Redimensionner la heatmap pour qu'elle soit plus grande (1/3 de la frame)
    scale = min(frame.shape[0] / 3 / h_hm, (frame.shape[1] / 3) / w_hm) * size_factor
    new_h_hm, new_w_hm = int(h_hm * scale), int(w_hm * scale)
    heatmap_resized = cv2.resize(heatmap_bright, (new_w_hm, new_h_hm), interpolation=cv2.INTER_AREA)

    # S'assurer que la région d'intérêt (ROI) reste dans les limites de la frame
    y_end_hm = min(y_hm + new_h_hm, frame.shape[0])
    x_end_hm = min(x_hm + new_w_hm, frame.shape[1])
    roi_hm = frame_copy[y_hm:y_end_hm, x_hm:x_end_hm]

    # Ajuster la heatmap redimensionnée à la taille de la ROI
    heatmap_resized = heatmap_resized[:y_end_hm - y_hm, :x_end_hm - x_hm]

    # Superposer avec transparence
    blended_hm = cv2.addWeighted(roi_hm, 1 - alpha, heatmap_resized, alpha, 0)
    frame_copy[y_hm:y_end_hm, x_hm:x_end_hm] = blended_hm

    return frame_copy, new_h_hm

def overlay_diagram_on_frame(frame, diagram_image, position=(10, 10), alpha=0.7, brightness_factor=1.5, size_factor=1.5, heatmap_height=None):
    """
    Superpose l'image des diagrammes de persistance sur la frame vidéo, dans le bandereau, avec luminosité et taille ajustées.

    Args:
        frame (np.ndarray): Frame vidéo (hauteur, largeur, 3).
        diagram_image (np.ndarray): Image des diagrammes (hauteur_dg, largeur_dg, 3).
        position (tuple): Position (x, y) où placer les diagrammes.
        alpha (float): Transparence des diagrammes.
        brightness_factor (float): Facteur de luminosité pour les diagrammes.
        size_factor (float): Facteur d'agrandissement.
        heatmap_height (int, optional): Hauteur de la heatmap pour position relative.

    Returns:
        np.ndarray: Frame avec diagrammes superposés.
    """
    if diagram_image is None:
        return frame

    frame_copy = frame.copy()
    x_dg, y_dg = position
    h_dg, w_dg, _ = diagram_image.shape

    # Si heatmap_height est fourni, ajuster la position
    if heatmap_height is not None:
        x_dg = x_dg
        y_dg = position[1] + heatmap_height + 10

    # Augmenter la luminosité des diagrammes
    diagram_bright = increase_brightness(diagram_image, brightness_factor)

    # Redimensionner les diagrammes pour qu'ils soient plus grands
    scale = min(frame.shape[0] / 3 / h_dg, (frame.shape[1] / 3) / w_dg) * size_factor
    new_w_dg = int(w_dg * scale)
    new_h_dg = int(h_dg * new_w_dg / w_dg)  # Conserver le ratio
    diagram_resized = cv2.resize(diagram_bright, (new_w_dg, new_h_dg), interpolation=cv2.INTER_AREA)

    # S'assurer que la région d'intérêt (ROI) reste dans les limites de la frame
    y_end_dg = min(y_dg + new_h_dg, frame.shape[0])
    x_end_dg = min(x_dg + new_w_dg, frame.shape[1])
    roi_dg = frame_copy[y_dg:y_end_dg, x_dg:x_end_dg]

    # Ajuster l'image redimensionnée à la taille de la ROI
    diagram_resized = diagram_resized[:y_end_dg - y_dg, :x_end_dg - x_dg]

    # Superposer avec transparence
    blended_dg = cv2.addWeighted(roi_dg, 1 - alpha, diagram_resized, alpha, 0)
    frame_copy[y_dg:y_end_dg, x_dg:x_end_dg] = blended_dg

    return frame_copy

def increase_brightness(image, brightness_factor=1.5):
    """
    Augmente la luminosité d'une image RGB.

    Args:
        image (np.ndarray): Image RGB (hauteur, largeur, 3).
        brightness_factor (float): Facteur de luminosité (> 1 pour augmenter, < 1 pour diminuer, par défaut : 1.5).

    Returns:
        np.ndarray: Image avec luminosité augmentée.
    """
    # Convertir en HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Augmenter la composante V (luminosité)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255)
    # Reconvertir en RGB
    bright_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return bright_image

def generate_persistence_diagram_image(diagrams_dict, color_dict=None, title="Diagrammes de Persistance"):
    """
    Génère une image des diagrammes de persistance à partir d'un dictionnaire, avec des couleurs par ID
    et une diagonale y=x en ligne hachée.

    Args:
        diagrams_dict (dict): Dictionnaire {ID: np.ndarray} contenant les diagrammes de persistance.
        color_dict (dict, optional): Dictionnaire {ID: couleur} pour colorer les diagrammes.
                                    Couleurs peuvent être des strings (ex. 'red') ou des tuples RGB (ex. (255, 0, 0)).
        title (str): Titre du diagramme.

    Returns:
        np.ndarray: Image des diagrammes (hauteur, largeur, 3) en RGB.
    """
    fig, ax = plt.subplots(figsize=(4, 3))

    if not diagrams_dict:
        ax.text(0.5, 0.5, "Aucun diagramme", horizontalalignment='center', verticalalignment='center')
        ax.set_title(title)
        ax.axis('off')
    else:
        max_val = 0
        for diagram_id, diagram in diagrams_dict.items():
            if diagram.size == 0:
                print(f"Diagramme vide pour ID {diagram_id}")
                continue

            # Extraire les temps de naissance et de mort
            birth = diagram[:, 0]
            death = diagram[:, 1]
            # Ignorer les points à l'infini
            finite_mask = np.isfinite(death)
            birth = birth[finite_mask]
            death = death[finite_mask]
            if len(birth) == 0:
                print(f"Aucun point fini dans le diagramme pour ID {diagram_id}")
                continue

            # Obtenir la couleur depuis color_dict
            color = color_dict.get(diagram_id, 'blue') if color_dict else 'blue'
            # Convertir les tuples RGB (0-255) en format Matplotlib (0-1) si nécessaire
            if isinstance(color, tuple) and len(color) == 3:
                # Si BGR (OpenCV), convertir en RGB
                color = (color[2], color[1], color[0]) if color_dict.get('format') == 'BGR' else color
                color = tuple(c / 255.0 for c in color)

            # Tracer les points avec la couleur unique
            ax.scatter(birth, death, s=10, label=f'ID {diagram_id}', c=color, alpha=0.6)
            max_val = max(max_val, np.max(birth), np.max(death))

        if max_val > 0:
            # Tracer la diagonale y=x en ligne hachée (tiret-point)
            ax.plot([0, max_val], [0, max_val], '-.', color='0.2', linewidth=1.5, alpha=0.5)
            ax.set_xlim(0, max_val * 1.1)
            ax.set_ylim(0, max_val * 1.1)
            ax.set_xlabel('Birth')
            ax.set_ylabel('Death')
            ax.legend()
        ax.set_title(title)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    image = rgba[:, :, :3]
    plt.close(fig)
    return image