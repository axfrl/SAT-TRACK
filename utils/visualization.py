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
from matplotlib import colormaps
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

def display_distance_matrix(matrix, labels_D, labels_P, title="Matrice de Distance"):
    """
    Affiche une matrice de distance dans une nouvelle fenêtre.

    Args:
        matrix (np.ndarray): Matrice à afficher.
        labels_D (list): Labels pour les lignes (ensemble D).
        labels_P (list): Labels pour les colonnes (ensemble P).
        title (str): Titre du graphique.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap='viridis', interpolation='nearest')
    
    # Labels des axes
    ax.set_xlabel('Pj (Cible)')
    ax.set_ylabel('Di (Source)')
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels_P)))
    ax.set_yticks(np.arange(len(labels_D)))
    ax.set_xticklabels(labels_P)
    ax.set_yticklabels(labels_D)
    
    # Ajout de la barre de couleur
    fig.colorbar(im, ax=ax, label='Distance')
    
    # Affichage des valeurs dans les cellules
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', color='white')
    
    plt.tight_layout()
    plt.show()

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