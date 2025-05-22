import torch
import torch.nn as nn
import cv2
import os
import time
import numpy as np
from queue import Queue
from threading import Thread, Event
import torch.nn.functional as F 
from engines.funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets, write_frames
from utils.visualization import vis_vertices_img
from structures.boxes import Boxes
from structures.instances import Instances
from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as mask_utils
from configs.base import WEIGHTS_DIR
from external.deep_sort_ import nn_matching
from external.deep_sort_.detection import Detection
from external.deep_sort_.tracker import Tracker
from models.human_models.hmar import HMAR
from models.predictor import Pose_transformer_v2
from utils.utils import (convert_pkl, get_prediction_interval,
                               progress_bar, smpl_to_pose_camera_vector)
from utils.utils_dataset import process_image, process_mask
from utils.utils_download import cache_url
from utils.postprocessor import Postprocessor
from sklearn.linear_model import Ridge
import gdown
from smplx.lbs import batch_rodrigues  # ou bien pytorch3d.transforms.axis_angle_to_matrix
from topology.persistence_analysis import compute_persistence_diagrams
import matplotlib.pyplot as plt
from gtda.homology import VietorisRipsPersistence
from matplotlib import cm

import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Fonction utilitaire SMPL45 → COCO17 (numpy) ---
COCO17_SMPL45_INDICES = [24, 26, 25, 28, 27, 16, 17, 18, 19,
                         20, 21,  1,  2,  4,  5,  7,  8]
COCO17_NAMES = [
    'nose','left_eye','right_eye','left_ear','right_ear',
    'left_shoulder','right_shoulder','left_elbow','right_elbow',
    'left_wrist','right_wrist','left_hip','right_hip',
    'left_knee','right_knee','left_ankle','right_ankle'
]

def smpl45_to_coco17(j2ds_smpl: np.ndarray) -> np.ndarray:
    """
    j2ds_smpl: np.array de forme (B, 45, 2) ou (45,2)
    return   : np.array de forme (B, 17, 2) ou (17,2)
    """
    # Gérer cas sans batch
    batched = (j2ds_smpl.ndim == 3)
    if not batched:
        j2ds_smpl = j2ds_smpl[None, ...]  # (1,45,2)
    # Sélection et reorder
    j2ds_coco = j2ds_smpl[:, COCO17_SMPL45_INDICES, :]  # (B,17,2)
    # Renvoyer à la forme initiale
    return j2ds_coco[0] if not batched else j2ds_coco

class TrackModel(nn.Module):
    def __init__(self, cfg, device, sat_model):
        super(TrackModel, self).__init__()

        self.sat_model = sat_model
        self.focal = sat_model.focal 
        self.input_size= sat_model.input_size
        self.cfg = cfg
        self.device = device
        self.eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
        self.history_keys    = ['appe', 'loca', 'pose', 'uv'] if self.cfg.render.enable else []
        self.prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if self.cfg.render.enable else []
        self.extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'class_name', 'conf', 'annotations']
        self.extra_keys_2    = ['smpl', 'camera', 'camera_bbox', '3d_joints', '2d_joints', 'mask', 'extra_data']
        self.history_keys    = self.history_keys + self.extra_keys_1 + self.extra_keys_2
        self.visual_store_   = self.eval_keys + self.history_keys + self.prediction_keys
        self.tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']

        self.tracked_frames = []
        self.final_visuals_dic = {}

        self.color_dict = {
            1: (0, 255, 255),    # Cyan électrique
            2: (255, 0, 255),    # Magenta électrique
            3: (0, 255, 0),      # Vert néon
            4: (255, 255, 0),    # Jaune vif
            5: (255, 165, 0),    # Orange fluo
            6: (0, 0, 255),      # Bleu électrique
            7: (255, 20, 147),   # Rose électrique
            8: (0, 191, 255),    # Bleu clair fluo
            9: (138, 43, 226),   # Violet vibrant
            10: (255, 105, 180), # Rose fluo
            11: (255, 0, 0),     # Rouge pur
            12: (0, 255, 128),   # Vert cyan
            13: (255, 128, 0),   # Orange vif
            14: (128, 0, 255),   # Violet électrique
            15: (255, 255, 128), # Jaune pâle fluo
            16: (0, 128, 255),   # Bleu ciel vibrant
            17: (255, 0, 128),   # Rose magenta
            18: (128, 255, 0),   # Vert citron
            19: (255, 64, 64),   # Rouge corail
            20: (0, 255, 64),    # Vert fluo clair
            21: (64, 0, 255),    # Bleu violet
            22: (255, 192, 0),   # Jaune doré
            23: (192, 0, 255),   # Violet magenta
            24: (0, 255, 192),   # Cyan clair
            25: (255, 0, 64),    # Rouge vif
            26: (64, 255, 0),    # Vert pomme
            27: (0, 64, 255),    # Bleu néon
            28: (255, 128, 128), # Rose pâle fluo
            29: (128, 255, 128), # Vert menthe
            30: (128, 128, 255), # Bleu lavande
            31: (255, 64, 0),    # Orange rougeâtre
            32: (0, 255, 255),   # Cyan pur
            33: (255, 0, 192),   # Magenta vif
            34: (64, 255, 64),   # Vert clair
            35: (192, 0, 192),   # Violet rose
            36: (0, 192, 255),   # Bleu turquoise
            37: (255, 96, 96),   # Rouge saumon
            38: (96, 255, 0),    # Vert chartreuse
            39: (0, 96, 255),    # Bleu azure
            40: (255, 255, 64),  # Jaune citron
            41: (255, 0, 96),    # Rose vif
            42: (96, 0, 255),    # Violet bleu
            43: (0, 255, 96),    # Vert turquoise
            44: (255, 160, 122), # Corail fluo
            45: (160, 32, 240),  # Violet orchidée
            46: (255, 215, 0),   # Or vif
            47: (0, 255, 160),   # Cyan vert
            48: (255, 48, 48),   # Rouge néon
            49: (48, 255, 48),   # Vert émeraude
            50: (48, 48, 255)    # Bleu saphir
        }

        # download wights and configs from Google Drive
        self.cached_download_from_drive()

        self.setup_hmr()

        # setup temporal pose predictor
        self.setup_predictor()
        
        # move to device
        self.to(self.device)
        
        # train or eval
        self.train() if(self.cfg.train) else self.eval()

        self.setup_postprocessor()
        
        self.setup_deepsort()

    def setup_hmr(self):
        print("Loading HMAR model...")
        self.HMAR = HMAR(self.cfg)
        self.HMAR.load_weights(self.cfg.hmr.hmar_path)

    def setup_predictor(self):
        print("Loading Predictor model...")
        self.pose_predictor = Pose_transformer_v2(self.cfg, self)
        self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)
        
    def setup_deepsort(self):
        print("Setting up DeepSort...")
        metric  = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
        self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])  

    def setup_postprocessor(self):
        # by default this will not be initialized
        self.postprocessor = Postprocessor(self.cfg, self)
    
    def generate_mask_from_vertices(self, vertices, cam_intrinsics, frame_size, radius=5, n_samples=100_000):
        """
        Génère un masque binaire approximant la silhouette à partir des vertices projetés.

        Args:
            vertices (np.ndarray): (N, 3) vertices du mesh (en mètres)
            cam_intrinsics (np.ndarray): (3, 3) matrice K de la caméra
            frame_size (tuple): (H, W) de l’image de sortie
            radius (int): Rayon du noyau circulaire autour de chaque point projeté
            n_samples (int): Nombre max de points à projeter

        Returns:
            mask (np.ndarray): (H, W) uint8 mask binaire (1 = humain)
        """
        H, W = map(int, frame_size)
        if vertices.shape[0] == 0:
            return np.zeros((H, W), dtype=np.uint8)

        # Sous-échantillonnage si nécessaire
        if vertices.shape[0] > n_samples:
            idx = np.random.choice(vertices.shape[0], n_samples, replace=False)
            vertices = vertices[idx]

        # Projection 3D -> 2D
        K = cam_intrinsics.astype(np.float32)
        verts_proj = vertices @ K.T
        pts2d = verts_proj[:, :2] / (verts_proj[:, 2:] + 1e-6)

        # Clipping dans l’image
        xs = np.clip(pts2d[:, 0], 0, W - 1).astype(np.int32)
        ys = np.clip(pts2d[:, 1], 0, H - 1).astype(np.int32)

        # Création d’un masque binaire
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[ys, xs] = 1

        # Dilatation circulaire pour obtenir un corps complet
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

        return mask
    
    def get_croped_image(self, image, bbox, bbox_pad, seg_mask, output_size=256):
        """
        GPU-native cropping + normalisation + concat mask
        Inputs:
        - image:   H×W×3 numpy uint8 BGR ou torch.uint8 tensor BGR
        - seg_mask: H×W numpy bool/uint8 ou torch tensor (0/1)
        - bbox, bbox_pad: listes [x0,y0,x1,y1] en pixels
        - output_size: int, taille carrée de sortie
        Returns:
        - masked_image: torch.FloatTensor (4×output_size×output_size) sur GPU
        - center_, scale_, rles, center_pad, scale_pad (comme avant)
        """

        # --- 1) RLE encoding (reste CPU & numpy) ---
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order='F', dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for r in rles:
            r["counts"] = r["counts"].decode("utf-8")

        # --- 2) Prépare les tenseurs et device ---
        device = torch.device("cuda")
        # image → float32 [0,1], BGR→RGB si besoin, shape 1×3×H×W
        if isinstance(image, np.ndarray):
            img_t = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).to(device).float() / 255.0
        else:
            img_t = image.to(device).permute(0,3,1,2).float() if image.ndim==4 else image.permute(2,0,1).unsqueeze(0).to(device).float()/255.0

        # seg_mask → float32 [0,1], shape 1×1×H×W
        if isinstance(seg_mask, np.ndarray):
            m_t = torch.from_numpy(seg_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        else:
            m_t = seg_mask.to(device).unsqueeze(0).unsqueeze(0).float()

        B, C, H, W = img_t.shape

        # --- 3) Calcul des centres et échelles ---
        center_pad = np.array([ (bbox_pad[0]+bbox_pad[2])/2, (bbox_pad[1]+bbox_pad[3])/2 ])
        scale_pad  = np.array([ bbox_pad[2]-bbox_pad[0],   bbox_pad[3]-bbox_pad[1] ])
        s = max(scale_pad)

        cx, cy = center_pad
        # Normalisation pour affine_grid (coords dans [-1,1])
        # scale_x = (bbox_pad_width) / W, translate_x = (2*cx/W -1)
        scale_x = scale_pad[0] / W
        scale_y = scale_pad[1] / H
        trans_x = 2 * (cx / W) - 1
        trans_y = 2 * (cy / H) - 1

        # Theta pour affine_grid : shape B×2×3
        theta = torch.tensor([[
            [ scale_x,     0.0, trans_x ],
            [    0.0, scale_y, trans_y ]
        ]], dtype=torch.float32, device=device)

        # --- 4) Génération de la grille et application ---
        grid = F.affine_grid(theta, size=(B, C, output_size, output_size), align_corners=False)
        img_crop = F.grid_sample(img_t,  grid, mode='bilinear',   padding_mode='zeros', align_corners=False)
        # pour le mask on réutilise la même grille
        m_crop   = F.grid_sample(m_t,   grid, mode='nearest',    padding_mode='zeros', align_corners=False)

        # --- 5) Normalisation image (au format ImageNet) ---
        # Les moyennes/std sont divisées par 255 pour rester sur [0,1]
        mean = torch.tensor([123.675, 116.280, 103.530], device=device) / 255.0
        std  = torch.tensor([ 58.395,  57.120,  57.375], device=device) / 255.0
        img_norm = (img_crop - mean[None,:,None,None]) / std[None,:,None,None]

        # --- 6) Concat image + mask ---
        # mask est 1×1×H×W, on garde ce canal unique
        masked_image = torch.cat([img_norm, m_crop], dim=1).squeeze(0)
        # masked_image : (4, output_size, output_size)

        # --- 7) Calcule center/scale original pour retourner ---
        center_ = np.array([ (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2 ])
        scale_  = np.array([ bbox[2]-bbox[0],     bbox[3]-bbox[1] ])

        return masked_image, center_, scale_, rles, center_pad, scale_pad

    def _get_croped_image(self, image, bbox, bbox_pad, seg_mask, output_size=256):
        """
        Args:
            image:      H×W×3 (BGR uint8)
            bbox:       [x0, y0, x1, y1] - sans padding
            bbox_pad:   [x0, y0, x1, y1] - avec padding
            seg_mask:   H×W (bool ou uint8)
            output_size: taille du carré en sortie (par défaut 256)

        Returns:
            masked_image: Tensor C×H×W (3 canaux image + 1 canal masque)
            center_:      centre du bbox original
            scale_:       échelle du bbox original
            rles:         encodage COCO du masque
            center_pad:   centre du bbox_pad
            scale_pad:    échelle du bbox_pad
        """

        # Encode mask (comme dans TAO)
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order='F', dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for rle in rles:
            rle["counts"] = rle["counts"].decode("utf-8")

        # Ensure seg_mask is 3-channel uint8
        if seg_mask.dtype != np.uint8:
            seg_mask = (seg_mask > 0).astype(np.uint8) * 255
        if len(seg_mask.shape) == 2:
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)

        # Centres et échelles
        center_ = np.array([(bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2])
        scale_  = np.array([(bbox[2] - bbox[0]),     (bbox[3] - bbox[1])])

        center_pad = np.array([(bbox_pad[2] + bbox_pad[0]) / 2, (bbox_pad[3] + bbox_pad[1]) / 2])
        scale_pad  = np.array([(bbox_pad[2] - bbox_pad[0]),     (bbox_pad[3] - bbox_pad[1])])

        # Transformation affine
        def get_affine_transform(center, scale, out_res):
            s = max(scale)
            src = np.array([
                [center[0] - s/2, center[1] - s/2],
                [center[0] - s/2, center[1] + s/2],
                [center[0] + s/2, center[1] - s/2]
            ], dtype=np.float32)
            dst = np.array([
                [0, 0],
                [0, out_res - 1],
                [out_res - 1, 0]
            ], dtype=np.float32)
            return cv2.getAffineTransform(src, dst)

        M = get_affine_transform(center_pad, scale_pad, output_size)

        # Image crop et normalisation
        cropped_img = cv2.warpAffine(image, M, (output_size, output_size), flags=cv2.INTER_LINEAR)
        cropped_mask = cv2.warpAffine(seg_mask, M, (output_size, output_size), flags=cv2.INTER_NEAREST)

        # Normalisation image
        mean = np.array([123.675, 116.280, 103.530], dtype=np.float32)
        std  = np.array([ 58.395,  57.120,  57.375], dtype=np.float32)
        img_f = cropped_img.astype(np.float32)
        img_f = (img_f - mean[None, None, :]) / std[None, None, :]

        # Passage en tensor
        image_tmp = torch.from_numpy(img_f.transpose(2, 0, 1))  # CxHxW
        mask_tmp = torch.from_numpy(cropped_mask.astype(np.float32).transpose(2, 0, 1))  # 3xHxW
        masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), dim=0)

        return masked_image, center_, scale_, rles, center_pad, scale_pad

    def _get_croped_image(self, image, bbox, bbox_pad, seg_mask):
        
        # Encode the mask for storing, borrowed from tao dataset
        # https://github.com/TAO-Dataset/tao/blob/master/scripts/detectors/detectron2_infer.py
        masks_decoded = np.array(np.expand_dims(seg_mask, 2), order='F', dtype=np.uint8)
        rles = mask_utils.encode(masks_decoded)
        for rle in rles: 
            rle["counts"] = rle["counts"].decode("utf-8")
        
        if seg_mask.dtype != np.uint8:
            seg_mask = (seg_mask > 0).astype(np.uint8) * 255

        if len(seg_mask.shape) == 2:
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_GRAY2BGR)
        
        center_      = np.array([(bbox[2] + bbox[0])/2, (bbox[3] + bbox[1])/2])
        scale_       = np.array([(bbox[2] - bbox[0]), (bbox[3] - bbox[1])])

        center_pad   = np.array([(bbox_pad[2] + bbox_pad[0])/2, (bbox_pad[3] + bbox_pad[1])/2])
        scale_pad    = np.array([(bbox_pad[2] - bbox_pad[0]), (bbox_pad[3] - bbox_pad[1])])
        mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_pad, 1.0*np.max(scale_pad))
        image_tmp    = process_image(image, center_pad, 1.0*np.max(scale_pad))

        # bbox_        = expand_bbox_to_aspect_ratio(bbox, target_aspect_ratio=(192,256))
        # center_x     = np.array([(bbox_[2] + bbox_[0])/2, (bbox_[3] + bbox_[1])/2])
        # scale_x      = np.array([(bbox_[2] - bbox_[0]), (bbox_[3] - bbox_[1])])
        # mask_tmp     = process_mask(seg_mask.astype(np.uint8), center_x, 1.0*np.max(scale_x))
        # image_tmp    = process_image(image, center_x, 1.0*np.max(scale_x))
        
        masked_image = torch.cat((image_tmp, mask_tmp[:1, :, :]), 0)
        
        return masked_image, center_, scale_, rles, center_pad, scale_pad

    def _generate_synthetic_masks(self, bboxes, img_height, img_width):
        """
        Generate synthetic binary masks from bounding boxes.
        
        Args:
            bboxes: Numpy array of shape (N, 4) with [x1, y1, x2, y2].
            img_height, img_width: Image dimensions.
        
        Returns:
            masks: Numpy array of shape (N, img_height, img_width) with binary masks.
        """
        masks = np.zeros((len(bboxes), img_height, img_width), dtype=np.uint8)
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_width, x2)
            y2 = min(img_height, y2)
            masks[i, y1:y2, x1:x2] = 1
        return masks

    def _generate_sam_masks(self, bboxes, image, img_height, img_width):
        sam = sam_model_registry["vit_h"](checkpoint="path/to/sam_vit_h.pth")
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        
        masks = []
        for bbox in bboxes:
            box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
            mask, _, _ = predictor.predict(box=box, multimask_output=False)
            masks.append(mask[0])
        return np.array(masks)

    def transl3d_to_weak_cam(self, pred_transl, eps=1e-6):
        """
        pred_transl : np.array shape (BS,3) = [t_x, t_y, t_z] issues de process_smpl
        focal       : float = model.focal
        input_size  : int   = model.input_size
        -----------------------------------------------
        Retour : torch.Tensor shape (BS,3) = [scale, tx, ty]
        """
        tx_3d = pred_transl[:, 0]
        ty_3d = pred_transl[:, 1]
        tz    = pred_transl[:, 2]

        scale = (2.0 * self.focal) / (tz * self.input_size + eps)
        tx = tx_3d * scale
        ty = ty_3d * scale

        cam_np = np.stack([scale, tx, ty], axis=1).astype(np.float32)
        return torch.from_numpy(cam_np)

    def get_human_features(self, sat_data, image, frame_name, t_, measurments, gt=None, ann=None, extra_data=None):
        """
        Get human features using SAT-HMR outputs directly and HMAR for appearance and UV maps.
        
        Args:
            sat_data: Dictionary from SAT-HMR's forward method (pred_boxes, pred_confs, pred_poses, etc.).
            image: Input image (numpy array or PIL Image).
            frame_name: Frame identifier.
            t_: Time step.
            measurments: Tuple (img_height, img_width, new_image_size, left, top).
            gt: Ground-truth flags (optional, default None).
            ann: Annotations (optional, default None).
            extra_data: Extra data (optional, default None).
        
        Returns:
            detection_data_list: List of Detection objects with human features.
        """
        img_height, img_width, new_image_size, left, top = measurments
        ratio = 1.0 / int(new_image_size) * self.cfg.render.res

        # Extract SAT-HMR outputs (batch_size=1)
        pred_boxes = sat_data['pred_boxes'][0]  # [num_queries, 4] (center_x, center_y, w, h)
        pred_confs = sat_data['pred_confs'][0].squeeze(-1)  # [num_queries]
        pred_poses = sat_data['pred_poses'][0]  # [num_queries, 72]
        pred_betas = sat_data['pred_betas'][0]  # [num_queries, 10]
        pred_j3ds = sat_data['pred_j3ds'][0]  # [num_queries, num_joints, 3]
        pred_j2ds = sat_data['pred_j2ds'][0]  # [num_queries, num_joints, 2]
        pred_transl = sat_data['pred_transl'][0]  # [num_queries, 3]
        pred_intrinsic = sat_data['pred_intrinsics'][0]
        pred_cam_xys = sat_data['pred_cam_xys'][0]

        def adjust_boxes_to_original(boxes, orig_w=1280, orig_h=720, input_size=1288):
            """
            Ajuste les bounding boxes de l'image 1288x1288 vers l'image originale.
            
            Args:
                boxes: Liste de bounding boxes [x1, y1, x2, y2] dans l'image 1288x1288
                orig_w: Largeur originale (default: 1280)
                orig_h: Hauteur originale (default: 720)
                input_size: Taille d'entrée du modèle (default: 1288)
            Returns:
                adjusted_boxes: Bounding boxes ajustées pour l'image originale
            """
            pad_top_bottom = (orig_w - orig_h) // 2  # 140 pour 1280x720
            scale_x = orig_w / input_size  # 1280 / 1288
            scale_y = (orig_h + 2 * pad_top_bottom) / input_size  # 1000 / 1288
            
            adjusted_boxes = []
            for box in boxes:
                x1, y1, x2, y2 = box
                # Étape 1 : Mapper vers l'image paddée
                x1_pad = x1 * scale_x
                y1_pad = y1 * scale_y
                x2_pad = x2 * scale_x
                y2_pad = y2 * scale_y
                # Étape 2 : Retirer le padding du haut
                x1_orig = x1_pad
                y1_orig = y1_pad - pad_top_bottom
                x2_orig = x2_pad
                y2_orig = y2_pad - pad_top_bottom
                # Étape 3 : Clipper
                x1_orig = max(0, min(x1_orig, orig_w))
                y1_orig = max(0, min(y1_orig, orig_h))
                x2_orig = max(0, min(x2_orig, orig_w))
                y2_orig = max(0, min(y2_orig, orig_h))
                adjusted_boxes.append([x1_orig, y1_orig, x2_orig, y2_orig])
            
            return np.array(adjusted_boxes)
        # Convert boxes to [x1, y1, x2, y2]
        cx, cy, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
        x1 = (cx - w / 2) * new_image_size
        y1 = (cy - h / 2) * new_image_size
        x2 = (cx + w / 2) * new_image_size
        y2 = (cy + h / 2) * new_image_size
        pred_bbox_1288 = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()
        # Étape 2 : Ajustement vers l'image originale (720x1280)

        pred_bbox = adjust_boxes_to_original(pred_bbox_1288, orig_w=img_width, orig_h=img_height, input_size=new_image_size)
        
        # Filter detections based on score and size thresholds
        NPEOPLE = len(pred_confs)
        masked_image_list = []
        center_list = []
        scale_list = []
        rles_list = []
        selected_ids = []
        pred_classes = np.zeros(NPEOPLE, dtype=np.int64)  # Class 0 (person)
        ground_truth_track_id = [1] * NPEOPLE  # Default track IDs
        ground_truth_annotations = [[]] * NPEOPLE  # Default annotations

        for p_ in range(NPEOPLE):
            if pred_confs[p_] < self.cfg.phalp.low_th_c:
                continue
            w = pred_bbox[p_][2] - pred_bbox[p_][0]
            h = pred_bbox[p_][3] - pred_bbox[p_][1]
            if w < self.cfg.phalp.small_w or h < self.cfg.phalp.small_h:
                continue
            
            # Generate synthetic mask
            mask = self._generate_synthetic_masks(pred_bbox[p_][None], img_height, img_width)[0]
            rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('utf-8')

            # Compute center, scale, and cropped image for HMAR
            center = np.array([(pred_bbox[p_][0] + pred_bbox[p_][2]) / 2, (pred_bbox[p_][1] + pred_bbox[p_][3]) / 2])
            scale = np.array([w, h]) / self.cfg.render.res
            bbox = pred_bbox[p_]
            bbox_pad = np.array([bbox[0] - w * 0.125, bbox[1] - h * 0.125, bbox[2] + w * 0.125, bbox[3] + h * 0.125])
            masked_image, center_pad, scale_pad, _, _, _ = self.get_croped_image(image, bbox, bbox_pad, mask)
            
            masked_image_list.append(masked_image)
            center_list.append(center_pad)
            scale_list.append(scale_pad)
            rles_list.append(rle)
            selected_ids.append(p_)
        
        if len(selected_ids) == 0:
            return []

        BS = len(selected_ids)
        
        # Run HMAR for appearance and UV maps
        if BS > 0:
            masked_image_list = torch.stack(masked_image_list, dim=0)
            with torch.no_grad():
                extra_args = {}
                hmar_out = self.HMAR(masked_image_list.cuda(), **extra_args)
                uv_vector = hmar_out['uv_vector']  # [BS, 256, 256, 3]
                appe_embedding = self.HMAR.autoencoder_hmar(uv_vector, en=True)  # [BS, embedding_dim]
                appe_embedding  = appe_embedding.view(appe_embedding.shape[0], -1)

        else:
            uv_vector = np.zeros((BS, 256, 256, 3))
            appe_embedding = torch.zeros(BS, 512)  # Dummy embedding

        # Prepare SMPL parameters and joints from SAT-HMR
        pred_cam_tensor = pred_transl[selected_ids]  # (BS, 3) – torch.Tensor
        pred_smpl_params = []
        for idx in selected_ids:
            aa        = pred_poses[idx]            # (72,)
            global_aa = aa[:3].unsqueeze(0)        # (1,3)
            body_aa   = aa[3:].view(-1,3)          # (23,3)

            # axis-angle → rotmat
            global_rot = batch_rodrigues(global_aa)[0].cpu().numpy()   # (3,3)
            body_rot   = batch_rodrigues(body_aa).cpu().numpy()       # (23,3,3)

            pred_smpl_params.append({
                'global_orient': global_rot[None, ...],  # (1,3,3)
                'body_pose'    : body_rot,               # (23,3,3)
                'betas'        : pred_betas[idx].cpu().numpy()  # (10,)
            })
        
        pred_cam_np = self.transl3d_to_weak_cam(pred_cam_tensor.cpu().numpy())  # (BS,3) array

        pred_joints_3d = pred_j3ds[selected_ids].cpu().numpy()  # [BS, num_joints, 3]
        pred_joints_2d = pred_j2ds[selected_ids].cpu().numpy()  # [BS, num_joints, 2]
        pred_cam = pred_transl[selected_ids].cpu().numpy()  # [BS, 3]
        pred_cam_weak = []
        # Compute pose embedding
        if self.cfg.phalp.pose_distance == "joints":
            pose_embedding = torch.from_numpy(pred_joints_3d).view(BS, -1)
        elif self.cfg.phalp.pose_distance == "smpl":
            pose_embedding_list = []
            for i in range(BS):
                pred_cam_weak.append(pred_cam_np[i])
                emb_np = smpl_to_pose_camera_vector(
                    pred_smpl_params[i],      # dict avec rotmats + betas
                    pred_cam_np[i]            # [scale, tx, ty]
                )
                pose_embedding_list.append(torch.from_numpy(emb_np[0]))
            pose_embedding = torch.stack(pose_embedding_list, dim=0)
        else:
            raise ValueError("Unknown pose distance")
        
        # Compute location embedding
        pred_joints_2d_ = pred_joints_2d.reshape(BS, -1) / self.cfg.render.res
        pred_cam_ = pred_cam.reshape(BS, -1)
        loca_embedding = torch.cat((torch.from_numpy(pred_joints_2d_), torch.from_numpy(pred_cam_),
                                torch.from_numpy(pred_cam_), torch.from_numpy(pred_cam_)), dim=1)

        # Compute full embedding (for legacy)
        full_embedding = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding), dim=1)
        
        j2ds_smpl = pred_j2ds[selected_ids].cpu().numpy()  # [BS, 45, 2]
        # convertis en 17 keypoints COCO
        j2ds_coco = smpl45_to_coco17(j2ds_smpl)            # [BS, 17, 2]

        # Create detection data list
        detection_data_list = []
        for i, p_ in enumerate(selected_ids):
            joints_2d = j2ds_coco[i]
            keypoints = []
            count_out = 0
            for j in range(joints_2d.shape[0]):
                x, y = joints_2d[j]
                if 0 <= x < img_width and 0 <= y < img_height:
                    vi = 1
                else:
                    vi = 0
                    count_out += 1
                keypoints.extend([x, y, vi])
            
            detection_data = {
                "bbox": np.array([pred_bbox[p_][0], pred_bbox[p_][1],
                                pred_bbox[p_][2] - pred_bbox[p_][0], pred_bbox[p_][3] - pred_bbox[p_][1]]),
                "mask": rles_list[i],
                "conf": pred_confs[p_].cpu().numpy(),
                "appe": appe_embedding[i].cpu().numpy(),
                "pose": pose_embedding[i].numpy(),
                "loca": loca_embedding[i].numpy(),
                "uv": uv_vector[i],
                "embedding": full_embedding[i].numpy(),
                "center": center_list[i] + np.array([left, top]) * ratio,
                "scale": scale_list[i] * ratio,
                "smpl": pred_smpl_params[i],
                "camera": pred_cam_[i],
                "camera_bbox": pred_cam_[i],  # Use transl as proxy
                "pred_intrinsic": pred_intrinsic,
                "pred_cam_xys": pred_cam_xys,
                "3d_joints": pred_joints_3d[i],
                "2d_joints": pred_joints_2d_[i],
                "size": [img_height, img_width],
                "img_path": frame_name,
                "img_name": frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                "class_name": pred_classes[p_],
                "time": t_,
                "ground_truth": gt[p_] if gt is not None else ground_truth_track_id[p_],
                "annotations": ann[p_] if ann is not None else ground_truth_annotations[p_],
                "extra_data": extra_data[p_] if extra_data is not None else None,
                "keypoints": keypoints
            }
            detection_data_list.append(Detection(detection_data))

        return detection_data_list

    def forward_for_tracking(self, vectors, attibute="A", time=1):

        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]

            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)
            
            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0) # (BS, 7, pose_dim)
                en_time          = en_time.unsqueeze(0) # (BS, 7)
                en_data          = en_data.unsqueeze(0) # (BS, 7, 6)
            
            with torch.no_grad():
                pose_pred = self.pose_predictor.predict_next(en_pose, en_data, en_time, time)
            
            return pose_pred.cpu()


        if(attibute=="L"):
            vectors_loca         = vectors[0]
            vectors_time         = vectors[1]
            vectors_conf         = vectors[2]

            en_loca              = torch.from_numpy(vectors_loca)
            en_time              = torch.from_numpy(vectors_time)
            en_conf              = torch.from_numpy(vectors_conf)
            time                 = torch.from_numpy(time)

            if(len(en_loca.shape)!=3):
                en_loca          = en_loca.unsqueeze(0)             
                en_time          = en_time.unsqueeze(0)             
            else:
                en_loca          = en_loca.permute(0, 1, 2)         

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy           = en_loca[:, :, :90]
            en_loca_xy           = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n            = en_loca[:, :, 90:]
            en_loca_n            = en_loca_n.view(BS, t_, 3, 3)

            new_en_loca_n = []
            for bs in range(BS):
                x0_                  = np.array(en_loca_xy[bs, :, 44, 0])
                y0_                  = np.array(en_loca_xy[bs, :, 44, 1])
                n_                   = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                t_                   = np.array(en_time[bs, :])

                loc_                 = torch.diff(en_time[bs, :], dim=0)!=0
                if(self.cfg.phalp.distance_type=="EQ_020" or self.cfg.phalp.distance_type=="EQ_021"):
                    loc_                 = 1
                else:
                    loc_                 = loc_.shape[0] - torch.sum(loc_)+1

                M = t_[:, np.newaxis]**[0, 1]
                time_ = 48 if time[bs]>48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi  = get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                x_p  = x_p[0]
                x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi  = get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                y_p  = y_p[0]
                y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi  = get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])
                
                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi/loc_, y_pi/loc_, np.exp(n_pi)/loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p
                
            new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
            xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt

    def get_uv_distance(self, t_uv, d_uv):
        t_uv         = torch.from_numpy(t_uv).cuda().float()
        d_uv         = torch.from_numpy(d_uv).cuda().float()
        d_mask       = d_uv[3:, :, :]>0.5
        t_mask       = t_uv[3:, :, :]>0.5

        mask_dt      = torch.logical_and(d_mask, t_mask)
        mask_dt      = mask_dt.repeat(4, 1, 1)
        mask_        = torch.logical_not(mask_dt)

        t_uv[mask_]  = 0.0
        d_uv[mask_]  = 0.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb        = t_emb.view(-1)/10**3
        d_emb        = d_emb.view(-1)/10**3
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy()/4/256/256/2

    def get_pose_distance(self, track_pose, detect_pose):
        """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`.""" 
        track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

        if(self.cfg.phalp.pose_distance=="smpl"):
            # remove additional dimension used for encoding location (last 3 elements)
            track_pose = track_pose[:, :-3]
            detect_pose = detect_pose[:, :-3]

        if len(track_pose) == 0 or len(detect_pose) == 0:
            return np.zeros((len(track_pose), len(detect_pose)))
        track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
        r2 = -2. * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))

        return r2

    def get_list_of_shots(self, list_of_frames):
        # https://github.com/Breakthrough/PySceneDetect
        list_of_shots    = []
        remove_tmp_video = False
        if(self.cfg.detect_shots):
            if(isinstance(list_of_frames[0], str)):
                # make a video if list_of_frames is frames
                video_tmp_name   = self.cfg.video.output_dir + "/_TMP/" + str(self.cfg.video_seq) + ".mp4"
                for ft_, fname_ in enumerate(list_of_frames):
                    im_ = cv2.imread(fname_)
                    if(ft_==0): 
                        video_file = cv2.VideoWriter(video_tmp_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, frameSize=(im_.shape[1], im_.shape[0]))
                    video_file.write(im_)
                video_file.release()
                remove_tmp_video = True
            elif(isinstance(list_of_frames[0], tuple)):
                video_tmp_name = list_of_frames[0][0]
            else:
                raise Exception("Unknown type of list_of_frames")
            
            # Detect scenes in a video using PySceneDetect.
            scene_list = detect(video_tmp_name, AdaptiveDetector())

            if(remove_tmp_video):
                os.system("rm " + video_tmp_name)

            for scene in scene_list:
                list_of_shots.append(scene[0].get_frames())
                list_of_shots.append(scene[1].get_frames())
            list_of_shots = np.unique(list_of_shots)
            list_of_shots = list_of_shots[1:-1]
            log.info("Detected shot change at frame"+ "s" * min(0,len(list_of_shots)-1) + ": " + ", ".join(map(str, list_of_shots)))

            return list_of_shots

    def cached_download_from_drive(self, additional_urls=None):
        """
        Check for existing model files in WEIGHTS_DIR and download only missing files.
        """
        WEIGHTS_DIR = "/home/alphafalcon/tracker/code/SAT-TRACK/weights"

        # Créer les sous-répertoires nécessaires
        os.makedirs(os.path.join(WEIGHTS_DIR, "smpl_data/smpl"), exist_ok=True)
        os.makedirs(os.path.join(WEIGHTS_DIR, "phalp/3D"), exist_ok=True)
        os.makedirs(os.path.join(WEIGHTS_DIR, "phalp/weights"), exist_ok=True)
        os.makedirs(os.path.join(WEIGHTS_DIR, "phalp/ava"), exist_ok=True)

        # Liste des fichiers à vérifier/télécharger
        download_files = {
            "SMPL_NEUTRAL.pkl": ["", os.path.join(WEIGHTS_DIR, "smpl_data/smpl")],  # Pas d'URL, fichier existant
            "smpl_mean_params.npz": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz", os.path.join(WEIGHTS_DIR, "smpl_data/smpl")],
            "SMPL_to_J19.pkl": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl", os.path.join(WEIGHTS_DIR, "smpl_data/smpl")],
            "texture.npz": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz", os.path.join(WEIGHTS_DIR, "smpl_data/smpl")],
            "head_faces.npy": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy", os.path.join(WEIGHTS_DIR, "smpl_data/smpl")],
            "mean_std.npy": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy", os.path.join(WEIGHTS_DIR, "smpl_data/smpl")],
            "bmap_256.npy": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/bmap_256.npy", os.path.join(WEIGHTS_DIR, "phalp/3D")],
            "fmap_256.npy": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/fmap_256.npy", os.path.join(WEIGHTS_DIR, "phalp/3D")],
            "hmar_v2_weights.pth": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth", os.path.join(WEIGHTS_DIR, "phalp/weights")],
            "pose_predictor.pth": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt", os.path.join(WEIGHTS_DIR, "phalp/weights")],
            "pose_predictor.yaml": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml", os.path.join(WEIGHTS_DIR, "phalp/weights")],
            "ava_labels.pkl": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl", os.path.join(WEIGHTS_DIR, "phalp/ava")],
            "ava_class_mapping.pkl": ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl", os.path.join(WEIGHTS_DIR, "phalp/ava")],
        }

        # Ajouter des URLs supplémentaires si fournies
        additional_urls = additional_urls if additional_urls is not None else {}
        download_files.update(additional_urls)

        # Vérifier et télécharger les fichiers manquants
        for file_name, url_info in download_files.items():
            file_path = os.path.join(url_info[1], file_name)
            if os.path.exists(file_path):
                print(f"Using existing {file_name} at {file_path}")
            elif url_info[0]:  # Télécharger seulement si une URL est fournie
                print(f"Downloading {file_name} to {file_path}")
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                gdown.download(url_info[0], file_path, quiet=False)
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"Failed to download {file_name} to {file_path}")
            else:
                raise FileNotFoundError(f"{file_name} not found in {file_path} and no download URL provided")