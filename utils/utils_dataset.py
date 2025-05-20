import cv2
import numpy as np
import torch


def process_image(img, center, scale, output_size=256):
    mean = np.array([123.675, 116.280, 103.530])
    std = np.array([58.395, 57.120, 57.375])

    img, _, _ = generate_image_patch(img, center[0], center[1], scale, scale, output_size, output_size, False, 1.0, 0.0)
    img_n = img[:, :, ::-1].copy().astype(np.float32)
    for n_c in range(3):
        img_n[:, :, n_c] = (img_n[:, :, n_c] - mean[n_c]) / std[n_c]
    return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))


def process_mask(img, center, scale, output_size=256):
    img, _, _ = generate_image_patch(img, center[0], center[1], scale, scale, output_size, output_size, False, 1.0, 0.0)
    img_n = img[:, :, ::-1].copy().astype(np.float32)
    return torch.from_numpy(np.transpose(img_n, (2, 0, 1)))

def unnormalize(img):
    img = img * torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1,3,1,1)
    img = img + torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1,3,1,1)
    return img

def normalize(img):
    img = img - torch.tensor([0.485, 0.456, 0.406], device=img.device).reshape(1,3,1,1)
    img = img / torch.tensor([0.229, 0.224, 0.225], device=img.device).reshape(1,3,1,1)
    return img

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y # np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans_inv = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans, trans_inv

def generate_image_patch(cvimg, c_x, c_y, bb_width, bb_height, patch_width, patch_height, do_flip, scale, rot):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans, trans_inv = gen_trans_from_patch_cv(c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot, inv=False)

    img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    return img_patch, trans, trans_inv

def smpl_to_coco_joints(smpl_joints):
    """
    Convertit des joints SMPL (45 x 3) en keypoints COCO 17 x 3.
    Entrée : smpl_joints de forme (B, 45, 3) ou (45, 3) en torch.Tensor ou numpy.ndarray.
    Sortie : points COCO de forme (B, 17, 3) ou (17, 3), même type que l'entrée.
    La liste des indices SMPL à extraire est :
      [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    correspondant à ['nose','left_eye','right_eye','left_ear','right_ear',
                     'L_Shoulder','R_Shoulder','L_Elbow','R_Elbow',
                     'L_Wrist','R_Wrist','L_Hip','R_Hip','L_Knee','R_Knee','L_Ankle','R_Ankle'].
    """
    # Indices des joints SMPL correspondant aux 17 points COCO
    coco_indices = [24, 26, 25, 28, 27, 16, 17, 18, 19, 20, 21, 1, 2, 4, 5, 7, 8]
    # Gérer entrée sans batch (45,3) en ajoutant une dimension
    has_batch = (smpl_joints.ndim == 3)
    if not has_batch:
        smpl_joints = smpl_joints[None, ...]  # deviens (1, 45, 3)
    # Conversion pour torch.Tensor ou numpy.ndarray
    if isinstance(smpl_joints, torch.Tensor):
        coco_joints = smpl_joints[:, coco_indices, :]
    else:
        coco_joints = smpl_joints[:, coco_indices, :]
    # Remettre à la forme (17,3) si pas de batch initial
    if not has_batch:
        coco_joints = coco_joints[0]
    return coco_joints

def encode_keypoints_xyv(joints):
    """
    joints: (K, 3) avec (x, y, visibility)
    retourne: flat list [x1,y1,v1, x2,y2,v2, …]
    """
    return joints[:, :3].reshape(-1).tolist()
