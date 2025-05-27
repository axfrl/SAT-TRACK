import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import math

def process_img(img, input_size, rot = 0., scale = 1.0):
        # resize
        img_size = torch.tensor(img.shape[:2])
        if img_size[1] >= img_size[0]:
            resize_rate = input_size/img_size[1]
            img = cv2.resize(img,dsize=(input_size,int(resize_rate*img_size[0])))
            img_size = torch.tensor([int(resize_rate*img_size[0]),input_size])
        else:
            resize_rate = input_size/img_size[0]
            img = cv2.resize(img,dsize=(int(resize_rate*img_size[1]),input_size))
            img_size = torch.tensor([input_size,int(resize_rate*img_size[1])])

        # rot and scale
        M  = cv2.getRotationMatrix2D((int(img_size[1]/2),int(img_size[0]/2)), rot, scale)
        img = cv2.warpAffine(img, M, dsize = (img.shape[1],img.shape[0]))
        
        return img, img_size

def create_empty_targets(device, img_size):
    """Create empty target dictionary for inference."""
    return [{
        'boxes': torch.zeros((0, 4), device=device),
        'labels': torch.zeros((0,), dtype=torch.int64, device=device),
        'poses': torch.zeros((0, 24*3), device=device),
        'betas': torch.zeros((0, 10), device=device),
        'transl': torch.zeros((0, 3), device=device),
        'j3ds': torch.zeros((0, 45, 3), device=device),
        'depths': torch.zeros((0, 1, 2), device=device),
        'cam_intrinsics': torch.eye(3, device=device).unsqueeze(0),
        '3d_valid': False,
        'age_valid': False,
        'detect_all_people': False,
        'img_size': torch.tensor(img_size, device=device)
    }]

def get_img(ori_img, use_sat: bool, input_size: int, device):
    img, img_size = process_img(ori_img, input_size)
    array2tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
    
    patch_size = 14
    if use_sat:
        patch_size = 56
    # pad image to support pooling
    pad_img = np.zeros((math.ceil(img.shape[0]/patch_size)*patch_size, math.ceil(img.shape[1]/patch_size)*patch_size, 3), dtype=img.dtype)
    pad_img[:img.shape[0], :img.shape[1]] = img
    assert max(pad_img.shape[:2]) == input_size
    pad_img = Image.fromarray(pad_img[:,:,::-1].copy())
    norm_img = array2tensor(pad_img).to(device)

    targets = create_empty_targets(device, img_size)

    return [norm_img], targets