import torch
from torchvision import transforms
import numpy as np

unNormalize = transforms.Normalize(
        mean=-np.array([0.485,0.456,0.406]) / np.array([0.229,0.224,0.225]),
        std=1 / np.array([0.229,0.224,0.225]))

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

def post_process(img, input_size, img_size):
    ori_img = tensor_to_BGR(unNormalize(img).cpu())
    ori_img[img_size[0]:,:,:] = 255
    ori_img[:,img_size[1]:,:] = 255
    ori_img[img_size[0]:,img_size[1]:,:] = 255
    ori_img = pad_img(ori_img, pad_color_offset=255)
    return ori_img
