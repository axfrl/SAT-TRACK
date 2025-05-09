import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import glob
import os
from queue import Empty
import math

def get_transform(input_size, orig_w=1280, orig_h=720, fov_deg=57.0, device='cuda:0'):
    """
    Transform avec padding vertical pour obtenir une image carrée, respectant le FOV horizontal.
    """
    # Padding vertical (pour rendre l'image carrée avant resize)
    pad_top_bottom = (orig_w - orig_h) // 2  # exemple : (1280 - 720) // 2 = 280

    class GPUPad:
        def __init__(self, padding):
            self.padding = padding  # (left, top, right, bottom)
        
        def __call__(self, img):
            left, top, right, bottom = self.padding
            return F.pad(img, (left, right, top, bottom), mode='constant', value=0)

    transform = transforms.Compose([
        GPUPad((0, pad_top_bottom, 0, pad_top_bottom)),
        transforms.Resize((input_size, input_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Image avec padding : nouvelle taille avant resize
    padded_h = orig_h + 2 * pad_top_bottom  # ex: 720 + 560 = 1280
    padded_w = orig_w                       # 1280

    # Intrinsèques ajustés à l’image paddée
    f_x = (padded_w / 2) / math.tan(math.radians(fov_deg / 2))
    f_y = f_x  # l’image est maintenant carrée
    c_x = padded_w / 2
    c_y = padded_h / 2

    # Mise à l’échelle vers input_size x input_size
    scale = input_size / padded_w  # padded_w == padded_h == 1280 dans ton cas

    f_x *= scale
    f_y *= scale
    c_x *= scale
    c_y *= scale

    K = torch.tensor([
        [f_x, 0,   c_x],
        [0,   f_y, c_y],
        [0,   0,   1  ]
    ], dtype=torch.float32, device=device)

    return transform, K

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

def preprocess_frame(frame, transform, device, orig_w=1280, orig_h=720):
    """
    Preprocess a video frame on the GPU by padding to square and resizing to model input size.
    
    Args:
        frame: Input frame (numpy array, HxWx3, BGR)
        transform: Torchvision transform pipeline (GPU-compatible)
        device: Target device (e.g., 'cuda:0')
        orig_w: Original frame width (default: 1280)
        orig_h: Original frame height (default: 720)
    Returns:
        tensor: Preprocessed frame (torch tensor, 1x3xinput_sizexinput_size)
    """
    frame_tensor = torch.from_numpy(frame).to(device).permute(2, 0, 1)
    frame_tensor = frame_tensor.flip(0)  # BGR to RGB
    frame_tensor = frame_tensor.float() / 255.0
    return transform(frame_tensor).unsqueeze(0)

def find_input_video(input_dir):
    """Find a single video file."""
    video_extensions = ['*.mp4', '*.avi', '*.mov']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(input_dir, ext)))
    if not video_files:
        raise ValueError(f"No video files found in {input_dir}")
    if len(video_files) > 1:
        print(f"Multiple videos found in {input_dir}. Using: {video_files[0]}")
    return video_files[0]

def get_output_video_path(input_video, output_dir):
    """Generate output video path."""
    video_name = os.path.basename(input_video)
    output_name = os.path.splitext(video_name)[0] + "_output.mp4"
    return os.path.join(output_dir, output_name)

def write_frames(output_video, frame_width, frame_height, fps, result_queue, stop_event):
    """Write frames to video in a separate thread."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
    while not stop_event.is_set():
        try:
            result = result_queue.get(timeout=1.0)
            if result is None:
                break
            frame_id, rendered_img = result
            out.write(rendered_img)
        except Empty:
            continue
    out.release()