import torch
import cv2
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import glob
import os
from queue import Empty

def get_transform(input_size, orig_w=1280, orig_h=720, device='cuda:0'):
    """
    Create a GPU-compatible transform pipeline for preprocessing.
    
    Args:
        input_size: Target size for model input (e.g., 1288)
        orig_w: Original frame width (default: 1280)
        orig_h: Original frame height (default: 720)
        device: Target device (default: 'cuda:0')
    Returns:
        transform: Torchvision transform pipeline
    """
    pad_top_bottom = (orig_w - orig_h) // 2  # 140 pixels top and bottom
    
    class GPUPad:
        def __init__(self, padding):
            self.padding = padding  # (left, top, right, bottom)
        
        def __call__(self, img):
            left, top, right, bottom = self.padding
            return F.pad(img, (left, right, top, bottom), mode='constant', value=0)

    transform = transforms.Compose([
        GPUPad((0, pad_top_bottom, 0, pad_top_bottom)),
        transforms.Resize((input_size, input_size)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

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

@torch.no_grad()
def preprocess_frame(frame, input_size, device):
    # Convert to tensor and normalize in one go
    frame_tensor = torch.from_numpy(frame).to(device=device, dtype=torch.float32).permute(2, 0, 1) / 255.0  # (3, H, W)
    
    # Convert BGR to RGB directly with slicing
    frame_tensor = frame_tensor.flip(0)  # Faster than indexing [2,1,0]
    
    # Get original dimensions
    _, h, w = frame_tensor.shape

    # Padding to square (height < width)
    if h < w:
        pad_h = w - h
        frame_tensor = F.pad(frame_tensor, (0, 0, pad_h // 2, pad_h - pad_h // 2), value=0)
    elif w < h:
        pad_w = h - w
        frame_tensor = F.pad(frame_tensor, (pad_w // 2, pad_w - pad_w // 2, 0, 0), value=0)
    
    # Resize with anti-aliasing if supported
    frame_tensor = F.interpolate(frame_tensor.unsqueeze(0), size=(input_size, input_size), mode='bilinear', align_corners=False)

    # Normalize in-place
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    frame_tensor.sub_(mean).div_(std)

    return frame_tensor  # shape: (1, 3, input_size, input_size)

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