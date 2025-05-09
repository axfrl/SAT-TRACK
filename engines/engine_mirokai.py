"""Model inference embedded on Mirokai robot using the python wheel pymirokai"""

import argparse
import asyncio
import logging
from pymirokai.robot import connect
from pymirokai.utils.get_local_ip import get_local_ip
from pymirokai.utils.run_until_interruption import run_until_interruption

logger = logging.getLogger("pymirokai")

import os
import time
import threading
import queue
import uuid
import cv2
import torch
import torch.nn.functional as F
import torch.profiler
import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table

from engines.funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets
from utils.visualization import (
    vis_vertices_img,
    vis_vertices_img_with_tracked_pose,
    generate_heatmap_image,
    generate_persistence_diagram_image,
    overlay_heatmap_on_frame,
    overlay_diagram_on_frame,
    add_left_border_to_frame
)
from tracker.SAT_TRACKER import TrackModel
"""from topology.persistence_analysis import (
    compute_persistence_diagrams,
    compute_cross_distance_matrix
)"""
from utils.utils import pose_camera_vector_to_smpl, smpl_to_pred_pose_shape

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

print("OpenCV version:", cv2.__version__)
print("FFMPEG? ", cv2.getBuildInformation() )


# RTSP stream URL (standard format)
rtsp_url = "rtsp://192.168.240.123:8554/head_color"

cap = None

class Engine:
    def __init__(self, args, mode='infer', gpu_id=0):
        self.mode = mode
        self.conf_thresh = args.sathmr.conf_thresh
        self.output_dir = args.video.output_dir
        self.live_stream = args.sathmr.live_stream
        self.use_fp16 = args.sathmr.use_fp16
        self.render_mode = args.sathmr.render_mode
        self.gpu_id = gpu_id
        self.tracker_on = args.tracker_on
        self.device = self._set_device(gpu_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self._prepare_models(args.sathmr)
        self.phalp_tracker = TrackModel(args, self.device, self.model)
        self.K = np.array([
            [372.33507966, 0., 437.45877085],
            [0., 344.35473032, 281.12838318],
            [0., 0., 1.]
        ], dtype=np.float32)

        self.dist_coeffs = np.array([[-0.81027268, 1.40567815, -0.06081206, 0.02541605, -0.70924915]], dtype=np.float32)

    def _set_device(self, gpu_id=0):
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            device = torch.device(f'cuda:{gpu_id}')
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(gpu_id)}")
            return device
        print("[WARNING] CUDA not available, falling back to CPU")
        return torch.device('cpu')

    def _prepare_models(self, sathmr_args):
        from models.sat_model import build_sat_model
        print(f'Preparing models on GPU {self.gpu_id}...')
        self.model, _ = build_sat_model(sathmr_args, set_criterion=False)
        if sathmr_args.pretrain:
            state_dict = torch.load(sathmr_args.pretrain_path, weights_only=True)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)
        if self.use_fp16:
            self.model = self.model.half()
        print(f'Model is on device: {self.device}')

    def _pad_frame_to_standard_resolution(self, frame, target_width=1920, target_height=1080):
        h, w = frame.shape[:2]
        top = (target_height - h) // 2
        bottom = target_height - h - top
        left = (target_width - w) // 2
        right = target_width - w - left
        padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded
    
    def _reader(self, cap, frame_queue, stop_event):
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        frame_queue.put(None)

    def _writer(self, output_path, width, height, fps, result_queue, stop_event):
        writer_initialized = False
        writer = None
        target_width, target_height = 1920, 1080
        n_frames_written = 0

        while not stop_event.is_set():
            try:
                result = result_queue.get(timeout=1)
                if result is None:
                    break

                padded_result = self._pad_frame_to_standard_resolution(result, target_width, target_height)

                if not writer_initialized:
                    print(f"[Writer] Initializing writer with size: {target_width}x{target_height}")
                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps,
                        (target_width, target_height)
                    )
                    writer_initialized = True

                writer.write(padded_result)
                n_frames_written += 1

            except queue.Empty:
                continue

        if writer is not None:
            writer.release()
        print(f"[Writer] Total frames written: {n_frames_written}")

    def _process_frame(self, frame, input_size, conf_thresh):
        h, w = frame.shape[:2]
        #print(h, w)
        transform, K = get_transform(input_size=input_size, orig_h=h, orig_w=w, device=self.device)
        tensor = preprocess_frame(frame, transform, self.device)
        if self.use_fp16:
            tensor = tensor.half()
        frame_id = 0
        # Check if tensor is on GPU
        if tensor.device.type != 'cuda' and self.device.type == 'cuda':
            print(f"[WARNING] Frame {frame_id}: Input tensor is on {tensor.device}, expected CUDA")

        # Profiler for model inference
        with torch.no_grad():
            outputs = self.model(tensor, create_empty_targets(self.device, [h, w]))

        # Tracker processing timing
        start_tracker = time.time()
        pad_h, pad_w = input_size - h, input_size - w
        left, top = pad_w // 2, pad_h // 2
        dets = self.phalp_tracker.get_human_features(
            sat_data=outputs,
            image=frame,
            frame_name=str(frame_id),
            t_=frame_id,
            measurments=(h, w, input_size, left, top)
        )

        #K = outputs['pred_intrinsics'][0].reshape(3,3).detach()

        if self.tracker_on:
            self.phalp_tracker.tracker.predict()
            self.phalp_tracker.tracker.update(dets, frame_id, str(frame_id), self.phalp_tracker.cfg.phalp.shot)

            detec_pose = {}
            for tr in self.phalp_tracker.tracker.tracks:
                if not tr.is_confirmed() or tr.time_since_update >= 5:
                    continue
                tid = tr.track_id
                history = tr.track_data['history']
                detec_pose[tid] = history[-1]['3d_joints']

        # Visualization timing
        confs = outputs['pred_confs'][0].view(-1)
        if (mask := confs > conf_thresh).any():
            verts = [outputs['pred_verts'][0,i].cpu().numpy() for i,v in enumerate(confs) if mask[i]]
            frame = vis_vertices_img(frame, verts, K, (w, h))
            if self.tracker_on:
                frame = vis_vertices_img_with_tracked_pose(frame, detec_pose, K, (w,h), self.phalp_tracker.color_dict)
        return frame

    def infer_video(self, input_video, output_video, input_size, conf_thresh, display=False):
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video {input_video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or np.isnan(fps):
            print("[Warning] Invalid FPS detected, using 30.0")
            fps = 30.0
        else:
            print(f"[INFO] Detected input FPS: {fps}")

        frame_queue = queue.Queue(maxsize=10)
        result_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()

        reader = threading.Thread(target=self._reader, args=(cap, frame_queue, stop_event), daemon=True)
        writer = threading.Thread(target=self._writer, args=(output_video, 1920, 1080, fps, result_queue, stop_event), daemon=True)
        reader.start(); writer.start()

        frame_count = 0
        
        self.model.eval()
        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            frame_count += 1
            processed = self._process_frame(frame, frame_count, input_size, conf_thresh)
            result_queue.put(processed)
            if self.live_stream:
                cv2.imshow('Output', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.model._print_timing_stats()
        result_queue.put(None)
        writer.join()
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

    async def infer(self, ip: str, api_key: str, input_size) -> None:
        """Run the robot, demonstrating various features."""
        cap = cv2.VideoCapture(rtsp_url)

        #print(f"Backend used: {cap.getBackendName()}")

        if not cap.isOpened():
            raise RuntimeError("Failed to open RTSP stream")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to retrieve frame")
                break

            processed = self._process_frame(frame, input_size, self.conf_thresh)

            if self.live_stream:
                cv2.imshow('Output', processed)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break