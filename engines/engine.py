import os
import time
import threading
import queue

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import json
from glob import glob

from .funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets
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
from topology.persistence_analysis import (
    compute_persistence_diagrams,
    compute_cross_distance_matrix
)
from utils.utils import pose_camera_vector_to_smpl, smpl_to_pred_pose_shape

class Engine:
    def __init__(self, args, mode='infer', gpu_id=0):
        self.mode = mode
        self.conf_thresh = args.sathmr.conf_thresh
        self.output_dir = args.video.output_dir
        self.live_stream = args.sathmr.live_stream
        self.use_fp16 = args.sathmr.use_fp16
        self.render_mode = args.sathmr.render_mode
        self.gpu_id = gpu_id
        self.device = self._set_device(gpu_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self._prepare_models(args.sathmr)
        self.phalp_tracker = TrackModel(args, self.device, self.model)
        self.input_size = 1288

    def _set_device(self, gpu_id=0):
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_id}')
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

    def _reader(self, cap, frame_queue, stop_event):
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            frame_queue.put(frame)
        frame_queue.put(None)

    def _pad_frame_to_standard_resolution(self, frame, target_width=1920, target_height=1080):
        h, w = frame.shape[:2]
        top = (target_height - h) // 2
        bottom = target_height - h - top
        left = (target_width - w) // 2
        right = target_width - w - left
        padded = cv2.copyMakeBorder(frame, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        return padded

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

    def _process_frame(self, frame, frame_id, input_size, conf_thresh, display):
        h, w = frame.shape[:2]
        transform = get_transform(input_size=input_size, orig_h=h, orig_w=w, device=self.device)
        tensor = preprocess_frame(frame, transform, self.device)
        if self.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(tensor, create_empty_targets(self.device, [h, w]))

        pad_h, pad_w = input_size - h, input_size - w
        left, top = pad_w // 2, pad_h // 2
        dets = self.phalp_tracker.get_human_features(
            sat_data=outputs,
            image=frame,
            frame_name=str(frame_id),
            t_=frame_id,
            measurments=(h, w, input_size, left, top)
        )
        K = outputs['pred_intrinsics'][0].reshape(3,3).detach().cpu()
        self.phalp_tracker.tracker.predict()
        self.phalp_tracker.tracker.update(dets, frame_id, str(frame_id), self.phalp_tracker.cfg.phalp.shot)

        detec_pose, hist_pose = {}, {}
        for tr in self.phalp_tracker.tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update >= 5:
                continue
            tid = tr.track_id
            history = tr.track_data['history']
            detec_pose[tid] = history[-1]['3d_joints']
            if len(history) > 3:
                hist_pose[tid] = history[-3]['3d_joints']

        #diag_hist = compute_persistence_diagrams(hist_pose)
        #diag_detec = compute_persistence_diagrams(detec_pose)
        #dist_mat = compute_cross_distance_matrix(list(diag_hist.values()), list(diag_detec.values()), epsilon=0.0)

        #heatmap = generate_heatmap_image(dist_mat, list(diag_hist.keys()), list(diag_detec.keys()))
        #diagram_img = generate_persistence_diagram_image(diag_detec, self.phalp_tracker.color_dict)

        confs = outputs['pred_confs'][0].view(-1)
        if (mask := confs > conf_thresh).any():
            verts = [outputs['pred_verts'][0,i].cpu().numpy() for i,v in enumerate(confs) if mask[i]]
            frame = vis_vertices_img(frame, verts, K, (w, h))
            frame = vis_vertices_img_with_tracked_pose(frame, detec_pose, K, (w,h), self.phalp_tracker.color_dict)

        #frame = add_left_border_to_frame(frame, 500, (1,1,1))
        #frame, hmap_h = overlay_heatmap_on_frame(frame, heatmap, position=(10,10), alpha=0.7, brightness_factor=1.5, size_factor=1.5)
        #final = overlay_diagram_on_frame(frame, diagram_img, position=(10,10), alpha=0.7, brightness_factor=1.5, size_factor=1.5, heatmap_height=hmap_h)
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
        t_start = time.time()

        while True:
            frame = frame_queue.get()
            if frame is None:
                break
            frame_count += 1
            processed = self._process_frame(frame, frame_count, input_size, conf_thresh, display)
            result_queue.put(processed)
            if display or self.live_stream:
                cv2.imshow('Output', processed)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        result_queue.put(None)
        writer.join()
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS)")

    @profile
    def create_posetrack_json_from_model(self, root_eval_dir, output_json_path):
        images = []
        annotations = []
        
        scene_dirs = sorted([d for d in os.listdir(root_eval_dir) if os.path.isdir(os.path.join(root_eval_dir, d))])

        for scene in tqdm(scene_dirs):
            scene_path = os.path.join(root_eval_dir, scene)
            img_paths = sorted(glob(os.path.join(scene_path, "*.jpg")))

            for frame_idx, img_path in enumerate(img_paths, start=1):
                image = cv2.imread(img_path)
                if image is None:
                    print(f"[Warning] Image introuvable: {img_path}")
                    continue

                detections = self.annote_posetrack_frame(image, frame_idx, self.input_size)

                images.append({
                    "id": frame_idx,
                    "file_name": os.path.join(scene, os.path.basename(img_path)),
                    "frame_id": frame_idx
                })

                for det in detections:
                    det["id"] = det["image_id"] * 1000 + det["track_id"]  # ID unique par annotation
                    annotations.append(det)

        categories = [{
            "id": 1,
            "name": "person",
            "keypoints": [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ],
            "skeleton": []
        }]

        out = {
            "images": images,
            "annotations": annotations,
            "categories": categories
        }

        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(out, f, indent=2)

        print(f"✅ JSON créé: {output_json_path}")
        print(f"   {len(images)} images, {len(annotations)} annotations, {len(scene_dirs)} scènes.")

    @profile
    def annote_posetrack_frame(self, frame, frame_id, input_size):
        h, w = frame.shape[:2]
        transform = get_transform(input_size=input_size, orig_h=h, orig_w=w, device=self.device)
        tensor = preprocess_frame(frame, transform, self.device)
        if self.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(tensor, create_empty_targets(self.device, [h, w]))

        pad_h, pad_w = input_size - h, input_size - w
        left, top = pad_w // 2, pad_h // 2

        dets = self.phalp_tracker.get_human_features(
            sat_data=outputs,
            image=frame,
            frame_name=str(frame_id),
            t_=frame_id,
            measurments=(h, w, input_size, left, top)
        )

        with torch.no_grad():
            self.phalp_tracker.tracker.predict()
            self.phalp_tracker.tracker.update(dets, frame_id, str(frame_id), self.phalp_tracker.cfg.phalp.shot)

        result = []
        for tr in self.phalp_tracker.tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update >= 5:
                continue
            tid = tr.track_id

            history = tr.track_data['history'][-1]
            keypoints = history['keypoints']
            conf = history['conf']
            bbox = history['bbox']

            track_dict = {
                "id": -1,  # sera mis dans la fonction appelante
                "bbox": list(bbox),
                "image_id": frame_id,
                "keypoints": keypoints,
                "scores": [conf] * (len(keypoints) // 3),
                "person_id": tid,
                "track_id": tid
            }

            result.append(track_dict)

        return result