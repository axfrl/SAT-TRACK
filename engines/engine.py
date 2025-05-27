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
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass

import json
from glob import glob
from utils.preprocess import get_img
from utils.postprocess import post_process

from .funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets
from utils.visualization import (
    vis_vertices_img,
    vis_pose_img,
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

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def display_bbox(image, bbox, box_color='red', linewidth=2):
    """
    Affiche une image avec une boîte englobante au format COCO.

    Args:
        image (np.ndarray): Image en format H x W x 3 (uint8 ou float)
        bbox (list or tuple): Bounding box [x_min, y_min, width, height]
        box_color (str): Couleur de la boîte (default: 'red')
        linewidth (int): Épaisseur du trait (default: 2)
    """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Créer un rectangle pour la bounding box
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h, linewidth=linewidth, edgecolor=box_color, facecolor='none')
    ax.add_patch(rect)

    plt.axis('off')
    plt.show()

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
        tensor, target = get_img(frame, True, self.input_size, self.device)
        if self.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(tensor, target)
                
        img = post_process(tensor[0], None, target[0]["img_size"])

        dets = self.phalp_tracker.get_human_features(
            sat_data=outputs,
            image=img,
            frame_name=str(frame_id),
            t_=frame_id,
            measurments=(h, w, input_size),
            conf_thresh=self.conf_thresh
        )

        K = outputs['pred_intrinsics'][0].reshape(3,3).detach().cpu()
        self.phalp_tracker.tracker.predict()
        self.phalp_tracker.tracker.update(dets, frame_id, str(frame_id), self.phalp_tracker.cfg.phalp.shot)

        detec_pose, hist_pose = {}, {}
        for tr in self.phalp_tracker.tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update >= 1:
                continue
            tid = tr.track_id
            history = tr.track_data['history']
            detec_pose[tid] = history[-1]['3d_joints']
            if len(history) > 3:
                hist_pose[tid] = history[-3]['3d_joints']

        confs = outputs['pred_confs'][0].view(-1)
        if (mask := confs > conf_thresh).any():
            verts = [outputs['pred_verts'][0,i].cpu().numpy() for i,v in enumerate(confs) if mask[i]]
            print(target[0]["img_size"][0], target[0]["img_size"][1])
            print(frame.shape)
            img_size = target[0]["img_size"]
            frame = vis_vertices_img(img, verts, K)
            frame = vis_pose_img(frame, detec_pose, K, self.phalp_tracker.color_dict)[:target[0]["img_size"][0], :target[0]["img_size"][1]]

        return frame
    
    def infer_video(
        self,
        input_path,
        output_dir,
        input_size,
        conf_thresh,
        display=False,
        default_fps=30.0,
    ):
        """
        Process a video given either a video file or a folder of image frames.

        Args:
            input_path (str): Path to input video file or folder of image frames.
            output_video (str): Path to save output video.
            input_size (tuple): Input size for processing (width, height).
            conf_thresh (float): Confidence threshold.
            display (bool): Whether to display frames in real-time.
            default_fps (float): FPS to use when reading from image folder.
        """
        # Prepare threading primitives
        frame_queue = queue.Queue(maxsize=10)
        result_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()

        # Determine input source
        if os.path.isdir(input_path):
            # Folder of image frames
            files = sorted(
                f for f in os.listdir(input_path)
                if os.path.splitext(f)[1].lower() in ['.jpg', '.jpeg', '.png', '.bmp']
            )
            if not files:
                raise RuntimeError(f"No image files found in folder: {input_path}")

            fps = default_fps
            print(f"[INFO] Reading {len(files)} frames from folder '{input_path}' at {fps} FPS")

            def reader():
                for fname in files:
                    if stop_event.is_set():
                        break
                    path = os.path.join(input_path, fname)
                    frame = cv2.imread(path)
                    if frame is None:
                        print(f"[WARNING] Could not read frame: {path}")
                        continue
                    frame_queue.put(frame)
                frame_queue.put(None)

            reader_target = reader
            def writer():
                index = 0
                while True:
                    frame = result_queue.get()
                    if frame is None:
                        break
                    out_path = os.path.join(output_dir, f"frame_{index:05d}.jpg")
                    cv2.imwrite(out_path, frame)
                    index += 1
        else:
            # Video file
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video: {input_path}")

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or np.isnan(fps):
                print(f"[WARNING] Invalid FPS in input; using default {default_fps}")
                fps = default_fps
            else:
                print(f"[INFO] Detected input FPS: {fps}")

            def reader():
                while not stop_event.is_set():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_queue.put(frame)
                frame_queue.put(None)

            reader_target = reader
            def writer():
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_dir, fourcc, fps, (1920, 1080))
                while True:
                    frame = result_queue.get()
                    if frame is None:
                        break
                    out.write(frame)
                out.release()
            
        # Start reader and writer threads
        reader_thread = threading.Thread(target=reader_target, daemon=True)
        writer_thread = threading.Thread(target=writer, daemon=True)

        reader_thread.start()
        writer_thread.start()

        # Main processing loop
        frame_count = 0
        start_time = time.time()

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
                    stop_event.set()
                    break

        # Signal writer to finish and clean up
        result_queue.put(None)
        writer_thread.join()
        stop_event.set()

        # Release resources
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        fps_processed = frame_count / elapsed if elapsed > 0 else 0
        print(f"Processed {frame_count} frames in {elapsed:.2f}s ({fps_processed:.2f} FPS)")


    def annote_posetrack_frame(self, frame, frame_id, input_size):
        h, w = frame.shape[:2]
        tensor, target = get_img(frame, True, self.input_size, self.device)
        if self.use_fp16:
            tensor = tensor.half()

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda"):
                outputs = self.model(tensor, target)

        img = post_process(tensor[0], None, target[0]["img_size"])

        dets = self.phalp_tracker.get_human_features(
            sat_data=outputs,
            image=img,
            frame_name=str(frame_id),
            t_=frame_id,
            measurments=(h, w, input_size),
            conf_thresh=self.conf_thresh
        )

        with torch.no_grad():
            self.phalp_tracker.tracker.predict()
            self.phalp_tracker.tracker.update(dets, frame_id, str(frame_id), self.phalp_tracker.cfg.phalp.shot)

        result = []
        for tr in self.phalp_tracker.tracker.tracks:
            if not tr.is_confirmed() or tr.time_since_update >= 1:
                continue
            tid = tr.track_id

            history = tr.track_data['history'][-1]
            keypoints = history['keypoints']
            conf = history['conf']
            bbox = history['bbox']
            
            #display_bbox(frame, [float(x) for x in bbox])
            track_dict = {
                "id": -1,  # sera mis dans la fonction appelante
                "bbox": [float(x) for x in bbox],
                "image_id": int(frame_id),
                "keypoints": [float(x) for x in keypoints],
                "scores": [float(conf) for _ in range(len(keypoints)//3)],
                "person_id": int(tid),
                "track_id": int(tid)
            }
            result.append(track_dict)

        return result
    
    def _process_scene(self, args):
        scene, root_eval_dir, input_size = args
        images = []
        annotations = []
        scene_path = os.path.join(root_eval_dir, scene)
        img_paths = sorted(glob(os.path.join(scene_path, "*.jpg")))

        for frame_idx, img_path in enumerate(img_paths, start=1):
            image = cv2.imread(img_path)
            if image is None:
                print(f"[Warning] Image introuvable: {img_path}")
                continue

            # appel à la méthode d'annotation
            detections = self.annote_posetrack_frame(image, frame_idx, input_size)

            images.append({
                        "scene": scene,
                        "frame_id": frame_idx,
                        "file_name": os.path.join('/images/val', scene, os.path.basename(img_path))
                    })

            for det in detections:
                det_record = det.copy()
                det_record["scene"] = scene
                det_record["frame_id"] = frame_idx
                annotations.append(det_record)

        return images, annotations

    def create_posetrack_json(self, root_eval_dir, output_json_dir, num_workers=4):
        scene_dirs = sorted([d for d in os.listdir(root_eval_dir)
                            if os.path.isdir(os.path.join(root_eval_dir, d))])

        args_list = [(scene, root_eval_dir, self.input_size) for scene in scene_dirs]

        os.makedirs(output_json_dir, exist_ok=True)

        # Traitement parallèle
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(self._process_scene, args_list),
                                total=len(args_list),
                                desc="Traitement des scènes"))

        for (scene, _, _), (images, annotations) in zip(args_list, results):
            image_id_map = {}
            for img in images:
                scene_str = img["scene"]
                frame_num = img["frame_id"]

                try:
                    scene_digits = int(scene_str[:6])
                except ValueError:
                    raise ValueError(f"Le nom de scène '{scene_str}' ne commence pas par 6 chiffres.")

                new_id = int(f"1{scene_digits:06d}{frame_num:04d}")
                key = (img["scene"], img["frame_id"])
                img["id"] = new_id-1
                img["image_id"] = new_id-1
                image_id_map[key] = new_id-1

            for ann in annotations:
                key = (ann["scene"], ann["frame_id"])
                ann["image_id"] = image_id_map[key]
                ann["id"] = ann["image_id"] * 1000 + ann["track_id"]
                ann.pop("frame_id")

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

            output_path = os.path.join(output_json_dir, f"{scene}.json")
            with open(output_path, "w") as f:
                json.dump(out, f, indent=2)

            print(f"✅ JSON scène écrit : {output_path} ({len(images)} images, {len(annotations)} annotations)")



