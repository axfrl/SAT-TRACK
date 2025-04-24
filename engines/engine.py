import torch
import cv2
import os
import time
from queue import Queue
from threading import Thread, Event
import torch.nn.functional as F 
from .funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets, write_frames
from utils.visualization import vis_vertices_img
from tracker.PHALP import PHALP

class Engine:
    class TrackModel(PHALP):
        def __init(self, cfg, device):
            super().__init__(cfg)

        def get_detections(self, image, frame_name, t_, additional_data=None, measurements=None):
            """
            Get detections using SAT-HMR's prediction head, replacing Detectron2 mask processing.
            
            Args:
                image: Input image (numpy array or PIL Image).
                frame_name: Frame identifier.
                t_: Time step (unused here).
                additional_data: Optional dictionary with ground-truth data.
                measurements: Tuple (img_height, img_width, new_image_size, left, top).
            
            Returns:
                pred_bbox: Predicted bounding boxes [x1, y1, x2, y2].
                pred_bbox: Duplicate for compatibility.
                pred_masks: Synthetic masks or None.
                pred_scores: Confidence scores.
                pred_classes: Class IDs (0 for people).
                ground_truth_track_id: Track IDs.
                ground_truth_annotations: Annotations.
            """
            # Initialize SAT-HMR model (load once during initialization, not here)
            if not hasattr(self, 'detector'):
                self.detector = SATHMRModel(
                    config_path="path/to/sat_hmr/config.yaml",
                    weights_path="path/to/sat_hmr/weights/sat_644.pth",
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
            
            img_height, img_width = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
            
            if frame_name in additional_data.keys():
                img_height, img_width, new_image_size, left, top = measurements
                gt_bbox = additional_data[frame_name]["gt_bbox"]
                ground_truth_track_id = additional_data[frame_name]["extra_data"].get('gt_track_id', [-1] * len(gt_bbox))
                ground_truth_annotations = additional_data[frame_name]["extra_data"].get('gt_class', [[]] * len(gt_bbox))
                
                inst = Instances((img_height, img_width))
                bbox_array = []
                class_array = []
                scores_array = []
                
                # Convert ground-truth boxes to [x1, y1, x2, y2] format
                for bbox_ in gt_bbox:
                    x1 = bbox_[0]
                    y1 = bbox_[1]
                    x2 = bbox_[2] + x1
                    y2 = bbox_[3] + y1
                    bbox_array.append([x1, y1, x2, y2])
                    class_array.append(0)  # Person class
                    scores_array.append(1.0)  # Assume max confidence for GT
                
                bbox_array = np.array(bbox_array)
                class_array = np.array(class_array)
                box = Boxes(torch.as_tensor(bbox_array, dtype=torch.float32))
                
                inst.pred_boxes = box
                inst.pred_classes = torch.as_tensor(class_array, dtype=torch.long)
                inst.scores = torch.as_tensor(scores_array, dtype=torch.float32)
                
                # Run SAT-HMR with provided bounding boxes
                outputs = self.detector.predict_with_bbox(image, inst)
                instances = outputs['instances']
                instances_people = instances[instances.pred_classes == 0]
                
                pred_bbox = instances_people.pred_boxes.tensor.cpu().numpy()
                pred_scores = instances_people.scores.cpu().numpy()
                pred_classes = instances_people.pred_classes.cpu().numpy()
                
                # Generate synthetic masks from bounding boxes
                pred_masks = self._generate_synthetic_masks(pred_bbox, img_height, img_width)
            
            else:
                # Convert image to tensor for SAT-HMR
                image_tensor = ToTensor()(image).to(self.detector.device)
                
                # Run SAT-HMR inference
                outputs = self.detector(image_tensor)
                instances = outputs['instances']
                instances = instances[instances.pred_classes == 0]
                instances = instances[instances.scores > self.cfg.phalp.low_th_c]
                
                pred_bbox = instances.pred_boxes.tensor.cpu().numpy()
                pred_scores = instances.scores.cpu().numpy()
                pred_classes = instances.pred_classes.cpu().numpy()
                
                # Generate synthetic masks
                pred_masks = self._generate_synthetic_masks(pred_bbox, img_height, img_width)
                
                ground_truth_track_id = [1] * len(pred_scores)
                ground_truth_annotations = [[]] * len(pred_scores)
            
            return pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, ground_truth_track_id, ground_truth_annotations

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







    def __init__(self, args, mode='infer', gpu_id=0):
        self.mode = mode
        self.conf_thresh = args.conf_thresh
        self.output_dir = args.output_dir
        self.live_stream = args.live_stream
        self.use_fp16 = args.use_fp16
        self.render_mode = args.render_mode  # 'points' or 'mesh'
        self.gpu_id = gpu_id
        self.device = self.set_device(gpu_id)
        os.makedirs(self.output_dir, exist_ok=True)
        self.prepare_models(args)


    
    def set_device(self, gpu_id=0):
        """Set device for a specific GPU or CPU."""
        if torch.cuda.is_available() and gpu_id < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_id}')
        return torch.device('cpu')
    
    def prepare_models(self, args):
        """Build and load the SAT-HMR model."""
        from models.sat_model import build_sat_model  # Delayed import to avoid circular dependencies
        print(f'Preparing models on GPU {self.gpu_id}...')
        self.model, _ = build_sat_model(args, set_criterion=False)
        if args.pretrain:
            print(f'Loading pretrained weights: {args.pretrain_path}')
            state_dict = torch.load(args.pretrain_path, weights_only=True)
            if 'encoder_pos_embeds' in state_dict:
                expected_size = (args.input_size // 14, args.input_size // 14)
                checkpoint_size = state_dict['encoder_pos_embeds'].shape[:2]
                if expected_size != checkpoint_size:
                    print(f"Adapting encoder_pos_embeds from {checkpoint_size} to {expected_size}")
                    pos_embeds = state_dict['encoder_pos_embeds'].permute(2, 0, 1).unsqueeze(0)
                    pos_embeds = F.interpolate(pos_embeds, size=expected_size, mode='bicubic', align_corners=False)
                    state_dict['encoder_pos_embeds'] = pos_embeds.squeeze(0).permute(1, 2, 0)
            self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model.to(self.device)
        if self.use_fp16:
            self.model = self.model.half()
        print(f'Model is on device: {self.device}')
        if 'cuda' in str(self.device):
            print(f'GPU Name: {torch.cuda.get_device_name(self.device)}')
        else:
            print('Warning: Model is running on CPU')

    def read_frames(self, cap, queue):
        """Read frames asynchronously."""
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            queue.put(frame)
        queue.put(None)

    def infer_video(self, input_video, output_video, input_size, conf_thresh, display):
        """Process a video or live stream on a single GPU."""
        device = self.device

        # Open video
        cap = cv2.VideoCapture(input_video)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video}")

        # Video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        # Frame queue for async reading (if live stream)
        frame_queue = Queue(maxsize=10)
        if self.live_stream:
            reader_thread = Thread(target=self.read_frames, args=(cap, frame_queue))
            reader_thread.start()

        # Result queue for video writing
        result_queue = Queue()
        stop_event = Event()
        if not self.live_stream:
            writer_thread = Thread(target=write_frames, args=(output_video, frame_width, frame_height, fps, result_queue, stop_event))
            writer_thread.start()

        frame_count = 0
        total_start_time = time.time()
        targets = create_empty_targets(device, [frame_height, frame_width])
        
        # Calculate padding to make the image square
        orig_w, orig_h = frame_width, frame_height
        transform = get_transform(input_size=input_size, orig_h=orig_h, orig_w=orig_w, device=device)
        
        while True:
            if self.live_stream:
                frame = frame_queue.get()
            else:
                ret, frame = cap.read()
                if not ret:
                    break
            if frame is None:
                break
            frame_count += 1

            # Preprocess frame
            t1 = time.time()
            input_tensor = preprocess_frame(frame, transform, device)
            if self.use_fp16:
                input_tensor = input_tensor.half()
            if frame_count == 1:
                print(f"GPU {self.gpu_id}: Input tensor is on device: {input_tensor.device}")
            
            t2 = time.time()

            # Model inference
            with torch.no_grad():
                outputs = self.model(input_tensor, targets)
            t3 = time.time()

            # Process outputs
            depths = outputs['pred_depths'][0, :, 0]
            confs = outputs['pred_confs'][0].view(-1)
            valid_mask = confs > conf_thresh

            if valid_mask.any():
                valid_depths = depths[valid_mask]
                valid_idxs = torch.arange(depths.shape[0], device=depths.device)[valid_mask]
                idx = valid_idxs[torch.argmin(valid_depths)].item()
            else:
                idx = torch.argmax(confs).item()

            pred_verts = outputs['pred_verts'][0, idx:idx+1].detach().cpu().numpy()
            cam_intrinsics = outputs['pred_intrinsics'][0].reshape(3, 3).detach().cpu()

            t4 = time.time()

            # Visualize vertices
            rendered_img = vis_vertices_img(frame, pred_verts, cam_intrinsics, frame_size=(frame_width, frame_height))
            t5 = time.time()

            # Put result in queue for writing
            result_queue.put((frame_count, rendered_img))
            t6 = time.time()

            # Display
            if display or self.live_stream:
                cv2.imshow('Inference Output', rendered_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        # Timing breakdown
        print(f"\nGPU {self.gpu_id}, Frame {frame_count} Timing Breakdown:")
        print(f"  Preprocessing: {t2 - t1:.4f} s")
        print(f"  Model Inference: {t3 - t2:.4f} s")
        print(f"  Post-processing: {t4 - t3:.4f} s")
        print(f"  Visualization: {t5 - t4:.4f} s")
        print(f"  Queue Output: {t6 - t5:.4f} s")
        print(f"  Total Frame Time: {t6 - t1:.4f} s")

        # Cleanup
        stop_event.set()
        cap.release()
        if not self.live_stream:
            writer_thread.join()
        cv2.destroyAllWindows()
        total_time = time.time() - total_start_time
        print(f"\nMain: Processed {frame_count} frames in {total_time:.2f} seconds ({frame_count/total_time:.2f} FPS)")