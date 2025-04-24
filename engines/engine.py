import torch
import cv2
import os
import time
from queue import Queue
from threading import Thread, Event
import torch.nn.functional as F 
from .funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets, write_frames
from utils.visualization import vis_vertices_img
#from tracker.PHALP import PHALP
#from structures.boxes import Boxes
#from structures.instances import Instances
#from structures import pairwise_iou
#from segment_anything import SamPredictor, sam_model_registry
#from pycocotools import mask as mask_utils
#from external.deep_sort_.detection import Detection
#from tracker.SAT_TRACKER import TrackModel

class Engine:
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
        #self.phalp_tracker = TrackModel(args, self.device, self.model)
    
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
            frame_name = input_video + str(frame_count)

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

            # Tracking
            #self.phalp_tracker.tracker.predict()
            #self.phalp_tracker.tracker.update(detections = [detection],
                                              #t_ = frame_count,
                                              #frame_name = frame_name,
                                              #self.phalp_tracker.cfg.phalp.shot)
            
            # Pose smoothing (post-processing)
            # TODO

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