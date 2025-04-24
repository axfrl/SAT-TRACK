import torch
import cv2
import os
import time
from queue import Queue
from threading import Thread, Event
import torch.nn.functional as F 
from engines.funcs.video_stream_funcs import get_transform, preprocess_frame, create_empty_targets, write_frames
from utils.visualization import vis_vertices_img
from tracker.PHALP import PHALP
from structures.boxes import Boxes
from structures.instances import Instances
from structures import pairwise_iou
from segment_anything import SamPredictor, sam_model_registry
from pycocotools import mask as mask_utils
from configs.base import CACHE_DIR
from external.deep_sort_ import nn_matching
from external.deep_sort_.detection import Detection
from external.deep_sort_.tracker import Tracker
from models.human_models.hmar import HMAR
from models.predictor import Pose_transformer_v2
from utils.utils import (convert_pkl, get_prediction_interval,
                               progress_bar, smpl_to_pose_camera_vector)
from utils.utils_dataset import process_image, process_mask
from utils.utils_download import cache_url

class TrackModel(nn.Module):
        def __init__(self, cfg, device, sat_model):
            super(TrackModel, self).__init__()

            self.sat_model = sat_model
            self.cfg = cfg
            self.device = device
            self.eval_keys       = ['tracked_ids', 'tracked_bbox', 'tid', 'bbox', 'tracked_time']
            self.history_keys    = ['appe', 'loca', 'pose', 'uv'] if self.cfg.render.enable else []
            self.prediction_keys = ['prediction_uv', 'prediction_pose', 'prediction_loca'] if self.cfg.render.enable else []
            self.extra_keys_1    = ['center', 'scale', 'size', 'img_path', 'img_name', 'class_name', 'conf', 'annotations']
            self.extra_keys_2    = ['smpl', 'camera', 'camera_bbox', '3d_joints', '2d_joints', 'mask', 'extra_data']
            self.history_keys    = self.history_keys + self.extra_keys_1 + self.extra_keys_2
            self.visual_store_   = self.eval_keys + self.history_keys + self.prediction_keys
            self.tmp_keys_       = ['uv', 'prediction_uv', 'prediction_pose', 'prediction_loca']

            self.tracked_frames = []
            self.final_visuals_dic = {}

            # download wights and configs from Google Drive
            self.cached_download_from_drive()

            self.setup_hmr()

            # setup temporal pose predictor
            self.setup_predictor()
            
            # move to device
            self.to(self.device)
            
            # train or eval
            self.train() if(self.cfg.train) else self.eval()
            
            # create nessary directories
            self.default_setup()

        def setup_hmr(self):
            print("Loading HMAR model...")
            self.HMAR = HMAR(self.cfg)
            self.HMAR.load_weights(self.cfg.hmr.hmar_path)

        def setup_predictor(self):
            print("Loading Predictor model...")
            self.pose_predictor = Pose_transformer_v2(self.cfg, self)
            self.pose_predictor.load_weights(self.cfg.pose_predictor.weights_path)
            
        def setup_deepsort(self):
            print("Setting up DeepSort...")
            metric  = nn_matching.NearestNeighborDistanceMetric(self.cfg, self.cfg.phalp.hungarian_th, self.cfg.phalp.past_lookback)
            self.tracker = Tracker(self.cfg, metric, max_age=self.cfg.phalp.max_age_track, n_init=self.cfg.phalp.n_init, phalp_tracker=self, dims=[4096, 4096, 99])  
        
        def setup_postprocessor(self):
            # by default this will not be initialized
            self.postprocessor = Postprocessor(self.cfg, self)

        def default_setup(self):
            # create subfolders for saving additional results
            try:
                os.makedirs(self.cfg.video.output_dir + '/results', exist_ok=True)  
                os.makedirs(self.cfg.video.output_dir + '/results_tracks', exist_ok=True)  
                os.makedirs(self.cfg.video.output_dir + '/_TMP', exist_ok=True)  
                os.makedirs(self.cfg.video.output_dir + '/_DEMO', exist_ok=True)  
            except: 
                pass
        
        def get_detections(self, image, sat_data, frame_name, t_, additional_data=None, measurements=None):
            """
            Get detections using SAT-HMR model, replacing Detectron2 mask processing.
            
            Args:
                image: Input image (numpy array or PIL Image).
                frame_name: Frame identifier.
                t_: Time step (unused here).
                additional_data: Optional dictionary with ground-truth data.
                measurements: Tuple (img_height, img_width, new_image_size, left, top).
            
            Returns:
                pred_bbox: Predicted bounding boxes [x1, y1, x2, y2].
                pred_bbox: Duplicate for compatibility.
                pred_masks: Synthetic masks.
                pred_scores: Confidence scores.
                pred_classes: Class IDs (0 for people).
                ground_truth_track_id: Track IDs.
                ground_truth_annotations: Annotations.
                smpl_params: Optional SMPL parameters (if PHALP can use them).
            """
            img_height, img_width = image.shape[:2] if isinstance(image, np.ndarray) else image.size[::-1]
            
            # Convert outputs to Instances
            pred_boxes = sat_data['pred_boxes'][0]  # [num_queries, 4] (center_x, center_y, w, h)
            pred_confs = sat_data['pred_confs'][0].squeeze(-1)  # [num_queries]
            pred_classes = torch.zeros_like(pred_confs, dtype=torch.long)  # Class 0 (person)
            
            # Convert boxes to [x1, y1, x2, y2]
            cx, cy, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
            x1 = (cx - w / 2) * img_width
            y1 = (cy - h / 2) * img_height
            x2 = (cx + w / 2) * img_width
            y2 = (cy + h / 2) * img_height
            pred_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
            
            instances = Instances((img_height, img_width))
            instances.pred_boxes = Boxes(pred_boxes)
            instances.scores = pred_confs
            instances.pred_classes = pred_classes
            instances.smpl_params = {
                'poses': sat_data['pred_poses'][0],
                'betas': sat_data['pred_betas'][0],
                'j3ds': sat_data['pred_j3ds'][0],
                'j2ds': sat_data['pred_j2ds'][0],
                'verts': sat_data['pred_verts'][0],
                'transl': sat_data['pred_transl'][0]
            }
            
            if frame_name in additional_data.keys() and False:
                img_height, img_width, new_image_size, left, top = measurements
                gt_bbox = additional_data[frame_name]["gt_bbox"]
                ground_truth_track_id = additional_data[frame_name]["extra_data"].get('gt_track_id', [-1] * len(gt_bbox))
                ground_truth_annotations = additional_data[frame_name]["extra_data"].get('gt_class', [[]] * len(gt_bbox))
                
                # Create Instances for ground-truth boxes
                gt_inst = Instances((img_height, img_width))
                bbox_array = []
                class_array = []
                scores_array = []
                
                # Convert ground-truth boxes to [x1, y1, x2, y2]
                for bbox_ in gt_bbox:
                    x1 = bbox_[0]
                    y1 = bbox_[1]
                    x2 = bbox_[2] + x1
                    y2 = bbox_[3] + y1
                    bbox_array.append([x1, y1, x2, y2])
                    class_array.append(0)  # Person class
                    scores_array.append(1.0)  # Max confidence for GT
                
                bbox_array = np.array(bbox_array)
                class_array = np.array(class_array)
                gt_inst.pred_boxes = Boxes(torch.as_tensor(bbox_array, dtype=torch.float32, device=self.detector.device))
                gt_inst.pred_classes = torch.as_tensor(class_array, dtype=torch.long, device=self.detector.device)
                gt_inst.scores = torch.as_tensor(scores_array, dtype=torch.float32, device=self.detector.device)
                
                # Match predicted boxes with ground-truth boxes
                # iou = pairwise_iou(gt_inst.pred_boxes, instances.pred_boxes)
                #max_iou, match_idx = iou.max(dim=1)
                # TODO



                
                valid = max_iou > 0.5  # IoU threshold for matching
                
                matched_instances = Instances((img_height, img_width))
                matched_instances.pred_boxes = instances.pred_boxes[match_idx[valid]]
                matched_instances.scores = instances.scores[match_idx[valid]]
                matched_instances.pred_classes = instances.pred_classes[match_idx[valid]]
                matched_instances.smpl_params = {
                    key: val[match_idx[valid]] for key, val in instances.smpl_params.items()
                }
                
                instances_people = matched_instances[matched_instances.pred_classes == 0]
                
                pred_bbox = instances_people.pred_boxes.tensor.cpu().numpy()
                pred_scores = instances_people.scores.cpu().numpy()
                pred_classes = instances_people.pred_classes.cpu().numpy()
                smpl_params = instances_people.smpl_params
                
                # Generate synthetic masks
                pred_masks = self._generate_synthetic_masks(pred_bbox, img_height, img_width)
            
            else:
                # Filter instances for people and score threshold
                instances_people = instances[instances.pred_classes == 0]
                instances_people = instances_people[instances_people.scores > self.cfg.phalp.low_th_c]
                
                pred_bbox = instances_people.pred_boxes.tensor.cpu().numpy()
                pred_scores = instances_people.scores.cpu().numpy()
                pred_classes = instances_people.pred_classes.cpu().numpy()
                smpl_params = instances_people.smpl_params
                
                # Generate synthetic masks
                pred_masks = self._generate_synthetic_masks(pred_bbox, img_height, img_width)
                
                ground_truth_track_id = [1] * len(pred_scores)
                ground_truth_annotations = [[]] * len(pred_scores)
            
            return pred_bbox, pred_bbox, pred_masks, pred_scores, pred_classes, ground_truth_track_id, ground_truth_annotations, smpl_params

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
        
        def _generate_sam_masks(self, bboxes, image, img_height, img_width):
            sam = sam_model_registry["vit_h"](checkpoint="path/to/sam_vit_h.pth")
            predictor = SamPredictor(sam)
            predictor.set_image(image)
            
            masks = []
            for bbox in bboxes:
                box = np.array([bbox[0], bbox[1], bbox[2], bbox[3]])
                mask, _, _ = predictor.predict(box=box, multimask_output=False)
                masks.append(mask[0])
            return np.array(masks)
        
        def get_human_features(self, sat_data, image, frame_name, t_, measurments, gt=None, ann=None, extra_data=None):
            """
            Get human features using SAT-HMR outputs directly and HMAR for appearance and UV maps.
            
            Args:
                sat_data: Dictionary from SAT-HMR's forward method (pred_boxes, pred_confs, pred_poses, etc.).
                image: Input image (numpy array or PIL Image).
                frame_name: Frame identifier.
                t_: Time step.
                measurments: Tuple (img_height, img_width, new_image_size, left, top).
                gt: Ground-truth flags (optional, default None).
                ann: Annotations (optional, default None).
                extra_data: Extra data (optional, default None).
            
            Returns:
                detection_data_list: List of Detection objects with human features.
            """
            img_height, img_width, new_image_size, left, top = measurments
            ratio = 1.0 / int(new_image_size) * self.cfg.render.res

            # Extract SAT-HMR outputs (batch_size=1)
            pred_boxes = sat_data['pred_boxes'][0]  # [num_queries, 4] (center_x, center_y, w, h)
            pred_confs = sat_data['pred_confs'][0].squeeze(-1)  # [num_queries]
            pred_poses = sat_data['pred_poses'][0]  # [num_queries, 72]
            pred_betas = sat_data['pred_betas'][0]  # [num_queries, 10]
            pred_j3ds = sat_data['pred_j3ds'][0]  # [num_queries, num_joints, 3]
            pred_j2ds = sat_data['pred_j2ds'][0]  # [num_queries, num_joints, 2]
            pred_transl = sat_data['pred_transl'][0]  # [num_queries, 3]

            # Convert boxes to [x1, y1, x2, y2]
            cx, cy, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]
            x1 = (cx - w / 2) * img_width
            y1 = (cy - h / 2) * img_height
            x2 = (cx + w / 2) * img_width
            y2 = (cy + h / 2) * img_height
            pred_bbox = torch.stack([x1, y1, x2, y2], dim=-1).cpu().numpy()

            # Filter detections based on score and size thresholds
            NPEOPLE = len(pred_confs)
            masked_image_list = []
            center_list = []
            scale_list = []
            rles_list = []
            selected_ids = []
            pred_classes = np.zeros(NPEOPLE, dtype=np.int64)  # Class 0 (person)
            ground_truth_track_id = [1] * NPEOPLE  # Default track IDs
            ground_truth_annotations = [[]] * NPEOPLE  # Default annotations

            for p_ in range(NPEOPLE):
                if pred_confs[p_] < self.cfg.phalp.low_th_c:
                    continue
                w = pred_bbox[p_][2] - pred_bbox[p_][0]
                h = pred_bbox[p_][3] - pred_bbox[p_][1]
                if w < self.cfg.phalp.small_w or h < self.cfg.phalp.small_h:
                    continue
                
                # Generate synthetic mask
                mask = self._generate_synthetic_masks(pred_bbox[p_][None], img_height, img_width)[0]
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('utf-8')

                # Compute center, scale, and cropped image for HMAR
                center = np.array([(pred_bbox[p_][0] + pred_bbox[p_][2]) / 2, (pred_bbox[p_][1] + pred_bbox[p_][3]) / 2])
                scale = np.array([w, h]) / self.cfg.render.res
                bbox = pred_bbox[p_]
                bbox_pad = np.array([bbox[0] - w * 0.125, bbox[1] - h * 0.125, bbox[2] + w * 0.125, bbox[3] + h * 0.125])
                masked_image, center_pad, scale_pad, _, _, _ = self.get_croped_image(image, bbox, bbox_pad, mask)
                
                masked_image_list.append(masked_image)
                center_list.append(center_pad)
                scale_list.append(scale_pad)
                rles_list.append(rle)
                selected_ids.append(p_)
            
            if len(selected_ids) == 0:
                return []

            BS = len(selected_ids)
            
            # Run HMAR for appearance and UV maps
            if BS > 0:
                masked_image_list = torch.stack(masked_image_list, dim=0)
                with torch.no_grad():
                    extra_args = {}
                    hmar_out = self.HMAR(masked_image_list.cuda(), **extra_args)
                    uv_vector = hmar_out['uv_vector'].cpu().numpy()  # [BS, 256, 256, 3]
                    appe_embedding = self.HMAR.autoencoder_hmar(uv_vector, en=True).view(BS, -1)  # [BS, embedding_dim]
            else:
                uv_vector = np.zeros((BS, 256, 256, 3))
                appe_embedding = torch.zeros(BS, 512)  # Dummy embedding

            # Prepare SMPL parameters and joints from SAT-HMR
            pred_smpl_params = [
                {
                    'global_orient': pred_poses[i, :3].cpu().numpy(),
                    'body_pose': pred_poses[i, 3:].cpu().numpy(),
                    'betas': pred_betas[i].cpu().numpy()
                } for i in selected_ids
            ]
            pred_joints_3d = pred_j3ds[selected_ids].cpu().numpy()  # [BS, num_joints, 3]
            pred_joints_2d = pred_j2ds[selected_ids].cpu().numpy()  # [BS, num_joints, 2]
            pred_cam = pred_transl[selected_ids].cpu().numpy()  # [BS, 3]

            # Compute pose embedding
            if self.cfg.phalp.pose_distance == "joints":
                pose_embedding = torch.from_numpy(pred_joints_3d).view(BS, -1)
            elif self.cfg.phalp.pose_distance == "smpl":
                pose_embedding = []
                for i in range(BS):
                    pose_embedding_ = smpl_to_pose_camera_vector(pred_smpl_params[i], pred_cam[i])
                    pose_embedding.append(torch.from_numpy(pose_embedding_[0]))
                pose_embedding = torch.stack(pose_embedding, dim=0)
            else:
                raise ValueError("Unknown pose distance")
            
            # Compute location embedding
            pred_joints_2d_ = pred_joints_2d.reshape(BS, -1) / self.cfg.render.res
            pred_cam_ = pred_cam.reshape(BS, -1)
            loca_embedding = torch.cat((torch.from_numpy(pred_joints_2d_), torch.from_numpy(pred_cam_),
                                    torch.from_numpy(pred_cam_), torch.from_numpy(pred_cam_)), dim=1)

            # Compute full embedding (for legacy)
            full_embedding = torch.cat((appe_embedding.cpu(), pose_embedding, loca_embedding), dim=1)

            # Create detection data list
            detection_data_list = []
            for i, p_ in enumerate(selected_ids):
                detection_data = {
                    "bbox": np.array([pred_bbox[p_][0], pred_bbox[p_][1],
                                    pred_bbox[p_][2] - pred_bbox[p_][0], pred_bbox[p_][3] - pred_bbox[p_][1]]),
                    "mask": rles_list[i],
                    "conf": pred_confs[p_].cpu().numpy(),
                    "appe": appe_embedding[i].cpu().numpy(),
                    "pose": pose_embedding[i].numpy(),
                    "loca": loca_embedding[i].numpy(),
                    "uv": uv_vector[i],
                    "embedding": full_embedding[i].numpy(),
                    "center": center_list[i] + np.array([left, top]) * ratio,
                    "scale": scale_list[i] * ratio,
                    "smpl": pred_smpl_params[i],
                    "camera": pred_cam_[i],
                    "camera_bbox": pred_cam_[i],  # Use transl as proxy
                    "3d_joints": pred_joints_3d[i],
                    "2d_joints": pred_joints_2d_[i],
                    "size": [img_height, img_width],
                    "img_path": frame_name,
                    "img_name": frame_name.split('/')[-1] if isinstance(frame_name, str) else None,
                    "class_name": pred_classes[p_],
                    "time": t_,
                    "ground_truth": gt[p_] if gt is not None else ground_truth_track_id[p_],
                    "annotations": ann[p_] if ann is not None else ground_truth_annotations[p_],
                    "extra_data": extra_data[p_] if extra_data is not None else None
                }
                detection_data_list.append(Detection(detection_data))

            return detection_data_list
        
        def track(self, frame, sat_data, t_, frame_name):

            img_height, img_width, _  = frame.shape
            new_image_size            = max(img_height, img_width)
            top, left                 = (new_image_size - img_height)//2, (new_image_size - img_width)//2,
            measurments               = [img_height, img_width, new_image_size, left, top]
            self.cfg.phalp.shot       = 1 # or 0 ???

            final_visuals_dic = self.final_visuals_dic
            tracked_frames = self.tracked_frames
            if(self.cfg.render.enable):
                # reset the renderer
                # TODO: add a flag for full resolution rendering
                self.cfg.render.up_scale = int(self.cfg.render.output_resolution / self.cfg.render.res)
                self.visualizer.reset_render(self.cfg.render.res*self.cfg.render.up_scale)
            
            ############ detection ##############
            pred_bbox, pred_bbox_pad, pred_masks, pred_scores, pred_classes, gt_tids, gt_annots = self.get_detections(image_frame, frame_name, t_, additional_data, measurments)

            ############ Run EXTRA models to attach to the detections ##############
            extra_data = self.run_additional_models(frame, pred_bbox, pred_masks, pred_scores, pred_classes, frame_name, t_, measurments, gt_tids, gt_annots)
            
            ############ HMAR ##############
            detections = self.get_human_features(frame, pred_masks, pred_bbox, pred_bbox_pad, pred_scores, frame_name, pred_classes, t_, measurments, gt_tids, gt_annots, extra_data)

            ############ tracking ##############
            self.tracker.predict()
            self.tracker.update(detections, t_, frame_name, self.cfg.phalp.shot)

            ############ record the results ##############
            final_visuals_dic.setdefault(frame_name, {'time': t_, 'shot': self.cfg.phalp.shot, 'frame_path': frame_name})
            if(self.cfg.render.enable): final_visuals_dic[frame_name]['frame'] = frame
            for key_ in self.visual_store_: final_visuals_dic[frame_name][key_] = []
            
            ############ record the track states (history and predictions) ##############
            for tracks_ in self.tracker.tracks:
                if(frame_name not in tracked_frames): tracked_frames.append(frame_name)
                if(not(tracks_.is_confirmed())): continue
                
                track_id        = tracks_.track_id
                track_data_hist = tracks_.track_data['history'][-1]
                track_data_pred = tracks_.track_data['prediction']

                final_visuals_dic[frame_name]['tid'].append(track_id)
                final_visuals_dic[frame_name]['bbox'].append(track_data_hist['bbox'])
                final_visuals_dic[frame_name]['tracked_time'].append(tracks_.time_since_update)

                for hkey_ in self.history_keys:     final_visuals_dic[frame_name][hkey_].append(track_data_hist[hkey_])
                for pkey_ in self.prediction_keys:  final_visuals_dic[frame_name][pkey_].append(track_data_pred[pkey_.split('_')[1]][-1])

                if(tracks_.time_since_update==0):
                    final_visuals_dic[frame_name]['tracked_ids'].append(track_id)
                    final_visuals_dic[frame_name]['tracked_bbox'].append(track_data_hist['bbox'])
                    
                    if(tracks_.hits==self.cfg.phalp.n_init):
                        for pt in range(self.cfg.phalp.n_init-1):
                            track_data_hist_ = tracks_.track_data['history'][-2-pt]
                            track_data_pred_ = tracks_.track_data['prediction']
                            frame_name_      = tracked_frames[-2-pt]
                            final_visuals_dic[frame_name_]['tid'].append(track_id)
                            final_visuals_dic[frame_name_]['bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_ids'].append(track_id)
                            final_visuals_dic[frame_name_]['tracked_bbox'].append(track_data_hist_['bbox'])
                            final_visuals_dic[frame_name_]['tracked_time'].append(0)

                            for hkey_ in self.history_keys:    final_visuals_dic[frame_name_][hkey_].append(track_data_hist_[hkey_])
                            for pkey_ in self.prediction_keys: final_visuals_dic[frame_name_][pkey_].append(track_data_pred_[pkey_.split('_')[1]][-1])
            smooth_pose = self.pose_predictor.smoo
            return 