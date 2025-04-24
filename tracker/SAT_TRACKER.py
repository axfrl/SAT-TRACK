import torch
import torch.nn as nn
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
from utils.postprocessor import Postprocessor


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

        self.setup_postprocessor()
        
        self.setup_deepsort()

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

    def forward_for_tracking(self, vectors, attibute="A", time=1):

        if(attibute=="P"):

            vectors_pose         = vectors[0]
            vectors_data         = vectors[1]
            vectors_time         = vectors[2]

            en_pose              = torch.from_numpy(vectors_pose)
            en_data              = torch.from_numpy(vectors_data)
            en_time              = torch.from_numpy(vectors_time)
            
            if(len(en_pose.shape)!=3):
                en_pose          = en_pose.unsqueeze(0) # (BS, 7, pose_dim)
                en_time          = en_time.unsqueeze(0) # (BS, 7)
                en_data          = en_data.unsqueeze(0) # (BS, 7, 6)
            
            with torch.no_grad():
                pose_pred = self.pose_predictor.predict_next(en_pose, en_data, en_time, time)
            
            return pose_pred.cpu()


        if(attibute=="L"):
            vectors_loca         = vectors[0]
            vectors_time         = vectors[1]
            vectors_conf         = vectors[2]

            en_loca              = torch.from_numpy(vectors_loca)
            en_time              = torch.from_numpy(vectors_time)
            en_conf              = torch.from_numpy(vectors_conf)
            time                 = torch.from_numpy(time)

            if(len(en_loca.shape)!=3):
                en_loca          = en_loca.unsqueeze(0)             
                en_time          = en_time.unsqueeze(0)             
            else:
                en_loca          = en_loca.permute(0, 1, 2)         

            BS = en_loca.size(0)
            t_ = en_loca.size(1)

            en_loca_xy           = en_loca[:, :, :90]
            en_loca_xy           = en_loca_xy.view(BS, t_, 45, 2)
            en_loca_n            = en_loca[:, :, 90:]
            en_loca_n            = en_loca_n.view(BS, t_, 3, 3)

            new_en_loca_n = []
            for bs in range(BS):
                x0_                  = np.array(en_loca_xy[bs, :, 44, 0])
                y0_                  = np.array(en_loca_xy[bs, :, 44, 1])
                n_                   = np.log(np.array(en_loca_n[bs, :, 0, 2]))
                t_                   = np.array(en_time[bs, :])

                loc_                 = torch.diff(en_time[bs, :], dim=0)!=0
                if(self.cfg.phalp.distance_type=="EQ_020" or self.cfg.phalp.distance_type=="EQ_021"):
                    loc_                 = 1
                else:
                    loc_                 = loc_.shape[0] - torch.sum(loc_)+1

                M = t_[:, np.newaxis]**[0, 1]
                time_ = 48 if time[bs]>48 else time[bs]

                clf = Ridge(alpha=5.0)
                clf.fit(M, n_)
                n_p = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                n_p = n_p[0]
                n_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                n_pi  = get_prediction_interval(n_, n_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=1.2)
                clf.fit(M, x0_)
                x_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                x_p  = x_p[0]
                x_p_ = (x_p-0.5)*np.exp(n_p)/5000.0*256.0
                x_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                x_pi  = get_prediction_interval(x0_, x_hat, t_, time_+1+t_[-1])

                clf  = Ridge(alpha=2.0)
                clf.fit(M, y0_)
                y_p  = clf.predict(np.array([1, time_+1+t_[-1]]).reshape(1, -1))
                y_p  = y_p[0]
                y_p_ = (y_p-0.5)*np.exp(n_p)/5000.0*256.0
                y_hat = clf.predict(np.hstack((np.ones((t_.size, 1)), t_.reshape((-1, 1)))))
                y_pi  = get_prediction_interval(y0_, y_hat, t_, time_+1+t_[-1])
                
                new_en_loca_n.append([x_p_, y_p_, np.exp(n_p), x_pi/loc_, y_pi/loc_, np.exp(n_pi)/loc_, 1, 1, 0])
                en_loca_xy[bs, -1, 44, 0] = x_p
                en_loca_xy[bs, -1, 44, 1] = y_p
                
            new_en_loca_n        = torch.from_numpy(np.array(new_en_loca_n))
            xt                   = torch.cat((en_loca_xy[:, -1, :, :].view(BS, 90), (new_en_loca_n.float()).view(BS, 9)), 1)

        return xt

    def get_uv_distance(self, t_uv, d_uv):
        t_uv         = torch.from_numpy(t_uv).cuda().float()
        d_uv         = torch.from_numpy(d_uv).cuda().float()
        d_mask       = d_uv[3:, :, :]>0.5
        t_mask       = t_uv[3:, :, :]>0.5

        mask_dt      = torch.logical_and(d_mask, t_mask)
        mask_dt      = mask_dt.repeat(4, 1, 1)
        mask_        = torch.logical_not(mask_dt)

        t_uv[mask_]  = 0.0
        d_uv[mask_]  = 0.0

        with torch.no_grad():
            t_emb    = self.HMAR.autoencoder_hmar(t_uv.unsqueeze(0), en=True)
            d_emb    = self.HMAR.autoencoder_hmar(d_uv.unsqueeze(0), en=True)
        t_emb        = t_emb.view(-1)/10**3
        d_emb        = d_emb.view(-1)/10**3
        return t_emb.cpu().numpy(), d_emb.cpu().numpy(), torch.sum(mask_dt).cpu().numpy()/4/256/256/2

    def get_pose_distance(self, track_pose, detect_pose):
        """Compute pair-wise squared l2 distances between points in `track_pose` and `detect_pose`.""" 
        track_pose, detect_pose = np.asarray(track_pose), np.asarray(detect_pose)

        if(self.cfg.phalp.pose_distance=="smpl"):
            # remove additional dimension used for encoding location (last 3 elements)
            track_pose = track_pose[:, :-3]
            detect_pose = detect_pose[:, :-3]

        if len(track_pose) == 0 or len(detect_pose) == 0:
            return np.zeros((len(track_pose), len(detect_pose)))
        track_pose2, detect_pose2 = np.square(track_pose).sum(axis=1), np.square(detect_pose).sum(axis=1)
        r2 = -2. * np.dot(track_pose, detect_pose.T) + track_pose2[:, None] + detect_pose2[None, :]
        r2 = np.clip(r2, 0., float(np.inf))

        return r2

    def get_list_of_shots(self, list_of_frames):
        # https://github.com/Breakthrough/PySceneDetect
        list_of_shots    = []
        remove_tmp_video = False
        if(self.cfg.detect_shots):
            if(isinstance(list_of_frames[0], str)):
                # make a video if list_of_frames is frames
                video_tmp_name   = self.cfg.video.output_dir + "/_TMP/" + str(self.cfg.video_seq) + ".mp4"
                for ft_, fname_ in enumerate(list_of_frames):
                    im_ = cv2.imread(fname_)
                    if(ft_==0): 
                        video_file = cv2.VideoWriter(video_tmp_name, cv2.VideoWriter_fourcc(*'mp4v'), 24, frameSize=(im_.shape[1], im_.shape[0]))
                    video_file.write(im_)
                video_file.release()
                remove_tmp_video = True
            elif(isinstance(list_of_frames[0], tuple)):
                video_tmp_name = list_of_frames[0][0]
            else:
                raise Exception("Unknown type of list_of_frames")
            
            # Detect scenes in a video using PySceneDetect.
            scene_list = detect(video_tmp_name, AdaptiveDetector())

            if(remove_tmp_video):
                os.system("rm " + video_tmp_name)

            for scene in scene_list:
                list_of_shots.append(scene[0].get_frames())
                list_of_shots.append(scene[1].get_frames())
            list_of_shots = np.unique(list_of_shots)
            list_of_shots = list_of_shots[1:-1]
            log.info("Detected shot change at frame"+ "s" * min(0,len(list_of_shots)-1) + ": " + ", ".join(map(str, list_of_shots)))

            return list_of_shots

    def cached_download_from_drive(self, additional_urls=None):
        """Download a file from Google Drive if it doesn't exist yet.
        :param url: the URL of the file to download
        :param path: the path to save the file to
        """

        os.makedirs(os.path.join(CACHE_DIR, "phalp"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/3D"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/weights"), exist_ok=True)
        os.makedirs(os.path.join(CACHE_DIR, "phalp/ava"), exist_ok=True)

        smpl_path = os.path.join(CACHE_DIR, "phalp/3D/models/smpl/SMPL_NEUTRAL.pkl")

        if not os.path.exists(smpl_path):
            # We are downloading the SMPL model here for convenience. Please accept the license
            # agreement on the SMPL website: https://smpl.is.tue.mpg.
            os.makedirs(os.path.join(CACHE_DIR, "phalp/3D/models/smpl"), exist_ok=True)
            os.system('wget https://github.com/classner/up/raw/master/models/3D/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')

            convert_pkl('basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('rm basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
            os.system('mv basicModel_neutral_lbs_10_207_0_v1.0.0_p3.pkl ' + smpl_path)

        additional_urls = additional_urls if additional_urls is not None else {}
        download_files = {
            "head_faces.npy"           : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/head_faces.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "mean_std.npy"             : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/mean_std.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "smpl_mean_params.npz"     : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/smpl_mean_params.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "SMPL_to_J19.pkl"          : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/SMPL_to_J19.pkl", os.path.join(CACHE_DIR, "phalp/3D")],
            "texture.npz"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/3D/texture.npz", os.path.join(CACHE_DIR, "phalp/3D")],
            "bmap_256.npy"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/bmap_256.npy", os.path.join(CACHE_DIR, "phalp/3D")],
            "fmap_256.npy"              : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/fmap_256.npy", os.path.join(CACHE_DIR, "phalp/3D")],

            "hmar_v2_weights.pth"      : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/hmar_v2_weights.pth", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.pth"       : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/pose_predictor_40006.ckpt", os.path.join(CACHE_DIR, "phalp/weights")],
            "pose_predictor.yaml"      : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/weights/config_40006.yaml", os.path.join(CACHE_DIR, "phalp/weights")],
            
            # data for ava dataset
            "ava_labels.pkl"           : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_labels.pkl", os.path.join(CACHE_DIR, "phalp/ava")],
            "ava_class_mapping.pkl"   : ["https://people.eecs.berkeley.edu/~jathushan/projects/phalp/ava/ava_class_mappping.pkl", os.path.join(CACHE_DIR, "phalp/ava")],

        } | additional_urls # type: ignore

        for file_name, url in download_files.items():
            if not os.path.exists(os.path.join(url[1], file_name)):
                print("Downloading file: " + file_name)
                # output = gdown.cached_download(url[0], os.path.join(url[1], file_name), fuzzy=True)
                output = cache_url(url[0], os.path.join(url[1], file_name))
                assert os.path.exists(os.path.join(url[1], file_name)), f"{output} does not exist"
