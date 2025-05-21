from dataclasses import dataclass, field
import os
from typing import Dict, Optional

import hydra
from omegaconf import MISSING

WEIGHTS_DIR = "/home/alphafalcon/tracker/code/SAT-TRACK/weights"
print("weights_dir", WEIGHTS_DIR)

@dataclass
class VideoConfig:
    source: str = '/home/alphafalcon/Videos/septime/input'
    output_dir: str = '/home/alphafalcon/Videos/septime/output'
    extract_video: bool = True
    base_path: Optional[str] = None
    start_frame: int = -1
    end_frame: int = 1300
    useffmpeg: bool = False
    start_time: str = '0s'
    end_time: str = '10s'

@dataclass
class PHALPConfig:
    predict: str = 'APL'
    pose_distance: str = 'smpl'
    distance_type: str = 'EQ_019'
    alpha: float = 0.1
    low_th_c: float = 0.2
    hungarian_th: float = 100.0
    track_history: int = 7
    max_age_track: int = 50
    n_init: int = 5
    encode_type: str = '4c'
    past_lookback: int = 1
    detector: str = 'vitdet'
    shot: int = 0
    start_frame: int = -1
    end_frame: int = 10
    small_w: int = 50
    small_h: int = 100

@dataclass
class PosePredictorConfig:
    config_path: str = f"{WEIGHTS_DIR}/phalp/weights/pose_predictor.yaml"
    weights_path: str = f"{WEIGHTS_DIR}/phalp/weights/pose_predictor.pth"
    mean_std: str = f"{WEIGHTS_DIR}/smpl_data/smpl/mean_std.npy"

@dataclass
class AVAConfig:
    ava_labels_path: str = f"{WEIGHTS_DIR}/phalp/ava/ava_labels.pkl"
    ava_class_mappping_path: str = f"{WEIGHTS_DIR}/phalp/ava/ava_class_mapping.pkl"

@dataclass
class HMRConfig:
    hmar_path: str = f"{WEIGHTS_DIR}/phalp/weights/hmar_v2_weights.pth"

@dataclass
class RenderConfig:
    enable: bool = True
    type: str = 'HUMAN_MESH'
    up_scale: int = 2
    res: int = 256
    side_view_each: bool = False
    metallicfactor: float = 0.0
    roughnessfactor: float = 0.7
    colors: str = "phalp"
    head_mask: bool = False
    head_mask_path: str = f"{WEIGHTS_DIR}/smpl_data/smpl/head_faces.npy"
    output_resolution: int = 1440
    fps: int = 30
    blur_faces: bool = False
    show_keypoints: bool = False

@dataclass
class PostProcessConfig:
    apply_smoothing: bool = True
    phalp_pkl_path: str = '_OUT/videos_v0'
    save_fast_tracks: bool = False
    
@dataclass
class SMPLConfig:
    MODEL_PATH: str = f"{WEIGHTS_DIR}/smpl_data/smpl/"
    GENDER: str = 'neutral'
    MODEL_TYPE: str = 'smpl'
    NUM_BODY_JOINTS: int = 23
    JOINT_REGRESSOR_EXTRA: str = f"{WEIGHTS_DIR}/smpl_data/smpl/SMPL_to_J19.pkl"
    TEXTURE: str = f"{WEIGHTS_DIR}/smpl_data/smpl/texture.npz"

@dataclass
class SMPLHeadConfig:
    TYPE: str = 'basic'
    POOL: str = 'max'
    SMPL_MEAN_PARAMS: str = f"{WEIGHTS_DIR}/smpl_data/smpl/smpl_mean_params.npz"
    IN_CHANNELS: int = 2048

@dataclass
class BackboneConfig:
    TYPE: str = 'resnet'
    NUM_LAYERS: int = 50
    MASK_TYPE: str = 'feat'

@dataclass
class TransformerConfig:
    HEADS: int = 1
    LAYERS: int = 1
    BOX_FEATS: int = 6

@dataclass
class ModelConfig:
    IMAGE_SIZE: int = 256
    SMPL_HEAD: SMPLHeadConfig = field(default_factory=SMPLHeadConfig)
    BACKBONE: BackboneConfig = field(default_factory=BackboneConfig)
    TRANSFORMER: TransformerConfig = field(default_factory=TransformerConfig)
    pose_transformer_size: int = 2048

@dataclass
class ExtraConfig:
    FOCAL_LENGTH: int = 5000

@dataclass
class SATHMRConfig:
    pretrain: bool = True
    pretrain_path: str = f"{WEIGHTS_DIR}/sat_hmr/sat_644.pth"
    infer_batch_size: int = 1
    infer_num_workers: int = 8
    distributed_infer: bool = True
    conf_thresh: float = 0.2
    display: bool = False
    live_stream: bool = True
    use_fp16: bool = False
    render_mode: str = 'points'
    input_size: int = 1288
    encoder: str = 'vitb'
    hidden_dim: int = 768
    nheads: int = 4
    dec_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.0
    num_queries: int = 50
    transformer_activation: str = "relu"
    sat_cfg: Dict = field(default_factory=lambda: {
        'use_sat': True,
        'share_patch_embed': False,
        'preprocess_pos_embed': False,
        'num_lvls': 3,
        'lvl_embed': True,
        'get_map_layer': 3,
        'use_additional_blocks': True,
        'conf_thresh': 0.3,
        'scale_thresh': 0.5
    })
    dn_cfg: Dict = field(default_factory=lambda: {
        'use_dn': True,
        'dn_number': 10,
        'tgt_embed_type': "params",
        'box_noise_scale': 0.4,
        'tgt_noise_scale': 0.2
    })
    mode: str = "infer"
    gpu_id: int = 0

@dataclass
class FullConfig:
    seed: int = 42
    track_dataset: str = "demo"
    device: str = "cuda"
    base_tracker: str = "PHALP"
    train: bool = False
    debug: bool = False
    use_gt: bool = False
    overwrite: bool = True
    task_id: int = -1
    num_tasks: int = 100
    verbose: bool = False
    detect_shots: bool = False
    video_seq: Optional[str] = None
    video: VideoConfig = field(default_factory=VideoConfig)
    phalp: PHALPConfig = field(default_factory=PHALPConfig)
    pose_predictor: PosePredictorConfig = field(default_factory=PosePredictorConfig)
    ava_config: AVAConfig = field(default_factory=AVAConfig)
    hmr: HMRConfig = field(default_factory=HMRConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    post_process: PostProcessConfig = field(default_factory=PostProcessConfig)
    SMPL: SMPLConfig = field(default_factory=SMPLConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    EXTRA: ExtraConfig = field(default_factory=ExtraConfig)
    sathmr: SATHMRConfig = field(default_factory=SATHMRConfig)
    hydra: Dict = field(default_factory=lambda: dict(
        mode=hydra.types.RunMode.RUN,
        run=dict(dir="${video.output_dir}"),
    ))