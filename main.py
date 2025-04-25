import hydra
from omegaconf import DictConfig, OmegaConf
from engines.engine import Engine
from engines.funcs.video_stream_funcs import find_input_video, get_output_video_path
from configs.base import FullConfig

# Register configuration with Hydra
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=FullConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig):
    # Validate input size for SAT-HMR
    if cfg.sathmr.sat_cfg.use_sat:
        patch_size = 14
        if cfg.sathmr.input_size % (patch_size * 4) != 0:
            print(f"Warning: input_size ({cfg.sathmr.input_size}) is not divisible by {patch_size * 4}.")

    # Get input and output video paths
    input_video = find_input_video(cfg.video.source)
    output_video = get_output_video_path(input_video, cfg.video.output_dir)

    # Initialize and run the engine with SAT-HMR configuration
    engine = Engine(cfg, mode=cfg.sathmr.mode, gpu_id=cfg.sathmr.gpu_id)
    engine.infer_video(input_video, output_video, cfg.sathmr.input_size, cfg.sathmr.conf_thresh, cfg.sathmr.display)

if __name__ == "__main__":
    main()