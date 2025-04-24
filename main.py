import argparse
import yaml
from engines.engine import Engine
from engines.funcs.video_stream_funcs import find_input_video, get_output_video_path

def load_config(config_path):
    """Load configuration from YAML."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return argparse.Namespace(**config)

def main(args):
    cfg = load_config(args.config)
    if cfg.sat_cfg['use_sat']:
        patch_size = 14
        if cfg.input_size % (patch_size * 4) != 0:
            print(f"Warning: input_size ({cfg.input_size}) is not divisible by {patch_size * 4}.")

    input_video = find_input_video(cfg.input_dir)
    output_video = get_output_video_path(input_video, cfg.output_dir)

    engine = Engine(cfg, mode='infer', gpu_id=args.gpu_id)
    engine.infer_video(input_video, output_video, cfg.input_size, cfg.conf_thresh, cfg.display)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAT-HMR Video Inference")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use (default: 0)")
    args = parser.parse_args()
    main(args)