# export_torchscript.py
import os
import torch
import hydra
from omegaconf import DictConfig
from models.sat_model import build_sat_model
from engines.funcs.video_stream_funcs import create_empty_targets
from configs.base import FullConfig
from hydra.core.config_store import ConfigStore

# Enregistrer la config Hydra
cs = ConfigStore.instance()
cs.store(name="config", node=FullConfig)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig):
    scfg = cfg.sathmr
    print("[TS Export] Building SAT-HMR model...")
    model, _ = build_sat_model(scfg, set_criterion=False)
    if scfg.pretrain:
        ckpt = torch.load(scfg.pretrain_path, map_location='cpu')
        sd = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(sd, strict=False)
    model.eval()

    # 1) Wrapper pour scripting
    class SatHmrScriptWrapper(torch.nn.Module):
        def __init__(self, sat_model, input_size: int):
            super().__init__()
            self.model = sat_model
            self.input_size = input_size

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # recréer les targets vides comme dans Engine
            targets = create_empty_targets(x.device, [self.input_size, self.input_size])
            out = self.model(x, targets)
            # par exemple on ne renvoie que les verts et confs
            return out['pred_verts'], out['pred_confs'], out['pred_intrinsics']

    wrapper = SatHmrScriptWrapper(model, input_size=scfg.input_size)

    # 2) Scripting (pas de trace !)
    print(f"[TS Export] Scripting wrapper...")
    scripted = torch.jit.script(wrapper)

    # 3) Sauvegarde
    ts_path = scfg.torchscript_path
    os.makedirs(os.path.dirname(ts_path), exist_ok=True)
    scripted.save(ts_path)
    print(f"[TS Export] Saved TorchScript model to {ts_path}")

if __name__ == "__main__":
    main()
