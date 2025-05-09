import hydra
from omegaconf import DictConfig
from engines.engine_mirokai import Engine
from pymirokai.utils.run_until_interruption import run_until_interruption
from configs.mirokai_config import FullConfig

# Enregistrement du schéma de config Hydra
cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name="config", node=FullConfig)

async def async_main(cfg: DictConfig):
    engine = Engine(cfg, mode=cfg.sathmr.mode, gpu_id=cfg.sathmr.gpu_id)
    await engine.infer(cfg.ip, cfg.api_key, cfg.sathmr.input_size)

@hydra.main(version_base="1.2", config_name="config")
def main(cfg: DictConfig) -> None:
    run_until_interruption(lambda: async_main(cfg))

if __name__ == "__main__":
    main()
