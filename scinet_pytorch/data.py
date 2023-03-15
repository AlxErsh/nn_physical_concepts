from omegaconf import OmegaConf

import numpy as np

from scinet_pytorch.modules.dataset import earth_elliptic_data

def generate_train_data(cfg) -> None:
    eccentricities = np.linspace(0, 0.03, 20)
    assert cfg.data.planet in ["earth", "mars"], "Planet is unknown!"
    if cfg.data.planet == "earth":
        earth_elliptic_data(
            eccentricities,
            series_length=cfg.data.time_series_length,
            N=int(cfg.data.N*cfg.data.train_freq),
            delta_t=cfg.data.delta_t,
            file_name=cfg.data.train_file
        )
        earth_elliptic_data(
            eccentricities,
            series_length=cfg.data.time_series_length,
            N=int(cfg.data.N*(1-cfg.data.train_freq)),
            delta_t=cfg.data.delta_t,
            file_name=cfg.data.valid_file
        )

def generate_inference_data(cfg) -> None:
    assert cfg.data.planet in ["earth", "mars"], "Planet is unknown!"
    if cfg.data.planet == "earth":
        eccentricities = [0.01671022]
        earth_elliptic_data(
            eccentricities,
            series_length=cfg.data.time_series_length,
            N=cfg.data.N,
            delta_t=cfg.data.delta_t,
            file_name=cfg.data.inference_file
        )

def main() -> None:
    cfg = OmegaConf.load("./scinet_pytorch/config.yaml")
    generate_train_data(cfg)
    generate_inference_data(cfg)

if __name__ == "__main__":
    main()