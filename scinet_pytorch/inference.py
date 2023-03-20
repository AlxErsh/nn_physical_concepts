import numpy as np

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from scinet_pytorch.modules.models import Network
from scinet_pytorch.modules.dataset import EllipticDataset
from scinet_pytorch.modules.modeling import Inverser

def run_inference() -> None:
    cfg = OmegaConf.load("./scinet_pytorch/config.yaml")
    model = Network(cfg.inference.model, cfg.inference.data.time_series_length, device=cfg.device)
    model.to(cfg.device)
    inference_dataset = EllipticDataset(cfg.inference.data.inference_file)
    inference_dataloader = DataLoader(inference_dataset, batch_size=cfg.train.batch_size)
    inverser = Inverser(model,
                        inference_dataloader,
                        n_epochs=cfg.inference.n_epochs,
                        device=cfg.device,
                        model_path=cfg.inference.model.weights_path)
    inverser.inverse()

def main() -> None:
    run_inference()

if __name__ == "__main__":
    main()