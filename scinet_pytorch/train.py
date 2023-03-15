import numpy as np

from omegaconf import OmegaConf

from torch.utils.data import DataLoader

from scinet_pytorch.modules.models import Network
from scinet_pytorch.modules.dataset import EllipticDataset
from scinet_pytorch.modules.modeling import Trainer

def main() -> None:
    cfg = OmegaConf.load("./scinet_pytorch/config.yaml")
    model = Network(cfg.train.model, cfg.data.time_series_length, device=cfg.device)
    model.to(cfg.device)
    train_dataset = EllipticDataset(cfg.data.train_file)
    val_dataset = EllipticDataset(cfg.data.valid_file)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size)
    trainer = Trainer(model, train_dataloader, val_dataloader,
                      n_epochs=cfg.train.n_epochs, device=cfg.device,
                      save_path=cfg.train.model.weights_path)
    trainer.fit()

if __name__ == "__main__":
    main()