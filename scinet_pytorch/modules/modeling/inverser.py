import numpy as np
import torch

class Inverser():
    def __init__(
        self,
        model,
        inference_dataloader,
        n_epochs:int=10,
        device:str='cpu',
        model_path:str='./resources/best_model.pth'
    ):
        self.model = model
        self.inference_dataloader = inference_dataloader
        self.inference_dataloader.dataset.ecc = torch.full_like(
            self.inference_dataloader.dataset.ecc,
            0.0
        )
        self.inference_dataloader.dataset.ecc.requires_grad_()
        self.n_epochs = n_epochs
        self.model_path = model_path
        self.device = device

        self._init_criterion()
        self._init_optimizer()

        self.load(self.model_path)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.8,
            patience=1,
            threshold_mode="abs",
            mode="min",
            verbose=True
        )

    def inverse(self):

        for epoch in range(self.n_epochs):
            inv_loss = self._inverse_one_epoch()
            print(f"Epoch: {epoch}, Inversion loss: {inv_loss}")
            ecc = torch.mean(self.inference_dataloader.dataset.ecc)
            print(f"Eccentricity: {ecc}")

            self.scheduler.step(metrics=inv_loss)

    def _inverse_one_epoch(self) -> float:
        self.model.eval()
        inference_loss = MeterLoss()
        general_loss = 0

        self.optimizer.zero_grad()
        
        for step, (data, _, ecc) in enumerate(self.inference_dataloader):
            data = data.float().to(self.device)
            ecc = ecc.float().to(self.device)

            outputs = self.model(data, ecc)

            loss = self.criterion(outputs, data)
            general_loss += loss

            inference_loss.update(loss)
        
        general_loss.backward()

        self.inference_dataloader.dataset.ecc.grad = torch.full_like(
            self.inference_dataloader.dataset.ecc.grad,
            torch.mean(self.inference_dataloader.dataset.ecc.grad)
        )

        self.optimizer.step()

        return inference_loss.avg

    def _init_criterion(self) -> None:
        self.criterion = torch.nn.MSELoss()

    def _init_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            [self.inference_dataloader.dataset.ecc],
            lr=0.001
        )
    
    def load(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path)['model_state_dict']
        )


class MeterLoss:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg