import numpy as np
import torch

class Trainer():
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        n_epochs:int=10,
        device:str='cpu',
        save_path:str='./resources/best_model.pth'
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.n_epochs = n_epochs
        self.save_path = save_path
        self.device = device

        self.best_loss = 10 ** 9

        self._init_criterion()
        self._init_optimizer()

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=0.5,
            patience=1,
            threshold_mode="abs",
            mode="min",
            verbose=True
        )

    def fit(self):

        for epoch in range(self.n_epochs):
            train_loss = self._train_one_epoch()
            valid_loss = self._valid_one_epoch()
            print(f"Epoch: {epoch}, Train loss: {train_loss}, Valid loss: {valid_loss}")

            if valid_loss < self.best_loss:
                self.best_loss = valid_loss
                self.save(self.save_path)

            self.scheduler.step(metrics=self.best_loss)

    def _train_one_epoch(self) -> float:
        self.model.train()
        train_loss = MeterLoss()
        for step, (data, _, ecc) in enumerate(self.train_dataloader):
            data = data.float().to(self.device)
            ecc = ecc.float().to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(data, ecc)

            loss = self.criterion(outputs, data)
            loss.backward()

            self.optimizer.step()

            train_loss.update(loss)

        return train_loss.avg

    def _valid_one_epoch(self) -> float:
        self.model.eval()
        valid_loss = MeterLoss()
        for step, (data, _, ecc) in enumerate(self.valid_dataloader):

            with torch.no_grad():
                data = data.float().to(self.device)
                ecc = ecc.float().to(self.device)

                outputs = self.model(data, ecc)

                loss = self.criterion(outputs, data)

                valid_loss.update(loss)

        return valid_loss.avg

    def _init_criterion(self) -> None:
        self.criterion = torch.nn.MSELoss()

    def _init_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def save(self, path: str) -> None:
        self.model.eval()
        torch.save(
            {
                "model_state_dict": self.model.state_dict()
            },
            path,
        )
        print("New best model saved!")


class MeterLoss:
    def __init__(self):
        self.avg = 0
        self.n = 0

    def update(self, val):
        self.n += 1
        # incremental update
        self.avg = val / self.n + (self.n - 1) / self.n * self.avg