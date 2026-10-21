import torch.nn as nn
from tqdm import tqdm
import numpy as np
import torch
from torchmetrics import F1Score, Accuracy
from downstream_models.downstream_model import DownstreamModel


class WeakMLPDownstream(nn.Module, DownstreamModel):
    def __init__(self, in_features, out_features=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, out_features),
        )

    def forward(self, x):
        return self.model(x)

    def train_epoch(
        self, epoch_idx, dataloader, optimizer, device="cpu", cat_label=None
    ):
        loop = tqdm(dataloader)
        loop.set_description(f"Training baseline downstream model epoch {epoch_idx}")

        loss_fn = self.get_loss_fn()

        for i, data in enumerate(loop):
            inputs, target = data
            inputs = inputs.float().to(device)

            if self.categorical_target:
                target = target.long().to(device)
                target = target.view(-1)
            else:
                target = target.float().to(device)

            outputs = self(inputs)

            loss = loss_fn(outputs, target)
            loop.set_postfix({"Loss": loss.item()})

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    def eval_model(self, dataloader, device="cpu", cat_label=None):
        loop = tqdm(dataloader)
        loop.set_description(f"Evaluation baseline downstream model")

        mean_loss = []
        loss_fn = self.get_loss_fn()

        self.train(False)
        o = None
        t = None
        for i, data in enumerate(loop):
            inputs, target = data
            inputs = inputs.float().to(device)

            if cat_label:
                target = target.long().to(device)
                target = target.view(-1)
            else:
                target = target.float().to(device)

            with torch.no_grad():
                outputs = self(inputs)
                loss = loss_fn(outputs, target)
                if cat_label:
                    if o is None:
                        o = outputs.to("cpu").numpy().tolist()
                        t = target.to("cpu").numpy().tolist()
                    else:
                        o.extend(outputs.to("cpu").numpy().tolist())
                        t.extend(target.to("cpu").numpy().tolist())

            mean_loss.append(loss.to("cpu").item())

            loop.set_postfix({"Loss": np.array(mean_loss).mean()})

        self.train(True)
        if cat_label:
            _, aa = torch.max(torch.tensor(o), dim=1)
            f1 = F1Score(
                task="multiclass", num_classes=self.out_features, average="macro"
            )
            acc = Accuracy(
                task="multiclass", num_classes=self.out_features, average="macro"
            )
            f1_score = f1(aa, torch.tensor(t)).detach().numpy()
            acc_score = acc(aa, torch.tensor(t)).detach().numpy()
            return np.array(mean_loss), [acc_score, f1_score]

        return np.array(mean_loss).mean(), None

    def get_loss_fn(self):
        return nn.CrossEntropyLoss() if self.categorical_target else RMSELoss()


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.eps = eps

    def forward(self, input_val, target_val):
        loss = torch.sqrt(self.mse_loss(input_val, target_val) + self.eps)
        return loss
