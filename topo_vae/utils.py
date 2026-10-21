#!/usr/bin/python3
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
import numpy as np
from sklearn.exceptions import NotFittedError


class TopoDataset(Dataset):
    def __init__(self, data, labels=None, flatten=True, scaler=None, **kwargs):
        if labels is not None:
            assert len(labels) == len(
                data
            ), "The length of labels and data are not equal"
            self.labels = labels
        if flatten:
            self.data = to_torch_tensor(data).flatten(start_dim=1).numpy()
        else:
            self.data = data
        if scaler is not None:
            try:
                self.data = scaler.transform(self.data)
            except NotFittedError:
                self.data = scaler.fit_transform(self.data)
        self.scaler = scaler

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if hasattr(self, "labels"):
            label = self.labels[index]
        else:
            label = 0
        if hasattr(self, "data_dist"):
            return index, self.data[index], label, self.data_dist[index]
        else:
            return index, self.data[index], label


def get_latent_representations(model, data_loader):
    labels = []
    data = []
    model.eval()
    model.to("cpu")
    with torch.no_grad():
        for x, _, y in data_loader:
            #             if x.device != model.device:
            #                 x.to(model.device)
            labels.append(y.numpy())
            mu, log_var = model.encode(x)
            z = model.reparameterize(mu, log_var)
            data.append(z.cpu().numpy())
    return np.concatenate(data, axis=0), np.concatenate(labels, axis=0)


class TopoScaler:
    def __init__(self, cat_col_idx, num_col_idx):
        self.is_fitted = False
        self.min_max_scaler = StandardScaler()
        self.cat_col_idx = cat_col_idx
        self.num_col_idx = num_col_idx
        self.cat_min_ = np.zeros_like(self.cat_col_idx)

    def fit(self, data):
        if len(self.num_col_idx) > 0:
            self.min_max_scaler.fit(data[:, self.num_col_idx])
        if len(self.cat_col_idx) > 0:
            self.cat_min_ = np.min(data[:, self.cat_col_idx], axis=0)
        self.is_fitted = True

    def transform(self, data):
        data_new = np.zeros_like(data)
        if len(self.num_col_idx) > 0:
            data_new[:, self.num_col_idx] = self.min_max_scaler.transform(
                data[:, self.num_col_idx]
            )
        if len(self.cat_col_idx) > 0:
            data_new[:, self.cat_col_idx] = data[:, self.cat_col_idx] - self.cat_min_
        return data_new

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data):
        data_new = np.zeros_like(data)
        if len(self.num_col_idx) > 0:
            data_new[:, self.num_col_idx] = self.min_max_scaler.inverse_transform(
                data[:, self.num_col_idx]
            )
        if len(self.cat_col_idx) > 0:
            data_new[:, self.cat_col_idx] = data[:, self.cat_col_idx] + self.cat_min_
        return data_new


def collate_with_matrix(samples):
    indicies, data, labels = zip(*samples)
    data, labels = to_torch_tensor(np.asarray(data)), to_torch_tensor(
        np.asarray(labels)
    )
    if len(data.shape) > 2:
        dist_data = torch.flatten(data, start_dim=1)
    else:
        dist_data = data
    x_dist = torch.cdist(dist_data, dist_data, p=2) / np.sqrt(dist_data.shape[1])
    return data, x_dist, labels


def to_torch_tensor(np_array) -> torch.Tensor:
    if torch.is_tensor(np_array):
        return np_array
    return torch.from_numpy(np_array).to(torch.float32)
