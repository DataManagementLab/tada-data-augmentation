from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

# from downstream_models.ft_transformer import FTTransformerWrapper
from downstream_models.mlp import MLPDownstream

# from downstream_models.xgboost import XGBDownstream
from downstream_models.weak_mlp import WeakMLPDownstream
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import config


def run_downstream(
    model_name,
    original_data,
    synthetic_data,
    validation_data,
    test_data=None,
    epochs=100,
    dataset_info=None,
    cat_label=False,
):
    if test_data is None:
        test_data = validation_data

    unique_vals, counts = np.unique(test_data["label"].values, return_counts=True)
    num_values = unique_vals.shape[0]
    mean_counts = int(np.mean(counts))

    test_data = (
        test_data.groupby("label", group_keys=False)[test_data.columns]
        .apply(
            lambda x: x.sample(n=mean_counts, random_state=123) if len(x) > mean_counts else x
        )
        .reset_index(drop=True)
    )
    test_data = test_data.sample(frac=1.0, random_state=123).reset_index(drop=True)

    scaler = MinMaxScaler()
    imputer = SimpleImputer(strategy="mean")

    training_dataset = DownstreamDataset(
        original_data,
        scaler=scaler,
        imputer=imputer,
        categorical_label=cat_label,
        use_fitted=False,
    )
    training_dataloader = DataLoader(training_dataset, batch_size=64)

    validation_dataset = DownstreamDataset(
        validation_data,
        scaler=scaler,
        imputer=imputer,
        use_fitted=True,  # False creates a new fitted scaler
        categorical_label=cat_label,
    )
    validation_dataloader = DataLoader(validation_dataset)

    test_dataset = DownstreamDataset(
        test_data,
        scaler=scaler,
        imputer=imputer,
        use_fitted=True,
        categorical_label=cat_label,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Training downstream model with original data")

    model, optimizer, is_torch = get_model_instance(
        model_name,
        in_features=original_data.shape[1] - 1,
        out_features=num_values if cat_label else 1,
        dataset_info=dataset_info,
    )

    if cat_label:
        cat_label_pass = scaler
    else:
        cat_label_pass = None

    if is_torch:
        for epoch in range(epochs):
            model.train_epoch(
                epoch,
                training_dataloader,
                optimizer,
                device=config.DOWNSTREAM_DEVICE,
                cat_label=cat_label_pass,
            )
    else:
        model.train_epoch(1, training_dataloader, validation_dataloader)
    original_data_loss, original_data_acc_f1 = model.eval_model(
        test_dataloader, device=config.DOWNSTREAM_DEVICE, cat_label=cat_label_pass
    )

    complete_data = pd.concat((original_data, synthetic_data), ignore_index=True)

    if cat_label:
        # synthetic_data = synthetic_data[synthetic_data["label"] >= 0]
        synthetic_data.loc[synthetic_data["label"] < 0, "label"] = 0
        # synthetic_data = synthetic_data[synthetic_data["label"] > 0.5]
        complete_data = pd.concat((original_data, synthetic_data), ignore_index=True)

        num_values_synth = np.unique(np.round(complete_data["label"].values)).shape[0]

        # --- Fix if categorical labels are missing in the middle ---
        num_values_synth = (
            np.max(np.unique(np.round(complete_data["label"].values))) + 1
        )
        # --- END ---

        num_values = int(max(num_values, num_values_synth))

    ####################################################################
    # complete_data["label"] = np.round(complete_data["label"].values)
    # unique_vals, counts = np.unique(complete_data["label"].values, return_counts=True)
    # min_val = int(np.mean(counts))

    # complete_data = (
    #    complete_data.groupby("label", as_index=False)
    #    .apply(
    #        lambda x: (
    #            x.sample(n=min_val, random_state=123) if len(x) > min_val else x
    #        )
    #    )
    #    .reset_index(drop=True)
    # )
    # complete_data = complete_data.sample(frac=1.0, random_state=123).reset_index(drop=True)
    ########################################################################

    training_dataset = DownstreamDataset(
        complete_data,
        scaler=scaler,
        imputer=imputer,
        use_fitted=False,
        categorical_label=cat_label,
    )
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)

    test_dataset = DownstreamDataset(
        test_data,
        scaler=scaler,
        imputer=imputer,
        use_fitted=True,
        categorical_label=cat_label,
    )

    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("Training downstream model with synthetic data")

    model, optimizer, is_torch = get_model_instance(
        model_name,
        in_features=original_data.shape[1] - 1,
        out_features=num_values if cat_label else 1,
        dataset_info=dataset_info,
    )

    if is_torch:
        for epoch in range(epochs):
            model.train_epoch(
                epoch,
                training_dataloader,
                optimizer,
                device=config.DOWNSTREAM_DEVICE,
                cat_label=cat_label_pass,
            )
    else:
        model.train_epoch(1, training_dataloader, validation_dataloader)

    synthetic_data_loss, synthetic_data_acc_f1 = model.eval_model(
        test_dataloader, device=config.DOWNSTREAM_DEVICE, cat_label=cat_label_pass
    )

    if cat_label:
        return original_data_acc_f1, synthetic_data_acc_f1

    return original_data_loss, synthetic_data_loss


def get_model_instance(
    model_name, in_features=None, out_features=None, dataset_info=None
):
    if model_name == "mlp":
        model = MLPDownstream(in_features, out_features=out_features).to(
            config.DOWNSTREAM_DEVICE
        )
        optimizer = Adam(model.parameters(), lr=0.001)
        return model, optimizer, True
    elif model_name == "xgboost":
        model = None  # XGBDownstream()
        return model, None, False
    elif model_name == "ft-transformer":
        model = None  # FTTransformerWrapper(dataset_info).to(config.DOWNSTREAM_DEVICE)
        optimizer = model.ft_transformer.make_default_optimizer()
        return model, optimizer, True
    elif model_name == "weak_mlp":
        model = WeakMLPDownstream(in_features, out_features).to(
            config.DOWNSTREAM_DEVICE
        )
        optimizer = Adam(model.parameters(), lr=0.001)
        return model, optimizer, True
    else:
        raise NotImplementedError


def get_data_cardinalities(dataset, cat_cols):
    cards = []
    for col in cat_cols:
        cards.append(np.max(dataset[col].values) - np.min(dataset[col].values))
    return cards


class DownstreamDataset(Dataset):
    def __init__(
        self,
        pandas_dataset: pd.DataFrame,
        label_col="label",
        scaler=None,
        imputer=None,
        use_fitted=False,
        categorical_label=False,
    ):
        self.label = np.array([pandas_dataset.loc[:, label_col].values]).T
        self.data = pandas_dataset.drop(columns=label_col).values

        if use_fitted:
            complete_data = scaler.transform(
                imputer.transform(np.concatenate((self.data, self.label), axis=1))
            )
        else:
            complete_data = scaler.fit_transform(
                imputer.fit_transform(np.concatenate((self.data, self.label), axis=1))
            )

        self.data = complete_data[:, :-1]

        if categorical_label:
            self.label = np.round(np.array(self.label, dtype=np.float32))
        else:
            self.label = complete_data[:, -1]

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], np.array([self.label[idx]])
