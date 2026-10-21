import json
from typing import Union, Dict, Any

from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import pytorch_lightning as pl
import matplotlib.pyplot as plt

from topo_vae.sampling import gaussian_sampling
from topo_vae.utils import *
from topo_vae.label_prediction import detect_label_column

from topo_vae.topo_vae_model import TopologicallyRegularizedVariationalAutoencoder

config: Dict[str, Any] = {
    "max_epochs": 600,  # number of training epochs
    "gpus": [0],  # ID of the used GPU
    "topo_every_n_batches": 1,  # indicates every i-th epoch the topo loss should be computed
    "topo_start_epoch": 0,  # epoch after which topo_vae loss is used for training as well
    "topo_weight": 3.0,  # topo_vae loss factor
    "hidden_dim": 400,  # hidden dimensionality of the TopoVAE
    "latent_dim": None,  # latent dimensionality of the TopoVAE if None -> latent_dim = num_cols/3
    "batch_size": 128,  # batch size used for training
    "lr": 1e-3,  # learning rate
    "balanced_sampling": True,  # true if sampling should be balanced with the label column
    "beta": 0.001,  # Beta used for kl-divergence weighting
    "delete_low_certainty_percantage": 0.6,  # delete samples which output probability is lower than set threshold
    "topo_stop": False,  # true if sampling should be stoped if data is balanced (not sampling till desired number of samples is reached)
}


class TopoVAE:
    def __init__(self, config_path: Union[str, None] = None) -> None:
        """
        TopoVAE model used for data augmentation.

        :param config_path: Path to configuration file. If none is given a standard configuration will be used.
        """
        self.topo_ae_model = None
        self.train_data = None
        self.train_loader = None
        self.categorical_columns = None
        self.column_names = None
        self.label_column_idx = None
        self.bin_label = None

        if config_path is not None:
            try:
                self.config = json.load(open(config_path, "r"))
            except FileNotFoundError:
                self.config = config.copy()
        else:
            self.config = config.copy()

    def train(
        self,
        train_data: pd.DataFrame,
        categorical_columns: Union[list[str], None] = None,
        one_hot_col_names: Union[list[str], None] = None,
        label_column_name: Union[str, None] = None,
    ) -> None:
        """
        Processes the complete training of the TopoAE model.

        :param train_data: The training data as a pandas Dataframe with shape (rows, cols).
        :param categorical_columns: List of names of categorical columns.
        :param one_hot_col_names: List of names of one hot columns.
        :param label_column_name: Name of label column.
        """
        if categorical_columns is None:
            categorical_columns = []
        self.categorical_columns = categorical_columns

        cols = train_data.columns
        self.column_names = cols

        org_data = train_data.dropna()

        if label_column_name is None:
            label_column_name = detect_label_column(
                org_data, self.categorical_columns
            ).values[0][0]

        if label_column_name in categorical_columns:
            self.bin_label = 0
        else:
            # n_bins = int(np.ceil(np.log2(len(decoded_augmented_data_for_max_class)) + 1))
            self.bin_label = 10

        self.label_column_idx = cols.tolist().index(label_column_name)

        org_labels = org_data[label_column_name].values
        org_data = org_data.values

        categorical_columns = [
            cols.tolist().index(name) for name in self.categorical_columns
        ]

        if one_hot_col_names is not None:
            if len(one_hot_col_names) > 0:
                one_hot_cols = []
                for l in one_hot_col_names:
                    one_hot_cols.append([cols.tolist().index(i) for i in l])
            else:
                one_hot_cols = None
        else:
            one_hot_cols = None

        self.train_data = org_data

        train_labels = org_labels

        num_samples_to_delete = 1 if self.train_data.shape[0] % 2 == 1 else 0

        random_indices = np.random.choice(
            self.train_data.shape[0], num_samples_to_delete, replace=False
        )

        self.train_data = np.delete(self.train_data, random_indices, axis=0)

        train_labels = np.delete(train_labels, random_indices, axis=0)

        self.train_data = to_torch_tensor(self.train_data)

        self.config["one_hot_cols"] = one_hot_cols
        if one_hot_cols is not None:
            one_hot_groups = [sorted(group) for group in one_hot_cols]

            representative_cols = [group[0] for group in one_hot_groups]

            cols_to_delete = sorted(
                col for group in one_hot_groups for col in group[1:]
            )

            for group in one_hot_groups:
                rep_col = group[0]
                self.train_data[:, rep_col] = np.argmax(
                    self.train_data[:, group], axis=1
                )

            self.train_data = np.delete(self.train_data, cols_to_delete, axis=1)

            old_to_new = {}
            new_idx = 0
            for old_idx in range(org_data.shape[1]):
                if old_idx in cols_to_delete:
                    continue
                old_to_new[old_idx] = new_idx
                new_idx += 1

            all_one_hot_members = {c for group in one_hot_groups for c in group}

            self.config["cat_cols"] = categorical_columns
            self.config["reg_cols"] = [
                i
                for i in range(org_data.shape[1])
                if i not in categorical_columns and i not in all_one_hot_members
            ]

            self.config["cat_cols"] = [
                old_to_new[i] for i in self.config["cat_cols"] if i in old_to_new
            ]
            self.config["reg_cols"] = [
                old_to_new[i] for i in self.config["reg_cols"] if i in old_to_new
            ]

            self.label_column_idx = old_to_new[self.label_column_idx]

            self.config["cat_cols"].extend(
                old_to_new[rep] for rep in representative_cols
            )
            self.config["cat_cols"] = sorted(set(self.config["cat_cols"]))

            categorical_columns_data = self.train_data[:, self.config["cat_cols"]]

            num_cat_vals_per_col = [
                int(np.max(categorical_columns_data[:, i].numpy())) + 1
                for i in range(categorical_columns_data.shape[1])
            ]
        else:
            self.config["cat_cols"] = categorical_columns
            self.config["reg_cols"] = [
                i for i in range(org_data.shape[1]) if i not in categorical_columns
            ]

            categorical_columns_data = org_data[:, self.config["cat_cols"]]

            num_cat_vals_per_col = [
                int(np.max(categorical_columns_data[:, i])) + 1
                for i in range(categorical_columns_data.shape[1])
            ]

        if self.config["latent_dim"] is None:
            self.config["latent_dim"] = (
                int(self.train_data.shape[1] / 3)
                if self.train_data.shape[1] >= 6
                else 2
            )

        self.config["num_cat_vals_per_col"] = num_cat_vals_per_col

        self.scaler = TopoScaler(
            cat_col_idx=self.config["cat_cols"], num_col_idx=self.config["reg_cols"]
        )

        train_dataset = TopoDataset(
            self.train_data,
            labels=train_labels,
            scaler=self.scaler,
            flatten=True,
        )

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            collate_fn=collate_with_matrix,
            shuffle=True,
            #            num_workers=6,
            #            persistent_workers=True,
            #            pin_memory=True,
        )

        #        torch.set_float32_matmul_precision('high')
        self.topo_ae_model = TopologicallyRegularizedVariationalAutoencoder(
            **self.config
        )
        #        self.topo_ae_model = torch.compile(self.topo_ae_model)

        trainer = pl.Trainer(
            accelerator="auto" if torch.cuda.is_available() else "cpu",
            devices=self.config["gpus"] if torch.cuda.is_available() else "auto",
            strategy="auto",
            max_epochs=self.config["max_epochs"],
            logger=False,
            enable_checkpointing=False,
        )
        trainer.fit(self.topo_ae_model, self.train_loader)

    def sample(
        self, sample_size: Union[float, int], plot_latent_space: bool = True
    ) -> pd.DataFrame:
        """
        Samples new data rows based on the data the TopoAE model was trained on.

        :param sample_size: Amount of rows to sample. Defined as a fraction of the original dataset size.
        :param plot_latent_space: True if latent space of original and augmented data should be plotted.
        :return: pd.Dataframe with new rows sampled and the same format as the original dataset.
        """
        decoder = self.topo_ae_model.decode

        self.train_loader.num_workers = 0

        latent, labels = get_latent_representations(
            self.topo_ae_model, self.train_loader
        )

        all_encoded_samples = None

        num_samples = (len(latent) / sample_size) - len(latent)
        if num_samples == 0:
            num_samples = 0.25 * len(latent)

        curr_size = 0
        base_points = latent.copy()

        sample_bar = tqdm(total=int(num_samples))
        sample_bar.set_description("Sampling data points: ")

        while curr_size < num_samples:
            encoded_samples = gaussian_sampling(base_points)

            ######################################################################
            ## Balanced Sampling
            if self.config["balanced_sampling"]:
                if all_encoded_samples is not None:
                    self.topo_ae_model.eval()
                    with torch.no_grad():
                        decoded_augmented_data_for_max_class = (
                            decoder(to_torch_tensor(all_encoded_samples))
                            .detach()
                            .numpy()
                        )
                    self.topo_ae_model.train()

                    decoded_augmented_data_for_max_class = (
                        self.scaler.inverse_transform(
                            decoded_augmented_data_for_max_class
                        )
                    )
                    decoded_augmented_data_for_max_class = np.concatenate(
                        (decoded_augmented_data_for_max_class, self.train_data.numpy()),
                        axis=0,
                    )
                else:
                    decoded_augmented_data_for_max_class = self.train_data.numpy()

                if self.bin_label > 0:
                    # equal-width binning
                    bin_edges = np.histogram_bin_edges(
                        decoded_augmented_data_for_max_class[:, self.label_column_idx],
                        bins=self.bin_label,
                    )
                    y_binned = pd.cut(
                        decoded_augmented_data_for_max_class[:, self.label_column_idx],
                        bins=bin_edges,
                        labels=False,
                        include_lowest=True,
                    )
                    labels_synth_max_class = y_binned
                else:
                    labels_synth_max_class = np.round(
                        decoded_augmented_data_for_max_class[:, self.label_column_idx]
                    )

                unique_labels_max_class, unique_labels_counts_max_class = np.unique(
                    np.round(labels_synth_max_class), return_counts=True
                )

                ind_ = np.argsort(unique_labels_counts_max_class)

                self.topo_ae_model.eval()
                with torch.no_grad():
                    if self.config["delete_low_certainty_percantage"] > 0:
                        with torch.no_grad():
                            decoded_augmented_data, probs = (
                                self.topo_ae_model.decode_sel(
                                    to_torch_tensor(encoded_samples)
                                )
                            )
                        decoded_augmented_data = decoded_augmented_data[
                            torch.max(probs, dim=1).values
                            > self.config["delete_low_certainty_percantage"]
                        ]
                        encoded_samples = encoded_samples[
                            torch.max(probs, dim=1).values
                            > self.config["delete_low_certainty_percantage"]
                        ]
                    else:
                        with torch.no_grad():
                            decoded_augmented_data = (
                                decoder(to_torch_tensor(encoded_samples))
                                .detach()
                                .numpy()
                            )
                self.topo_ae_model.train()

                decoded_augmented_data = self.scaler.inverse_transform(
                    decoded_augmented_data
                )

                if self.bin_label > 0:
                    # equal-width binning
                    bin_edges = np.histogram_bin_edges(
                        decoded_augmented_data[:, self.label_column_idx],
                        bins=self.bin_label,
                    )
                    y_binned = pd.cut(
                        decoded_augmented_data[:, self.label_column_idx],
                        bins=bin_edges,
                        labels=False,
                        include_lowest=True,
                    )
                    labels_synth = y_binned
                else:
                    labels_synth = np.round(
                        decoded_augmented_data[:, self.label_column_idx]
                    )

                unique_labels, unique_labels_counts = np.unique(
                    np.round(labels_synth), return_counts=True
                )

                num_class_not_sample = unique_labels_counts_max_class[ind_][-1]

                num_to_sample = num_class_not_sample - unique_labels_counts_max_class

                encoded_samples_new = np.concatenate(
                    (encoded_samples, labels_synth[:, None]), axis=1
                )
                encoded_samples_final = []

                if np.sum(num_to_sample) > 0:
                    for i in range(len(unique_labels)):
                        c = unique_labels[i]
                        cat_rows = encoded_samples_new[encoded_samples_new[:, -1] == c]
                        if i >= len(num_to_sample) or num_to_sample[i] <= 0:
                            continue
                        curr_num_to_sample = (
                            np.min(unique_labels_counts)
                            if np.min(unique_labels_counts) < num_to_sample[i]
                            else num_to_sample[i]
                        )
                        samples = cat_rows[
                            np.random.choice(
                                cat_rows.shape[0],
                                size=(
                                    curr_num_to_sample
                                    if curr_num_to_sample < len(cat_rows)
                                    else len(cat_rows)
                                ),
                                replace=False,
                            )
                        ]

                        encoded_samples_final.extend(samples[:, :-1])
                else:
                    if self.config["topo_stop"]:
                        break
                    min_to_sample = np.min(unique_labels_counts)
                    for i in range(len(unique_labels)):
                        c = unique_labels[i]

                        cat_rows = encoded_samples_new[encoded_samples_new[:, -1] == c]

                        samples = cat_rows[
                            np.random.choice(
                                cat_rows.shape[0], size=min_to_sample, replace=False
                            )
                        ]
                        encoded_samples_final.extend(samples[:, :-1])

                if len(encoded_samples_final) > 0 and self.config["balanced_sampling"]:
                    encoded_samples = np.array(encoded_samples_final)
            ########################################################################

            if len(encoded_samples) == 0:
                break

            if all_encoded_samples is None:
                all_encoded_samples = encoded_samples
            else:
                all_encoded_samples = np.concatenate(
                    (all_encoded_samples, encoded_samples), axis=0
                )

            sample_bar.update(encoded_samples.shape[0])

            curr_size += len(encoded_samples)

            # Sample incrementally
            if curr_size < num_samples:
                base_points = np.concatenate((base_points, encoded_samples), axis=0)

        all_encoded_samples = np.array(all_encoded_samples)
        sample_bar.close()

        self.topo_ae_model.eval()
        with torch.no_grad():
            decoded_augmented_data = (
                decoder(to_torch_tensor(all_encoded_samples)).detach().numpy()
            )
        self.topo_ae_model.train()

        denormalized_data = self.scaler.inverse_transform(decoded_augmented_data)

        if self.config["one_hot_cols"] is not None:
            one_hot_groups = sorted(self.config["one_hot_cols"], key=lambda g: g[0])

            pieces = []
            prev_end = 0
            cumulative_offset = 0

            for group in one_hot_groups:
                rep_col_current = group[0] - cumulative_offset
                num_classes = len(group)

                pieces.append(denormalized_data[:, prev_end:rep_col_current])

                one_hot_encoding = np.array(
                    denormalized_data[:, rep_col_current], dtype=int
                )
                one_hot_vec = torch.nn.functional.one_hot(
                    torch.tensor(one_hot_encoding), num_classes
                ).numpy()
                pieces.append(one_hot_vec)

                prev_end = rep_col_current + 1
                cumulative_offset += num_classes - 1

            pieces.append(denormalized_data[:, prev_end:])

            denormalized_data = np.concatenate(pieces, axis=1)

        augmented_df = pd.DataFrame(denormalized_data, columns=self.column_names)

        if plot_latent_space:
            augmented_labels = augmented_df["label"].values

            complete_labels = np.concatenate([augmented_labels, labels], axis=0)
            complete_latent = np.concatenate([all_encoded_samples, latent], axis=0)

            fig = plt.figure()
            plt.set_cmap("Spectral")

            ax1 = fig.add_subplot(221)
            ax1.scatter(
                all_encoded_samples[:, 0],
                all_encoded_samples[:, 1],
                c=augmented_labels,
                s=0.1,
                alpha=0.8,
            )

            ax1.set_title("Sampled datapoints latent")

            ax2 = fig.add_subplot(222)
            ax2.scatter(
                complete_latent[:, 0],
                complete_latent[:, 1],
                c=complete_labels,
                s=0.1,
                alpha=0.8,
            )
            ax2.set_title("Combined latent")

            ax3 = fig.add_subplot(223)
            ax3.scatter(latent[:, 0], latent[:, 1], c=labels, s=0.1, alpha=0.8)
            ax3.set_title("Original data latent")

            plt.show()

            plt.scatter(latent[:, 0], latent[:, 1], c=labels, s=0.1, alpha=0.8)
            plt.savefig("test.svg", format="svg", transparent=True)

        return augmented_df
