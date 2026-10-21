import numpy as np
import pandas as pd
from config import get_dataset_info
import os

permute_data = False

datasets = get_dataset_info()

for dataset in datasets:
    data_path = datasets[dataset]["datapath"]
    save_path = datasets[dataset]["savepath"]

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    train = []
    test = []
    val = []

    names = []

    if len(datasets[dataset]["str_cols"]) != 0:
        c_test = np.load(data_path + "C_test.npy", allow_pickle=True)
        c_train = np.load(data_path + "C_train.npy", allow_pickle=True)
        c_val = np.load(data_path + "C_val.npy", allow_pickle=True)

        train.append(c_train)
        test.append(c_test)
        val.append(c_val)

        for i in range(c_train.shape[1]):
            names.append("categorical_" + str(i))

    n_test = np.load(data_path + "N_test.npy", allow_pickle=True)
    n_train = np.load(data_path + "N_train.npy", allow_pickle=True)
    n_val = np.load(data_path + "N_val.npy")

    train.append(n_train)
    test.append(n_test)
    val.append(n_val)

    for i in range(n_train.shape[1]):
        names.append("numerical_" + str(i))

    if datasets[dataset]["has_q"]:
        q_test = np.load(data_path + "q_test.npy", allow_pickle=True)
        q_train = np.load(data_path + "q_train.npy", allow_pickle=True)
        q_val = np.load(data_path + "q_val.npy", allow_pickle=True)

        train.append(np.array([q_train]).T)
        test.append(np.array([q_test]).T)
        val.append(np.array([q_val]).T)

        names.append("q")

    y_test = np.load(data_path + "y_test.npy", allow_pickle=True)
    y_train = np.load(data_path + "y_train.npy", allow_pickle=True)
    y_val = np.load(data_path + "y_val.npy", allow_pickle=True)

    train.append(np.array([y_train]).T)
    test.append(np.array([y_test]).T)
    val.append(np.array([y_val]).T)

    if permute_data:
        permute_idx = np.random.permutation(names)
        permute_idx = permute_idx.tolist()
        permute_idx.append("label")

    names.append("label")

    train_data = np.concatenate(train, axis=1)
    test_data = np.concatenate(test, axis=1)
    val_data = np.concatenate(val, axis=1)

    print("Loading data finished")

    if permute_data:
        pd.DataFrame(train_data, columns=names).reindex(permute_idx, axis=1).to_csv(
            save_path + "train.csv", index=False
        )
        print("Save train finished")
        pd.DataFrame(test_data, columns=names).reindex(permute_idx, axis=1).to_csv(
            save_path + "test.csv", index=False
        )
        print("Save test finished")
        pd.DataFrame(val_data, columns=names).reindex(permute_idx, axis=1).to_csv(
            save_path + "val.csv", index=False
        )
        print("Save validation finished")
    else:
        pd.DataFrame(train_data, columns=names).to_csv(
            save_path + "train.csv", index=False
        )
        print("Save train finished")
        pd.DataFrame(test_data, columns=names).to_csv(
            save_path + "test.csv", index=False
        )
        print("Save test finished")
        pd.DataFrame(val_data, columns=names).to_csv(save_path + "val.csv", index=False)
        print("Save validation finished")
