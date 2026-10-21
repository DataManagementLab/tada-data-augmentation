import numpy as np
import pandas as pd
import config
from downstream_models.run_downstream import run_downstream
from utils import categories_to_numerical, get_additional_info
import time


def main():
    datasets = config.get_dataset_info()
    benchmarks = config.BENCHMARKS

    for benchmark in benchmarks:

        for dataset in datasets:
            real_data = pd.read_csv(datasets[dataset]["datapath"] + "train.csv")
            validation_data = pd.read_csv(datasets[dataset]["datapath"] + "val.csv")
            test_data = pd.read_csv(datasets[dataset]["datapath"] + "test.csv")

            # Names of the columns that are discrete
            discrete_columns = datasets[dataset]["cat_cols"]

            str_columns = datasets[dataset]["str_cols"]
            preprocessed_real_data = categories_to_numerical(
                pd.concat((real_data, validation_data, test_data), axis=0),
                str_columns,
                datasets[dataset]["savepath"] + "mappings.json",
            )

            preprocessed_real_data = categories_to_numerical(
                real_data, str_columns, datasets[dataset]["savepath"] + "mappings.json"
            )

            get_additional_info(datasets, dataset, preprocessed_real_data)

            preprocessed_validation_data = categories_to_numerical(
                validation_data,
                str_columns,
                datasets[dataset]["savepath"] + "mappings.json",
            )
            preprocessed_test_data = categories_to_numerical(
                test_data, str_columns, datasets[dataset]["savepath"] + "mappings.json"
            )

            for sample_size in config.get_sample_sizes():
                if sample_size == 1:
                    sampled_data = preprocessed_real_data
                    sample_length = int(np.round(len(preprocessed_real_data) * 0.25))
                else:
                    sampled_data = preprocessed_real_data.sample(
                        frac=sample_size, random_state=123
                    )
                    if config.STRATIFIED_SAMPLING:
                        sampled_data = preprocessed_real_data.groupby("label").apply(
                            lambda x: x.sample(frac=sample_size, random_state=123)
                        )
                    sample_length = len(preprocessed_real_data) - len(sampled_data)

                if config.SAVE_DATASETS and not config.USE_EXISTING_DATASETS:
                    pd.DataFrame(sampled_data).to_csv(
                        datasets[dataset]["savepath"]
                        + dataset
                        + "_original_"
                        + str(sample_size)
                        + ".csv",
                        index=False,
                    )

                model_run_time = 0
                if not config.USE_EXISTING_DATASETS:
                    model_start_time = time.time()
                    synthetic_data = benchmark.train_and_sample(
                        data_path=(
                            datasets[dataset]["savepath"]
                            + dataset
                            + "_original_"
                            + str(sample_size)
                            + ".csv"
                        ),
                        cat_cols=discrete_columns,
                        sample_size=sample_length,
                        epochs=config.NUM_EPOCHS,
                        sample_fac=sample_size,
                        one_hot_cols=datasets[dataset]["one_hot_cols"],
                    )
                    model_run_time = time.time() - model_start_time
                    time_file = open("results/time.txt", "a")
                    time_file.write("Model: " + benchmark.get_model_name() + "\n")
                    time_file.write("Dataset: " + dataset + "\n")
                    time_file.write("Sample size: " + str(sample_size) + "\n")
                    time_file.write("Time: " + str(model_run_time) + "\n")
                    time_file.write("-----------------\n")
                    time_file.close()

                if config.SAVE_DATASETS and not config.USE_EXISTING_DATASETS:
                    pd.DataFrame(synthetic_data).to_csv(
                        datasets[dataset]["savepath"]
                        + dataset
                        + "_"
                        + benchmark.get_model_name()
                        + "_synthetic_"
                        + str(sample_size)
                        + ".csv",
                        index=False,
                    )
                print("Model: " + benchmark.get_model_name())
                print("Downstream: " + config.DOWNSTREAM_MODEL)
                print("Dataset: " + dataset)
                print("Sample size: " + str(sample_size))
                print("-----------------")

                if config.USE_EXISTING_DATASETS:
                    sampled_data = pd.read_csv(
                        datasets[dataset]["savepath"]
                        + dataset
                        + "_original_"
                        + str(sample_size)
                        + ".csv"
                    )
                    synthetic_data = pd.read_csv(
                        datasets[dataset]["savepath"]
                        + dataset
                        + "_"
                        + benchmark.get_model_name()
                        + "_synthetic_"
                        + str(sample_size)
                        + ".csv"
                    )

                categorical_label = True if "label" in discrete_columns else False
                original_data_results, synthetic_data_results = run_downstream(
                    model_name=config.DOWNSTREAM_MODEL,
                    original_data=sampled_data,
                    synthetic_data=pd.DataFrame(synthetic_data),
                    validation_data=preprocessed_validation_data,
                    test_data=preprocessed_test_data,
                    epochs=config.DOWNSTREAM_EPOCHS,
                    dataset_info=datasets[dataset],
                    cat_label=categorical_label,
                )

                if categorical_label:
                    original_acc = original_data_results[0]
                    original_f1 = original_data_results[1]

                    synthetic_acc = synthetic_data_results[0]
                    synthetic_f1 = synthetic_data_results[1]

                    results_file = open("results/results_acc.txt", "a")
                    results_file.write("Model: " + benchmark.get_model_name() + "\n")
                    results_file.write("Downstream: " + config.DOWNSTREAM_MODEL + "\n")
                    results_file.write("Dataset: " + dataset + "\n")
                    results_file.write("Sample size: " + str(sample_size) + "\n")
                    results_file.write("Original accuracy: " + str(original_acc) + "\n")
                    results_file.write(
                        "Synthetic accuracy: " + str(synthetic_acc) + "\n"
                    )
                    results_file.write("Time: " + str(model_run_time) + "\n")
                    results_file.write("-----------------\n")
                    results_file.close()

                    results_file = open("results/results_f1.txt", "a")
                    results_file.write("Model: " + benchmark.get_model_name() + "\n")
                    results_file.write("Downstream: " + config.DOWNSTREAM_MODEL + "\n")
                    results_file.write("Dataset: " + dataset + "\n")
                    results_file.write("Sample size: " + str(sample_size) + "\n")
                    results_file.write("Original f1: " + str(original_f1) + "\n")
                    results_file.write("Synthetic f1: " + str(synthetic_f1) + "\n")
                    results_file.write("Time: " + str(model_run_time) + "\n")
                    results_file.write("-----------------\n")
                    results_file.close()
                else:
                    results_file = open("results/results_rmse.txt", "a")
                    results_file.write("Model: " + benchmark.get_model_name() + "\n")
                    results_file.write("Downstream: " + config.DOWNSTREAM_MODEL + "\n")
                    results_file.write("Dataset: " + dataset + "\n")
                    results_file.write("Sample size: " + str(sample_size) + "\n")
                    results_file.write(
                        "Original loss: " + str(original_data_results) + "\n"
                    )
                    results_file.write(
                        "Synthetic loss: " + str(synthetic_data_results) + "\n"
                    )
                    results_file.write("Time: " + str(model_run_time) + "\n")
                    results_file.write("-----------------\n")
                    results_file.close()


if __name__ == "__main__":
    main()
    exit()
    from topo_vae.clean_interface import TopoAE

    data = np.load("./data/mammoth/splits/prepared/train_data_1.npy")
    labels = np.load("data/mammoth/splits/prepared/train_labels_1.npy")
    dataset_name = "mammoth"

    topo_ae = TopoAE(set_size=1000)
    topo_ae.train(data, labels, dataset_name)
    topo_ae.sample(1.0, ["l1", "l2", "label"])
