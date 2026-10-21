from tqdm import tqdm
import pandas as pd
import json
import os


def categories_to_numerical(dataset: pd.DataFrame, cat_cols, mappings_file_path):
    if os.path.exists(mappings_file_path):
        with open(mappings_file_path) as json_file:
            mappings_file = json.load(json_file)

        loop = tqdm(range(1))
        loop.set_description(
            f"Converting categorical values to numerical (using existing mapping file)"
        )

        for i, _ in enumerate(loop):
            dataset = dataset.replace(mappings_file)

        return dataset

    mappings = {}

    loop = tqdm(cat_cols)
    loop.set_description(f"Converting categorical values to numerical")

    for i, col in enumerate(loop):
        vals = dataset[col].unique()
        counter = 0
        mapping = {}

        for val in vals:
            mapping[val] = counter
            dataset = dataset.replace(val, counter)
            counter = counter + 1

        mappings[col] = mapping

    with open(mappings_file_path, "w") as outfile:
        json.dump(mappings, outfile)

    return dataset


def numerical_to_categories(dataset, mapping_file):
    with open(mapping_file) as json_file:
        mappings = json.load(json_file)
    for column_name in mappings:
        for categorical_value in mappings[column_name]:
            dataset[column_name] = dataset[column_name].replace(
                mappings[column_name][categorical_value], categorical_value
            )

    return dataset


def get_additional_info(datasets_conf, dataset_name, dataset):
    cat_col_index = []
    cardinalities = []
    for cat_col in datasets_conf[dataset_name]["cat_cols"]:
        if cat_col is "label":
            continue
        cat_col_index.append(dataset.columns.get_loc(cat_col))
        cardinalities.append(dataset[cat_col].round().nunique(dropna=False) + 1)

    datasets_conf[dataset_name]["num_features"] = (
        len(dataset.columns) - 1 - len(cat_col_index)
    )
    datasets_conf[dataset_name]["cat_col_index"] = cat_col_index
    datasets_conf[dataset_name]["cardinalities"] = cardinalities
