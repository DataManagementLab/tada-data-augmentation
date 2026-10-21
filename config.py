# from model_benchmarks.ctgan_benchmark import CTGANBenchmark
# from model_benchmarks.tvae_benchmark import TVAEBenchmark
# from model_benchmarks.great_benchmark import GReaTBenchmark
# from model_benchmarks.realtabformer_benchmark import RealTabFormerBenchmark
# from model_benchmarks.mixup_benchmark import MixUpBenchmark
# from model_benchmarks.ctabganplus_benchmark import CTABGANPlusBenchmark
# from model_benchmarks.copulagan_benchmark import CopulaGANBenchmark
# from model_benchmarks.tabpfn_benchmark import TabPFNBenchmark
from model_benchmarks.topo_benchmark import TopoBenchmark
import torch

BENCHMARKS = [
    # CTGANBenchmark(),
    # TVAEBenchmark(),
    # GReaTBenchmark(),
    # MixUpBenchmark(),
    # RealTabFormerBenchmark(),
    # CTABGANPlusBenchmark(),
    # CopulaGANBenchmark(),
    # TabPFNBenchmark(),
    TopoBenchmark()
]  # List of benchmark method, change for different model to benchmark (see model_benchmark.py to implement new models)

NUM_EPOCHS = 100  # Number of epochs the model should be trained per benchmark
DOWNSTREAM_EPOCHS = (
    50  # Number of epochs the downstream model should be trained per benchmark
)
DOWNSTREAM_MODEL = (
    "mlp"  # Name of the downstream model either "mlp", "xgboost" or "ft-transformer"
)
SAVE_DATASETS = True
USE_EXISTING_DATASETS = False
DOWNSTREAM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STRATIFIED_SAMPLING = False

SELECTED_DATASETS = [
    "adult", "aloi", "california_housing", "covtype", "helena", "jannis", "microsoft", "year", "shelter-animal-outcomes"
]  # Names of the datasets for which the benchmark should be run
# All datasets: ["adult", "aloi", "california_housing", "covtype", "helena", "jannis", "microsoft", "year", "shelter-animal-outcomes"]
# Great works on : "adult", "california_housing", "shelter-animal-outcomes"
SAMPLE_PORTIONS = [
    0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1
]  # All used portions: [0.01, 0.05, 0.1, 0.2, 0.5, 0.75, 1]


def get_dataset_info_all():
    """
    :return: Dict with all datasets which should be used in the benchmark.
    Contains path where the train data is located,
    path where the sampled data should be saved and a list with all categorical columns
    """
    return {
        "adult": {
            "datapath": "data/adult/",
            "savepath": "data/adult/splits/",
            "cat_cols": [
                "categorical_0",
                "categorical_1",
                "categorical_2",
                "categorical_3",
                "categorical_4",
                "categorical_5",
                "categorical_6",
                "categorical_7",
                "label",
            ],
            "str_cols": [
                "categorical_0",
                "categorical_1",
                "categorical_2",
                "categorical_3",
                "categorical_4",
                "categorical_5",
                "categorical_6",
                "categorical_7",
            ],
            "has_q": False,
            "one_hot_cols": [],
        },
        "aloi": {
            "datapath": "data/aloi/",
            "savepath": "data/aloi/splits/",
            "cat_cols": [],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [],
        },
        "california_housing": {
            "datapath": "data/california_housing/",
            "savepath": "data/california_housing/splits/",
            "cat_cols": [],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [],
        },
        "covtype": {
            "datapath": "data/covtype/",
            "savepath": "data/covtype/splits/",
            "cat_cols": ["label"],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [
                ["numerical_10", "numerical_11", "numerical_12", "numerical_13"],
                [
                    "numerical_14",
                    "numerical_15",
                    "numerical_16",
                    "numerical_17",
                    "numerical_18",
                    "numerical_19",
                    "numerical_20",
                    "numerical_21",
                    "numerical_22",
                    "numerical_23",
                    "numerical_24",
                    "numerical_25",
                    "numerical_26",
                    "numerical_27",
                    "numerical_28",
                    "numerical_29",
                    "numerical_30",
                    "numerical_31",
                    "numerical_32",
                    "numerical_33",
                    "numerical_34",
                    "numerical_35",
                    "numerical_36",
                    "numerical_37",
                    "numerical_38",
                    "numerical_39",
                    "numerical_40",
                    "numerical_41",
                    "numerical_42",
                    "numerical_43",
                    "numerical_44",
                    "numerical_45",
                    "numerical_46",
                    "numerical_47",
                    "numerical_48",
                    "numerical_49",
                    "numerical_50",
                    "numerical_51",
                    "numerical_52",
                    "numerical_53",
                ],
            ],
        },
        "helena": {
            "datapath": "data/helena/",
            "savepath": "data/helena/splits/",
            "cat_cols": [],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [],
        },
        "jannis": {
            "datapath": "data/jannis/",
            "savepath": "data/jannis/splits/",
            "cat_cols": ["label"],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [],
        },
        "microsoft": {
            "datapath": "data/microsoft/",
            "savepath": "data/microsoft/splits/",
            "cat_cols": [],
            "str_cols": [],
            "has_q": True,
            "one_hot_cols": [],
        },
        "year": {
            "datapath": "data/year/",
            "savepath": "data/year/splits/",
            "cat_cols": [],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [],
        },
        "shelter-animal-outcomes": {
            "datapath": "data/shelter-animal-outcomes/",
            "savepath": "data/shelter-animal-outcomes/splits/",
            "cat_cols": [
                "AnimalType",
                "SexuponOutcome",
                "AgeuponOutcome",
                "Breed",
                "Color",
                "label",
            ],
            "str_cols": [
                "AnimalType",
                "SexuponOutcome",
                "AgeuponOutcome",
                "Breed",
                "Color",
                "label",
            ],
            "has_q": False,
            "one_hot_cols": [],
        },
        "mammoth": {
            "datapath": "data/mammoth/",
            "savepath": "data/mammoth/splits/",
            "cat_cols": ["label"],
            "str_cols": [],
            "has_q": False,
            "one_hot_cols": [],
        },
    }


def get_dataset_info():
    all_datasets_info = get_dataset_info_all()
    dataset_info = {}
    for dataset in SELECTED_DATASETS:
        dataset_info[dataset] = all_datasets_info[dataset]
    return dataset_info


def get_sample_sizes():
    """
    :return: List with fractions which should be used in the benchmark.
    The sampled data is always 1-(fraction*dataset_length) if 1 sampled data is 0.25
    """
    return SAMPLE_PORTIONS
