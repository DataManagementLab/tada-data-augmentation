import pandas as pd
from ctgan import TVAE
from model_benchmarks.model_benchmark import ModelBenchmark


class TVAEBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("tvae")

    def train_and_sample(
        self,
        data_path,
        cat_cols,
        sample_size,
        epochs,
        sample_fac=None,
        one_hot_cols=None,
    ):
        data = pd.read_csv(data_path)
        tvae = TVAE(epochs=epochs)
        tvae.fit(data, cat_cols)

        # Create synthetic data
        synthetic_data = tvae.sample(sample_size)
        return synthetic_data
