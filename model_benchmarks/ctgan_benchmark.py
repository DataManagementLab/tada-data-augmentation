import pandas as pd
from ctgan import CTGAN
from model_benchmarks.model_benchmark import ModelBenchmark


class CTGANBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("ctgan")

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
        ctgan = CTGAN(epochs=epochs)
        ctgan.fit(data, cat_cols)

        # Create synthetic data
        synthetic_data = ctgan.sample(sample_size)
        return synthetic_data
