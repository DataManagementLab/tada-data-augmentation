from sdv.metadata import Metadata
from sdv.single_table import CopulaGANSynthesizer
from model_benchmarks.model_benchmark import ModelBenchmark
import pandas as pd


class CopulaGANBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("copula_gan")

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
        metadata = Metadata.detect_from_dataframe(data, table_name="data")
        model = CopulaGANSynthesizer(metadata)
        model.fit(data)

        # Create synthetic data
        synthetic_data = model.sample(num_rows=sample_size)
        return synthetic_data
