import pandas as pd
from realtabformer import REaLTabFormer
from model_benchmarks.model_benchmark import ModelBenchmark


class RealTabFormerBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("realtabformer")

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
        rtf = REaLTabFormer(
            model_type="tabular",
            gradient_accumulation_steps=4,
            logging_steps=100,
            numeric_max_len=11,
        )
        rtf.fit(pd.DataFrame(data).fillna(0))

        # Create synthetic data
        synthetic_data = rtf.sample(n_samples=sample_size)
        return synthetic_data
