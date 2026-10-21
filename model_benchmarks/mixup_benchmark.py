import pandas as pd
from model_benchmarks.model_benchmark import ModelBenchmark


class MixUpBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("mixup")

    def train_and_sample(
        self,
        data_path,
        cat_cols,
        sample_size,
        epochs,
        sample_fac=None,
        one_hot_cols=None,
    ):
        alpha = 0.2
        synthetic_data = []
        df = pd.read_csv(data_path)
        for i in range(sample_size):
            samples = df.sample(n=2)
            synthetic_sample = (
                alpha * samples.iloc[0].values + (1 - alpha) * samples.iloc[1].values
            )
            synthetic_data.append(synthetic_sample)
        synthetic_data = pd.DataFrame(synthetic_data, columns=df.columns)
        return synthetic_data
