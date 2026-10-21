from be_great import GReaT
from model_benchmarks.model_benchmark import ModelBenchmark
import pandas as pd
import shutil


class GReaTBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("great")

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
        great = GReaT(llm="distilgpt2", batch_size=32, epochs=epochs, fp16=True)
        great.fit(data)

        # Create synthetic data
        synthetic_data = great.sample(n_samples=sample_size)
        shutil.rmtree("./trainer_great")
        return synthetic_data
