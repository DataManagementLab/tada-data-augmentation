import pandas as pd
from model_benchmarks.model_benchmark import ModelBenchmark
from topo_vae.clean_interface import TopoVAE


class TopoBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("topo")

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

        topo_ae = TopoVAE()
        topo_ae.train(
            train_data=data,
            categorical_columns=cat_cols,
            one_hot_col_names=one_hot_cols,
        )

        df = topo_ae.sample(sample_fac)

        if len(df) > sample_size:
            df = df.sample(n=sample_size, axis="rows")

        return df
