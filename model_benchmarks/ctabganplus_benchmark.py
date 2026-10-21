import pandas as pd
import numpy as np
from model_benchmarks.model_benchmark import ModelBenchmark
from model_benchmarks.ctabganplus.ctabgan import CTABGAN


class CTABGANPlusBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("ctabganplus")

    def train_and_sample(
        self,
        data_path,
        cat_cols,
        sample_size,
        epochs,
        sample_fac=None,
        one_hot_cols=None,
    ):
        general = []
        for i in range(pd.read_csv(data_path).shape[1] - len(cat_cols)):
            general.append("numerical_" + str(i))
        model = CTABGAN(
            data_path,
            categorical_columns=cat_cols,
            general_columns=general,
            mixed_columns={},
            integer_columns=[],
            problem_type={None: None},
        )

        model.fit()
        sample_reps = int(np.round(sample_size / pd.read_csv(data_path).shape[0]))
        df = None
        if sample_reps == 0:
            sample_reps = 1

        for _ in range(sample_reps):
            data = model.generate_samples()
            if df is None:
                df = data
            else:
                df = pd.concat((df, data), axis="rows", ignore_index=True)
        return df
