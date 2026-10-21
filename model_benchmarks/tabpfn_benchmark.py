import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor, unsupervised
from model_benchmarks.model_benchmark import ModelBenchmark


class TabPFNBenchmark(ModelBenchmark):
    def __init__(self):
        super().__init__("tabpfn")

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

        org_cols = data.columns.tolist()
        shuffled_cols = np.random.permutation(data.columns)

        data_np = data[shuffled_cols].values

        clf = TabPFNClassifier(n_estimators=1, random_state=0)
        reg = TabPFNRegressor(n_estimators=1, random_state=0)
        model = unsupervised.TabPFNUnsupervisedModel(
            tabpfn_clf=clf,
            tabpfn_reg=reg,
        )
        col_idx = [shuffled_cols.tolist().index(name) for name in cat_cols]
        model.set_categorical_features(col_idx)
        model.fit(data_np)

        sample_reps = int(np.round(sample_size / 1000))
        df = None
        if sample_reps == 0:
            sample_reps = 1

        i = 0
        failed_counter = 0
        while i < sample_reps:
            try:
                synthetic_data = model.generate_synthetic_data(n_samples=1000).numpy()
            except Exception as e:
                failed_counter += 1
                if failed_counter == 3:
                    raise e
                continue

            synthetic_data = pd.DataFrame(synthetic_data, columns=shuffled_cols)
            if df is None:
                df = synthetic_data
            else:
                df = pd.concat((df, synthetic_data), axis="rows", ignore_index=True)
            i += 1
            failed_counter = 0

        df = df[org_cols]

        return df
