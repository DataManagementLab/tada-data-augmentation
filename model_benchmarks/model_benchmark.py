class ModelBenchmark:
    def __init__(self, name):
        self.name = name

    def get_model_name(self):
        """
        :return: Name of the used model as a String
        """
        return self.name

    def train_and_sample(
        self,
        data_path,
        cat_cols,
        sample_size,
        epochs,
        sample_fac=None,
        one_hot_cols=None,
    ):
        """
        Main benchmark method which trains a new instance of the model,
        samples data and creates a Dataframe with the sampled data.
        :param data_path: Path to the dataset to be trained on
        :param cat_cols: List of categorical columns
        :param sample_size: Number of data samples to be created
        :param epochs: Number of epochs for the training
        :param sample_fac: Sample size factor
        :param one_hot_cols: List of one hot encoded columns
        :return: Synthetic dataset as pd.Dataframe
        """
        raise NotImplementedError
