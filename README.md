# TADA

Tabular data augmentation with a topologically regularized Variational Autoencoder (TADA), plus a benchmarking suite for comparing it against other tabular synthesis/augmentation methods on downstream tasks.

The project has two main contributions:

1. **A benchmark suite** for evaluating tabular data augmentation/synthesis methods by training them on a sample of a dataset, generating synthetic rows, and comparing downstream-model performance (accuracy/F1 for classification, RMSE for regression) on original vs. augmented data.
2. **TADA**, a Variational Autoencoder regularized with a topological signature loss (based on persistent homology) that preserves the topological structure of the data manifold while augmenting/oversampling tabular datasets, with a simple wrapper interface for standalone use.

## Repository layout

```
config.py                    # Benchmark configuration (datasets, models, sample sizes, ...)
run_benchmark.py             # Entry point that runs the configured benchmark suite

model_benchmarks/            # One wrapper per synthesis/augmentation method, all implementing
                              # the ModelBenchmark interface (model_benchmark.py)
  topo_benchmark.py          #   - wraps TADA for the benchmark
  ctgan_benchmark.py, tvae_benchmark.py, great_benchmark.py, ...

downstream_models/           # Downstream models used to score original vs. synthetic data
  run_downstream.py          #   - orchestrates downstream training/evaluation
  mlp.py

topo_vae/                    # The TopoVAE model and its public interface
  clean_interface.py         #   - TopoVAE wrapper class (train / sample)
  topo_vae_model.py          #   - the PyTorch Lightning VAE + topological signature loss
  sampling.py, label_prediction.py, utils.py
  config.json                #   - default TopoVAE hyperparameters

data/                        # Datasets (train/val/test splits) used by the benchmark
                            Data from: https://github.com/yandex-research/rtdl-revisiting-models#how-to-reproduce-the-results
                            The provided numpy files were converted into csv files 
                            with the column names categorical_i and numerical_i, as well as, label
                            (vis_and_data_prep/merging_data.py is doing this automatically)
results/                     # Benchmark output (accuracy, F1, RMSE, timing)
vis_and_data_prep/           # Plotting and data preparation helper scripts
```

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is used automatically when available (`torch.cuda.is_available()`), both for TopoVAE training and the downstream models; otherwise everything falls back to CPU.

## The benchmark

The benchmark trains each configured augmentation/synthesis model on a fraction of each configured dataset, generates synthetic data to fill the rest, and then trains a downstream model on the original vs. the augmented dataset to compare performance.

### Running it

```bash
python run_benchmark.py
```

### Configuring it (`config.py`)

- `BENCHMARKS` — list of `ModelBenchmark` instances to run (e.g. `TopoBenchmark()`, `CTGANBenchmark()`, `TVAEBenchmark()`, ...). Uncomment/add entries to benchmark additional methods.
- `SELECTED_DATASETS` — which datasets (by name) to run the benchmark on. Available datasets are defined in `get_dataset_info_all()`, each with its data path, categorical/string columns, and any one-hot-encoded column groups.
- `SAMPLE_PORTIONS` — fractions of each dataset to sample as the "real" training data; the rest is synthesized to compare against the full dataset.
- `NUM_EPOCHS` — number of training epochs for the augmentation model.
- `DOWNSTREAM_MODEL` — downstream evaluator: `"mlp"`
- `DOWNSTREAM_EPOCHS` — training epochs for the downstream model.
- `STRATIFIED_SAMPLING` — whether sampling from the real dataset is stratified by label.
- `SAVE_DATASETS` / `USE_EXISTING_DATASETS` — persist sampled/synthetic datasets to disk, or reuse previously generated ones instead of re-running the augmentation model.
- `DOWNSTREAM_DEVICE` — device used for downstream model training.

### Adding a new method to benchmark

Implement `ModelBenchmark` (see `model_benchmarks/model_benchmark.py`) with a `train_and_sample(...)` method that trains on a dataset and returns a synthetic `pd.DataFrame`, then add an instance to `BENCHMARKS` in `config.py`.

### Results

Results are appended to plain-text files in `results/`:

- `results_acc.txt`, `results_f1.txt` — classification metrics (original vs. synthetic) per model/dataset/sample size
- `results_rmse.txt` — regression metrics
- `time.txt` — training time per model/dataset/sample size

## TADA

TADA is a Variational Autoencoder trained with an additional topological signature loss that encourages the latent space to preserve the persistent-homology structure of the input data, then used to oversample/augment a tabular dataset via latent-space sampling around encoded points, with optional class-balancing.

### Usage

The public interface lives in `topo_vae/clean_interface.py`:

```python
import pandas as pd
from topo_vae.clean_interface import TopoVAE

train_data = pd.read_csv("data/adult/train.csv")

topo_ae = TopoVAE()  # or TopoVAE(config_path="path/to/config.json")
topo_ae.train(
    train_data=train_data,
    categorical_columns=["categorical_0", "label"],  # names of categorical columns
    one_hot_col_names=None,                           # optional groups of one-hot columns
    label_column_name=None,                           # auto-detected if not given
)

augmented_data = topo_ae.sample(
    sample_size=1.0,          # fraction of the original dataset size to add
    plot_latent_space=True,   # visualize original vs. sampled latent space
)
```

- `train(...)` preprocesses the data (handles categorical/one-hot columns, scaling, label detection), then fits the topologically regularized VAE with PyTorch Lightning.
- `sample(...)` draws new points around the encoded training data in latent space, optionally balancing classes and filtering low-confidence decodes, and decodes them back into the original data format (including re-expanding one-hot columns).

### Configuration

Default hyperparameters are defined in `topo_vae/clean_interface.py` (and can be overridden via a JSON file passed as `config_path`), including `max_epochs`, `hidden_dim`, `latent_dim`, `batch_size`, `lr`, `topo_weight`, `topo_start_epoch`, `beta` (KL weighting), `balanced_sampling`, and `delete_low_certainty_percantage`.

### Persistent homology backend

The topological signature loss (`topo_vae/topo_vae_model.py`) supports a pure-Python persistent homology calculation
