# Reproducing the evaluations

This directory contains the compact inputs, settings, and numerical results needed to reproduce
the central evaluations.

## Contents

- `configs/benchmark_settings.yaml`: model mapping and shared numerical settings
- `manifests/`: exact molecule selections used in each benchmark
- `data/geometry/`: initial and KS-optimized XYZ structures for the geometry benchmarks
- `results/`: compact per-molecule and aggregate numerical results

The model names map to the four directories in
[the Hugging Face repository](https://huggingface.co/sciai-lab/structures25-nuclear-gradients):

| Model          | Hugging Face directory | Evaluation ID |
| -------------- | ---------------------- | ------------: |
| Graphormer     | `Graphormer/`          |           519 |
| Graphormer NGL | `Graphormer-NGL/`      |           565 |
| eSEN           | `eSEN/`                |           479 |
| eSEN NGL       | `eSEN-NGL/`            |           597 |

## Environment

Install the project and download the models:

```bash
export DFT_DATA="/path/to/data"
export DFT_MODELS="/path/to/models"
uv sync --extra cuda
source .venv/bin/activate
hf download sciai-lab/structures25-nuclear-gradients \
  --local-dir "$DFT_MODELS/structures25-nuclear-gradients"
```

Use `--extra cpu` instead of `--extra cuda` for CPU execution. The full evaluations are
computationally expensive and use double-precision model inference.

The inference files can be checked independently of the datasets:

```python
from pathlib import Path

from mldft.utils.instantiators import instantiate_model

model_root = Path("/path/to/structures25-nuclear-gradients")
model = instantiate_model(
    model_root / "eSEN-NGL" / "eSEN-NGL.safetensors",
    device="cpu",
)
```

## Data generation

The `mldft_ks` and `mldft_labelgen` entry points generate the KS calculations and orbital-free
labels. Dataset configurations are under `configs/datagen`. Nuclear-gradient labels are added with:

```bash
.venv/bin/python scripts/add_nuclear_gradients.py \
  --dataset QM9_perturbed_fock --n_molecules NUMBER_OF_MOLECULES
```

The exact molecule selections used for evaluation are recorded under `manifests`. Paths below
assume that generated datasets use the directory names in `configs/benchmark_settings.yaml`.

## Preparing the model directories

The evaluation entry points use the original training-run directory convention. For each downloaded
model, create a `checkpoints` directory containing links to its safetensors file and portable model
configuration. For example:

```bash
model_dir="$DFT_MODELS/structures25-nuclear-gradients/eSEN-NGL"
mkdir -p "$model_dir/checkpoints"
ln -s ../eSEN-NGL.safetensors "$model_dir/checkpoints/eSEN-NGL.safetensors"
ln -s ../model.yaml "$model_dir/checkpoints/model.yaml"
```

Use the model directory as `run_path` and the safetensors filename as `ckpt_choice`.

## Nuclear gradients at reference densities

Run the analytic gradients at the saved KS densities for each model and dataset. The following is
the eSEN NGL/QM9 command; substitute the model and split paths from
`configs/benchmark_settings.yaml` for the other combinations:

```bash
.venv/bin/python -m mldft.ofdft.evaluation.analytical_forces \
  run_path="$DFT_MODELS/structures25-nuclear-gradients/eSEN-NGL" \
  ckpt_choice=eSEN-NGL.safetensors \
  split_file_path="$DFT_DATA/QM9_perturbed_fock/split.pkl" \
  split=test n_molecules=250 molecule_choice=seeded_random seed=0 \
  device=cuda model_dtype=torch.float64 gradient_method=analytic
```

The resulting aggregate values are in `results/force_summary.csv` with
`mode=reference_density`; the underlying molecule-level values are split by dataset across
`results/force_per_molecule_{qm7x,qm9,qmugsx}.csv`.

## Nuclear gradients at self-consistent densities

The corresponding evaluation after density optimization uses:

```bash
.venv/bin/python -m mldft.ofdft.evaluation.analytical_forces_at_gs \
  run_path="$DFT_MODELS/structures25-nuclear-gradients/eSEN-NGL" \
  ckpt_choice=eSEN-NGL.safetensors \
  split_file_path="$DFT_DATA/QM9_perturbed_fock/split.pkl" \
  split=test n_molecules=250 molecule_choice=seeded_random seed=0 \
  device=cuda model_dtype=torch.float64 \
  density_optimizer.hard_projection=true \
  density_optimizer.eigenvalue_threshold=0.001
```

Use hard projection with an eigenvalue threshold of `0.001` for both eSEN models and disable hard
projection for both Graphormer models, as listed in `configs/benchmark_settings.yaml`.

The aggregate values are in `results/force_summary.csv` with `mode=self_consistent`; the
molecule-level values are split by dataset across
`results/force_per_molecule_{qm7x,qm9,qmugsx}.csv`.

## Numerical validation of analytic gradients

The finite-difference validation uses the 25 QM9 molecules listed in
`manifests/numerical_gradients_qm9.txt`, the eSEN NGL model, a displacement of
0.001 Bohr, and density convergence tolerances of `1e-3`, `1e-4`, and `1e-6`:

```bash
.venv/bin/python -m mldft.ofdft.evaluation.gradient_comparison \
  run_path="$DFT_MODELS/structures25-nuclear-gradients/eSEN-NGL" \
  ckpt_choice=eSEN-NGL.safetensors \
  split_file_path="$DFT_DATA/QM9_perturbed_fock/split.pkl" \
  split=val n_molecules=25 molecule_choice=seeded_random seed=0 \
  epsilon=0.001 methods='[analytic,numerical,fully_numerical]' \
  density_optimizer.convergence_tolerance=0.0001 device=cuda \
  save_gradients=true
```

Repeat with the other two convergence tolerances. The compact per-molecule output for all three
runs is in `results/numerical_gradients.csv`.

## Geometry optimization

Each geometry dataset under `data/geometry` contains the initial structures and corresponding
KS-optimized XYZ structures. Generate the PySCF reference checkpoints before running ML geometry
optimization:

```bash
.venv/bin/python scripts/perform_ks_geometry_optimization.py \
  evaluation/data/geometry/QM7X
```

For the tight QM7X subset, add `--geometric_cfg gaussian_very_tight.yaml`.

Then run, for example:

```bash
mldft_geomopt \
  run_path="$DFT_MODELS/structures25-nuclear-gradients/eSEN-NGL" \
  ckpt_choice=eSEN-NGL.safetensors \
  data_path=evaluation/data/geometry/QM7X n_molecules=100 \
  gradients=analytic maxsteps=50 \
  density_optimizer.convergence_tolerance=0.0001
```

Use the loose geomeTRIC configuration, 50 geometry steps, and a density convergence tolerance of
`1e-4` for the main QM7X, QMugs, and constitutional-isomer benchmarks. For `QM7X_tight`, use
`geometric_config=gaussian_very_tight`, `maxsteps=300`, and
`density_optimizer.convergence_tolerance=1e-6`. The compact per-molecule metrics are in
`results/geometry_optimization.csv`.

## Density optimization

Density optimization on the QMugs size-extrapolation set uses PBE/6-31G(2df,p), SAD
initialization, and a density-gradient convergence tolerance of `1e-4`:

```bash
mldft_denop \
  run_path="$DFT_MODELS/structures25-nuclear-gradients/eSEN-NGL" \
  ckpt_choice=eSEN-NGL.safetensors \
  split_file_path="$DFT_DATA/QMUGSLargeBins/split.pkl" \
  split=test n_molecules=-1 device=cuda \
  density_optimizer.convergence_tolerance=0.0001
```

The run covers all 850 molecules in `QMUGSLargeBins`. The reported comparison retains the 800
molecules with at least 21 heavy atoms. `results/density_optimization.csv` contains this filtered
set and records the heavy-atom count explicitly.

The STRUCTURES25 rows use the released QMugs model from training run 214 with the same optimizer
settings and reproduce its reported aggregate values after rounding.
