# PRANA

This repository includes in the implmentation of PRANA

Preprint is available [here](https://www.medrxiv.org/content/10.64898/2026.07.12.26357860v1).

Abstract: Polygenic risk scores (PRSs), which quantify inherited susceptibility to complex traits and diseases, have emerged as valuable tools for risk stratification and precision medicine. Despite their promise, PRS developed on European cohorts often demonstrate substantially reduced predictive accuracy in non-European populations, due to differences in genetic architecture. The disproportionate representation of European ancestry cohorts in genome-wide association studies (GWAS) leads to inequitable deployment of PRS technologies across diverse populations. Here, we introduce PRANA (Polygenic Risk Adaptation via Neural-network Architecture), a deep learning framework that adapts an existing PRS developed on one population to other ancestries. Unlike methods that require large-scale GWAS in the target population, PRANA leverages pre-trained PRS models derived from European cohorts and adapts them using modestly sized cohorts from the target population.

We evaluated PRANA on seven complex traits in South Asian, East Asian and Ashkenazi Jewish populations, as well as in selected smaller East Asian subpopulations where the scarcity of training data poses a particular challenge. PRANA mostly improved predictive performance of the baseline PRS models, and, in most cases, outperformed existing cross-ancestry multi-PRS approaches. These results highlight PRANA as a scalable and practical strategy to reduce disparities in genomic risk prediction and advance the equitable application of PRS in diverse populations.

## Layout

```
constants.py            path/config loader (reads config/config.json)
utils.py                device handling, auto_to_device
config/config.json       dataset/output path configuration - edit this for your environment
torch_dataset_cv.py      CV genotype loader (used by main.py)
nn_models/prana.py       the PRANA model definition (class PRANA)
main.py                  generic training/evaluation entry point - works with any dataset
                          that follows the standard CV fold layout
```

## Requirements

`main.py` needs: `torch`, `pandas`, `numpy`, `scikit-learn`.

## Configuration

Edit `config/config.json` before running anything:

- `BASE_PROFILE` / `PRS_DATASETS` - root folder containing your dataset directories
- `PRS_OUTPUT` - where model checkpoints are written
- `bcac_pca` / `ukbb_pca` - optional overrides used only when `--dataset` starts with
  `bcac` or `ukbb`

## Dataset layout expected by `main.py`

`main.py` trains on a single CV fold and expects the following standard layout:

```
<BASE_DIR>/<dataset>/<rep>/<imp>/ds___<fold>_<n_folds>_train
<BASE_DIR>/<dataset>/<rep>/<imp>/ds___<fold>_<n_folds>_train.frq
<BASE_DIR>/<dataset>/<rep>/<imp>/ds___<fold>_<n_folds>_validation
<BASE_DIR>/<dataset>/<rep>/pheno_<pheno_suffix>__<fold>_<n_folds>_train
...
```

Passing `--fold` equal to `--n_folds` trains on the outer "both"/"test" split instead of an
inner train/validation split (see `build_fold_subsets` in `main.py`). `--fold` and
`--n_folds` are combined into a `<fold>_<n_folds>` label for inner folds (e.g. `"1_3"` for
fold 1 of 3) - this is both the exact suffix `create_cv_repetitions.py` uses in dataset
filenames and the fold component of saved checkpoint filenames (see `fold_label()`).

The `<fold>_<n_folds>_train.frq` file holds per-SNP allele frequencies (e.g. from running
`plink --freq` on the training genotype file) - PRANA needs these to initialize its input
layers. If it's missing you'll get `RuntimeError: No allele-frequency file found...`.

You'll also need a GWAS summary-statistics file: a tab-separated table with the SNP id as
the index column and a per-SNP effect-size column (`--gwas_beta_col`, default `BETA`) - this
initializes the model's PRS layer.

## `main.py` parameters

Below are PRANA's primary parameters:

| Flag | Default | Meaning |
| --- | --- | --- |
| `-ds`, `--dataset` (required) | - | Dataset folder name under `BASE_PROFILE`. |
| `-i`, `--imp` (required) | - | Imputation subfolder holding the genotype files. |
| `-r`, `--rep` (required) | - | CV repetition folder name (e.g. `rep_<base_rep>_<i>`). |
| `-f`, `--fold` (required) | - | Inner fold number to run. Combined with `--n_folds` into the `<fold>_<n_folds>` label used in dataset/checkpoint filenames (see `fold_label()`). Set equal to `--n_folds` to train on the outer "both"/"test" split instead of an inner train/validation split. |
| `-nf`, `--n_folds` | `3` | Number of folds the CV split was created with. |
| `-p`, `--pheno_suffix` | `""` | Phenotype name used in the dataset's `pheno_<name>__<fold>` files. |
| `-g`, `--gwas_path` (required) | - | TSV of GWAS summary statistics, SNP id as the index column - initializes the model's PRS layer. |
| `--gwas_beta_col` | `BETA` | Column in `--gwas_path` holding the per-SNP effect size. |

Full list of parameteres is available at any time via
`python main.py --help`.**

```bash
python main.py \
    --dataset <dataset_name> \
    --imp <imputation_folder> \
    --rep <rep_folder> \
    --fold 1 --n_folds 3 \
    --pheno_suffix <phenotype_name> \
    --gwas_path /path/to/gwas_summary_stats.tsv
```
