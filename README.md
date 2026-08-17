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
prs_losses.py            PRANA's training objectives - notably ngl_r2_loss, the customized
                          loss the model is trained with (see "Loss function" below)
prs_metrics.py           PRS evaluation metrics (Nagelkerke R2, OR per 1 SD, AUROC/AUPRC);
                          ngl_r2_loss builds on generate_glm from here
main.py                  generic training/evaluation entry point - works with any dataset
                          that follows the standard CV fold layout
```

## Requirements

`main.py` needs: `torch`, `pandas`, `numpy`, `scikit-learn`, `scipy`, `statsmodels`
(`statsmodels` and `scipy` are pulled in by the `ngl_r2` loss via `prs_metrics.py`).

## Loss function

PRANA is not trained on plain cross-entropy. Its objective (`--loss ngl_r2`, the default) is
a **differentiable surrogate of the Nagelkerke pseudo-R²** of a logistic regression of the
phenotype on `[PRANA score + covariates]`, implemented as `ngl_r2_loss` in `prs_losses.py`:

1. An inner logistic-regression head is fit on the standardized PRANA score plus the PCs,
   by gradient descent, so it stays differentiable w.r.t. the score
   (`train_logistic_regression`).
2. Its log-likelihood is compared against the null-model log-likelihood from a statsmodels
   GLM to form the Nagelkerke R².
3. The loss is `(margin - R² · sign(corr(score, label)))²`, passed through a LeakyReLU and
   normalized - so the network directly maximizes the metric the paper reports rather than
   per-sample classification accuracy. A `PerfectSeparationError` in the GLM yields a zero
   loss for that batch.

`--loss bce` (class-weighted BCE on the fused output) and `--loss bce_prs` (class-weighted
BCE on the raw PRS branch alone) are available as baselines, but they are *not* the
published objective.

## What gets adapted, and at what learning rate

Two further details of the training recipe are easy to miss but materially change the
result (`configure_trainable_params` and `derive_lr` in `main.py`):

**The base PRS betas are frozen.** `mult_prs_weights` — the per-SNP effect sizes loaded
from `--gwas_path`, the model's largest tensor — is held fixed. This is the method itself:
PRANA adapts an existing PRS to a new ancestry through the layers *around* the score rather
than refitting the score. Conversely `scaler_adapter`, the mixing weight in
`h_out = h_prs + h_out1 * scaler_adapter`, is made trainable (the model constructor
declares it `requires_grad=False`). Pass `--train_prs_weights` and/or
`--freeze_adapter_scaler` to override either.

**The learning rate is derived from the model, not tuned.** With `--lr` unset, it is
`10 * sd(non-zero base PRS betas) / n_snps`. This scales the step size to the magnitude of
the weights being adapted, so it transfers across GWAS panels of different sizes and
effect-size scales without retuning. Passing `--lr` explicitly overrides it.

Both defaults reproduce `main_scz_cv.py` / `main_cimba_cv.py` / `main_ukb_cv.py`. Running
with `--train_prs_weights --freeze_adapter_scaler` reproduces `main_bcac_cv.py` /
`main_bcac_loo.py`, which never applied the freezing step.

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
| `--loss` | `ngl_r2` | Training objective: `ngl_r2` (PRANA's customized differentiable Nagelkerke R² loss - see above), or the `bce` / `bce_prs` baselines. |
| `-l`, `--lr` | *derived* | Learning rate. Unset means `10 * sd(non-zero base PRS betas) / n_snps`. |
| `--train_prs_weights` | off | Also adapt the per-SNP base PRS betas, instead of keeping the score fixed. |
| `--freeze_adapter_scaler` | off | Pin the adapter mixing weight at 1.0 instead of learning it. |

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
