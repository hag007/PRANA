"""Generic training/evaluation entry point for the PRANA model.

Unlike dataset-specific scripts that hardcode GWAS file paths, SNP index files, and
column names for one particular dataset, this script derives the SNP ids and allele
frequencies straight from the loaded genotype fold itself, so the same command works
for any dataset that follows the standard CV layout:

    <BASE_DIR>/<dataset>/<rep>/<imp>/ds___<fold>_<n_folds>_train
    <BASE_DIR>/<dataset>/<rep>/<imp>/ds___<fold>_<n_folds>_train.frq   (e.g. from `plink --freq`)
    <BASE_DIR>/<dataset>/<rep>/<imp>/ds___<fold>_<n_folds>_validation
    <BASE_DIR>/<dataset>/<rep>/pheno_<pheno_suffix>__<fold>_<n_folds>_train
    ...

Example:
    python main.py \
        --dataset <dataset_name> --imp <imputation_folder> --rep <rep_folder> \
        --fold 1 --n_folds 3 --pheno_suffix <phenotype_name> \
        --gwas_path /path/to/gwas_summary_stats.tsv --gwas_beta_col BETA
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

import constants
import utils
from utils import tensor_dtype, auto_to_device
from torch_dataset_cv import load_dataloader, param_builder_cv
from nn_models.prana import PRANA


def fold_label(fold, n_folds):
    """The fold identifier baked into dataset and checkpoint filenames: bare "<n_folds>"
    for the outer held-out fold, or "<fold>_<n_folds>" for an inner fold (e.g. "1_3" for
    inner fold 1 of a 3-fold split). This is exactly the suffix used in every
    pheno_*/ds___* filename - so it's used both to build dataset subset names below and,
    in main(), the checkpoint filename.
    """
    return str(n_folds) if str(fold) == str(n_folds) else f'{fold}_{n_folds}'


def build_fold_subsets(fold, n_folds):
    """Map a fold id to the {train, validation[, test]} subset names.

    Inner folds ("<fold>" != "<n_folds>") are split into a train/validation pair. The
    outer fold ("<fold>" == "<n_folds>") trains on the full "both" split and is evaluated
    on the held-out "test" split. When an inner fold's validation split is not itself the
    held-out test split, the test split is evaluated too.
    """
    label = fold_label(fold, n_folds)
    test_subset = f'{n_folds}_test'
    if str(fold) == str(n_folds):
        train_subset = f'{label}_both'
        validation_subset = f'{label}_test'
    else:
        train_subset = f'{label}_train'
        validation_subset = f'{label}_validation'

    subsets = {"train": train_subset, "validation": validation_subset}
    if validation_subset != test_subset:
        subsets["test"] = test_subset
    return subsets


def load_gwas_betas(gwas_path, beta_col):
    """Load a SNP-indexed effect-size column from a GWAS summary statistics file."""
    return pd.read_csv(gwas_path, sep='\t', index_col=0)[beta_col]


def load_mafs(dataset, imp, rep, subset, train_geno_set):
    """Load the per-SNP allele frequencies PRANA needs to initialize its input layers.

    GenotypeDataset (torch_dataset_cv.py) only populates train_geno_set.maf when a file
    literally named "ds.frq" sits next to the genotype data - but the real per-fold MAF
    files this project's pipeline produces (e.g. via `plink --freq`) are named after the
    fold itself, "ds___<subset>.frq" (this is the exact path the original main_scz_cv.py /
    main_bcac_cv.py / main_cimba_cv.py / main_ukb_cv.py hardcoded for their "df_mafs" load).
    So look there first, and only fall back to train_geno_set.maf for setups where a bare
    "ds.frq" genuinely exists.
    """
    genotype_file_name, _, _, _ = param_builder_cv(dataset=dataset, imp=imp, rep=rep, subset=subset)
    maf_path = f'{genotype_file_name}.frq'
    if os.path.exists(maf_path):
        return pd.read_csv(maf_path, sep='\s+', index_col=1)
    if hasattr(train_geno_set, 'maf'):
        return train_geno_set.maf
    raise RuntimeError(
        f"No allele-frequency file found. Looked for '{maf_path}' (per-fold, e.g. produced "
        f"by `plink --freq` on the training genotype file) and for a bare 'ds.frq' next to "
        f"the genotype data - neither exists. PRANA needs per-SNP MAFs to initialize its "
        f"input layers."
    )


def create_model(df_mafs, rsids, df_gwas, n_pcs, dropout):
    net = PRANA(data_suffix="", df_gwas=df_gwas, rsids=rsids, df_mafs=df_mafs, p=dropout, n_pcs=n_pcs)
    net = net.to(device=utils.device)
    return net


def compute_pos_weight(labels):
    labels = np.asarray(labels, dtype=np.float32)
    n_pos = max(labels.sum(), 1.0)
    n_neg = max(len(labels) - labels.sum(), 1.0)
    return auto_to_device(float(n_neg / n_pos))


def train_epoch(net, loader, optimizer, pos_weight):
    net.train()
    total_loss, n_batches = 0.0, 0
    for inputs, labels, cov, sample_ids in loader:
        optimizer.zero_grad()
        inputs = inputs.float().to(utils.device)
        cov = cov.float().to(utils.device)
        labels = labels.float().to(utils.device)

        prediction, _ = net(inputs, cov=cov)
        loss = nn.functional.binary_cross_entropy_with_logits(
            prediction.flatten(), labels, pos_weight=pos_weight)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(net, loader):
    net.eval()
    all_preds, all_labels = [], []
    for inputs, labels, cov, sample_ids in loader:
        inputs = inputs.float().to(utils.device)
        cov = cov.float().to(utils.device)
        prediction, _ = net(inputs, cov=cov)
        all_preds.append(prediction.flatten().cpu())
        all_labels.append(labels.float())

    preds = torch.cat(all_preds)
    labels = torch.cat(all_labels)
    bce = nn.functional.binary_cross_entropy_with_logits(preds, labels).item()
    auroc = float('nan')
    if labels.unique().numel() > 1:
        auroc = roc_auc_score(labels.numpy(), preds.numpy())
    return {"bce": bce, "auroc": auroc}


def main(args):
    torch.set_default_dtype(getattr(torch, tensor_dtype))

    subsets = build_fold_subsets(args.fold, args.n_folds)
    loaders, geno_sets = {}, {}
    for role, subset in subsets.items():
        loader, geno_set = load_dataloader(
            args.dataset, args.imp, args.rep, subset,
            pheno_suffix=args.pheno_suffix, batch_size=args.batch_size,
            force_reloading=args.force_reloading)
        loaders[role] = loader
        geno_sets[role] = geno_set

    df_gwas = load_gwas_betas(args.gwas_path, args.gwas_beta_col)
    df_mafs = load_mafs(args.dataset, args.imp, args.rep, subsets["train"], geno_sets["train"])
    net = create_model(df_mafs, rsids_of(geno_sets["train"]), df_gwas, args.n_pcs, args.dropout)

    pos_weight = compute_pos_weight(geno_sets["train"].labels)
    if args.optimizer == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Training PRANA on {args.dataset}/{args.rep}/fold {args.fold} "
          f"({len(geno_sets['train'])} train samples, {len(rsids_of(geno_sets['train']))} SNPs)")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(net, loaders["train"], optimizer, pos_weight)
        msg = f"epoch {epoch}/{args.epochs} - train_bce={train_loss:.4f}"

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            for role, loader in loaders.items():
                if role == "train":
                    continue
                metrics = evaluate(net, loader)
                msg += f" - {role}_bce={metrics['bce']:.4f} - {role}_auroc={metrics['auroc']:.4f}"

            # No extension: this matches the model_<dataset>_<rep>_<fold_label>_<epoch>
            # convention any downstream evaluation tooling should expect when locating
            # saved checkpoints (see fold_label's docstring).
            ckpt_path = os.path.join(
                args.output_dir,
                f"model_{args.dataset}_{args.rep}_{fold_label(args.fold, args.n_folds)}_{epoch}")
            torch.save(net.state_dict(), ckpt_path)

        print(msg)


def rsids_of(geno_set):
    return geno_set.snp_ids


def get_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate PRANA on a single CV fold of a genotype dataset.")

    parser.add_argument('-ds', '--dataset', required=True,
                         help="dataset folder name under BASE_PROFILE, e.g. 'dbg-scz19'")
    parser.add_argument('-i', '--imp', required=True,
                         help="imputation subfolder holding the genotype files, "
                              "e.g. 'impute2_1kg_eur2_r07_1'")
    parser.add_argument('-r', '--rep', required=True,
                         help="CV repetition folder name, e.g. 'rep_103_1'")
    parser.add_argument('-f', '--fold', required=True,
                         help="inner fold number to run, e.g. '1' (combined with "
                              "--n_folds into the '<fold>_<n_folds>' label used in dataset "
                              "filenames - see fold_label()). Set equal to --n_folds to "
                              "train on the outer 'both'/'test' split instead")
    parser.add_argument('-nf', '--n_folds', type=int, default=3)
    parser.add_argument('-p', '--pheno_suffix', default="",
                         help="phenotype name used in the dataset's pheno_<name>__<fold> files")

    parser.add_argument('-g', '--gwas_path', required=True,
                         help="TSV of GWAS summary statistics with SNP id as the index column")
    parser.add_argument('--gwas_beta_col', default="BETA",
                         help="column in --gwas_path holding the per-SNP effect size")

    parser.add_argument('--n_pcs', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('-b', '--batch_size', type=int, default=1000)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('-l', '--lr', type=float, default=5e-2)
    parser.add_argument('--optimizer', choices=["sgd", "adam"], default="sgd")
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--force_reloading', action='store_true',
                         help="ignore any cached .pkl files and reload the raw genotype data")

    parser.add_argument('-m', '--model_name', default="prana")
    parser.add_argument('-o', '--output_dir', default=None,
                         help="defaults to {PRS_OUTPUT}/models/{model_name}")

    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(constants.PRS_OUTPUT, "models", args.model_name)
    return args


if __name__ == '__main__':
    main(get_args())
