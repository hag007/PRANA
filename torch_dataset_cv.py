import os
import io

import numpy as np
import pandas as pd
from numpy import dtype

import constants
from torch.utils.data import Dataset

import utils
from utils import auto_to_device,  tensor_dtype
import torch

from constants import config_json as config

class GenotypeDataset(Dataset):

    def __init__(self, dataset, imp, rep, subset, pheno="pheno", ds_version="", range=None, filter=None, excluded_samples=None, force_reloading=True, shuffle=False, n_pcs=6):

        # Extract data file names
        if rep and subset:
            #  Fold data
            genotype_file_name , phenotype_file_name, pop_file_name , cov_local_file_name = param_builder_cv(dataset=dataset, imp=imp, rep=rep, subset=subset, pheno=pheno)
        else:
            # Full dataset
            genotype_file_name , phenotype_file_name, pop_file_name , cov_local_file_name = param_builder(dataset=dataset, imp=imp, pheno=pheno, ds_version=ds_version)

        # Override cov file name, in needed
        if dataset.split('_')[0]=='bcac' and 'bcac_pca' in config:
            cov_file_name=config['bcac_pca']
        elif dataset.split('_')[0] == 'ukbb' and 'ukbb_pca' in config:
            cov_file_name=config['ukbb_pca']
        else:
            cov_file_name=cov_local_file_name

        # Extract MAF
        maf_file=f'{os.path.dirname(genotype_file_name)}/ds.frq'
        if os.path.exists(maf_file):
            self.maf = pd.read_csv(f'{os.path.dirname(genotype_file_name)}/ds.frq', sep='\s+', index_col=1)

        # Format file names
        pkl_suffix=f"{range[0]}_{range[1]}"
        folder='/'.join(genotype_file_name.split('/')[0:-1])
        genotype_pkl_name=f"{folder}/{os.path.basename(genotype_file_name)}_{pkl_suffix}.pkl"
        pheno_pkl_name=f"{folder}/pheno{os.path.basename(genotype_file_name)[2:]}_{pkl_suffix}.pkl"
        sample_ids_pkl_name = f"{folder}/sample_ids{os.path.basename(genotype_file_name)[2:]}_{pkl_suffix}.pkl"
        cov_pkl_name = f"{folder}/cov{os.path.basename(genotype_file_name)[2:]}_{pkl_suffix}.pkl"

        # Load from cache
        if not force_reloading and os.path.exists(genotype_pkl_name) and os.path.exists(pheno_pkl_name) and os.path.exists(sample_ids_pkl_name) and os.path.exists(cov_pkl_name):
            print(f'dataset was loaded from pkls: {genotype_pkl_name}, {pheno_pkl_name}, {sample_ids_pkl_name}, {os.path.exists(cov_pkl_name)}')
            self.samples=torch.load(genotype_pkl_name, map_location="cpu").astype(np.float_) # utils.device
            self.in_shape = self.samples.shape[1]
            self.samples_raw=self.samples

            self.labels = torch.load(pheno_pkl_name, map_location="cpu")  # utils.device
            self.labels-=self.labels.min()
            self.sample_ids = torch.load(sample_ids_pkl_name, map_location="cpu")
            df_cov = pd.read_csv(cov_file_name, delim_whitespace=True, index_col=0, header=None)
            df_cov.index = df_cov.index.astype(str)
            self.cov=df_cov.loc[self.sample_ids].values[:,1:n_pcs+1].astype(getattr(np, tensor_dtype))
            self.snp_ids = open(f'{folder}/{os.path.basename(genotype_file_name)}_index.tsv', 'r').read().split()  # [1:]

            ## Transform genotypes to one-hot encoding
            value_padding=np.repeat([[-1,0,1,2]], self.samples.shape[0], axis=0)
            self.samples=np.hstack((value_padding, self.samples))
            self.samples = np.apply_along_axis(\
                lambda a: pd.get_dummies(a).values.T,\
                1, self.samples).astype(np.float_)[:,-3:, 4:]

            return


        # Load from raw data
        print(f'read dataset with no cache: {genotype_file_name} {phenotype_file_name}')
        self.labels = np.array([])
        self.samples = pd.DataFrame()
        hdr = open(f'{folder}/{os.path.basename(genotype_file_name)}_header.tsv', 'r').read().split()# [1:]
        idx = open(f'{folder}/{os.path.basename(genotype_file_name)}_index.tsv', 'r').read().split()# [1:]
        dta = np.loadtxt(f'{folder}/{os.path.basename(genotype_file_name)}_data.tsv', delimiter="\t", dtype=getattr(np, tensor_dtype))[:480312].T

        # Load pheno
        df_pheno=pd.read_csv(phenotype_file_name, delim_whitespace=True, index_col=0)
        df_pheno.index=df_pheno.index.astype(str)
        if excluded_samples is not None:
            df_pheno=df_pheno.drop(excluded_samples)

        # Load cov
        df_cov = pd.read_csv(cov_file_name, delim_whitespace=True, index_col=0)
        df_cov.index=df_cov.index.astype(str)

        # Drop samples with missing pheno/cov/genotypes
        df_pheno = df_pheno.reindex(df_cov.index).dropna()
        df_cov = df_cov.reindex(df_pheno.index).dropna()
        df_pheno = df_pheno.reindex(hdr).dropna()
        df_cov = df_cov.reindex(hdr).dropna()
        dta=dta[pd.Series(hdr, dtype=str).isin(df_pheno.index)]
        hdr=np.array(hdr)[pd.Series(hdr, dtype=str).isin(df_pheno.index)]

        # Shuffle and take a fraction of of samples (fixed to all samples)
        sample_size=dta.shape[0]
        if shuffle:
            drawn_samples=np.random.choice(np.arange(sample_size), int(range[1] * sample_size)-int(range[0] * sample_size), replace=False)
        else:
            drawn_samples=np.arange(int(range[0]*sample_size),int(range[1]*sample_size))
        self.samples=dta[drawn_samples]
        self.sample_ids=hdr[drawn_samples]
        self.cov = df_cov.values[drawn_samples,1:n_pcs+1].astype(getattr(np, tensor_dtype))
        self.labels = df_pheno.values[:,-1][drawn_samples].astype(getattr(np, tensor_dtype))

        # Normalize labels
        self.labels-=self.labels.min()

        # Set auxiliary attributes
        self.snp_ids=idx
        self.in_shape = self.samples.shape[1]

        # Save cache
        print(f"tensor shapes: genotypes - {self.samples.shape}, labels -{self.labels.shape}, sample_ids - {self.sample_ids.shape}, cov - {self.cov.shape}")
        torch.save(self.samples, genotype_pkl_name, pickle_protocol=4)
        torch.save(self.labels, pheno_pkl_name, pickle_protocol=4)
        torch.save(self.sample_ids, sample_ids_pkl_name, pickle_protocol=4)
        torch.save(self.cov, cov_pkl_name, pickle_protocol=4)

        ## Transform genotypes to one-hot encoding
        value_padding = np.repeat([[-1, 0, 1, 2]], self.samples.shape[0], axis=0)
        self.samples = np.hstack((value_padding, self.samples))
        self.samples = np.apply_along_axis( \
            lambda a: pd.get_dummies(a).values.T, \
            1, self.samples).astype(np.float_)[:, -3:, 4:]

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        if type(idx) == torch.Tensor:
            idx = idx.item()
        result = self.samples[idx, :], self.labels[idx], self.cov[idx] , self.sample_ids[idx]
        return result

def load_dataloader(dataset, imp, rep=None, subset=None, pheno_suffix="", interval=(0,1), batch_size=1000, ds_version="", filter=None, excluded_samples=None, force_reloading=False, shuffle=False):
    geno_set = GenotypeDataset(dataset, imp, rep, subset, pheno=pheno_suffix, range=interval, ds_version=ds_version, filter=filter, excluded_samples=excluded_samples, force_reloading=force_reloading, shuffle=shuffle)
    loader = torch.utils.data.DataLoader(geno_set, batch_size=batch_size, shuffle=True) # Originally shuffle=True  # , pin_memory=False, num_workers=num_workers)
    loader.name=dataset
    return loader, geno_set

def itr_merge(itrs):
    itrs=[enumerate(a) for a in itrs]
    cur_elems=[next(itr, None) for itr in itrs]
    while any(cur_elems):
        for i_dataset, cur_elem in enumerate(cur_elems):
            if not cur_elem is None:
                yield i_dataset, cur_elem[1]
        cur_elems = [next(itr, None) for itr in itrs]

def param_builder_cv(dataset, imp="original", rep="105_1", subset="1_5", pheno="pheno", pop="pop.panel"):
    return os.path.join(constants.BASE_DIR, dataset, rep, imp, f'ds___{subset}'), os.path.join(constants.BASE_DIR, dataset, rep, f'pheno_{pheno}__{subset}'), \
           os.path.join(constants.BASE_DIR, dataset, rep, f'pheno_{pheno}__{subset}'), os.path.join(constants.BASE_DIR, dataset, rep, imp, f'ds___{subset}.eigenvec')

def param_builder(dataset, imp="original", ds_version="ds", pheno="pheno", pop="pop.panel", cov="ds.eigenvec"):
    return os.path.join(constants.BASE_DIR, dataset, imp, ds_version), os.path.join(constants.BASE_DIR, dataset, pheno), \
           os.path.join(constants.BASE_DIR, dataset, pop), os.path.join(constants.BASE_DIR, dataset, imp, cov)
