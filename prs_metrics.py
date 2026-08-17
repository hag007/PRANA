import os
import sys
import io
import subprocess
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import sklearn
import numpy as np
import scipy as sp

from constants import *
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy as sp

import argparse


def Nagelkerke_Rsquare(ytrain, Xtrain):
    model = generate_glm(ytrain, Xtrain)
    N=Xtrain.shape[0]
    llf=model.llf
    llnull=model.llnull
    naglkerke_rsquare=(1-np.exp((llnull-llf)*(2/N)))/(1-np.exp(llnull*(2/N)))
    return naglkerke_rsquare

def or_per_1sd(ytrain, Xtrain):
    model = generate_glm(ytrain, Xtrain)
    return np.exp(model.params['score']), get_or_se(model)


def generate_glm(ytrain, Xtrain):
    try:
        # print("generate_glm", Xtrain, Xtrain.loc[ytrain == 0, 'score'].mean(), Xtrain.loc[ytrain == 0, 'score'].std(), sep='\n')
        Xtrain = sm.add_constant(Xtrain)
        Xtrain = standardize_scores(ytrain, Xtrain)
        log_reg = sm.GLM(ytrain, Xtrain, family=sm.families.Binomial())
        model = log_reg.fit(disp=0)
    except Exception as e:
        raise type(e)(f"{e}\nAdditional Info: ytrain shape={ytrain.shape}, Xtrain shape={Xtrain.shape}") from e
    return model


def standardize_scores(ytrain, Xtrain):
    if Xtrain['score'].std() == 0:
        Xtrain['score'] = 0
    else:
        n_null_scores = Xtrain['score'].isnull().sum()
        if n_null_scores > 0:
            print(f"Warning: there are {Xtrain['score'].isnull().sum()} before standardization")
        Xtrain['score'] = (Xtrain['score'] - Xtrain.loc[ytrain == 0, 'score'].mean()) / \
                          Xtrain.loc[ytrain == 0, 'score'].std()
    n_null_scores = Xtrain['score'].isnull().sum()
    if n_null_scores > 0:
        print(f"There are {Xtrain['score'].isnull().sum()}. before standardization. dropping these lines")
        Xtrain = Xtrain.dropna()
    return Xtrain


def get_or_se(model):
    # Extract the model coefficients and variance-covariance matrix
    params = model.params
    cov_or = model.cov_params().loc['score','score']

    # Compute odds ratio and its standard error
    odds_ratio = np.exp(params['score'])
    var_diag = np.sqrt(cov_or)
    return odds_ratio * var_diag


def compute_or_percentile(ytrain, Xtrain, resolution=0.9):
    scores=Xtrain['score']
    scores_case=scores[ytrain==1]
    scores_control=scores[ytrain==0]
    p_top=scores.quantile(resolution)
    p10=scores.quantile(0.1)
    p40=scores.quantile(0.4)
    p60=scores.quantile(0.6)
    a=(scores_case>p_top).sum()
    d=(scores_control>p_top).sum()
    b=((scores_case>p40) & (scores_case<=p60)).sum()
    c=((scores_control>p40) & (scores_control<=p60)).sum()
    oddsratio=(a/d)/(b/c)
    print((a,d),(b,c))
    se_log_or=(1/a+1/b+1/c+1/d)**0.5
    ci_min = np.exp(np.log(oddsratio) - 1.96*(se_log_or))
    ci_max = np.exp(np.log(oddsratio) + 1.96*(se_log_or))
    return oddsratio, (ci_min, ci_max)

def auroc(ytrain, Xtrain):
        model=generate_glm(ytrain, Xtrain)
        Xtrain=pd.concat((pd.DataFrame(columns=['constant'], data=[1]*len(Xtrain), index=Xtrain.index), Xtrain), axis=1)
        return sklearn.metrics.roc_auc_score(ytrain, Xtrain['score']) # model.predict(Xtrain)

def auprc(ytrain, Xtrain):
        model=generate_glm(ytrain, Xtrain)
        Xtrain=pd.concat((pd.DataFrame(columns=['constant'], data=[1]*len(Xtrain), index=Xtrain.index), Xtrain), axis=1)
        return sklearn.metrics.average_precision_score(ytrain, Xtrain['score']) # model.predict(Xtrain)


def compute_metrics(df_scores, df_pcs, df_pheno):
    y = (df_pheno['label'] - 1).dropna()
    df_cov = pd.concat((df_scores, df_pcs), axis=1).dropna()
    or_top=compute_or_percentile(y, df_cov)
    df_results= \
        {'or_top': or_top[0],
         'or_top_min': or_top[1][0],
         'or_top_max': or_top[1][1],
         'nagel_r': Nagelkerke_Rsquare(y, df_cov),
         'or_per_1sd': or_per_1sd(y, df_cov),
         'auroc': auroc(y, df_cov) }
    df_results=pd.DataFrame([df_results])
    return df_results

def load_profile_file(discovery, target, imp, profile_file_name):
    df_pheno= pd.read_csv(os.path.join(PRS_DATASETS, target, f"pheno"), sep=r'\s+', index_col=0, dtype={0: str})
    df_pcs = pd.read_csv(os.path.join(PRS_DATASETS, target, imp, "ds.eigenvec"), sep=r'\s+', header=None, index_col=0, dtype={0: str}).iloc[:,1:]
    df_scores = pd.read_csv(os.path.join(PRS_PRSS, f'{discovery}_{target}', imp, profile_file_name), sep=r'\s+', index_col=0, dtype={0: str})[["SCORE"]]
    df_scores.rename(columns={"SCORE" : "score"}, inplace=True)
    df_scores=df_scores.reindex(df_pheno.index).dropna()
    df_pcs=df_pcs.reindex(df_pheno.index).dropna()
    df_pheno = df_pheno.reindex(df_scores.index).dropna()
    assert len(df_scores) == len(df_pheno) and len(df_pcs) == len(df_pheno)

    return compute_metrics(df_scores, df_pcs, df_pheno)


if __name__=="__main__":

    parser = argparse.ArgumentParser(
        prog='ProgramName',
        description='What the program does',
        epilog='Text at the bottom of help')
    parser.add_argument('-d', '--discovery', default="bca_313")
    parser.add_argument('-t', '--target', default="bcac_onco_afr")
    parser.add_argument('-i', '--imp', default="impX_313")
    parser.add_argument('-p', '--profile', default="prs.mono.pt.profile")


    args = parser.parse_args()
    for k in args.__dict__.keys():
        exec(f"{k}=args.{k}")
    print(load_profile_file(discovery, target, imp, profile))