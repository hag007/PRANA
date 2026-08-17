import utils
import torch
from torch import nn
import numpy as np
import pandas as pd
from prs_metrics import standardize_scores as standardize_scores_r2
from prs_metrics import Nagelkerke_Rsquare, generate_glm
import statsmodels.api as sm
from statsmodels.tools.sm_exceptions import PerfectSeparationError
activation=None # torch.nn.Sigmoid()

def soft_quantile(scores, q, beta=10):
    """Approximate quantile using a softmax-weighted sum with q consideration."""
    sorted_scores, _ = torch.sort(scores)
    n = len(sorted_scores)

    # Compute soft ranks centered around the desired quantile
    ranks = torch.linspace(0, 1, steps=n, device=scores.device)
    weights = torch.nn.functional.softmax(-beta * (ranks - q).abs(), dim=0)  # Weigh elements near q

    return torch.sum(weights * sorted_scores)

def compute_or_percentile_loss(ytrain, scores, resolution=0.9, beta=10):
    scores_case = scores[ytrain == 1]
    scores_control = scores[ytrain == 0]

    # Compute soft quantiles
    p_top = soft_quantile(scores, resolution)
    p40 = soft_quantile(scores, 0.4)
    p60 = soft_quantile(scores, 0.6)

    # Use sigmoid-based soft thresholding instead of hard Boolean masks
    a = torch.sum(torch.sigmoid(beta * (scores_case - p_top)))
    d = torch.sum(torch.sigmoid(beta * (scores_control - p_top)))
    b = torch.sum(torch.sigmoid(beta * (scores_case - p40)) - torch.sigmoid(beta * (scores_case - p60)))
    c = torch.sum(torch.sigmoid(beta * (scores_control - p40)) - torch.sigmoid(beta * (scores_control - p60)))

    eps = 1e-8
    oddsratio = (a / (d + eps)) / ((b / (c + eps)) + eps)

    print((a, d), (b, c))

    # Standard error approximation for log OR
    se_log_or = torch.sqrt((1 / (a + eps)) + (1 / (b + eps)) + (1 / (c + eps)) + (1 / (d + eps)))

    # Compute confidence intervals
    ci_min = torch.exp(torch.log(oddsratio) - 1.96 * se_log_or)
    ci_max = torch.exp(torch.log(oddsratio) + 1.96 * se_log_or)

    return oddsratio, (ci_min, ci_max)

import torch

class LogisticRegressionModel(torch.nn.Module):
    """Logistic Regression model with multiple features."""
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1, bias=True)  # Multiple features
        with torch.no_grad():
            self.linear.bias.copy_(utils.auto_to_device([1]))


    def forward(self, x):
        return self.linear(x)  # Sigmoid for probability output


def standardize_scores(ytrain, Xtrain):
    a = 0
    mean_control = torch.mean(Xtrain[ytrain == 0, a])
    std_control = torch.std(Xtrain[ytrain == 0, a], unbiased=False) + 1e-8
    new_Xtrain= Xtrain * 1
    new_Xtrain[:, a]= (Xtrain[:, a] - mean_control) / std_control
    return new_Xtrain

def train_logistic_regression(ytrain, Xtrain, num_steps=100, lr=0.01):
    """
    Trains a logistic regression model while maintaining differentiability with `Xtrain`.
    Uses manual gradient updates with `torch.autograd.grad()` to avoid breaking computation graphs.

    Args:
        ytrain: (Tensor) Binary labels (0 or 1).
        Xtrain: (Tensor) Feature values (all variables included).
        num_steps: (int) Number of gradient steps for logistic regression.
        lr: (float) Learning rate.

    Returns:
        log_or: (Tensor) The trained log(OR per 1 SD), which remains differentiable with `Xtrain`.
    """

    Xtrain = Xtrain.clone().detach()

    Xtrain = standardize_scores(ytrain, Xtrain)

    mean_control = torch.mean(Xtrain[ytrain == 0, 0])
    std_control = torch.std(Xtrain[ytrain == 0, 0], unbiased=False) + 1e-8
    Xtrain[:, 0] = (Xtrain[:, 0] - mean_control) / std_control

    input_dim = Xtrain.shape[1]
    log_or_model = LogisticRegressionModel(input_dim)
    log_or_model.to(device=utils.device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(log_or_model.parameters(), lr=lr)

    # Train logistic regression model
    for epoch in range(num_steps):
        optimizer.zero_grad()

        # Forward pass
        y_pred = log_or_model(Xtrain).squeeze()  # Get probabilities
        loss = criterion(y_pred, ytrain.float())  # Compute BCE loss

        # Backward pass
        loss.backward()
        optimizer.step()

    # for _ in range(num_steps):
    #     logits = log_or_model(Xtrain).squeeze()
    #     loss = criterion(logits, ytrain.float())
    #
    #     grads = torch.autograd.grad(loss, log_or_model.parameters(), create_graph=True)
    #
    #     with torch.no_grad():
    #         for param, grad in zip(log_or_model.parameters(), grads):
    #             param -= lr * grad

    return log_or_model


def or_per_1sd_loss(ytrain, Xtrain, beta=1.0):
    """
    Computes a differentiable loss function based on OR per 1 SD.

    Args:
        ytrain: (Tensor) Binary labels (0 or 1).
        Xtrain: (Tensor) Output of the main neural network (used as input to log OR model).
        beta: (float) Regularization strength.

    Returns:
        loss: (Tensor) Loss based on OR per 1 SD.
    """
    Xtrain = Xtrain.clone().requires_grad_(True)

    mean_control = torch.mean(Xtrain[ytrain == 0, 0])
    std_control = torch.std(Xtrain[ytrain == 0, 0], unbiased=False) + 1e-8
    Xtrain[:, 0] = (Xtrain[:, 0] - mean_control) / std_control

    log_or_model= train_logistic_regression(ytrain, Xtrain, num_steps=1000)
    log_or = log_or_model.linear.weight[0]

    # ✅ Extract the relevant value outside
    loss_prs = -log_or[0]  # ✅ Keeps differentiability

    return loss_prs

def or_per_1sd_loss2(ytrain, Xtrain, beta=1.0):
    Xtrain = Xtrain.clone().requires_grad_(True)

    log_model_no_scores = train_logistic_regression(ytrain, Xtrain[:, 1:])
    log_model_with_scores = train_logistic_regression(ytrain, Xtrain)

    preds = nn.Sigmoid()(log_model_with_scores(Xtrain)).flatten()
    scores_loss = (ytrain - preds) ** 2

    N = len(ytrain)
    n_pos = (ytrain == 1).sum().clamp(min=1)
    n_neg = (ytrain == 0).sum().clamp(min=1)

    w_pos = N / (2.0 * n_pos.float())
    w_neg = N / (2.0 * n_neg.float())

    scores_weight = torch.where(ytrain == 1, w_pos, w_neg)

    return (scores_loss).mean(), None, None

def ngl_r2_loss(ytrain, Xtrain, margin=1, negative_slope=0): # margin=0.2, negative_slope=0.1
    normalization_factor=margin+(1-margin)*negative_slope
    N = len(ytrain)
    model_f= train_logistic_regression(ytrain, Xtrain)
    model_null=train_logistic_regression(ytrain, utils.auto_to_device([[1.0]]*N))
    Xtrain_std = Xtrain # .clone().detach()
    Xtrain_std = standardize_scores(ytrain, Xtrain_std)
    # ypred_f=nn.Sigmoid()(model_f(Xtrain_std).flatten())
    pos_weight = torch.tensor((ytrain == 0).sum() / ytrain.sum().clamp(min=1), device=ytrain.device)
    # pos_weight = n_neg / n_pos
    llf = -nn.BCEWithLogitsLoss()(model_f(Xtrain_std).flatten(), ytrain)*len(Xtrain)
    # ypred_null=nn.Sigmoid()(model_null(Xtrain_std).flatten())
    const=utils.auto_to_device([model_null.linear.weight[0][0]+model_null.linear.bias]*N)
    llnull = -nn.BCEWithLogitsLoss()(const, ytrain)*len(Xtrain)

    X_df=pd.DataFrame(Xtrain.cpu().detach().numpy())
    X_df.columns=['score']+list(X_df.columns[1:])

    try:
        log_model = generate_glm(ytrain.cpu().detach().numpy(), X_df)
            # Nagelkerke_Rsquare(ytrain.detach().numpy(),X_df)
        # X_df_s = sm.add_constant(X_df)
        # X_df_s = standardize_scores_r2(ytrain.detach().numpy(), X_df_s)
        # ypred=log_model.predict(X_df_s)
        # return (torch.nn.LeakyReLU(negative_slope=negative_slope)((ngl_loss.mean())))/normalization_factor

        ngl = (1 - torch.exp((torch.tensor(log_model.llnull, device=utils.device) - llf) * (2 / N))) / (1 - torch.exp(torch.tensor(log_model.llnull, device=utils.device) * (2 / N)))
        # ngl = (1 - torch.exp((llnull - llf) * (2 / N))) / (1 - torch.exp(llnull * (2 / N)))
        
        corr_sign = np.sign(np.corrcoef(np.stack([X_df['score'].values, ytrain.cpu().detach().numpy()]))[0, 1])
        ngl_loss=(margin-ngl*corr_sign)**2


        return (torch.nn.LeakyReLU(negative_slope=negative_slope)(ngl_loss))/normalization_factor
    except PerfectSeparationError as e:
        return torch.tensor(0.0, device=utils.device, requires_grad=True)




def tml_split(labels, prediction):
    positive_prediction = prediction.flatten()[labels == 1]
    negative_prediction = prediction.flatten()[labels == 0]
    min_len = min(len(positive_prediction), len(negative_prediction))
    i_anchor = np.random.choice(np.arange(len(positive_prediction)), min_len // 2, replace=False)
    anchor = positive_prediction[i_anchor]  # nn.Sigmoid()(positive_prediction[i_anchor])
    i_positive = np.delete(np.arange(len(positive_prediction)), i_anchor)[:min_len // 2]
    positive = positive_prediction[i_positive]  # nn.Sigmoid()(positive_prediction[i_positive])
    i_negative = np.random.choice(np.arange(len(negative_prediction)), min_len // 2, replace=False)
    negative = negative_prediction[i_negative]  # nn.Sigmoid()(negative_prediction[i_negative])
    return anchor, positive, negative

def triplet_margin_loss(anchor, positive, negative, margin=1, negative_slope=0.1, activation=activation):
    if activation:
        anchor=activation(anchor)
        positive = activation(positive)
        negative = activation(negative)
    positivity_factor = (1 - margin) * negative_slope
    normalization_factor = 1 / margin
    return (torch.nn.LeakyReLU(negative_slope=negative_slope)(margin + (torch.abs(positive - anchor) - torch.abs((negative - anchor))).mean())+positivity_factor)*normalization_factor

def mrl_split(labels, prediction):
    positive_prediction = prediction.flatten()[labels == 1]
    negative_prediction = prediction.flatten()[labels == 0]
    min_len = min(len(positive_prediction), len(negative_prediction))
    i_positive = np.random.choice(np.arange(len(positive_prediction)), min_len, replace=False)
    positive = positive_prediction[i_positive]  # nn.Sigmoid()(positive_prediction[i_anchor])
    i_negative = np.random.choice(np.arange(len(negative_prediction)), min_len, replace=False)
    negative = negative_prediction[i_negative]  # nn.Sigmoid()(negative_prediction[i_negative])
    return positive, negative

def margin_ranking_loss(positive, negative, margin=1, negative_slope=0.1, activation=activation):
    if activation:
        positive=activation(positive)
        negative = activation(negative)
    positivity_factor=(1 - margin) * negative_slope
    normalization_factor=1/margin
    return (torch.nn.LeakyReLU(negative_slope=negative_slope)(((margin + negative-positive).mean()))+positivity_factor)*normalization_factor


def kl_divergence_to_standard_normal(x):
    mu = torch.mean(x)
    sigma = torch.std(x) + 1e-8  # avoid division by zero

    # KL Divergence formula for normal distribution to standard normal
    kl_div = torch.log(1 / sigma) + 0.5 * (sigma**2 + mu**2) - 0.5
    return kl_div

def prs_distribution_loss(labels, prediction, margin=0.5, negative_slope=0.1, activation=activation):
    normalization_factor = margin + (1 - margin) * negative_slope
    pos_preds = prediction[labels == 1]
    neg_preds = prediction[labels == 0]

    # Normality loss using KL divergence
    kl_pos = kl_divergence_to_standard_normal(pos_preds)
    kl_neg = kl_divergence_to_standard_normal(neg_preds)
    normality_loss = 0.1*(kl_pos + kl_neg) + \
                     torch.nn.LeakyReLU(negative_slope=negative_slope)(neg_preds.mean() - pos_preds.mean() + margin)
    return normality_loss


def mse(prediction, labels):
    if activation:
        prediction=activation(prediction)
    return nn.MSELoss(reduction='mean')(prediction, labels)


def vae_loss(output, input, mu, logvar, recon_weight=0.5, kl_weight=0.00002, factor=5):
    # BCE loss
    # recon_loss = nn.NLLLoss(reduction='sum')(output1.flatten()+ 1e-10, inputs.flatten().type(torch.long))
    recon_loss = nn.MSELoss(reduction='mean')(nn.Sigmoid()(output.flatten()), nn.Sigmoid()(
        input.flatten()))  # nn.BCEWithLogitsLoss(reduction='mean')(output1.flatten()+1e-10,  inputs.flatten())
    # kl divergence loss
    kl_loss = 0.5 * torch.mean(torch.exp(logvar) + mu ** 2 - 1.0 - logvar)
    # total loss
    return recon_weight * recon_loss * factor, kl_weight * kl_loss * factor


def weight_distance_l1_loss(current, initial, weight=1.0, factor=1.0):
    return nn.L1Loss(reduction='mean')(current, initial) / initial.abs().mean() * weight * factor


def weight_distance_mse_loss(current, initial, weight=20.0, factor=1.0):
    return nn.MSELoss(reduction='mean')(current.flatten(), initial.flatten()) / initial.abs().mean() * weight * factor



def pairwise_auc_loss(scores, labels):
    pos_idx = labels == 1
    neg_idx = labels == 0

    pos_scores = scores[pos_idx]
    neg_scores = scores[neg_idx]

    # All pairwise differences: s_pos - s_neg
    diffs = pos_scores[:, None] - neg_scores[None, :]  # shape [n_pos, n_neg]

    # Logistic loss over differences
    loss = torch.log1p(torch.exp(-diffs)).mean()
    return loss
