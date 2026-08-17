import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import utils
import numpy as np
import torch
from utils import auto_to_device
from torch.autograd import Variable
import pandas as pd

class PRANA(nn.Module):

    def create_fc_layer(self, name, fc_in, fc_out, sd=None, p=0.5):
        if sd is None:
            sd=1/fc_in
        layer=nn.Linear(fc_in, fc_out)
        layer.name=name
        self.__setattr__(name, layer)
        self.__setattr__(f"{name}_batchnorm", nn.BatchNorm1d(fc_out))
        self.__setattr__(f"{name}_drp", nn.Dropout(p=p))
        with torch.no_grad():
            layer.weight.data.normal_(0.0,sd)
            layer.bias.data.uniform_(-sd, sd)


    def create_lstm_layer(self, name, fc_in, fc_out, p=0.5):
        self.__setattr__(name, nn.LSTM(input_size=fc_in, hidden_size=fc_out, num_layers=1, batch_first=True, dropout=p))
        # self.__setattr__(f"{name}_batchnorm", nn.BatchNorm1d(fc_out))
        # self.__setattr__(f"{name}_drp", nn.Dropout(p=p))

    def create_prs_layer(self, name, fc_out, weights, p=0.0001):
        layer=nn.Linear(len(weights), fc_out)
        weights=torch.broadcast_to(weights,(fc_out,len(weights)))
        weight_noise = 0 # auto_to_device(torch.randn(weights.shape)) * torch.std(weights)
        bias_noise = 0 # auto_to_device(torch.randn(fc_out)) * torch.std(weights)
        layer.name = name
        with torch.no_grad():
            layer.weight.copy_(weights+weight_noise)
            layer.bias.copy_(bias_noise)
        self.__setattr__(name, layer)
        self.__setattr__(f"{name}_batchnorm", nn.BatchNorm1d(fc_out))
        self.__setattr__(f"{name}_drp", nn.Dropout(p=p))


    def __init__(self, data_suffix, df_gwas=None, rsids=None, df_mafs=None, p=0.3, n_pcs=6):
        super(PRANA, self).__init__()
        df_gwas=df_gwas.reindex(rsids).fillna(0)
        self.prs_weights = utils.auto_to_device(df_gwas.reindex(rsids).values).float()
        self.mult_prs_weights = Parameter(utils.auto_to_device(df_gwas.reindex(rsids).values).float()[:,None])
        self.snp_weight_no_zeros=self.mult_prs_weights.detach().cpu().numpy().flatten()
        self.snp_weight_no_zeros=self.snp_weight_no_zeros[self.snp_weight_no_zeros!=0]
        self.snp_weight_no_zeros_fraction=len(self.snp_weight_no_zeros)/len(self.mult_prs_weights)
        self.snp_weight_sd=self.snp_weight_no_zeros.std()
        self.prior_final_layer= Parameter((utils.auto_to_device([1,0,1])+utils.auto_to_device(torch.abs(torch.randn(3))/10)).float()[:,None])
        self.fuse_scale = 1.0
        self.initial_fuse_weight= Parameter(utils.auto_to_device(0.0/self.fuse_scale)) #  0.04
        self.mafs=df_mafs['MAF'].reindex(rsids)
        self.mafs=self.mafs.fillna(self.mafs.mean())
        self.mafs = utils.auto_to_device(self.mafs.values).float()
        self.in_convs=[]
        self.cov_convs=[]
        self.n_pcs=n_pcs
        n_snps = len(rsids)
        geno_mean=self.mafs*2
        geno_sd=torch.sqrt(2*self.mafs*(1-self.mafs))


        self.scaler_bias = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.scaler_adapter = nn.Parameter(torch.tensor(1.0), requires_grad=False) # self.snp_weight_no_zeros_fraction
        self.scaler_pcs = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.scaler_prs = nn.Parameter(torch.tensor(-2.0), requires_grad=True)
        self.scaler_fuse = nn.Parameter(torch.tensor(0.95), requires_grad=True)

        for i in range(n_snps):
            cur_conv=nn.Conv1d(3, 1, 1, 1)
            cur_conv_name=f"in_conv_{i+1}"
            cur_conv.name=cur_conv_name
            self.__setattr__(cur_conv_name, cur_conv)
            with torch.no_grad():
                cur_conv.weight.copy_(utils.auto_to_device(torch.reshape(utils.auto_to_device([0,1,2]),(-1,1))-geno_mean[i]))
                cur_conv.bias.copy_(utils.auto_to_device([geno_mean[i]]))

            self.in_convs.append(cur_conv)


        for i in range(n_snps):
            cur_conv=nn.Conv1d(self.n_pcs+1, 1, 1, 1) # +1
            cur_conv_name=f"cov_conv_{i+1}"
            cur_conv.name = cur_conv_name
            self.__setattr__(cur_conv_name, cur_conv)

            with torch.no_grad():
                # cur_conv.weight.copy_(utils.auto_to_device(torch.abs(cur_conv.weight)))
                cur_conv.weight.copy_(utils.auto_to_device(torch.reshape(utils.auto_to_device(np.concatenate((np.random.normal(size=self.n_pcs)/10,[0.1]))),(-1,1))))
                cur_conv.bias.copy_(torch.abs(cur_conv.bias)) # utils.auto_to_device([np.random.normal()/10]))
            self.cov_convs.append(cur_conv)

        self.create_fc_layer(f"fc_cov2fuse", fc_in=self.n_pcs, fc_out=1, p=p)

        self.in_dropout=nn.Dropout(p=p)
        self.in_batchnorm = nn.BatchNorm1d(n_snps) # nn.BatchNorm1d(p=p)

        fc_latent=2

        # self.conv_fuse_c1_cov = nn.Conv1d(2, 1, 1, 1, device=utils.device)
        # self.conv_fuse_c1_cov.name="conv_fuse_c1_cov"
        # with torch.no_grad():
        #     self.conv_fuse_c1_cov.weight.copy_(utils.auto_to_device([[1],[0.1]])+utils.auto_to_device(torch.abs(self.conv_fuse_c1_cov.weight))) # utils.auto_to_device([[0.9],[0.1]])+
        #     self.conv_fuse_c1_cov.bias.copy_(utils.auto_to_device([0])) # utils.auto_to_device([0]))

        self.create_fc_layer(f"fc_1_1", fc_in=n_snps, fc_out=fc_latent, sd=self.snp_weight_sd/len(self.mult_prs_weights)**0.5, p=p) # sd=self.snp_weight_sd/10
        self.create_fc_layer(f"fc_1_2", fc_in=n_snps, fc_out=fc_latent, p=p)
        self.create_fc_layer(f"fc_latent2out", fc_in=fc_latent, fc_out=1, p=p)
        self.create_fc_layer(f"fc_cov2out", fc_in=self.n_pcs, fc_out=1, p=p)
        self.create_fc_layer("fc_dec", fc_in=fc_latent, fc_out=n_snps, p=p)
        self.create_fc_layer("fc_out1", fc_in=fc_latent, fc_out=1, p=p)
        with torch.no_grad():
            self.fc_out1.weight.copy_(utils.auto_to_device(torch.abs(self.fc_out1.weight))) # ,0.5 utils.auto_to_device([0.1,0,0.1])+utils.auto_to_device(torch.abs(self.fc_out1.weight)/10)
            self.fc_out1.bias.copy_(utils.auto_to_device(torch.abs(self.fc_out1.bias))) # utils.auto_to_device([0]))
        self.create_fc_layer("fc_out2", fc_in=32, fc_out=8, p=p)
        with torch.no_grad():
            self.fc_out2.weight.copy_(utils.auto_to_device(torch.abs(self.fc_out2.weight))) # ,0.5
            self.fc_out2.bias.copy_(utils.auto_to_device(torch.abs(self.fc_out2.bias))) # utils.auto_to_device([0]))
        self.create_fc_layer("fc_out", fc_in=3, fc_out=1, p=p)
        with torch.no_grad():
            self.fc_out.weight.copy_(utils.auto_to_device([1,1,1])) # (utils.auto_to_device([0.1,0,0.1])+utils.auto_to_device(torch.abs(self.fc_out.weight)/10)) # utils.auto_to_device([1,0,1])+utils.auto_to_device(torch.abs(self.fc_out.weight)/10)
            self.fc_out.bias.copy_(utils.auto_to_device(torch.abs(self.fc_out.bias))) # utils.auto_to_device([0]))

        # input_dim=3
        # self.attention = nn.Sequential(
        #     nn.Linear(input_dim, 16),  # Hidden layer to learn feature interactions
        #     nn.Tanh(),                # Non-linear activation
        #     nn.Linear(16, 1), # Produce attention scores for each input
        #     nn.Sigmoid() # Softmax(dim=1)         # Normalize scores
        # )


    def forward_fc_layer(self, name, x, activation=F.sigmoid, do_activation=True, do_batchnorm=True, do_dropout=True):
        h=self.__getattr__(name)(x)
        if do_batchnorm:
            h=self.__getattr__(f"{name}_batchnorm")(h)
        if do_activation:
            h = activation(h)
        if do_dropout:
            h = self.__getattr__(f"{name}_drp")(h)
        return h

    def forward_prs_layer(self, name, x, activation=torch.nn.Sigmoid(), do_activation=False, do_batchnorm=False, do_dropout=False):
        return self.forward_fc_layer(name, x, activation=activation, do_activation=do_activation, do_batchnorm=do_batchnorm, do_dropout=do_dropout)

    def forward_lstm_layer(self, name, x, activation=F.sigmoid):
        h, (h_state, c_state)=self.__getattr__(name)(x)
        # h=self.__getattr__(f"{name}_batchnorm")(h_out)

        return h

    def forward(self, x, is_main=True, cov=None):

        self.mult_prs_weights=self.mult_prs_weights
        # x=self.in_dropout(x)
        h_0s=[]
        for i, cur_conv in enumerate(self.in_convs):
            h_0s.append(cur_conv(x.transpose(2,0)[i]))
        h_0s=torch.stack(h_0s).transpose(2,0)
        c1_rsp = h_0s.reshape(h_0s.shape[0], -1)
        # c1_rsp=self.in_dropout(c1_rsp)

        # h_covs=[]
        # for i, cur_conv in enumerate(self.cov_convs):
        #     # h_covs.append(cur_conv(cov[:,:self.n_pcs].reshape(x.shape[0],cov[:,:self.n_pcs].shape[1],1)))
        #     cov_in=torch.hstack((cov[:,:self.n_pcs], c1_rsp[:,[i]])).reshape(x.shape[0],cov[:,:self.n_pcs].shape[1]+1,1)

        #     h_covs.append(cur_conv(cov_in))
        # h_covs = torch.hstack(h_covs)
        # covs_rsp = h_covs.reshape(h_covs.shape[0], -1)

        # h_cov2fuse_raw = self.forward_fc_layer('fc_cov2fuse', cov[:,:self.n_pcs], do_dropout=False, do_batchnorm=False, do_activation=False)
        # h_cov2fuse = nn.Sigmoid()(h_cov2fuse_raw)

        # stacked_c1_cov=torch.stack((covs_rsp, c1_rsp), 1)


        # self.fuse_weight= nn.Sigmoid()(self.initial_fuse_weight*self.fuse_scale)   #    h_cov2fuse*self.fuse_scale# *self.initial_fuse_weight
        # self.fuse_weight=self.scaler_fuse
        fused_c1_cov=c1_rsp # self.fuse_weight * covs_rsp + (1 - self.fuse_weight) * c1_rsp # self.conv_fuse_c1_cov(stacked_c1_cov).reshape(x.shape[0],-1) # covs_rsp+0.05*c1_rsp

        h_prs_raw=torch.matmul(fused_c1_cov, self.mult_prs_weights) # covs_rsp+c1_rsp
        h_prs = h_prs_raw # (nn.Sigmoid()(self.scaler_prs)/nn.Sigmoid()(torch.tensor(-2.0)))*h_prs_raw+self.scaler_bias # nn.Sigmoid()(h_prs_raw)

        # c1_rsp=self.in_batchnorm(c1_rsp)

        h_mu = self.forward_fc_layer("fc_1_1", fused_c1_cov, do_dropout=True, do_batchnorm=False, do_activation=False) # covs_rsp+c1_rsp
        # h_logvar = self.forward_fc_layer("fc_1_2", fused_c1_cov, do_dropout=False, do_batchnorm=False, do_activation=False) # covs_rsp+c1_rsp
        # std = torch.exp(0.5 * h_logvar)
        # eps = torch.randn_like(std)
        # h_latent=eps.mul(std).add_(h_mu)
        # h_dec = self.forward_fc_layer("fc_dec", h_latent, do_dropout=False, do_batchnorm=False, do_activation=False) # , activation=F.leaky_relhu)

        h_latent2out = self.forward_fc_layer('fc_latent2out', h_mu, do_dropout=False, do_batchnorm=False, do_activation=False)

        # h_cov2out_raw = self.forward_fc_layer('fc_cov2out', cov[:,:self.n_pcs], do_dropout=False, do_batchnorm=False, do_activation=False)
        # h_cov2out = h_cov2out_raw # nn.Sigmoid()(h_cov2out_raw)
        # cat_layer=torch.cat((h_latent2out, h_cov2out, h_prs),1) # h_cov2out , h_prs

        h_out1_raw = self.forward_fc_layer('fc_out1', h_mu, do_dropout=False, do_batchnorm=False, do_activation=False)
        h_out1=h_out1_raw # nn.Sigmoid()(h_out1_raw)
        # h_out2 = self.forward_fc_layer('fc_out2', h_out1, do_dropout=False, do_batchnorm=False, do_activation=False)

        # cat_layer=torch.cat((h_out1, h_cov2out, h_prs),1) # h_cov2out , h_prs
        # h_out = (1-nn.Sigmoid()(self.scaler_adapter))*(self.scaler_prs*h_prs+self.scaler_bias)+nn.Sigmoid()(self.scaler_adapter)*(h_out1) # (h_prs + h_out1*nn.ReLU()(self.scaler_adapter) + h_cov2out*nn.ReLU()(self.scaler_pcs)) # +h_cov2out # self.forward_fc_layer('fc_out', cat_layer, do_dropout=False, do_batchnorm=False, do_activation=False) # (h_out1+h_cov2out+h_prs)/3
        # w=nn.Tanh()(self.scaler_adapter)
        h_out = h_prs+h_out1*self.scaler_adapter # w*(h_out1) # (h_prs + h_out1*nn.ReLU()(self.scaler_adapter) + h_cov2out*nn.ReLU()(self.scaler_pcs)) # +h_cov2out # self.forward_fc_layer('fc_out', cat_layer, do_dropout=False, do_batchnorm=False, do_activation=False) # (h_out1+h_cov2out+h_prs)/3
        # h_out = (1-nn.Sigmoid()(self.scaler_adapter))*(h_prs)+nn.Sigmoid()(self.scaler_adapter)*(h_out1) # (h_prs + h_out1*nn.ReLU()(self.scaler_adapter) + h_cov2out*nn.ReLU()(self.scaler_pcs)) # +h_cov2out # self.forward_fc_layer('fc_out', cat_layer, do_dropout=False, do_batchnorm=False, do_activation=False) # (h_out1+h_cov2out+h_prs)/3

        # # # cat_layer_mult=torch.einsum('nw,wc->nw', cat_layer, self.prior_final_layer)
        # # Compute attention scores
        # attention_scores = self.attention(cat_layer)
        # # Apply scores to the features
        # weighted_features = cat_layer * attention_scores
        # # Aggregate (e.g., sum or mean the weighted features)
        # h_out=weighted_features.sum(dim=1, keepdim=True)+h_prs
        # # h_out= self.attention(cat_layer)


        cat_raw=torch.cat((h_out1_raw, h_out1_raw, h_prs_raw),1) # cat_raw=torch.cat((h_out1_raw, h_cov2out_raw, h_prs_raw),1)
        cat_layer=torch.cat((h_out1_raw, h_out1_raw, h_prs),1)
        return h_out, (None, None, h_mu, None, cat_layer, cat_raw) # return h_out, (h_dec, covs_rsp, h_mu, h_logvar, cat_layer, cat_raw)
