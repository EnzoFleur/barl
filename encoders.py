import torch
import torch.nn as nn
from transformers import DistilBertModel, GPT2Model, BertModel
import torch.nn.functional as F
import numpy as np
import os

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

DISTILBERT_PATH = os.path.join("..","distilBERT", "distilbert-base-uncased") if os.path.isdir(os.path.join("..", "distilBERT")) else 'distilbert-base-uncased'
BERT_PATH = os.path.join("..","BERT", "bert-base-uncased") if os.path.isdir(os.path.join("..", "BERT")) else 'bert-base-uncased'
GPT2_PATH = os.path.join("..", "GPT2") if os.path.isdir(os.path.join("..", "GPT2")) else 'gpt2'

TEMPOBERT_PATH = os.path.join("..", "TempoBERT")

from typing import Union, List

import torch
import torch.nn as nn

class EarlyStopper:
    def __init__(self, patience=3, min_delta=1e-2):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss-self.min_validation_loss > self.min_validation_loss*self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def build_linear(in_dim, out_dim, activation="linear"):
    lin_layer = nn.Linear(in_dim, out_dim)
    if activation == 'silu':
        gain = 1.
    else:
        gain = torch.nn.init.calculate_gain(activation)
    torch.nn.init.xavier_normal_(lin_layer.weight, gain=gain)
    return lin_layer

def build_mlp(in_dim: int,
              h_dim: Union[int, List],
              n_layers: int = None,
              out_dim: int = None,
              dropout_p: float = 0.2,
              activation: str = 'relu') -> nn.Sequential:
    """Builds an MLP.
    Parameters
    ----------
    in_dim: int,
        Input dimension of the MLP
    h_dim: int,
        Hidden layer dimension of the MLP
    out_dim: int, default None
        Output size of the MLP. If None, a Linear layer is returned, with ReLU
    dropout_p: float, default 0.2,
        Dropout probability
    """

    if isinstance(h_dim, list) and n_layers is not None:
        print("n_layers should be None if h_dim is a list. Skipping")

    if isinstance(h_dim, int):
        h_dim = [h_dim]
        if n_layers is not None:
            h_dim = h_dim * n_layers

    sizes = [in_dim] + h_dim
    mlp_size_tuple = list(zip(*(sizes[:-1], sizes[1:])))

    if isinstance(dropout_p, float):
        dropout_p = [dropout_p] * len(mlp_size_tuple)

    layers = []

    for idx, (prev_size, next_size) in enumerate(mlp_size_tuple):
        # layers.append(nn.BatchNorm1d(prev_size))
        layers.append(build_linear(prev_size, next_size, activation))
        if activation == 'leaky_relu':
            layers.append(nn.LeakyReLU())
        elif activation == 'relu':
            layers.append(nn.ReLU())
        elif activation == 'selu':
            layers.append(nn.SELU())
        elif activation == 'silu':
            layers.append(nn.SiLU())
        else:
            raise ValueError(f"Unknown activation function: {activation}")
        layers.append(nn.Dropout(dropout_p[idx]))

    if out_dim is not None:
        layers.append(build_linear(sizes[-1], out_dim))

    return nn.Sequential(*layers)

class VarBrownianEncoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, loss,
                tokenizer,
                finetune, method, L=10, beta=0.0, n_axis=None, dynamic="global", test_ids=None):
        super(VarBrownianEncoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.finetune = finetune
        self.tokenizer = tokenizer
        self.loss = loss
        self.n_axis = n_axis
        self.L = L
        self.beta = beta
        self.dynamic = dynamic
        self.test_ids = test_ids

        self.method = method 

        print(self.method)

        if self.tokenizer == "DistilBERT":
            self.encoder = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        elif self.tokenizer == "BERT":
            self.encoder = BertModel.from_pretrained(BERT_PATH)
        elif "TempoBERT" in self.tokenizer:
            self.encoder = BertModel.from_pretrained(self.tokenizer)
        elif self.tokenizer == "GPT2":
            self.encoder = GPT2Model.from_pretrained(GPT2_PATH)
        
        for param in self.encoder.parameters():
            param.requires_grad = self.finetune

        self.mlp = build_mlp(in_dim=768, h_dim=self.hidden_dim, n_layers=3, out_dim=self.latent_dim, activation='leaky_relu')
        self.var_time_mlp = build_mlp(in_dim=self.latent_dim+1, h_dim=self.latent_dim*2, out_dim=self.latent_dim, n_layers=2)

        self.C_eta = nn.Linear(1, 1)

        self.a_eta = nn.Parameter(torch.rand(1))
        self.b_eta = nn.Parameter(torch.rand(1))
        # self.var_time = nn.Parameter(torch.rand(1))

        self.alpha_emb = nn.Embedding(self.n_axis, 1)

        self.params = nn.ModuleDict({
            'encoder': nn.ModuleList([self.encoder]),
            'classifier': nn.ModuleList([self.mlp, self.C_eta])
        })

        if self.n_axis is not None:
            self.z_estart = nn.Embedding(self.n_axis, latent_dim)
            self.z_eend = nn.Embedding(self.n_axis, latent_dim)

            self.params['classifier'].extend([self.z_estart, self.z_eend])

    def reparameterize(self, mean, var, logvar=False):
        
        eps = torch.normal(mean=0.0, std=1.0, size=mean.shape).to(device)

        if logvar:
            return eps * torch.sqrt(torch.exp(var)) + mean
        else:
            return eps * torch.sqrt(var) + mean
    
    def logistic_classifier(self, x, apply_sigmoid=True):

        logits = -torch.exp(self.a_eta) * x + self.b_eta

        if apply_sigmoid:
            logits = torch.sigmoid(logits)

        return logits
    
    def forward(self, input_ids, attention_mask, axis, label, t_, t, T, Tmax, criterion):

        if self.dynamic == "local":
            t = t-t_
            Tmax = T-t_

        z_t, z_0, z_T = self.encode_doc(input_ids, attention_mask, axis)

        z_t = self.mlp(z_t)

        z_hat = z_0 * (1- t/Tmax).unsqueeze(-1) + z_T * (t/Tmax).unsqueeze(-1)

        loss = 0.0
        prior_loss = torch.tensor(0.0).to(device)

        if self.loss == "L2":

            loss+= torch.sum(label * torch.sum(criterion(z_t, z_hat), dim=1) - (1-label) * torch.sum(criterion(z_t, z_hat), dim=1))

        elif self.loss == "BCE_var":

            z_hat_var = torch.exp(self.var_time_mlp(z_hat)) + self.alpha_emb(axis) * (t*(Tmax - t)/Tmax/365).unsqueeze(-1)

            # z_hat_var = torch.cat([(t*(Tmax - t)/Tmax/365).unsqueeze(-1), z_hat], dim=1)
            # z_hat_var = self.var_time * self.var_time_mlp(z_hat_var)

            for _ in range(self.L):
                z_emb = z_t
                z_hat_emb = self.reparameterize(z_hat, z_hat_var)

                distance = torch.norm(z_emb - z_hat_emb, dim=1)

                probs = self.logistic_classifier(distance, apply_sigmoid=False)

                loss += criterion(probs, label.float())

                loss *= 1/self.L

            prior_loss += 0.5 * torch.sum(torch.square(z_t) -1)
            prior_loss += 0.5 * torch.sum(torch.square(z_hat) + torch.exp(z_hat_var) - z_hat_var -1)

        elif self.loss == "BCE_time":
            z_hat_var = (t*(Tmax - t)/Tmax/365).unsqueeze(-1).repeat(1,self.latent_dim)

            for _ in range(self.L):
                z_hat_emb = self.reparameterize(z_hat, z_hat_var)

                distance = torch.norm(z_t - z_hat_emb, dim=1)

                probs = self.logistic_classifier(distance, apply_sigmoid=False)

                loss += criterion(probs, label.float())

                loss *= 1/self.L

            prior_loss += 0.5 * torch.sum(torch.square(z_hat) + torch.exp(z_hat_var) - z_hat_var -1)

        elif self.loss == "BCE":
                distance = torch.norm(z_t - z_hat, dim=1)

                probs = self.logistic_classifier(distance, apply_sigmoid=False)

                loss += criterion(probs, label.float())

        prior_loss *= self.beta

        return loss + prior_loss, loss.item(), prior_loss.item()

    def compute_masked_means(self, outputs, masks):
        # we don't want to include padding tokens
        # outputs : B x T x D
        # masks   : B x T
        dim = outputs.size(2)
        masks_dim = masks.unsqueeze(2).repeat(1, 1, dim)
        # masked_outputs : B x T x D
        masked_outputs = outputs * masks_dim  # makes the masked entries 0
        # masked_outputs: B x D / B x 1 => B x D
        partition = torch.sum(masks, dim=1, keepdim=True)
        masked_outputs = torch.sum(masked_outputs, dim=1) / partition
        return masked_outputs
    
    def encode(self, input_ids, attention_mask, label=None):

        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output[0]

        hidden_state = self.compute_masked_means(hidden_state, attention_mask)

        latent_state = self.mlp(hidden_state)
        
        if label is None:
            return latent_state
        else:
            z_start = self.z_estart(label)
            z_end = self.z_eend(label)
        
            return latent_state, z_start, z_end

    def encode_doc(self, input_ids, attention_mask, label=None):
        
        encoder_output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = encoder_output[0]

        hidden_state = self.compute_masked_means(hidden_state, attention_mask)

        z_start = self.z_estart(label)
        z_end = self.z_eend(label)

        return hidden_state, z_start, z_end

class FDistilBert(torch.nn.Module):
    def __init__(self, na):
        super().__init__()

        self.na = na
        self.distilBERT = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        self.drop = torch.nn.Dropout(0.2)
        self.out = torch.nn.Linear(768, na)

    def forward(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = distilbert_output[0]
        embed = self.drop(hidden_state[:,0])
        output = self.out(embed)
        return output    

    def forward_test(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = distilbert_output[0]
        embed = hidden_state[:,0]
        prediction = self.out(embed)
        return embed, prediction
    
class BFDistilBert(torch.nn.Module):
    def __init__(self, n1, n2):
        super().__init__()
        
        self.na = n1
        self.nt = n2
        self.distilBERT = DistilBertModel.from_pretrained(DISTILBERT_PATH)
        self.drop = torch.nn.Dropout(0.2)
        self.out1 = torch.nn.Linear(768, n1)
        self.out2 = torch.nn.Linear(768, n2)

    def forward(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = distilbert_output[0]
        embed = self.drop(hidden_state[:,0])
        output1 = self.out1(embed)
        output2 = self.out2(embed)
        return output1, output2    

    def forward_test(self, ids, mask):
        distilbert_output = self.distilBERT(ids, mask)
        hidden_state = distilbert_output[0]
        embed = hidden_state[:,0]

        out1 = self.out1(embed)
        out2 = self.out2(embed)
        return embed, out1, out2
