import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Beta
from scipy.special import betainc

class UpsilonLayer(nn.Module):
    def __init__(self, id):
        super(UpsilonLayer, self).__init__()
        self.id = id

    def forward(self, x):
        if self.id == 1:
            return torch.clamp(torch.norm(x[0], dim=1, keepdim=True) * torch.norm(x[1], dim=1, keepdim=True), min=1e-6)
        elif self.id == 2:
            return torch.clamp(torch.abs(torch.sum(x[0] * x[1], dim=-1, keepdim=True)), min=1e-6)
        elif self.id == 3:
            return torch.clamp(torch.norm(x[0] + x[1], dim=1, keepdim=True), min=1e-6)

def get_upsilon_layer(id):
    return UpsilonLayer(id)

class LBD(nn.Module):
    def __init__(self, num_users, num_items, min_rating, max_rating, opt):
        super(LBD,self).__init__()

        self.num_users = num_users
        self.num_items = num_items
        self.num_hidden = opt.num_hidden
        self.upsilon_layer = get_upsilon_layer(opt.upsilon_layer_id)
        self.bin_size = opt.bin_size
        self.min_rating = min_rating
        self.max_rating = max_rating

        self.rating_range = torch.arange(self.min_rating, self.max_rating + self.bin_size, self.bin_size)

        self.num_bins = int((self.max_rating - self.min_rating) / self.bin_size + 1)
        self.uid_bin_size_terms = nn.Embedding(self.num_users, self.num_bins)
        self.iid_bin_size_terms = nn.Embedding(self.num_items, self.num_bins)


        self.uid_features = nn.Embedding(num_users, self.num_hidden + 1)
        self.iid_features = nn.Embedding(num_items, self.num_hidden + 1)

        self.uid_confidence_features = self.uid_features
        self.iid_confidence_features = self.iid_features

        self.uid_alpha_emb = nn.Embedding(self.num_users, 1)
        self.uid_beta_emb  = nn.Embedding(self.num_users, 1)
        self.iid_alpha_emb = nn.Embedding(self.num_items, 1)
        self.iid_beta_emb  = nn.Embedding(self.num_items, 1)

        self.global_bias_add = nn.Parameter(torch.tensor(opt.global_bias))

    def incomplete_beta(self, x, a, b, n):
        t = torch.linspace(0, 1, n).to(x.device)  # 구간을 0에서 1까지 나누기
        t = t.unsqueeze(1).unsqueeze(2)  # t를 (n, 1, 1) 형태로 만들어 브로드캐스팅 가능하게 하기
        x = x.unsqueeze(0)  # x를 (1, 64, 4) 형태로 만들어 브로드캐스팅 가능하게 하기
        a = a.unsqueeze(1)  # a를 (64, 1) 형태로 만들어 브로드캐스팅 가능하게 하기
        b = b.unsqueeze(1)

        dt = t[1] - t[0]  # 각 구간의 길이
        integrand = (t * x) ** (a - 1) * (1 - t * x) ** (b - 1)  # integrand 함수 값 계산
        integrand = integrand * (t <= x).float()  # t가 x 이하인 부분만 계산

        integral = torch.trapz(integrand, t, dim=0)  # 사다리꼴 적분법으로 적분 값 계산
        return integral

    def complete_beta(self, a, b, n):
        t = torch.linspace(0, 1, n).to(a.device)  # 구간을 0에서 1까지 나누기
        dt = t[1] - t[0]  # 각 구간의 길이
        integrand = t ** (a - 1) * (1 - t) ** (b - 1)  # integrand 함수 값 계산
        integral = torch.trapz(integrand, t)  # 사다리꼴 적분법으로 적분 값 계산
        return integral

    def regularized_incomplete_beta(self, x, a, b, n=64):
        Bx_ab = self.incomplete_beta(x, a, b, n)  # incomplete beta 함수 값 계산 64 x 4
        B_ab = self.complete_beta(a, b, n).unsqueeze(1)  # complete beta 함수 값 계산
        return Bx_ab / B_ab  # regularized incomplete beta 값 계산

    def compute_mu_upsilon(self, uid_features, iid_features, uid_confidence_features, iid_confidence_features):
        dot = torch.sum(uid_features[:, :-1] * iid_features[:, :-1], dim=1, keepdim=True)
        uid_norm = torch.norm(uid_features[:, :-1], dim=1, keepdim=True)
        iid_norm = torch.norm(iid_features[:, :-1], dim=1, keepdim=True)
        len_prod = uid_norm * iid_norm

        mu = torch.clamp(0.5 + 0.5 * dot / torch.clamp(len_prod, min=1e-6), min=1e-6, max=1 - 1e-6) # equation 6
        upsilon = self.upsilon_layer([uid_confidence_features[:, :-1], iid_confidence_features[:, :-1]]) # equation 7
        return uid_features, iid_features, uid_confidence_features, iid_confidence_features, mu, upsilon

    def compute_alpha_beta(self, mu, upsilon, uid_input, iid_input):
        alpha = torch.maximum(torch.tensor(1e-2), mu * upsilon)
        beta = torch.maximum(torch.tensor(1e-2), upsilon - alpha)

        alpha = self.global_bias_add + alpha
        beta = self.global_bias_add + beta

        alpha = torch.maximum(torch.tensor(1e-2), alpha + self.uid_alpha_emb(uid_input) + self.iid_alpha_emb(iid_input))
        beta = torch.maximum(torch.tensor(1e-2), beta + self.uid_beta_emb(uid_input) + self.iid_beta_emb(iid_input))
        return alpha, beta

    def compute_edges(self, uid_input, iid_input):
        ui_bin_size_terms = torch.exp(self.uid_bin_size_terms(uid_input) + self.iid_bin_size_terms(iid_input)) # exp(i+j)
        ui_bin_size_terms_norm = ui_bin_size_terms / torch.sum(ui_bin_size_terms, dim=-1, keepdim=True)
        edges = torch.cumsum(ui_bin_size_terms_norm, dim=-1)
        return edges

    def compute_bin_mass(self, alpha, beta, edges):
        cdf = self.regularized_incomplete_beta(edges[:, :-1], alpha, beta)
        cdf = torch.cat([cdf, torch.ones_like(alpha)], dim=-1)
        mass = torch.cat([cdf[:, :1], cdf[:, 1:] - cdf[:, :-1]], dim=-1)
        return mass

    def compute_bin_mean(self, inputs):
        output = torch.sum(inputs * self.rating_range, dim=-1)
        return output

    def forward(self, uid_input, iid_input):
        uid_features = self.uid_features(uid_input)
        iid_features = self.iid_features(iid_input)

        uid_confidence_features = self.uid_confidence_features(uid_input)
        iid_confidence_features = self.iid_confidence_features(iid_input)

        uid_features, iid_features, uid_confidence_features, iid_confidence_features, mu, upsilon = self.compute_mu_upsilon(uid_features, iid_features, uid_confidence_features, iid_confidence_features)
        alpha, beta = self.compute_alpha_beta(mu, upsilon, uid_input, iid_input)
        edges = self.compute_edges(uid_input, iid_input)

        beta_bins_mass = self.compute_bin_mass(alpha, beta, edges)
        beta_bins_mean = self.compute_bin_mean(beta_bins_mass)

        return beta_bins_mass