import torch


def gaussian_nll(mu, sigma, target):
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(target[..., :2]).sum(-1).mean()