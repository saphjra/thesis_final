import torch


def gaussian_nll(mu, sigma, target):
    # Ensure sigma is positive and not too small to avoid numerical instability
    sigma = torch.clamp(sigma, min=1e-4)
    dist = torch.distributions.Normal(mu, sigma)
    return -dist.log_prob(target[..., :2]).sum(-1).mean()
