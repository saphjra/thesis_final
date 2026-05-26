import torch


def gaussian_nll(mu, sigma, target):
    # Ensure sigma is positive and not too small to avoid numerical instability
    print("target: \n", target)
    sigma = torch.clamp(sigma, min=1e-8)
    print(sigma)
    print(mu)
    dist = torch.distributions.Normal(mu, sigma)
    print("dist", dist)
    nll = -dist.log_prob(target).sum(-1).mean()
    print(nll)
    total_loss = nll
    return total_loss


def gaussian_nll_with_sigma_regularization(mu, sigma, target, sigma_reg_weight=0.1):
    """
    Gaussian NLL with regularization to prevent sigma from collapsing to near-zero.

    This prevents the model from predicting unrealistically small uncertainties.
    #TOdo however if I normalize the gaze data this is not smart as then the data is already in the range [0,1] ...
    """
    # Clamp sigma to reasonable range [0.01, 1.0]
    sigma = torch.clamp(sigma, min=1e-8, max=0.1)

    # Compute standard NLL
    dist = torch.distributions.Normal(mu, sigma)
    nll = -dist.log_prob(target).sum(-1).mean()

    # Regularization: penalize small sigma values
    # -log(sigma) is large when sigma is small, so this term grows as sigma shrinks
    sigma_reg = -torch.log(sigma).mean()

    total_loss = nll + sigma_reg_weight * sigma_reg

    return total_loss


def gmm_nll(pi_logits, mu, log_sx, log_sy, rho_raw, target):
    """
    Accepts raw (pre-activation) outputs from your model head.
    Activations are applied here, inside the loss, so gradients
    flow through them cleanly.

    Args:
        pi_logits : (B, T, K)       raw mixture weights, pre-softmax
        mu        : (B, T, K, 2)    predicted means [x, y]
        log_sx    : (B, T, K)       log sigma x  (unconstrained)
        log_sy    : (B, T, K)       log sigma y  (unconstrained)
        rho_raw   : (B, T, K)       pre-tanh correlation
        target    : (B, T, 2)       ground truth gaze [x, y]

    Returns:
        scalar loss
    """
    B, T, K = pi_logits.shape

    # ── activations ──────────────────────────────────────────────
    pi = torch.softmax(pi_logits, dim=-1)  # (B,T,K) sums to 1
    sx = torch.exp(log_sx).clamp(min=1e-4)  # (B,T,K) strictly positive
    sy = torch.exp(log_sy).clamp(min=1e-4)  # (B,T,K)
    rho = torch.tanh(rho_raw) * 0.99  # (B,T,K) in (-0.99, 0.99)
    # *0.99 keeps det > 0

    # ── assert shapes (remove after debugging) ───────────────────
    assert mu.shape == (B, T, K, 2), f"mu shape {mu.shape}"
    assert target.shape == (B, T, 2), f"target shape {target.shape}"

    # ── bivariate gaussian log-prob ───────────────────────────────
    tx = target[..., 0].unsqueeze(-1)  # (B,T,1) → broadcasts over K
    ty = target[..., 1].unsqueeze(-1)  # (B,T,1)

    dx = (tx - mu[..., 0]) / sx  # (B,T,K)
    dy = (ty - mu[..., 1]) / sy  # (B,T,K)

    det = 1.0 - rho**2  # (B,T,K) always > 0 now
    z = dx**2 + dy**2 - 2.0 * rho * dx * dy  # (B,T,K)

    # log N(target | mu_k, Sigma_k)
    log_gauss = (
        -0.5 * z / det
        - torch.log(sx)
        - torch.log(sy)
        - 0.5 * torch.log(det)
        - torch.log(torch.tensor(2.0 * torch.pi))  # scalar constant
    )  # (B,T,K)

    # ── mixture log-prob via logsumexp ────────────────────────────
    log_pi = torch.log(pi + 1e-8)  # (B,T,K)
    log_mix = torch.logsumexp(
        log_pi + log_gauss,  # (B,T,K) → (B,T)
        dim=-1,
    )

    return -log_mix.mean()  # scalar


# ── minimal sanity check ──────────────────────────────────────────
if __name__ == "__main__":
    B, T, K = 4, 128, 5
    pi_logits = torch.randn(B, T, K)
    mu = torch.randn(B, T, K, 2)
    log_sx = torch.zeros(B, T, K)  # exp(0) = 1.0
    log_sy = torch.zeros(B, T, K)
    rho_raw = torch.zeros(B, T, K)  # tanh(0) = 0, independent
    target = torch.rand(B, T, 2)

    loss = gmm_nll(pi_logits, mu, log_sx, log_sy, rho_raw, target)
    print(f"loss: {loss.item():.4f}")  # should be ~2.8 (≈ log(2π))
    assert not torch.isnan(loss), "nan in loss!"
    assert not torch.isinf(loss), "inf in loss!"
    print("all checks passed")
