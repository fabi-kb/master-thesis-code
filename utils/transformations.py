import torch
import torch.nn.functional as F


def symexp(x):
    return torch.sign(x) * torch.expm1(torch.abs(x))

def symlog(x):
    return torch.sign(x) * torch.log1p(torch.abs(x))


def create_symexp_bins(num_bins, limit, device):
    if num_bins % 2 == 1:
        half = torch.linspace(-limit, 0, (num_bins - 1) // 2 + 1, device=device)
        half = symexp(half)
        bins = torch.cat([half, -half[:-1].flip(0)], dim=0)

    else: 
        half = torch.linspace(-limit, 0, num_bins // 2, device=device)
        half = symexp(half)
        bins = torch.cat([half, -half.flip(0)], dim=0)

    return bins

def twohot(y, bins):
    # Ensure y matches bins dtype
    y = y.to(bins.dtype)

    # Compute nearest bin indices
    y_exp = y.unsqueeze(-1)                # [B, T, 1]
    # Compute lower & upper bin indices
    upper = torch.searchsorted(bins, y_exp.squeeze(-1), right=False)    # [B, T]
    upper = upper.clamp(max=bins.numel() - 1)                           # [B, T]
    lower = (upper - 1).clamp(min=0)                                      # [B, T]
    
    bin_lower = bins[lower]                          # [B, T]
    bin_upper = bins[upper]                          # [B, T]
    
    denom = (bin_upper - bin_lower).clamp(min=1e-8)       # [B, T]
    w_higher = ((y - bin_lower) / denom).clamp(0, 1)     # [B, T]
    w_lower = 1 - w_higher                             # [B, T]
    
    K = bins.numel()
    
    # One-hot (B,T,K)
    lower_oh = F.one_hot(lower, K).to(y.dtype)  # [B, T, K]
    upper_oh = F.one_hot(upper, K).to(y.dtype)  # [B, T, K]
    
    # Weights (B,T,K)
    w_lower = w_lower.unsqueeze(-1)                   # [B, T, 1]
    w_higher = w_higher.unsqueeze(-1)                   # [B, T, 1]
    targets = lower_oh * w_lower + upper_oh * w_higher  # [B, T, K]
    
    return targets


def expected_value(logits, bins):

    # Compute probabilities from logits
    probs = torch.softmax(logits, dim=-1)
    num_bins = probs.shape[-1] 

    # Compute expected reward, naive implementation would result in non zero result
    if num_bins % 2 == 1:
        m = (num_bins - 1) // 2
        negative_bins = bins[:m]
        positive_bins = bins[m+1:]
        zero_bin = bins[m:m+1]
        negative_probs = probs[..., :m]
        positive_probs = probs[..., m+1:]
        zero_probs = probs[..., m:m+1]
        expected_value = ((negative_probs * negative_bins).flip(-1) + (positive_probs * positive_bins)).sum(dim=-1)
        expected_value = expected_value + (zero_probs * zero_bin).sum(-1)

    else:
        m = num_bins // 2
        negative_bins = bins[:m]
        positive_bins = bins[m:]
        negative_probs = probs[..., :m]
        positive_probs = probs[..., m:]
        expected_value = ((negative_probs * negative_bins).flip(-1) + (positive_probs * positive_bins)).sum(dim=-1)
    
    return expected_value

def log_prob(logits, targets):

    log_pred = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    log_prob = (targets * log_pred).sum(dim=-1)
    return log_prob
