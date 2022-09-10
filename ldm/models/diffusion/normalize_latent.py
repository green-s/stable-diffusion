import torch


def normalize_latent(x, max_val=1.7, quantile_val=0.975):
    x = x.detach().clone()
    for i in range(x.shape[0]):
        if x[[i], :].std() > 1.0:
            x[[i], :] = x[[i], :] / x[[i], :].std()
        s = torch.quantile(torch.abs(x[[i], :]), quantile_val)
        s = torch.maximum(s, torch.ones_like(s) * max_val)
        x[[i], :] = x[[i], :] / (s / max_val)
    return x
