"""Unmodified head, causal raw features and reference math from retained training sources."""
import math
import torch
from torch import nn
from torch.nn import functional as F

class ExplicitLayerNorm(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.bias = nn.Parameter(torch.zeros(dimension))

    def forward(self, x):
        c = x - x.mean(-1, keepdim=True)
        return c * torch.rsqrt(c.square().mean(-1, keepdim=True) + 1e-05) * self.weight + self.bias

class CausalBlock(nn.Module):

    def __init__(self, dilation):
        super().__init__()
        self.dilation = dilation
        self.norm = ExplicitLayerNorm(64)
        self.conv = nn.Conv1d(64, 64, 3, dilation=dilation)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        z = self.norm(x).transpose(1, 2)
        z = self.conv(F.pad(z, (2 * self.dilation, 0))).transpose(1, 2)
        return x + self.drop(F.gelu(z))

def raw_features(history, current, ticks):
    hmean = torch.nanmean(history, 1)
    hsd = torch.nanmean((history - hmean[:, None]).square(), 1).clamp_min(1e-06).sqrt()
    p = current.reshape(-1, 64, 16)
    pmean = torch.nanmean(p, 2)
    psd = torch.nanmean((p - pmean[..., None]).square(), 2).clamp_min(1e-06).sqrt()
    endpoint = ticks[:, None] - 1008 + 16 * torch.arange(64, device=ticks.device)[None]
    fraction = (endpoint[..., None] - 15 + torch.arange(16, device=ticks.device)[None, None] >= 0).float().mean(2)
    online_endpoint = endpoint.clamp_min(0).float()
    features = torch.stack([(pmean - hmean[:, None]) / hsd[:, None], torch.log(psd / hsd[:, None]), (p[..., -1] - hmean[:, None]) / hsd[:, None], fraction, torch.log1p(online_endpoint) / math.log(1000), online_endpoint / 1000], -1)
    return torch.nan_to_num(features, nan=0, posinf=8, neginf=-8).clamp(-8, 8)

class PatchHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(ExplicitLayerNorm(3840), nn.Linear(3840, 64), nn.GELU(), nn.Dropout(0.1))
        self.blocks = nn.Sequential(CausalBlock(1), CausalBlock(2), CausalBlock(4))
        self.output = nn.Linear(70, 1)

    def forward(self, reference, current, raw):
        mean, sd = (reference[:, 0, None].expand_as(current), reference[:, 1, None].expand_as(current))
        delta = current - mean
        features = torch.cat([current, mean, sd, delta, (delta / (sd + 0.1)).clamp(-20, 20)], -1)
        x = self.blocks(self.project(features))
        return self.output(torch.cat([x, raw], -1))[..., 0]

def reference(patches, history):
    valid = torch.isfinite(history).reshape(-1, 64, 16).any(-1).to(patches.dtype)
    denominator = valid.sum(1, keepdim=True).clamp_min(1)
    mean = (patches * valid[..., None]).sum(1) / denominator
    variance = ((patches - mean[:, None]).square() * valid[..., None]).sum(1) / denominator
    return torch.stack([mean, variance.clamp_min(0).sqrt()], 1)
