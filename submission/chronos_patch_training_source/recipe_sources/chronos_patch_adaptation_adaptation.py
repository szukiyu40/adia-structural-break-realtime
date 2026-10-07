"""Share the unchanged first ten blocks; independently optimize the tail."""
import copy
import torch
from torch import nn
from common import Encoder, PatchHead, ORIGINAL


def arguments(encoder, hidden, mask):
    group = torch.arange(len(hidden), device=hidden.device)
    return dict(position_ids=torch.arange(hidden.shape[1], device=hidden.device).unsqueeze(0),
                attention_mask=encoder._expand_and_invert_time_attention_mask(mask, hidden.dtype),
                group_time_mask=encoder._construct_and_invert_group_time_mask(group, mask, hidden.dtype),
                output_attentions=False)


class SharedPrefix(nn.Module):
    def __init__(self, fold, device):
        super().__init__()
        self.retained = Encoder(fold, device)
        self.eval()

    @property
    def model(self):
        return self.retained.original.encoder_model

    @torch.no_grad()
    def forward(self, context):
        m = self.model
        patches, mask, loc_scale = m._prepare_patched_context(context, None)
        assert mask.shape[1] == 64
        z = m.input_patch_embedding(patches)
        reg_ids = torch.full((len(context), 1), m.config.reg_token_id, device=z.device)
        z = torch.cat([z, m.shared(reg_ids)], 1)
        mask = torch.cat([mask.to(m.dtype), torch.ones_like(reg_ids).to(m.dtype)], 1)
        future, _ = m._prepare_patched_future(None, None, loc_scale, 1, len(context))
        z = torch.cat([z, m.input_patch_embedding(future)], 1)
        mask = torch.cat([mask, torch.ones(len(context), 1, dtype=m.dtype, device=z.device)], 1)
        args = arguments(m.encoder, z, mask)
        z = m.encoder.dropout(z)
        for block in m.encoder.block[:10]:
            z = block(z, **args)[0]
        return z.detach(), mask.detach()


class Tail(nn.Module):
    def __init__(self, prefix, trainable):
        super().__init__()
        self.blocks = copy.deepcopy(prefix.model.encoder.block[10:])
        self.norm = copy.deepcopy(prefix.model.encoder.final_layer_norm)
        self.requires_grad_(trainable)
        self.eval()  # Gradients remain enabled, encoder dropout stays disabled.

    def forward(self, hidden, mask, encoder):
        args = arguments(encoder, hidden, mask)
        for block in self.blocks:
            hidden = block(hidden, **args)[0]
        return self.norm(hidden)[:, :64]


def reference(patches, history):
    valid = torch.isfinite(history).reshape(-1, 64, 16).any(-1).to(patches.dtype)
    denominator = valid.sum(1, keepdim=True).clamp_min(1)
    mean = (patches*valid[..., None]).sum(1)/denominator
    variance = ((patches-mean[:, None]).square()*valid[..., None]).sum(1)/denominator
    return torch.stack([mean, variance.clamp_min(0).sqrt()], 1)


def retained_head(fold, device):
    head = PatchHead().to(device)
    saved = torch.load(ORIGINAL/f'fold{fold}/localization/head.pt', map_location='cpu', weights_only=True)
    head.load_state_dict(saved['head'], strict=True)
    return head

