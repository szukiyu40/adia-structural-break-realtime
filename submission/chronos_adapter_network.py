"""Same network as Colab training; constructor accepts a local checkpoint."""
import torch
from torch import nn
from chronos import Chronos2Pipeline
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"

class Detector(nn.Module):
    def __init__(self, device, model_path, dtype=torch.float32):
        super().__init__()
        pipeline = Chronos2Pipeline.from_pretrained(model_path, revision=REVISION, local_files_only=True,
            device_map=device, dtype=dtype, trust_remote_code=False)
        self.encoder_model = pipeline.model
        for p in self.encoder_model.parameters():
            p.requires_grad_(False)
        for block in self.encoder_model.encoder.block[-2:]:
            for p in block.parameters(): p.requires_grad_(True)
        for p in self.encoder_model.encoder.final_layer_norm.parameters():
            p.requires_grad_(True)
        d = int(self.encoder_model.config.d_model)
        assert d == 768 and self.encoder_model.chronos_config.use_reg_token
        self.head = nn.Sequential(nn.LayerNorm(4*d+6), nn.Linear(4*d+6,64),
            nn.GELU(), nn.Dropout(.1), nn.Linear(64,1)).to(device)
        self.encoder_model.eval()

    def forward(self, history, current, ticks, *, history_embedding=None, return_history_embedding=False, return_current_embedding=False):
        n = len(current)
        if history_embedding is None:
            context = torch.cat([history, current], 0)
            # Every view is a distinct group: no cross-series or cross-view attention.
            group_ids = torch.arange(2*n, device=context.device)
            enc, _, _, patches = self.encoder_model.encode(context, group_ids=group_ids)
            reg = enc.last_hidden_state[:, patches]
            h, c = reg[:n], reg[n:]
        else:
            # The immutable normal history was encoded at this series' first update.
            # Preserve the current view's original group identifier; its group is independent.
            assert history_embedding.shape == (n, self.encoder_model.config.d_model)
            group_ids = torch.arange(n, 2*n, device=current.device)
            enc, _, _, patches = self.encoder_model.encode(current, group_ids=group_ids)
            h, c = history_embedding, enc.last_hidden_state[:, patches]
        hc = torch.nan_to_num(history)
        cc = torch.nan_to_num(current)
        hm = hc.sum(1)/torch.isfinite(history).sum(1).clamp_min(1)
        cm = cc.sum(1)/torch.isfinite(current).sum(1).clamp_min(1)
        hs = torch.sqrt(torch.nanmean((history-hm[:,None]).square(),1).clamp_min(1e-6))
        cs = torch.sqrt(torch.nanmean((current-cm[:,None]).square(),1).clamp_min(1e-6))
        scalars = torch.stack([cm-hm, torch.log(cs/hs), torch.abs(cm-hm),
            torch.log1p(ticks.float())/7, (cc[:,-1]-hm)/hs,
            torch.isfinite(current).float().mean(1)],1).clamp(-8,8)
        features = torch.cat([h,c,c-h,torch.abs(c-h),scalars],1)
        logits = self.head(features).squeeze(1)
        if return_current_embedding:
            return logits, c
        if return_history_embedding:
            cached = h.detach().clone() if history_embedding is None else history_embedding
            return logits, cached
        return logits
