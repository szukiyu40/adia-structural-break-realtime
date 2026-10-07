"""Full-data patch-localization Chronos with per-series GPU state and CUDA replay."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':16:8')
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from chronos_adapter_network import Detector, REVISION
from chronos_patch_network import PatchHead, raw_features, reference
from chronos_single_group import _simplify_single_group_attention


class PatchDetector(nn.Module):
    def __init__(self, root, dtype=torch.float16, optimized=True):
        super().__init__()
        original = Detector('cuda', str(root/'base'), dtype=dtype)
        self.encoder_model = original.encoder_model
        self.head = PatchHead().cuda()
        checkpoint = torch.load(root/'patch_detector.pt', map_location='cpu', weights_only=True)
        assert checkpoint['fold'] == -1 and checkpoint['condition'] == 'tail_and_head'
        self.head.load_state_dict(checkpoint['head'], strict=True)
        tail = checkpoint['tail']; expected = set()
        for i in range(2):
            prefix = f'blocks.{i}.'
            state = {k[len(prefix):]: v for k, v in tail.items() if k.startswith(prefix)}
            self.encoder_model.encoder.block[10+i].load_state_dict(state, strict=True)
            expected.update(prefix+k for k in state)
        norm = {k[5:]: v for k, v in tail.items() if k.startswith('norm.')}
        self.encoder_model.encoder.final_layer_norm.load_state_dict(norm, strict=True)
        expected.update('norm.'+k for k in norm)
        assert set(tail) == expected
        self.requires_grad_(False); self.eval()
        if optimized:
            _simplify_single_group_attention(self)

    def patches(self, context):
        assert context.shape == (1, 1024)
        group = torch.zeros(1, device=context.device, dtype=torch.int64)
        encoded, _, _, count = self.encoder_model.encode(context, group_ids=group)
        assert count == 64
        return encoded.last_hidden_state[:, :64].float()

    def forward(self, history, current, ticks, normal_reference):
        patches = self.patches(current)
        features = raw_features(history, current, ticks)
        logits = self.head(normal_reference, patches, features)[:, -1]
        return torch.sigmoid(logits), patches


class Resources:
    def __init__(self, folder):
        root = Path(folder)
        self.meta = json.loads((root/'metadata.json').read_text())
        assert self.meta['revision'] == REVISION
        assert self.meta['weight'] == .125 and self.meta['stride'] == 4
        assert self.meta['training_series'] == 10000
        for name, digest in self.meta['sha256'].items():
            with (root/name).open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == digest, name
        assert torch.cuda.is_available()
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.device = 'cuda'
        self.model = PatchDetector(root)
        self.graph_runner = GraphRunner(self.model)


class GraphRunner:
    def __init__(self, model):
        self.history = torch.zeros((1, 1024), device='cuda', dtype=torch.float32)
        self.current = torch.zeros_like(self.history)
        self.ticks = torch.zeros(1, device='cuda', dtype=torch.int64)
        self.normal = torch.zeros((1, 2, 768), device='cuda', dtype=torch.float32)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            for _ in range(3):
                model(self.history, self.current, self.ticks, self.normal)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=stream):
            self.probability, self.current_patches = model(
                self.history, self.current, self.ticks, self.normal)

    def encode_history(self, history):
        self.history.copy_(torch.from_numpy(history[None]))
        self.current.copy_(torch.from_numpy(history[None]))
        self.ticks.zero_(); self.normal.zero_()
        self.graph.replay()
        return reference(self.current_patches, self.history).detach().clone()

    def predict(self, history, current, tick, normal_reference):
        self.history.copy_(torch.from_numpy(history[None]))
        self.current.copy_(torch.from_numpy(current[None]))
        self.ticks.fill_(tick); self.normal.copy_(normal_reference)
        self.graph.replay()
        return self.probability.item()


class State:
    def __init__(self, history, resources):
        self.resources = resources
        history = np.asarray(history, dtype='f4')[-1024:]
        assert history.ndim == 1 and len(history) and np.isfinite(history).all()
        self.history = np.full(1024, np.nan, dtype='f4')
        self.history[-len(history):] = history
        self.current = self.history.copy()
        self.tick = -1; self.score = .5
        self.normal_reference = None

    def update(self, value):
        assert np.isfinite(value)
        self.tick += 1
        self.current[:-1] = self.current[1:]
        self.current[-1] = value
        if self.tick % 4 == 0:
            with torch.inference_mode():
                if self.normal_reference is None:
                    self.normal_reference = self.resources.graph_runner.encode_history(self.history)
                self.score = self.resources.graph_runner.predict(
                    self.history, self.current, self.tick, self.normal_reference)
            assert np.isfinite(self.score) and 0 <= self.score <= 1
        return self.score
