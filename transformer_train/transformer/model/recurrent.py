import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_block import Transformer, norm


class RecurrenceSampler:
    def __init__(self, mean=5.0, sigma=0.5, r_min=1, r_max=16, seed=0):
        self.mean = float(mean)
        self.sigma = float(sigma)
        self.r_min = int(r_min)
        self.r_max = int(r_max)
        self.rng = random.Random(seed)
        self.counts = {}

    def sample(self):
        mu = math.log(max(self.mean, 1e-6)) - self.sigma ** 2 / 2.0
        lam = math.exp(self.rng.gauss(mu, self.sigma))
        k, p, target = 0, 1.0, math.exp(-lam)
        while p > target:
            p *= self.rng.random()
            k += 1
        r = max(self.r_min, min(self.r_max, k - 1))
        self.counts[r] = self.counts.get(r, 0) + 1
        return r

    def histogram(self):
        return dict(sorted(self.counts.items()))


class RecurrentTransformer(Transformer):

    def __init__(self, config, tokenizer=None, pad_vocab_size_to=64):
        super().__init__(config, tokenizer, pad_vocab_size_to)
        rc = config['recurrent']
        self.n_prelude = rc['n_prelude']
        self.n_recurrent = rc['n_recurrent']
        self.n_coda = rc['n_coda']
        total = self.n_prelude + self.n_recurrent + self.n_coda
        assert total == config['n_layer'], (
            f"n_prelude + n_recurrent + n_coda = {total} but n_layer = {config['n_layer']}"
        )
        assert self.n_recurrent >= 1
        self.r_default = rc.get('r_default', 4)
        self.r_max = rc.get('r_max', 16)
        self.backprop_depth = rc.get('backprop_depth', 8)
        self.state_init = rc.get('state_init', 'random')
        self.state_init_std = rc.get('state_init_std', 0.02)
        self.state_generator = None

        n_embd = config['n_embd']
        self.use_adapter = rc.get('adapter', True)
        if self.use_adapter:
            self.adapter = nn.Linear(2 * n_embd, n_embd, bias=False)
            torch.nn.init.uniform_(
                self.adapter.weight, -(3 ** 0.5) * n_embd ** -0.5, (3 ** 0.5) * n_embd ** -0.5
            )
        else:
            self.adapter = None

        self.prelude_idx = list(range(self.n_prelude))
        self.recurrent_idx = list(range(self.n_prelude, self.n_prelude + self.n_recurrent))
        self.coda_idx = list(range(self.n_prelude + self.n_recurrent, total))

    def cache_num_layers(self, r=None):
        r = self.r_max if r is None else r
        return self.n_prelude + self.n_recurrent * r + self.n_coda

    def init_state(self, e):
        if self.state_init != 'random':
            return e
        if self.state_generator is None:
            return torch.randn_like(e) * self.state_init_std
        return self.state_init_std * torch.randn(
            e.shape, generator=self.state_generator, device=e.device, dtype=e.dtype
        )

    def extra_parameters(self):
        return [] if self.adapter is None else [self.adapter.weight]

    def _run(self, x, indices, cos_sin, kv_cache, cache_base, x0):
        for offset, i in enumerate(indices):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            slot = None if cache_base is None else cache_base + offset
            x = self.h[i](x, cos_sin, self.window_sizes[i], kv_cache, slot)
        return x

    def _inject(self, s, e):
        if self.adapter is None:
            return s + e
        return self.adapter(torch.cat([s, e], dim=-1))

    def forward(self, idx, targets=None, kv_cache=None, r=None, loss_reduction='mean'):
        B, T = idx.size()
        r = self.r_default if r is None else int(r)
        assert r >= 1

        assert T <= self.cos.size(1), f"Sequence length {T} exceeds rotary embeddings cache {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices"
        assert self.cos.dtype == torch.bfloat16, "Rotary embeddings must be in bfloat16"
        if self.wte.weight.dtype != self.body_dtype():
            assert torch.is_autocast_enabled(idx.device.type), (
                f"embedding_dtype={self.config.get('embedding_dtype')!r} keeps wte in "
                f"{self.wte.weight.dtype} while the blocks are in {self.body_dtype()}. "
                f"That split only works inside autocast, which is not active here. "
                f"Wrap this call in torch.amp.autocast, or drop embedding_dtype from the "
                f"config to let wte follow the blocks."
            )

        if kv_cache is not None:
            need = self.cache_num_layers(r)
            assert kv_cache.num_layers >= need, (
                f"kv cache holds {kv_cache.num_layers} layer slots but r={r} needs "
                f"{need} (prelude {self.n_prelude} + {self.n_recurrent}x{r} + coda "
                f"{self.n_coda}). Size the cache with cache_num_layers(r) for the same r "
                f"you generate with."
            )

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        assert T0 + T <= self.cos.size(1), (
            f"position {T0}+{T} exceeds the rotary cache of {self.cos.size(1)}; the "
            f"slice would silently come back short and break the attention shapes"
        )
        cos_sin = self.cos[:, T0:T0 + T], self.sin[:, T0:T0 + T]

        x = norm(self.wte(idx))
        x0 = x

        use_cache = kv_cache is not None
        e = self._run(x, self.prelude_idx, cos_sin, kv_cache, 0 if use_cache else None, x0)

        s = self.init_state(e)

        n_grad = min(self.backprop_depth, r)
        n_nograd = r - n_grad

        for step in range(n_nograd):
            base = self.n_prelude + step * self.n_recurrent if use_cache else None
            with torch.no_grad():
                s = self._inject(s, e)
                s = self._run(s, self.recurrent_idx, cos_sin, kv_cache, base, e)
            s = s.detach()

        for step in range(n_nograd, r):
            base = self.n_prelude + step * self.n_recurrent if use_cache else None
            s = self._inject(s, e)
            s = self._run(s, self.recurrent_idx, cos_sin, kv_cache, base, e)

        coda_base = self.n_prelude + r * self.n_recurrent if use_cache else None
        x = self._run(s, self.coda_idx, cos_sin, kv_cache, coda_base, e)

        if kv_cache is not None:
            kv_cache.advance(T)

        x = norm(x)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config['vocab_size']]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            return F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1),
                ignore_index=-1, reduction=loss_reduction,
            )
        return logits
