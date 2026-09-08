import torch

class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        self.kv_shape = (num_layers, 2, batch_size, seq_len, num_heads, head_dim)
        self.kv_cache = None
        self.cache_seqlens = None
        self.pos = 0
        self.batch_size = batch_size
        self.num_layers = num_layers

    def reset(self):
        self.pos = 0
        if self.cache_seqlens is not None:
            self.cache_seqlens.zero_()

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            if ix in [0, 1, 4, 5]:
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 3:
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        self.kv_cache[:, :, :, :other.pos, :, :] = other.kv_cache[:, :, :, :other.pos, :, :]
        if other.cache_seqlens is not None:
            pos_value = other.cache_seqlens[0].item() if other.cache_seqlens.numel() > 0 else other.pos
            self.cache_seqlens = torch.full((self.batch_size,), pos_value, dtype=torch.int32, device=device)
        else:
            self.cache_seqlens = torch.full((self.batch_size,), other.pos, dtype=torch.int32, device=device)
        self.pos = other.pos

    def get_layer_cache(self, layer_idx):
        if self.kv_cache is None:
            return None, None
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(
                f"layer slot {layer_idx} outside a cache of {self.num_layers} slots; "
                f"a negative index would otherwise wrap and silently read the wrong layer."
            )
        return self.kv_cache[layer_idx, 0], self.kv_cache[layer_idx, 1]

    def advance(self, T_new):
        self.pos += T_new
        if self.cache_seqlens is not None:
            self.cache_seqlens.fill_(self.pos)

    def init_cache(self, dtype, device):
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
            self.cache_seqlens = torch.zeros(self.batch_size, dtype=torch.int32, device=device)
