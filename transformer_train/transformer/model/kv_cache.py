import torch

class KVCache:
    def __init__(self, batch_size, num_heads, seq_len, head_dim, num_layers):
        """
        FA3-native KV Cache format: (B, T, H, D)
        This avoids transpose operations when using Flash Attention 3.
        """
        # FA3-native format: (num_layers, 2, batch_size, seq_len, num_heads, head_dim)
        self.kv_shape = (num_layers, 2, batch_size, seq_len, num_heads, head_dim)
        self.kv_cache = None
        self.cache_seqlens = None  # FA3 requires int32 tensor for position tracking
        self.pos = 0  # current position in time in the cache
        self.batch_size = batch_size
        self.num_layers = num_layers

    def reset(self):
        self.pos = 0
        if self.cache_seqlens is not None:
            self.cache_seqlens.zero_()

    def get_pos(self):
        return self.pos

    def prefill(self, other):
        # 1) validate the shapes
        assert self.kv_cache is None, "Cannot prefill a non-empty KV cache"
        assert other.kv_cache is not None, "Cannot prefill with a None KV cache"
        for ix, (dim1, dim2) in enumerate(zip(self.kv_shape, other.kv_shape)):
            # ix 0: num_layers, 1: k/v, 2: batch_size, 3: seq_len, 4: num_heads, 5: head_dim
            if ix in [0, 1, 4, 5]:
                # num_layers, k/v, num_heads, head_dim must match
                assert dim1 == dim2, f"Dim {ix} mismatch: {dim1} != {dim2}"
            elif ix == 2:
                # batch_size can be expanded
                assert dim1 == dim2 or dim2 == 1, f"Batch dim mismatch: {dim1} != {dim2}"
            elif ix == 3:
                # seq_len: self must be longer than other
                assert dim1 >= dim2, f"Seq len mismatch: {dim1} < {dim2}"
        # 2) initialize the cache
        dtype, device = other.kv_cache.dtype, other.kv_cache.device
        self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
        # 3) copy the data over
        # Shape: (num_layers, 2, batch_size, seq_len, num_heads, head_dim)
        # Copy only the filled portion (first other.pos tokens in seq_len dimension)
        self.kv_cache[:, :, :, :other.pos, :, :] = other.kv_cache[:, :, :, :other.pos, :, :]
        # 4) initialize cache_seqlens (required for FA3)
        # Expand batch_size if needed (from 1 to num_samples)
        if other.cache_seqlens is not None:
            # Copy the value from other, but expand to new batch_size
            pos_value = other.cache_seqlens[0].item() if other.cache_seqlens.numel() > 0 else other.pos
            self.cache_seqlens = torch.full((self.batch_size,), pos_value, dtype=torch.int32, device=device)
        else:
            # Initialize with current position
            self.cache_seqlens = torch.full((self.batch_size,), other.pos, dtype=torch.int32, device=device)
        # 5) update the pos
        self.pos = other.pos

    def get_layer_cache(self, layer_idx):
        """
        Get k_cache and v_cache for a specific layer (FA3-native format).
        Returns: (k_cache, v_cache) both of shape (B, T_max, H, D)
        """
        if self.kv_cache is None:
            return None, None
        return self.kv_cache[layer_idx, 0], self.kv_cache[layer_idx, 1]

    def advance(self, T_new):
        """
        Advance the cache position by T_new tokens.
        Called after the last layer processes the tokens.
        """
        self.pos += T_new
        if self.cache_seqlens is not None:
            self.cache_seqlens.fill_(self.pos)

    def init_cache(self, dtype, device):
        """
        Initialize the cache tensors on first use.
        FA3-native format: (B, T_max, H, D)
        """
        if self.kv_cache is None:
            self.kv_cache = torch.empty(self.kv_shape, dtype=dtype, device=device)
            # FA3 requires int32 tensor for cache position tracking
            self.cache_seqlens = torch.zeros(self.batch_size, dtype=torch.int32, device=device)

