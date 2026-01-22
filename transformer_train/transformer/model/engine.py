import torch
import torch.nn.functional as F
from collections import deque
from .kv_cache import KVCache


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0, generated_tokens=None):
    """
    Sample next token from logits.
    
    Args:
        logits: (batch_size, vocab_size) tensor of logits
        rng: Random number generator
        temperature: Sampling temperature (0.0 = greedy)
        top_k: Top-k sampling (None = no limit)
        top_p: Nucleus sampling threshold (1.0 = disabled)
        repetition_penalty: Penalty for repeated tokens (>1.0 = penalize, 1.0 = no penalty)
        generated_tokens: List of previously generated token IDs for repetition penalty
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    assert top_p > 0.0 and top_p <= 1.0, "top_p must be in (0, 1]"
    assert repetition_penalty > 0.0, "repetition_penalty must be positive"
    
    # Apply repetition penalty
    if repetition_penalty != 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
        # Create a set of unique generated tokens for efficiency
        unique_tokens = set(generated_tokens)
        # Apply penalty to logits of previously generated tokens
        for token_id in unique_tokens:
            if 0 <= token_id < logits.size(-1):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty
    
    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    
    # Apply temperature
    logits = logits / temperature
    
    # Apply top_k filtering if specified
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        # Create a mask to zero out non-top-k logits
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
        logits = logits_filtered
    
    # Apply top_p (nucleus) sampling
    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create mask for tokens to keep (cumulative prob <= top_p)
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 0] = False
        
        # Create a mask to zero out filtered tokens
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
    
    # Sample from the filtered distribution
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=rng)


class Engine:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(self, tokens, num_samples=1, max_tokens=None, temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0, seed=42):
        assert isinstance(tokens, list) and isinstance(tokens[0], int), "expecting list of ints"
        device = self.model.get_device()
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        # Get the special tokens we need
        get_special = lambda s: self.tokenizer.encode_special(s) if hasattr(self.tokenizer, 'encode_special') else None
        assistant_end = get_special("<|assistant_end|>")
        bos = self.tokenizer.get_bos_token_id() if hasattr(self.tokenizer, 'get_bos_token_id') else None

        # 1) Run a batch 1 prefill of the prompt tokens
        m = self.model.config
        kv_model_kwargs = {
            "num_heads": m.get('n_kv_head', m['n_head']),
            "head_dim": m['n_embd'] // m['n_head'],
            "num_layers": m['n_layer']
        }
        kv_cache_prefill = KVCache(
            batch_size=1,
            seq_len=len(tokens),
            **kv_model_kwargs,
        )
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        logits = self.model.forward(ids, kv_cache=kv_cache_prefill)
        logits = logits[:, -1, :]
        
        # Track generated tokens for repetition penalty (per sample)
        # Start with empty lists since we haven't generated anything yet
        generated_tokens_per_sample = [[] for _ in range(num_samples)]
        
        # Prefill: no repetition penalty yet (nothing generated)
        next_ids = sample_next_token(
            logits, rng, temperature, top_k, top_p, 
            1.0, None  # No repetition penalty on first token
        )  # (B, 1)
        sampled_tokens = next_ids[:, 0].tolist()

        # 2) Replicate the KV cache for each sample/row
        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else m.get('block_size', 1024)
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill  # no need to keep this memory around

        # 3) Initialize states for each sample
        row_states = [tokens.copy() for _ in range(num_samples)]
        completed = [False] * num_samples  # Track completion status

        # 4) Main generation loop
        num_generated = 0
        first_iteration = True
        while True:
            # Stop condition: we've reached max tokens
            if max_tokens is not None and num_generated >= max_tokens:
                break

            # Get sampled tokens - either from prefill or from forward pass
            if first_iteration:
                # Use the tokens we already sampled from prefill
                sampled_tokens = [sampled_tokens[0]] * num_samples  # Broadcast first token to all rows
                first_iteration = False
            else:
                # Forward the model and get the next token for each row
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)  # (B, T, vocab_size)
                logits = logits[:, -1, :]  # (B, vocab_size) at last time step
                
                # Sample tokens for each row (with repetition penalty per sample)
                sampled_tokens = []
                for i in range(num_samples):
                    sample_logits = logits[i:i+1, :]  # (1, vocab_size)
                    sample_generated = generated_tokens_per_sample[i] if repetition_penalty != 1.0 else None
                    next_id = sample_next_token(
                        sample_logits, rng, temperature, top_k, top_p,
                        repetition_penalty, sample_generated
                    )  # (1, 1)
                    sampled_tokens.append(next_id[0, 0].item())

            # Process each row: choose the next token
            token_column = []  # contains the next token id along each row
            token_masks = []  # contains the mask (was it sampled (1) or forced (0)?) along each row
            for i, state in enumerate(row_states):
                next_token = sampled_tokens[i]
                token_masks.append(1)  # all tokens are sampled for now
                token_column.append(next_token)
                # Update the state of this row to include the next token
                state.append(next_token)
                # Track generated tokens for repetition penalty
                if repetition_penalty != 1.0:
                    generated_tokens_per_sample[i].append(next_token)
                # Mark as completed on special tokens
                if assistant_end and next_token == assistant_end:
                    completed[i] = True
                if bos and next_token == bos:
                    completed[i] = True

            # Check if all samples are completed
            if all(completed):
                break

            # Yield the token column
            yield token_column, token_masks
            num_generated += 1
            # Prepare ids for next iteration
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        assistant_end = self.tokenizer.encode_special("<|assistant_end|>") if hasattr(self.tokenizer, 'encode_special') else None
        bos = self.tokenizer.get_bos_token_id() if hasattr(self.tokenizer, 'get_bos_token_id') else None
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if (assistant_end and token == assistant_end) or (bos and token == bos):
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            # Stop if all rows are completed
            if all(completed):
                break
        return results, masks

