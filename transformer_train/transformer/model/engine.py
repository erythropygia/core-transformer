import torch
import torch.nn.functional as F
from .kv_cache import KVCache


@torch.inference_mode()
def sample_next_token(logits, rng, temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0, generated_tokens=None):
    assert temperature >= 0.0, "temperature must be non-negative"
    assert top_p > 0.0 and top_p <= 1.0, "top_p must be in (0, 1]"
    assert repetition_penalty > 0.0, "repetition_penalty must be positive"

    if repetition_penalty != 1.0 and generated_tokens is not None and len(generated_tokens) > 0:
        unique_tokens = set(generated_tokens)
        for token_id in unique_tokens:
            if 0 <= token_id < logits.size(-1):
                if logits[0, token_id] > 0:
                    logits[0, token_id] /= repetition_penalty
                else:
                    logits[0, token_id] *= repetition_penalty

    if temperature == 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature

    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
        logits_filtered = torch.full_like(logits, float('-inf'))
        logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
        logits = logits_filtered

    if top_p < 1.0:
        probs = F.softmax(logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')

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

        stop_ids = set(self.tokenizer.get_stop_token_ids()) if hasattr(self.tokenizer, 'get_stop_token_ids') else set()

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

        generated_tokens_per_sample = [[] for _ in range(num_samples)]

        next_ids = sample_next_token(
            logits, rng, temperature, top_k, top_p, 
            1.0, None
        )
        sampled_tokens = next_ids[:, 0].tolist()

        kv_length_hint = (len(tokens) + max_tokens) if max_tokens is not None else m.get('block_size', 1024)
        kv_cache_decode = KVCache(
            batch_size=num_samples,
            seq_len=kv_length_hint,
            **kv_model_kwargs,
        )
        kv_cache_decode.prefill(kv_cache_prefill)
        del kv_cache_prefill

        row_states = [tokens.copy() for _ in range(num_samples)]
        completed = [False] * num_samples

        num_generated = 0
        first_iteration = True
        while True:
            if max_tokens is not None and num_generated >= max_tokens:
                break

            if first_iteration:
                sampled_tokens = [sampled_tokens[0]] * num_samples
                first_iteration = False
            else:
                logits = self.model.forward(ids, kv_cache=kv_cache_decode)
                logits = logits[:, -1, :]

                sampled_tokens = []
                for i in range(num_samples):
                    sample_logits = logits[i:i+1, :]
                    sample_generated = generated_tokens_per_sample[i] if repetition_penalty != 1.0 else None
                    next_id = sample_next_token(
                        sample_logits, rng, temperature, top_k, top_p,
                        repetition_penalty, sample_generated
                    )
                    sampled_tokens.append(next_id[0, 0].item())

            token_column = []
            token_masks = []
            for i, state in enumerate(row_states):
                next_token = sampled_tokens[i]
                token_masks.append(1)
                token_column.append(next_token)
                state.append(next_token)
                if repetition_penalty != 1.0:
                    generated_tokens_per_sample[i].append(next_token)
                if next_token in stop_ids:
                    completed[i] = True

            if all(completed):
                break

            yield token_column, token_masks
            num_generated += 1
            ids = torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)

    def generate_batch(self, tokens, num_samples=1, **kwargs):
        stop_ids = set(self.tokenizer.get_stop_token_ids()) if hasattr(self.tokenizer, 'get_stop_token_ids') else set()
        results = [tokens.copy() for _ in range(num_samples)]
        masks = [[0] * len(tokens) for _ in range(num_samples)]
        completed = [False] * num_samples
        for token_column, token_masks in self.generate(tokens, num_samples, **kwargs):
            for i, (token, mask) in enumerate(zip(token_column, token_masks)):
                if not completed[i]:
                    if token in stop_ids:
                        completed[i] = True
                    else:
                        results[i].append(token)
                        masks[i].append(mask)
            if all(completed):
                break
        return results, masks
