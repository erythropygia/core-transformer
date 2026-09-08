import json
import os

from safetensors import safe_open
from safetensors.torch import load_file

from .config import MODEL_CONFIG
from .common import print0
from .tokenizer import create_tokenizer
from .model.transformer_block import Transformer
from .model.recurrent import RecurrentTransformer


def load_model_from_checkpoint(checkpoint_path, device, eval_mode=True):
    print0(f"Loading model from: {checkpoint_path}")

    with safe_open(checkpoint_path, framework="pt") as f:
        metadata = f.metadata() or {}

    config = json.loads(metadata.get('config', '{}')) or MODEL_CONFIG.copy()

    tokenizer = create_tokenizer(tokenizer_dir=metadata.get('tokenizer_path', 'tokenizer'))
    config['vocab_size'] = tokenizer.vocab_size

    recurrent = bool(config.get('recurrent', {}).get('enabled', False))
    cls = RecurrentTransformer if recurrent else Transformer
    model = cls(config, tokenizer).to(device)

    state = load_file(checkpoint_path)
    prefix = '_orig_mod.'
    state = {(k[len(prefix):] if k.startswith(prefix) else k): v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    parameters = {name for name, _ in model.named_parameters()}
    absent = [k for k in result.missing_keys if k in parameters]
    if absent:
        raise RuntimeError(
            f"{checkpoint_path} is missing {len(absent)} trained tensors for "
            f"{cls.__name__}: {absent[:8]}{' ...' if len(absent) > 8 else ''}. "
            f"strict=False would have loaded a partly random model and reported nothing."
        )
    if result.unexpected_keys:
        print0(f"  ignoring {len(result.unexpected_keys)} unused tensors in checkpoint: "
               f"{result.unexpected_keys[:4]}")
    if recurrent:
        rc = config['recurrent']
        print0(f"  RecurrentTransformer: prelude {rc['n_prelude']} / recurrent "
               f"{rc['n_recurrent']} / coda {rc['n_coda']} | r_default {rc.get('r_default', 4)}")
    if eval_mode:
        model.eval()

    meta = dict(metadata)
    meta['model_config'] = config
    return model, tokenizer, meta


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    if not os.path.isdir(checkpoint_dir):
        return None
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.safetensors')]
    if not files:
        return None
    latest = max(files, key=lambda f: os.path.getmtime(os.path.join(checkpoint_dir, f)))
    return os.path.join(checkpoint_dir, latest)
