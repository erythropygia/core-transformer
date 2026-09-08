import importlib

_EXPORTS = {
    'train': ('.training.train', 'train'),
    'cleanup_memory': ('.utils', 'cleanup_memory'),
    'get_gpu_memory_info': ('.utils', 'get_gpu_memory_info'),
    'get_memory_usage': ('.utils', 'get_memory_usage'),
    'MODEL_CONFIG': ('.config', 'MODEL_CONFIG'),
    'TRAINING_CONFIG': ('.config', 'TRAINING_CONFIG'),
    'TEST_PROMPTS': ('.config', 'TEST_PROMPTS'),
    'SHOW_SPECIAL_TOKENS': ('.config', 'SHOW_SPECIAL_TOKENS'),
    'create_tokenizer': ('.tokenizer', 'create_tokenizer'),
    'TransformerDataset': ('.data.dataset', 'TransformerDataset'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        return getattr(importlib.import_module(module, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
