import importlib

_EXPORTS = {'train': ('.train', 'train')}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        return getattr(importlib.import_module(module, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
