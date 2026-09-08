import importlib

_EXPORTS = {
    'TransformerDataset': ('.dataset', 'TransformerDataset'),
    'load_and_preprocess_data': ('.dataset', 'load_and_preprocess_data'),
    'SFTDataset': ('.sft_dataset', 'SFTDataset'),
    'tokenizing_distributed_data_loader': ('.dataloader', 'tokenizing_distributed_data_loader'),
    'tokenizing_distributed_data_loader_with_state': ('.dataloader', 'tokenizing_distributed_data_loader_with_state'),
    'tokenizing_distributed_data_loader_bos_bestfit': ('.dataloader', 'tokenizing_distributed_data_loader_bos_bestfit'),
    'create_mid_datasets': ('.dataset_utils', 'create_mid_datasets'),
    'create_sft_datasets': ('.dataset_utils', 'create_sft_datasets'),
    'list_parquet_files': ('.dataset_utils', 'list_parquet_files'),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        module, attribute = _EXPORTS[name]
        return getattr(importlib.import_module(module, __name__), attribute)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
