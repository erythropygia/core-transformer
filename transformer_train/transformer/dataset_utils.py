import os
from typing import List, Iterator, Dict, Any
from datasets import load_dataset


def list_parquet_files(data_dir=None):
    if data_dir is None:
        # Try common locations
        possible_dirs = ["base_data", "data", "pretraining_data"]
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                data_dir = dir_path
                break
    
    if data_dir is None or not os.path.exists(data_dir):
        return []
    
    parquet_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    return parquet_files


# -----------------------------------------------------------------------------
# Base Training Datasets (Pretraining - Parquet files)
# -----------------------------------------------------------------------------

def get_base_dataset_config():
    return {
        'type': 'parquet',  # 'parquet' or 'huggingface'
        'data_dir': 'base_data',  # Directory containing parquet files
        'huggingface_dataset': None,  # If type is 'huggingface', specify dataset name
        'huggingface_config': None,  # Optional config name
        'text_column': 'text',  # Column name in parquet/dataset
        'max_samples': None,  # None = use all
    }


# -----------------------------------------------------------------------------
# Mid Training Datasets (Structured tasks - Turkish equivalents)
# -----------------------------------------------------------------------------

class TurkishDataset:    
    def __init__(self, split="train", **kwargs):
        self.split = split
        self.kwargs = kwargs
    
    def __iter__(self):
        raise NotImplementedError
    
    def __len__(self):
        raise NotImplementedError


class TurkishWikipedia(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
    
    def __iter__(self):
        try:
            dataset = load_dataset("wikipedia", "20220301.tr", split=self.split)
            count = 0
            for item in dataset:
                if self.max_samples and count >= self.max_samples:
                    break
                text = item.get('text', '')
                if text and len(text) > 100:
                    yield text
                    count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish Wikipedia: {e}")
            return


class TurkishNews(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
    
    def __iter__(self):
        try:
            # Try to load a Turkish news dataset
            # You can replace this with your actual dataset
            dataset = load_dataset("musabg/wikipedia-tr-summarization", split=self.split)
            count = 0
            for item in dataset:
                if self.max_samples and count >= self.max_samples:
                    break
                text = item.get('text', '')
                if text:
                    yield text
                    count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish news: {e}")
            return


class TurkishQA(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
    
    def __iter__(self):
        # Placeholder - replace with actual Turkish QA dataset
        # Format: "Soru: ... Cevap: ..."
        examples = [
            "Soru: Türkiye'nin başkenti neresidir? Cevap: Türkiye'nin başkenti Ankara'dır.",
            "Soru: İstanbul hangi kıtada yer alır? Cevap: İstanbul hem Avrupa hem de Asya kıtalarında yer alır.",
        ]
        count = 0
        for example in examples:
            if self.max_samples and count >= self.max_samples:
                break
            yield example
            count += 1


class TurkishMath(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
    
    def __iter__(self):
        # Placeholder - replace with actual Turkish math dataset
        examples = [
            "Soru: 5 + 3 kaçtır? Cevap: 5 + 3 = 8",
            "Soru: 10 * 4 kaçtır? Cevap: 10 * 4 = 40",
        ]
        count = 0
        for example in examples:
            if self.max_samples and count >= self.max_samples:
                break
            yield example
            count += 1


class TaskMixture:    
    def __init__(self, datasets: List[TurkishDataset]):
        self.datasets = datasets
    
    def __iter__(self):
        # Cycle through datasets
        import itertools
        for dataset in itertools.cycle(self.datasets):
            try:
                yield next(iter(dataset))
            except StopIteration:
                continue


# -----------------------------------------------------------------------------
# SFT Datasets (Conversation format)
# -----------------------------------------------------------------------------

class ConversationDataset:    
    def __init__(self, split="train", **kwargs):
        self.split = split
        self.kwargs = kwargs
    
    def __iter__(self):
        raise NotImplementedError


class TurkishChat(ConversationDataset):    
    def __init__(self, split="train", max_samples=None, **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
    
    def __iter__(self):
        # Placeholder - replace with actual Turkish chat dataset
        # Format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        examples = [
            {
                "messages": [
                    {"role": "user", "content": "Merhaba, nasılsın?"},
                    {"role": "assistant", "content": "Merhaba! Ben iyiyim, teşekkürler. Sen nasılsın?"}
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Türkiye'nin başkenti neresidir?"},
                    {"role": "assistant", "content": "Türkiye'nin başkenti Ankara'dır."}
                ]
            },
        ]
        count = 0
        for example in examples:
            if self.max_samples and count >= self.max_samples:
                break
            yield example
            count += 1


def get_mid_dataset_config():
    return {
        'datasets': [
            {'type': 'wikipedia', 'split': 'train', 'max_samples': 10000},
            {'type': 'news', 'split': 'train', 'max_samples': 5000},
            {'type': 'qa', 'split': 'train', 'max_samples': 2000},
            {'type': 'math', 'split': 'train', 'max_samples': 1000},
        ]
    }


def get_sft_dataset_config():
    return {
        'datasets': [
            {'type': 'chat', 'split': 'train', 'max_samples': 5000},
        ]
    }


def create_mid_datasets(config: Dict[str, Any]) -> TaskMixture:
    datasets = []
    for ds_config in config.get('datasets', []):
        ds_type = ds_config.get('type')
        split = ds_config.get('split', 'train')
        max_samples = ds_config.get('max_samples')
        
        if ds_type == 'wikipedia':
            datasets.append(TurkishWikipedia(split=split, max_samples=max_samples))
        elif ds_type == 'news':
            datasets.append(TurkishNews(split=split, max_samples=max_samples))
        elif ds_type == 'qa':
            datasets.append(TurkishQA(split=split, max_samples=max_samples))
        elif ds_type == 'math':
            datasets.append(TurkishMath(split=split, max_samples=max_samples))
        else:
            print(f"Warning: Unknown dataset type: {ds_type}")
    
    return TaskMixture(datasets)


def create_sft_datasets(config: Dict[str, Any]) -> List[ConversationDataset]:
    datasets = []
    for ds_config in config.get('datasets', []):
        ds_type = ds_config.get('type')
        split = ds_config.get('split', 'train')
        max_samples = ds_config.get('max_samples')
        
        if ds_type == 'chat':
            datasets.append(TurkishChat(split=split, max_samples=max_samples))
        else:
            print(f"Warning: Unknown SFT dataset type: {ds_type}")
    
    return datasets

