import os
from typing import List, Iterator, Dict, Any
import pyarrow.parquet as pq


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
    def __init__(self, split="train", max_samples=None, data_dir="mid_data/wikipedia", text_column="text", **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
        self.data_dir = data_dir
        self.text_column = text_column
    
    def __iter__(self):
        try:
            parquet_files = list_parquet_files(self.data_dir)
            if not parquet_files:
                print(f"Warning: No parquet files found in {self.data_dir}")
                return
            
            count = 0
            for parquet_file in parquet_files:
                if self.max_samples and count >= self.max_samples:
                    break
                
                pf = pq.ParquetFile(parquet_file)
                for batch in pf.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    for _, row in df.iterrows():
                        if self.max_samples and count >= self.max_samples:
                            break
                        text = row.get(self.text_column, '')
                        if text and len(str(text)) > 50:  # Minimum length filter
                            yield str(text)
                            count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish Wikipedia from {self.data_dir}: {e}")
            return


class TurkishNews(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, data_dir="mid_data/news", text_column="text", **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
        self.data_dir = data_dir
        self.text_column = text_column
    
    def __iter__(self):
        try:
            parquet_files = list_parquet_files(self.data_dir)
            if not parquet_files:
                print(f"Warning: No parquet files found in {self.data_dir}")
                return
            
            count = 0
            for parquet_file in parquet_files:
                if self.max_samples and count >= self.max_samples:
                    break
                
                pf = pq.ParquetFile(parquet_file)
                for batch in pf.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    for _, row in df.iterrows():
                        if self.max_samples and count >= self.max_samples:
                            break
                        text = row.get(self.text_column, '')
                        if text and len(str(text)) > 50:  # Minimum length filter
                            yield str(text)
                            count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish news from {self.data_dir}: {e}")
            return


class TurkishQA(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, data_dir="mid_data/qa", question_column="question", answer_column="answer", **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
        self.data_dir = data_dir
        self.question_column = question_column
        self.answer_column = answer_column
    
    def __iter__(self):
        try:
            parquet_files = list_parquet_files(self.data_dir)
            if not parquet_files:
                print(f"Warning: No parquet files found in {self.data_dir}")
                return
            
            count = 0
            for parquet_file in parquet_files:
                if self.max_samples and count >= self.max_samples:
                    break
                
                pf = pq.ParquetFile(parquet_file)
                for batch in pf.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    for _, row in df.iterrows():
                        if self.max_samples and count >= self.max_samples:
                            break
                        question = row.get(self.question_column, '')
                        answer = row.get(self.answer_column, '')
                        if question and answer:
                            # Format: "Soru: ... Cevap: ..."
                            text = f"Soru: {str(question)} Cevap: {str(answer)}"
                            yield text
                            count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish QA from {self.data_dir}: {e}")
            return


class TurkishMath(TurkishDataset):    
    def __init__(self, split="train", max_samples=None, data_dir="mid_data/math", question_column="soru", answer_column="solution", **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
        self.data_dir = data_dir
        self.question_column = question_column
        self.answer_column = answer_column
    
    def __iter__(self):
        try:
            parquet_files = list_parquet_files(self.data_dir)
            if not parquet_files:
                print(f"Warning: No parquet files found in {self.data_dir}")
                return
            
            count = 0
            for parquet_file in parquet_files:
                if self.max_samples and count >= self.max_samples:
                    break
                
                pf = pq.ParquetFile(parquet_file)
                for batch in pf.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    for _, row in df.iterrows():
                        if self.max_samples and count >= self.max_samples:
                            break
                        question = row.get(self.question_column, '')
                        answer = row.get(self.answer_column, '')
                        if question and answer:
                            # Format: "Soru: ... Cevap: ..."
                            text = f"Soru: {str(question)} Cevap: {str(answer)}"
                            yield text
                            count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish math from {self.data_dir}: {e}")
            return


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
    def __init__(self, split="train", max_samples=None, data_dir="sft_data", messages_column="messages", **kwargs):
        super().__init__(split, **kwargs)
        self.max_samples = max_samples
        self.data_dir = data_dir
        self.messages_column = messages_column
    
    def __iter__(self):
        try:
            parquet_files = list_parquet_files(self.data_dir)
            if not parquet_files:
                print(f"Warning: No parquet files found in {self.data_dir}")
                return
            
            count = 0
            for parquet_file in parquet_files:
                if self.max_samples and count >= self.max_samples:
                    break
                
                pf = pq.ParquetFile(parquet_file)
                for batch in pf.iter_batches(batch_size=1000):
                    df = batch.to_pandas()
                    for _, row in df.iterrows():
                        if self.max_samples and count >= self.max_samples:
                            break
                        
                        # Parquet'ten messages kolonunu al
                        messages = row.get(self.messages_column, [])
                        
                        # Validate messages format
                        if isinstance(messages, list) and len(messages) >= 2:
                            # Ensure all messages have 'role' and 'content'
                            valid_messages = []
                            for msg in messages:
                                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                                    # Clean content
                                    content = str(msg.get('content', '')).strip()
                                    if content:
                                        valid_messages.append({
                                            "role": str(msg.get('role', '')).strip(),
                                            "content": content
                                        })
                            
                            # Only yield if we have at least 2 valid messages
                            if len(valid_messages) >= 2:
                                yield {"messages": valid_messages}
                                count += 1
        except Exception as e:
            print(f"Warning: Could not load Turkish chat from {self.data_dir}: {e}")
            return




def create_mid_datasets(config: Dict[str, Any]) -> TaskMixture:
    datasets = []
    for ds_config in config.get('datasets', []):
        ds_type = ds_config.get('type')
        split = ds_config.get('split', 'train')
        max_samples = ds_config.get('max_samples')
        
        if ds_type == 'wikipedia':
            datasets.append(TurkishWikipedia(
                split=split, 
                max_samples=max_samples,
                data_dir=ds_config.get('data_dir', 'mid_data/wikipedia'),
                text_column=ds_config.get('text_column', 'text')
            ))
        elif ds_type == 'news':
            datasets.append(TurkishNews(
                split=split, 
                max_samples=max_samples,
                data_dir=ds_config.get('data_dir', 'mid_data/news'),
                text_column=ds_config.get('text_column', 'text')
            ))
        elif ds_type == 'qa':
            datasets.append(TurkishQA(
                split=split, 
                max_samples=max_samples,
                data_dir=ds_config.get('data_dir', 'mid_data/qa'),
                question_column=ds_config.get('question_column', 'question'),
                answer_column=ds_config.get('answer_column', 'answer')
            ))
        elif ds_type == 'math':
            datasets.append(TurkishMath(
                split=split, 
                max_samples=max_samples,
                data_dir=ds_config.get('data_dir', 'mid_data/math'),
                question_column=ds_config.get('question_column', 'soru'),
                answer_column=ds_config.get('answer_column', 'solution')
            ))
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
            datasets.append(TurkishChat(
                split=split, 
                max_samples=max_samples,
                data_dir=ds_config.get('data_dir', 'sft_data'),
                messages_column=ds_config.get('messages_column', 'messages')
            ))
        else:
            print(f"Warning: Unknown SFT dataset type: {ds_type}")
    
    return datasets

