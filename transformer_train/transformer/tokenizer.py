import os
import pickle
from functools import lru_cache
from typing import List, Optional, Union
import torch

# Special tokens for chat format (we'll use these for chat inference later)
SPECIAL_TOKENS = [
    # every document begins with the Beginning of Sequence (BOS) token that delimits documents
    "<|bos|>",
    # End of Sequence (EOS) token marks the end of a document
    "<|eos|>",
    # tokens below are only used during finetuning to render Conversations into token ids
    "<|user_start|>",  # user messages
    "<|user_end|>",
    "<|assistant_start|>",  # assistant messages
    "<|assistant_end|>",
    "<|python_start|>",  # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>",  # python REPL outputs back to assistant
    "<|output_end|>",
]

# NOTE: this split pattern deviates from GPT-4 in that we use \p{N}{1,2} instead of \p{N}{1,3}
# This is better for smaller vocab sizes
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# Try to import rustbpe and tiktoken
try:
    import rustbpe
    import tiktoken
    RUSTBPE_AVAILABLE = True
except ImportError:
    RUSTBPE_AVAILABLE = False
    raise ImportError(
        "rustbpe and tiktoken are required. "
        "Install rustbpe (build from source or install as package) and tiktoken: pip install tiktoken"
    )


class RustBPETokenizer:
    def __init__(self, enc, bos_token="<|bos|>", eos_token="<|eos|>"):
        if not RUSTBPE_AVAILABLE:
            raise ImportError("rustbpe and tiktoken not available. Install with: pip install rustbpe tiktoken")
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)
        self.eos_token_id = self.encode_special(eos_token)

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        if not RUSTBPE_AVAILABLE:
            raise ImportError("rustbpe and tiktoken not available. Install with: pip install rustbpe tiktoken")
        # 1) train using rustbpe
        tokenizer = rustbpe.Tokenizer()
        # the special tokens are inserted later in __init__, we don't train them here
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        # 2) construct the associated tiktoken encoding for inference
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,  # dict[bytes, int] (token bytes -> merge priority rank)
            special_tokens=special_tokens,  # dict[str, int] (special token name -> token id)
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        if not RUSTBPE_AVAILABLE:
            raise ImportError("rustbpe and tiktoken not available. Install with: pip install rustbpe tiktoken")
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Tokenizer file not found: {pickle_path}")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        if not RUSTBPE_AVAILABLE:
            raise ImportError("rustbpe and tiktoken not available. Install with: pip install rustbpe tiktoken")
        # https://github.com/openai/tiktoken/blob/eedc8563/tiktoken_ext/openai_public.py
        enc = tiktoken.get_encoding(tiktoken_name)
        # tiktoken calls the special document delimiter token "<|endoftext|>"
        return cls(enc, "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id
    
    def get_eos_token_id(self):
        return self.eos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8, add_special_tokens=False):
        # For backward compatibility, if add_special_tokens is True, prepend BOS
        if add_special_tokens and prepend is None:
            prepend = self.get_bos_token_id()

        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids, skip_special_tokens=False):
        # Tiktoken already handles special tokens correctly
        # Manual replacement is redundant and can cause issues
        text = self.enc.decode(ids)
        return text

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()
    
    def render_conversation(self, conversation, max_tokens=2048):
        import copy
        # ids, masks that we will return and a helper function to help build them up.
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        # sometimes the first message is a system message...
        # => just merge it with the second (user) message
        if conversation["messages"][0]["role"] == "system":
            # some conversation surgery is necessary here for now...
            conversation = copy.deepcopy(conversation) # avoid mutating the original
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", "System message must be followed by a user message"
            messages[1]["content"] = messages[0]["content"] + "\n\n" + messages[1]["content"]
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        # fetch all the special tokens we need
        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        # now we can tokenize the conversation
        add_tokens(bos, 0)
        for i, message in enumerate(messages):

            # some sanity checking here around assumptions, to prevent footguns
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, f"Message {i} is from {message['role']} but should be from {must_be_from}"

            # content can be either a simple string or a list of parts (e.g. containing tool calls)
            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), "User messages are simply expected to be strings"
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    # simple string => simply add the tokens
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            # string part => simply add the tokens
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            # python tool call => add the tokens inside <|python_start|> and <|python_end|>
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            # python output => add the tokens inside <|output_start|> and <|output_end|>
                            # none of these tokens are supervised because the tokens come from Python at test time
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        # truncate to max_tokens tokens MAX (helps prevent OOMs)
        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        import copy
        # We have some surgery to do: we need to pop the last message (of the Assistant)
        conversation = copy.deepcopy(conversation) # avoid mutating the original
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", "Last message must be from the Assistant"
        messages.pop() # remove the last message (of the Assistant) inplace

        # Now tokenize the conversation
        ids, mask = self.render_conversation(conversation)

        # Finally, to prime the Assistant for a completion, append the Assistant start token
        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


# -----------------------------------------------------------------------------
# Main tokenizer factory function (backward compatible)
def create_tokenizer(tokenizer_dir: str = None, model_path: str = None):
    if not RUSTBPE_AVAILABLE:
        raise ImportError(
            "rustbpe and tiktoken are required. "
            "Please install rustbpe (build from source or install as package) and tiktoken: pip install tiktoken"
        )
    
    # Try tokenizer_dir first
    if tokenizer_dir and os.path.exists(tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        if os.path.exists(pickle_path):
            return RustBPETokenizer.from_directory(tokenizer_dir)
    
    # Try common default locations
    default_dirs = [
        "tokenizer",
        "turkish_tokenizer",
        "out/tokenizer",
    ]
    for dir_path in default_dirs:
        if os.path.exists(dir_path):
            pickle_path = os.path.join(dir_path, "tokenizer.pkl")
            if os.path.exists(pickle_path):
                return RustBPETokenizer.from_directory(dir_path)
    
    # Legacy support: try SentencePiece if model_path is provided
    if model_path and os.path.exists(model_path):
        try:
            import sentencepiece as smp
            print("Warning: Using legacy SentencePiece tokenizer. Please train a RustBPE tokenizer instead.")
            sp = smp.SentencePieceProcessor()
            sp.load(model_path)
            
            class TokenizerWrapper:
                def __init__(self, sp_processor):
                    self.sp = sp_processor
                    
                def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
                    if add_special_tokens:
                        return [self.sp.bos_id()] + self.sp.encode_as_ids(text)
                    else:
                        return self.sp.encode_as_ids(text)
                
                def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
                    if skip_special_tokens:
                        filtered_ids = [
                            tid for tid in token_ids 
                            if tid not in [self.sp.pad_id(), self.sp.bos_id(), self.sp.eos_id(), self.sp.unk_id()]
                        ]
                        return self.sp.decode_ids(filtered_ids)
                    else:
                        return self.sp.decode_ids(token_ids)
                
                @property
                def vocab_size(self) -> int:
                    return self.sp.vocab_size()
                
                def get_vocab_size(self) -> int:
                    return self.sp.vocab_size()
            
            return TokenizerWrapper(sp)
        except ImportError:
            pass
    
    raise FileNotFoundError(
        f"Could not find tokenizer. Please provide tokenizer_dir or ensure tokenizer.pkl exists. "
        f"Expected locations: {default_dirs}"
    )


def get_token_bytes(tokenizer, device="cpu"):
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    
    # Get special token IDs
    special_tokens = tokenizer.get_special_tokens()
    special_token_ids = set()
    for token in special_tokens:
        try:
            token_id = tokenizer.encode_special(token)
            if isinstance(token_id, list):
                special_token_ids.update(token_id)
            else:
                special_token_ids.add(token_id)
        except:
            pass
    
    for token_id in range(vocab_size):
        if token_id in special_token_ids:
            token_bytes[token_id] = 0  # Special tokens don't count
        else:
            try:
                # Decode the token and count bytes
                decoded = tokenizer.decode([token_id], skip_special_tokens=False)
                # Count UTF-8 bytes
                token_bytes[token_id] = len(decoded.encode('utf-8'))
            except:
                token_bytes[token_id] = 0  # Fallback to 0 if decoding fails
    
    return token_bytes
