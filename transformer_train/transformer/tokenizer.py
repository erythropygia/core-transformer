import os
import pickle

import torch

DOC_SEP = "<|endoftext|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
THINK_START = "<think>"
THINK_END = "</think>"
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
TOOL_RESPONSE_START = "<tool_response>"
TOOL_RESPONSE_END = "</tool_response>"
PAD = "<|pad|>"

SPECIAL_TOKENS = [
    DOC_SEP,
    IM_START,
    IM_END,
    THINK_START,
    THINK_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESPONSE_START,
    TOOL_RESPONSE_END,
    PAD,
]

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

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
    def __init__(self, enc):
        self.enc = enc
        self.doc_sep_id = self.encode_special(DOC_SEP)
        self.im_start_id = self.encode_special(IM_START)
        self.im_end_id = self.encode_special(IM_END)
        self.think_start_id = self.encode_special(THINK_START)
        self.think_end_id = self.encode_special(THINK_END)
        self.tool_call_start_id = self.encode_special(TOOL_CALL_START)
        self.tool_call_end_id = self.encode_special(TOOL_CALL_END)
        self.tool_response_start_id = self.encode_special(TOOL_RESPONSE_START)
        self.tool_response_end_id = self.encode_special(TOOL_RESPONSE_END)
        self.pad_id = self.encode_special(PAD)
        self.bos_token_id = self.doc_sep_id
        self.eos_token_id = self.doc_sep_id
        self.special_token_ids = {self.encode_special(t) for t in SPECIAL_TOKENS}

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rustbpe.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN)
        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc)

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    @classmethod
    def from_pretrained(cls, tiktoken_name):
        enc = tiktoken.get_encoding(tiktoken_name)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def get_eos_token_id(self):
        return self.eos_token_id

    def get_pad_token_id(self):
        return self.pad_id

    def get_stop_token_ids(self):
        return [self.doc_sep_id, self.im_end_id]

    def encode(self, text, prepend=None, append=None, num_threads=8, add_special_tokens=False):
        if add_special_tokens and append is None:
            append = self.eos_token_id
        prepend_id = None
        append_id = None
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.encode_special(prepend)
        if append is not None:
            append_id = append if isinstance(append, int) else self.encode_special(append)

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend_id is not None:
                ids.insert(0, prepend_id)
            if append_id is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend_id is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append_id is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids, skip_special_tokens=False):
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_token_ids]
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):
        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    @property
    def vocab_size(self) -> int:
        return self.get_vocab_size()

    def _render_part(self, part, ids, mask, supervised):
        ptype = part.get("type", "text")
        text = part.get("text", "")
        if ptype == "text":
            self._add(ids, mask, self.encode(text), supervised)
        elif ptype == "think":
            self._add(ids, mask, [self.think_start_id], supervised)
            self._add(ids, mask, self.encode(text), supervised)
            self._add(ids, mask, [self.think_end_id], supervised)
        elif ptype == "tool_call":
            self._add(ids, mask, [self.tool_call_start_id], supervised)
            self._add(ids, mask, self.encode(text), supervised)
            self._add(ids, mask, [self.tool_call_end_id], supervised)
        elif ptype == "tool_response":
            self._add(ids, mask, [self.tool_response_start_id], 0)
            self._add(ids, mask, self.encode(text), 0)
            self._add(ids, mask, [self.tool_response_end_id], 0)
        else:
            raise ValueError(f"Unknown part type: {ptype}")

    @staticmethod
    def _add(ids, mask, token_ids, mask_val):
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        ids.extend(token_ids)
        mask.extend([mask_val] * len(token_ids))

    def _render_header(self, ids, mask, role):
        self._add(ids, mask, [self.im_start_id], 0)
        self._add(ids, mask, self.encode(role + "\n"), 0)

    def render_conversation(self, conversation, max_tokens=2048):
        messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"
        ids, mask = [], []
        newline = self.encode("\n")

        for message in messages:
            role = message["role"]
            content = message["content"]
            supervised = 1 if role == "assistant" else 0
            self._render_header(ids, mask, role)

            reasoning = message.get("reasoning")
            if role == "assistant" and reasoning:
                self._render_part({"type": "think", "text": reasoning}, ids, mask, supervised)
                self._add(ids, mask, newline, supervised)

            if isinstance(content, str):
                self._add(ids, mask, self.encode(content), supervised)
            elif isinstance(content, list):
                for part in content:
                    self._render_part(part, ids, mask, supervised)
            else:
                raise ValueError(f"Unknown content type: {type(content)}")

            self._add(ids, mask, [self.im_end_id], supervised)
            self._add(ids, mask, newline, 0)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation, enable_thinking=False):
        messages = list(conversation["messages"])
        if messages and messages[-1]["role"] == "assistant":
            messages = messages[:-1]
        ids, _ = self.render_conversation({"messages": messages})
        ids.append(self.im_start_id)
        ids.extend(self.encode("assistant\n"))
        if enable_thinking:
            ids.append(self.think_start_id)
        return ids


DEFAULT_TOKENIZER_DIRS = ["tokenizer", "turkish_tokenizer", "out/tokenizer"]


def create_tokenizer(tokenizer_dir: str = None, model_path: str = None):
    candidates = []
    if tokenizer_dir:
        candidates.append(tokenizer_dir)
    candidates.extend(DEFAULT_TOKENIZER_DIRS)
    for d in candidates:
        if os.path.exists(os.path.join(d, "tokenizer.pkl")):
            return RustBPETokenizer.from_directory(d)
    raise FileNotFoundError(
        f"tokenizer.pkl not found in any of {candidates}. "
        f"Train one with tokenizer_train/train_tokenizer.py"
    )


def get_token_bytes(tokenizer, device="cpu"):
    vocab_size = tokenizer.get_vocab_size()
    token_bytes = torch.zeros(vocab_size, dtype=torch.int64, device=device)
    special_ids = getattr(tokenizer, "special_token_ids", set())
    for token_id in range(vocab_size):
        if token_id in special_ids:
            continue
        try:
            token_bytes[token_id] = len(tokenizer.decode([token_id]).encode("utf-8"))
        except Exception:
            token_bytes[token_id] = 0
    return token_bytes
