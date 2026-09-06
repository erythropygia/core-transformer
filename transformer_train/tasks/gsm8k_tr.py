import re
from datasets import load_dataset
from .common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
def extract_answer(completion):
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K_TR(Task):
    def __init__(self, subset="main", split="train", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K_TR subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K_TR split must be train|test"

        try:
            self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        except Exception as e:
            print(f"Warning: Could not load GSM8K dataset: {e}")
            from datasets import Dataset
            self.ds = Dataset.from_dict({"question": [], "answer": []})

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row['question']
        answer = row['answer']

        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                inner = part[2:-2]
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                assistant_message_parts.append({"type": "python", "text": expr})
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                assistant_message_parts.append({"type": "text", "text": part})

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"

        if isinstance(assistant_message['content'], list):
            full_text = ""
            for part in assistant_message['content']:
                if part.get('type') == 'text':
                    full_text += part['text']
            ground_truth = extract_answer(full_text)
        else:
            ground_truth = extract_answer(assistant_message['content'])

        predicted = extract_answer(assistant_response)

        if ground_truth is None or predicted is None:
            return False
        return ground_truth == predicted
