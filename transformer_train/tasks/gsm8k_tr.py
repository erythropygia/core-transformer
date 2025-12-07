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
        
        # Placeholder: Gerçek Türkçe dataset eklenebilir
        # Şimdilik İngilizce GSM8K kullanıyoruz ama Türkçe'ye çevrilebilir
        try:
            self.ds = load_dataset("openai/gsm8k", subset, split=split).shuffle(seed=42)
        except Exception as e:
            print(f"Warning: Could not load GSM8K dataset: {e}")
            # Fallback: Empty dataset
            from datasets import Dataset
            self.ds = Dataset.from_dict({"question": [], "answer": []})

    @property
    def eval_type(self):
        return 'generative'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """ Get a single problem from the dataset. """
        row = self.ds[index]
        question = row['question'] # string of the question prompt
        answer = row['answer'] # string of the full solution and the answer after #### marker
        
        # Create and return the Conversation object
        # This is tricky because GSM8K uses tool calls, which we need to parse here.
        assistant_message_parts = []
        parts = re.split(r'(<<[^>]+>>)', answer)
        for part in parts:
            if part.startswith('<<') and part.endswith('>>'):
                # This is a calculator tool call
                inner = part[2:-2]  # Remove << >>
                # Split on = to get expression and result
                if '=' in inner:
                    expr, result = inner.rsplit('=', 1)
                else:
                    expr, result = inner, ""
                # Add the tool call as a part
                assistant_message_parts.append({"type": "python", "text": expr})
                # Add the result as a part
                assistant_message_parts.append({"type": "python_output", "text": result})
            else:
                # Regular text in between tool calls
                assistant_message_parts.append({"type": "text", "text": part})
        
        messages = [
            {"role": "user", "content": question}, # note: simple string
            {"role": "assistant", "content": assistant_message_parts}, # note: list of parts (as dicts)
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert isinstance(assistant_response, str), "Assuming simple string response for now"
        # First extract the ground truth answer
        assistant_message = conversation['messages'][-1]
        assert assistant_message['role'] == "assistant", "Last message must be from the Assistant"
        
        # Extract ground truth answer from assistant message
        if isinstance(assistant_message['content'], list):
            # Content is a list of parts (with tool calls)
            # Find the #### marker in the text parts
            full_text = ""
            for part in assistant_message['content']:
                if part.get('type') == 'text':
                    full_text += part['text']
            ground_truth = extract_answer(full_text)
        else:
            # Content is a simple string
            ground_truth = extract_answer(assistant_message['content'])
        
        # Extract predicted answer from assistant_response
        predicted = extract_answer(assistant_response)
        
        # Compare (normalize both to strings for comparison)
        if ground_truth is None or predicted is None:
            return False
        return ground_truth == predicted

