from datasets import load_dataset
from .common import Task, render_mc

class ARC_TR(Task):
    def __init__(self, subset="ARC-Easy", split="train", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["ARC-Easy", "ARC-Challenge"], "ARC_TR subset must be ARC-Easy or ARC-Challenge"
        assert split in ["train", "validation", "test"], "ARC_TR split must be train|validation|test"
        
        try:
            self.ds = load_dataset("allenai/ai2_arc", subset, split=split).shuffle(seed=42)
        except Exception as e:
            print(f"Warning: Could not load ARC dataset: {e}")
            from datasets import Dataset
            self.ds = Dataset.from_dict({
                "question": [],
                "choices": {"text": [], "label": []},
                "answerKey": []
            })

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"]["text"] # the text of each choice
        answer_string = row["answerKey"] # e.g. "A", "B", "C", "D"
        letters = row["choices"]["label"] # e.g. ["A", "B", "C", "D"]
        assert answer_string in letters, f"ARC answer {answer_string} must be one of {letters}" # sanity check
        
        # create and return the Conversation object
        user_message = render_mc(question, letters, choices)
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": answer_string}
        ]
        conversation = {
            "messages": messages,
            "letters": letters, # useful during evaluation
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in conversation['letters'], f"ARC answer {assistant_response} is expected to be one of {conversation['letters']}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message

