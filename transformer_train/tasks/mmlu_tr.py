from datasets import load_dataset
from .common import Task, render_mc

class MMLU_TR(Task):
    letters = ('A', 'B', 'C', 'D')

    def __init__(self, subset="all", split="train", **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], f"subset {subset} must be all|auxiliary_train"
        assert split in ["train", "validation", "dev", "test"], f"split {split} must be train|validation|dev|test"
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"
        self.subset = subset
        self.split = split
        
        try:
            self.ds = load_dataset("cais/mmlu", subset, split=split).shuffle(seed=42)
            if subset == "auxiliary_train":
                # I don't understand why but the auxiliary_train rows have some weird additional 'train' wrapper
                self.ds = self.ds.map(lambda row: row['train'], remove_columns=['train'])
        except Exception as e:
            print(f"Warning: Could not load MMLU dataset: {e}")
            from datasets import Dataset
            self.ds = Dataset.from_dict({
                "question": [],
                "choices": [],
                "answer": [],
                "subject": []
            })

    @property
    def eval_type(self):
        return 'categorical'

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"] # the question text
        choices = row["choices"] # the text of each choice
        answer = row["answer"] # index of the answer, e.g. 0,1,2,3 (for A,B,C,D)
        subject = row["subject"] # e.g. "college_biology", "college_chemistry", etc.
        assert len(choices) == 4, "MMLU should have 4 choices"
        
        # create and return the Conversation object
        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        conversation = {
            "messages": messages,
            "subject": subject, # might be useful later for grouping metrics by subject
            "letters": self.letters, # useful during evaluation
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        assert assistant_response in self.letters, f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        assistant_message = conversation['messages'][-1]['content'] # e.g. "A"
        return assistant_response == assistant_message

