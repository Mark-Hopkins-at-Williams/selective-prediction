import datasets
from transformers import AutoTokenizer
from spred.loader import Loader
from torch.utils.data import DataLoader, random_split
from spred.task import TaskFactory


class TokenizerCache:
    def __init__(self):
        self.tokenizers = dict()

    def load(self, tokenizer_name):
        if tokenizer_name not in self.tokenizers:
            self.tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        return self.tokenizers[tokenizer_name]


tokenizer_cache = TokenizerCache()


class GlueLoader(Loader):

    def __init__(self, dataset, shuffle, bsz, output_sz):
        super().__init__()
        self.dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bsz)
        self.output_sz = output_sz


    def __iter__(self):
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def input_size(self):
        return 512  # this is max length of the sentences, not the embedding size

    def output_size(self):
        return self.output_sz


class GlueTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.initialize_datasets()

    def initialize_datasets(self):
        try:
            _ = self.train
        except AttributeError:
            self.task_name = self.config['task']['name']
            self.task_to_keys = {
                "cola": ("sentence", None),
                "mnli": ("premise", "hypothesis"),
                "mrpc": ("sentence1", "sentence2"),
                "qnli": ("question", "sentence"),
                "qqp": ("question1", "question2"),
                "rte": ("sentence1", "sentence2"),
                "sst2": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                "wnli": ("sentence1", "sentence2"),
            }
            tokenizer = self.config['network']['base_model']
            self.bsz = self.config['bsz']
            raw_datasets = datasets.load_dataset('glue', self.task_name)
            self.tokenizer = tokenizer_cache.load(tokenizer)
            tokenized = raw_datasets.map(self.tokenize_function, batched=True)
            obsolete_cols = [key for key in self.task_to_keys[self.task_name]
                             if key is not None] + ["idx"]
            tokenized = tokenized.remove_columns(obsolete_cols)
            tokenized = tokenized.rename_column("label", "labels")
            tokenized.set_format("torch")
            train_dataset = tokenized['train']
            train_size = int(.5 * len(train_dataset))
            cotrain_size = len(train_dataset) - train_size
            self.train, self.cotrain = random_split(train_dataset,
                                                    [train_size, cotrain_size])
            self.test = tokenized['validation']

            is_regression = self.task_name == "stsb"
            if not is_regression:
                label_list = raw_datasets["train"].features["label"].names
                self.output_size = len(label_list)
            else:
                self.output_size = 1

    def tokenize_function(self, examples):
        key1, key2 = self.task_to_keys[self.task_name]
        if key2 is not None:
            return self.tokenizer(examples[key1], examples[key2],
                                  padding="max_length", truncation=True)
        else:
            return self.tokenizer(examples[key1],
                                  padding="max_length", truncation=True)

    def train_loader_factory(self):
        self.initialize_datasets()
        return GlueLoader(self.train, shuffle=True, bsz=self.bsz, output_sz=self.output_size)

    def validation_loader_factory(self):
        self.initialize_datasets()
        return GlueLoader(self.cotrain, shuffle=True, bsz=self.bsz, output_sz=self.output_size)

    def test_loader_factory(self):
        self.initialize_datasets()
        return GlueLoader(self.test, shuffle=False, bsz=self.bsz, output_sz=self.output_size)