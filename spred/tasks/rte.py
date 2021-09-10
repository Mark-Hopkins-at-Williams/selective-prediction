import datasets
from transformers import AutoTokenizer
from spred.loader import Loader
from torch.utils.data import DataLoader
from spred.task import TaskFactory


class TokenizerCache:
    def __init__(self):
        self.tokenizers = dict()

    def load(self, tokenizer_name):
        if tokenizer_name not in self.tokenizers:
            self.tokenizers[tokenizer_name] = AutoTokenizer.from_pretrained(tokenizer_name)
        return self.tokenizers[tokenizer_name]


tokenizer_cache = TokenizerCache()


class RteLoader(Loader):

    def __init__(self, bsz, split, tokenizer):
        super().__init__()
        self.split = split
        self.bsz = bsz
        raw_datasets = datasets.load_dataset('glue', 'rte')
        self.tokenizer = tokenizer_cache.load(tokenizer)
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        dataset = tokenized_datasets[split]
        shuffle = (split == "train")
        self.dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bsz)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def input_size(self):
        return 512  # this is max length of the sentences, not the embedding size

    def output_size(self):
        return 2


class RteTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        bsz = self.config['trainer']['bsz']
        tokenizer = self.config['network']['base_model']
        return RteLoader(bsz, split="train", tokenizer=tokenizer)

    def val_loader_factory(self):
        bsz = self.config['trainer']['bsz']
        tokenizer = self.config['network']['base_model']
        return RteLoader(bsz, split="validation", tokenizer=tokenizer)