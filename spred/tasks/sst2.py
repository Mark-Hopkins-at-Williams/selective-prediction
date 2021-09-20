import datasets
from transformers import AutoTokenizer
from spred.loader import Loader
from torch.utils.data import DataLoader
from spred.task import TaskFactory


class Sst2Loader(Loader):

    def __init__(self, bsz, split, tokenizer):
        super().__init__()
        self.split = split
        self.bsz = bsz
        raw_datasets = datasets.load_dataset('glue', 'sst2')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)

        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

        tokenized_datasets.set_format("torch")
        dataset = tokenized_datasets[split]
        shuffle = (split == "train")
        self.dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bsz)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence"], padding="max_length", truncation=True)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch

    def __len__(self):
        return len(self.dataloader)

    def input_size(self):
        return 512  # this is max length of the sentences, not the embedding size

    def output_size(self):
        return 2


class Sst2TaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        bsz = self.config['bsz']
        tokenizer = self.config['network']['base_model']
        return Sst2Loader(bsz, split="train", tokenizer=tokenizer)

    def validation_loader_factory(self):
        bsz = self.config['bsz']
        tokenizer = self.config['network']['base_model']
        return Sst2Loader(bsz, split="validation", tokenizer=tokenizer)

    def test_loader_factory(self):
        bsz = self.config['bsz']
        tokenizer = self.config['network']['base_model']
        return Sst2Loader(bsz, split="validation", tokenizer=tokenizer)