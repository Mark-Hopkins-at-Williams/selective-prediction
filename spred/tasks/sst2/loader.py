import datasets
from transformers import AutoTokenizer
from spred.loader import Loader
from torch.utils.data import DataLoader


class Sst2Loader(Loader):

    def __init__(self, bsz, split, tokenizer="roberta"):
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
