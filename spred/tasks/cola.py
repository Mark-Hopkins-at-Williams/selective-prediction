import datasets
from transformers import AutoTokenizer
from spred.loader import Loader
from torch.utils.data import DataLoader
from spred.task import TaskFactory


class ColaLoader(Loader):

    def __init__(self, bsz, split):
        super().__init__()
        self.split = split
        self.bsz = bsz
        raw_datasets = datasets.load_dataset('glue', 'cola')
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
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

    def restart(self):
        return ColaLoader(self.bsz, self.split)


class ColaTaskFactory(TaskFactory):

    def __init__(self, config):
        super().__init__(config)
        self.architecture = self.config['network']['architecture']

    def train_loader_factory(self):
        bsz = self.config['bsz']
        return ColaLoader(bsz, split="train")

    def val_loader_factory(self):
        bsz = self.config['bsz']
        return ColaLoader(bsz, split="validation")