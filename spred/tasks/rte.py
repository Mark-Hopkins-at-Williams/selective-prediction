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


class RteLoader(Loader):

    def __init__(self, dataset, shuffle, bsz):
        super().__init__()
        # self.dataloader = DataLoader(dataset, shuffle=shuffle, batch_size=bsz)

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
        self.bsz = None
        self.tokenizer = None
        self.train = None
        self.cotrain = None
        self.test = None
        self.initialize_datasets()

    def initialize_datasets(self):
        try:
            _ = self.train
        except AttributeError:
            tokenizer = self.config['network']['base_model']
            self.bsz = self.config['bsz']
            raw_datasets = datasets.load_dataset('glue', 'rte')
            self.tokenizer = tokenizer_cache.load(tokenizer)
            tokenized_datasets = raw_datasets.map(self.tokenize_function, batched=True)
            tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
            tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
            tokenized_datasets.set_format("torch")
            train_dataset = tokenized_datasets['train']
            train_size = int(.5 * len(train_dataset))
            cotrain_size = len(train_dataset) - train_size
            self.train, self.cotrain = random_split(train_dataset, [train_size, cotrain_size])
            self.test = tokenized_datasets['validation']

    def tokenize_function(self, examples):
        return self.tokenizer(examples["sentence1"], examples["sentence2"], padding="max_length", truncation=True)

    def train_loader_factory(self):
        self.initialize_datasets()
        return RteLoader(self.train, shuffle=True, bsz=self.bsz)

    def validation_loader_factory(self):
        self.initialize_datasets()
        return RteLoader(self.cotrain, shuffle=True, bsz=self.bsz)

    def test_loader_factory(self):
        self.initialize_datasets()
        return RteLoader(self.test, shuffle=False, bsz=self.bsz)