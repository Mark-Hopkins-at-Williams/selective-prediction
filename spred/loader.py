from abc import ABC, abstractmethod
import torch
from torch import tensor
import random


class Loader(ABC):
    @abstractmethod
    def __iter__(self):
        ...

    @abstractmethod
    def __len__(self):
        ...

    @abstractmethod
    def input_size(self):
        ...

    @abstractmethod
    def output_size(self):
        ...


class CalibrationLoader(Loader):
    def __init__(self, predictor, base_loader):
        super().__init__()
        self.predictor = predictor
        self.base_loader = base_loader
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def __iter__(self):
        for batch in self.base_loader:
            revised_batch = {k: v.to(self.device) for k, v in batch.items()}
            self.predictor.eval()
            with torch.no_grad():
                model_out = self.predictor(revised_batch, compute_conf=False)
                outputs = model_out['outputs']
            preds = outputs.argmax(dim=1)
            golds = revised_batch['labels']
            correctness = (preds==golds).long()
            revised_batch['labels'] = correctness
            yield revised_batch

    def __len__(self):
        return len(self.base_loader)

    def input_size(self):
        return self.base_loader.input_size()

    def output_size(self):
        return 2


class BalancedLoader(Loader):
    def __init__(self, base_loader):
        self.base_loader = base_loader

    def __iter__(self):
        for batch in self.base_loader:
            rows_by_label = dict()
            lbls = batch['labels']
            for value in lbls.unique().cpu().numpy():
                mask = lbls == value
                row_indices = tensor(range(len(mask)))[mask]
                rows_by_label[value] = list(row_indices.numpy())
            sample_size = max([len(rows_by_label[lbl]) for lbl in rows_by_label])
            choices = []
            for lbl in rows_by_label:
                choices += random.choices(rows_by_label[lbl], k=sample_size)
            choices = tensor(choices)
            balanced_batch = dict()
            for key in batch:
                balanced_batch[key] = batch[key][choices]
            yield balanced_batch


    def __len__(self):
        return len(self.base_loader)

    def input_size(self):
        return self.base_loader.input_size()

    def output_size(self):
        return self.base_loader.output_size()