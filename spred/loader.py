from abc import ABC, abstractmethod
import torch


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
        return None

    def output_size(self):
        return None

