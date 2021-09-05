import torch
from torch.nn import functional
from spred.util import cudaify
from tqdm import tqdm
from abc import ABC, abstractmethod
from datasets import load_metric


class Decoder(ABC):

    def __init__(self):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def get_loss(self):
        if self.running_loss_denom == 0:
            return None
        return self.running_loss_total / self.running_loss_denom

    @abstractmethod
    def make_predictions(self, outputs, labels, conf):
        ...

    def __call__(self, net, data, loss_f=None):
        net.eval()
        self.running_loss_total = 0.0
        self.running_loss_denom = 0
        for batch in tqdm(data, total=len(data)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs, conf = net(batch)
            if loss_f is not None:
                loss = loss_f(outputs, conf, batch['labels'])
                self.running_loss_total += loss.item()
                self.running_loss_denom += len(batch)
            for pred in self.make_predictions(outputs, batch['labels'], conf):
                yield pred


class InterfaceADecoder(Decoder):

    def make_predictions(self, outputs, labels, conf):
        preds = outputs.argmax(dim=1)
        for p, g, c in zip(preds, labels, conf):
            yield {'pred': p.item(), 'gold': g.item(),
                   'confidence': c.item(), 'abstain': False}


class InterfaceBDecoder(Decoder):

    def make_predictions(self, outputs, labels, conf):
        output = functional.softmax(outputs.clamp(min=-25, max=25), dim=1)
        preds = output.argmax(dim=-1)
        abs_i = output.shape[1] - 1
        preds[preds == abs_i] = -1
        no_abstain_preds = output[:, :-1].argmax(dim=-1)
        for element in zip(no_abstain_preds, labels, conf, preds):
            p, g, c, p2 = element
            result = {'pred': p.item(), 'gold': g.item(),
                      'confidence': c.item(), 'abstain': p2.item() == -1}
            yield result


class InterfaceCDecoder(Decoder):

    def make_predictions(self, outputs, labels, conf):
        preds = outputs[:, :-1].argmax(dim=-1)
        for p, g, c in zip(preds, labels, conf):
            result = {'pred': p.item(), 'gold': g.item(),
                      'confidence': c.item(), 'abstain': False}
            yield result
