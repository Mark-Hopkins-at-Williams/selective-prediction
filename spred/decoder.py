import torch
from torch.nn import functional
from spred.util import cudaify
from tqdm import tqdm


class Decoder:

    def get_loss(self):
        if self.running_loss_denom == 0:
            return None
        return self.running_loss_total / self.running_loss_denom

    def make_predictions(self, outputs, labels, conf):
        raise NotImplementedError("**ABSTRACT METHOD**")

    def __call__(self, net, data, loss_f=None):
        net.eval()
        self.running_loss_total = 0.0
        self.running_loss_denom = 0
        for images, labels in tqdm(data, total=len(data)):
            with torch.no_grad():
                outputs, conf = net(cudaify(images))
            if loss_f is not None:
                loss = loss_f(outputs, conf, cudaify(labels))
                self.running_loss_total += loss.item()
                self.running_loss_denom += 1  # TODO: why 1 and not len(images)?
            for pred in self.make_predictions(outputs, labels, conf):
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
