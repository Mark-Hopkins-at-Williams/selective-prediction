import torch
from torch.nn import functional
from tqdm import tqdm
from datasets import load_metric
from spred.evaluate import Evaluator
from spred.util import softmax


def validate_and_analyze(model, validation_loader, task_name=None):
    decoder = Decoder(model.include_abstain())
    model.eval()
    results = list(decoder(model, validation_loader))
    validation_loss = decoder.get_loss()
    eval_result = Evaluator(results, validation_loss, task_name).get_result()
    return eval_result


class Decoder:

    def __init__(self, include_abstain_output):
        self.device = (torch.device("cuda") if torch.cuda.is_available()
                       else torch.device("cpu"))
        self.include_abstain_output = include_abstain_output

    def get_loss(self):
        if self.running_loss_denom == 0:
            return None
        return self.running_loss_total / self.running_loss_denom

    def make_predictions(self, outputs, labels, conf):
        if not self.include_abstain_output:
            preds = outputs.argmax(dim=1)
            for p, g, c in zip(preds, labels, conf):
                yield {'pred': p.item(), 'gold': g.item(),
                       'confidence': c.item(), 'non_abstain_prob': 1.0}
        else:
            output = softmax(outputs)
            abstain_probs = output[:, -1]
            preds = output.argmax(dim=-1)
            abs_i = output.shape[1] - 1
            preds[preds == abs_i] = -1
            no_abstain_preds = output[:, :-1].argmax(dim=-1)
            for element in zip(no_abstain_preds, labels, conf, abstain_probs):
                p, g, c, p2 = element
                result = {'pred': p.item(), 'gold': g.item(),
                          'confidence': c.item(), 'non_abstain_prob': (1.0-p2).item()}
                yield result

    def __call__(self, model, data, loss_f=None):
        self.running_loss_total = 0.0
        self.running_loss_denom = 0
        for batch in tqdm(data, total=len(data)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            model.eval()
            with torch.no_grad():
                model_out = model(batch, compute_conf=True)
                outputs, loss, conf = model_out['outputs'], model_out['loss'], model_out['confidences']
            self.running_loss_total += loss.item()
            self.running_loss_denom += 1
            for pred in self.make_predictions(outputs, batch['labels'], conf):
                yield pred

